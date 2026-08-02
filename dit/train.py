"""Train a DiT on ImageNet (streamed or downloaded from the HuggingFace Hub).

Single-GPU and multi-GPU (torchrun) are both supported:

    python -m dit.train --data-repo ILSVRC/imagenet-1k --image-size 256
    torchrun --nproc_per_node=8 -m dit.train --global-batch-size 256
"""

import argparse
import copy
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import build_dataset, collate
from .diffusion import FlowMatching
from .model import DIT_MODELS
from .plotting import save_loss_curve, write_history_csv

logger = logging.getLogger("dit")


# -----------------------------------------------------------------------------
# Distributed / device helpers
# -----------------------------------------------------------------------------


def setup_distributed():
    """Initialise torch.distributed if launched via torchrun. Returns (rank, world, local)."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def pick_device(local_rank):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_main(rank):
    return rank == 0


# -----------------------------------------------------------------------------
# EMA
# -----------------------------------------------------------------------------


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1 - decay)


def requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def main(args):
    rank, world_size, local_rank = setup_distributed()
    device = pick_device(local_rank)
    torch.manual_seed(args.seed * world_size + rank)

    logging.basicConfig(
        level=logging.INFO if is_main(rank) else logging.WARNING,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results_dir = Path(args.results_dir)
    if is_main(rank):
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
        logger.info("results -> %s", results_dir)

    # -- data --------------------------------------------------------------
    dataset, num_classes = build_dataset(
        repo=args.data_repo,
        split=args.data_split,
        image_size=args.image_size,
        random_flip=not args.no_flip,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    if args.num_classes is not None:
        num_classes = args.num_classes

    if args.global_batch_size % world_size != 0:
        raise ValueError("global-batch-size must be divisible by the world size")
    local_batch_size = args.global_batch_size // world_size

    sampler = None
    if not args.streaming and world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
        )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=(sampler is None and not args.streaming),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    logger.info("dataset %s: %s classes", args.data_repo, num_classes)

    # -- model -------------------------------------------------------------
    if args.model not in DIT_MODELS:
        raise ValueError(f"unknown model {args.model}; choose from {list(DIT_MODELS)}")
    model = DIT_MODELS[args.model](
        image_size=args.image_size,
        in_channels=3,
        num_classes=num_classes,
        class_dropout_prob=args.class_dropout_prob,
        # x-prediction: the network outputs a clean-image estimate and there
        # is no reverse-process variance to learn.
        learn_sigma=False,
        bottleneck_dim=args.bottleneck_dim if args.bottleneck_dim > 0 else None,
    ).to(device)

    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("%s: %.1fM parameters, %d tokens", args.model, n_params / 1e6,
                model.x_embedder.num_patches)

    # The paper holds SNR fixed across resolutions by scaling the noise
    # endpoint proportionally: 1.0 at 256, 2.0 at 512, 4.0 at 1024.
    noise_scale = (
        args.noise_scale if args.noise_scale > 0 else args.image_size / 256.0
    )
    flow = FlowMatching(
        noise_scale=noise_scale,
        t_mu=args.t_mu,
        t_sigma=args.t_sigma,
        denom_clip=args.denom_clip,
    )
    logger.info(
        "flow matching: noise_scale %.2f, logit-normal t (mu %.2f, sigma %.2f)",
        noise_scale, args.t_mu, args.t_sigma,
    )

    ddp_model = model
    if world_size > 1:
        ddp_model = DDP(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    history = []  # (step, v_loss, x_mse) at each logging interval
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        # Carry the curve across the resume so the final plot covers the run.
        history = [tuple(row) for row in ckpt.get("history", [])]
        logger.info("resumed from %s at step %d", args.resume, start_step)

    # EMA starts from the (possibly resumed) weights.
    if not args.resume:
        ema.load_state_dict(model.state_dict())

    # -- loop --------------------------------------------------------------
    model.train()
    step = start_step
    running_loss, running_x_mse, running_n, t0 = 0.0, 0.0, 0, time.time()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        if args.streaming and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        logger.info("epoch %d", epoch)

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Linear warmup on the raw lr, applied before the step it governs.
            if args.warmup_steps > 0 and step < args.warmup_steps:
                for g in opt.param_groups:
                    g["lr"] = args.lr * (step + 1) / args.warmup_steps

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                # t is drawn from the logit-normal prior inside training_losses.
                terms = flow.training_losses(ddp_model, x, model_kwargs={"y": y})
                loss = terms["loss"].mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            update_ema(ema, model, decay=args.ema_decay)

            running_loss += loss.item()
            running_x_mse += terms["x_mse"].mean().item()
            running_n += 1
            step += 1

            if step % args.log_every == 0:
                stats = torch.tensor(
                    [running_loss, running_x_mse], device=device
                ) / max(running_n, 1)
                if world_size > 1:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    stats = stats / world_size
                imgs_per_sec = (
                    args.log_every * args.global_batch_size / (time.time() - t0)
                )
                logger.info(
                    "step %d | v-loss %.4f | x-mse %.4f | %.1f img/s",
                    step, stats[0].item(), stats[1].item(), imgs_per_sec,
                )
                history.append((step, stats[0].item(), stats[1].item()))
                running_loss, running_x_mse, running_n, t0 = 0.0, 0.0, 0, time.time()

            if step % args.ckpt_every == 0 and is_main(rank):
                path = results_dir / f"ckpt_{step:08d}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "args": vars(args),
                        "num_classes": num_classes,
                        "history": history,
                    },
                    path,
                )
                logger.info("saved %s", path)

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    if is_main(rank):
        path = results_dir / "ckpt_final.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "args": vars(args),
                "num_classes": num_classes,
                "history": history,
            },
            path,
        )
        logger.info("saved %s", path)

        write_history_csv(history, results_dir / "loss_history.csv")
        curve = results_dir / "loss_curve.jpg"
        if save_loss_curve(history, curve, log_every=args.log_every):
            logger.info("saved %s", curve)

    if world_size > 1:
        dist.destroy_process_group()


def build_parser():
    p = argparse.ArgumentParser(description="Train a DiT on ImageNet")
    # data
    p.add_argument("--data-repo", default="ILSVRC/imagenet-1k",
                   help="HuggingFace dataset repo id")
    p.add_argument("--data-split", default="train")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--streaming", action="store_true",
                   help="stream from the Hub instead of downloading the full dataset")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--num-classes", type=int, default=None,
                   help="override the class count inferred from the dataset")
    p.add_argument("--num-workers", type=int, default=8)
    # model
    p.add_argument("--model", default="DiT-B/16", choices=list(DIT_MODELS))
    p.add_argument("--class-dropout-prob", type=float, default=0.1)
    p.add_argument("--bottleneck-dim", type=int, default=128,
                   help="low-rank patch-embedding dim; 0 disables the bottleneck")
    # flow matching
    p.add_argument("--noise-scale", type=float, default=0.0,
                   help="noise endpoint std; 0 means auto (image_size / 256)")
    p.add_argument("--t-mu", type=float, default=-0.8,
                   help="mean of the logit-normal t sampler; lower = noisier")
    p.add_argument("--t-sigma", type=float, default=0.8)
    p.add_argument("--denom-clip", type=float, default=0.05,
                   help="lower bound on (1 - t) when converting to v-space")
    # optim
    p.add_argument("--global-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--epochs", type=int, default=1400)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    # bookkeeping
    p.add_argument("--results-dir", default="results")
    p.add_argument("--resume", default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
