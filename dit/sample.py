"""Sample images from a trained DiT checkpoint.

    python -m dit.sample --ckpt results/ckpt_final.pt --class-labels 207 360 88 \
        --cfg-scale 4.0 --num-steps 50 --out samples.png
"""

import argparse

import torch
from torchvision.utils import save_image

from .diffusion import GaussianDiffusion
from .model import DIT_MODELS


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path, device, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    bottleneck = targs.get("bottleneck_dim", 128)
    model = DIT_MODELS[targs["model"]](
        image_size=targs["image_size"],
        in_channels=3,
        num_classes=ckpt.get("num_classes", 1000),
        class_dropout_prob=targs["class_dropout_prob"],
        learn_sigma=not targs["no_learn_sigma"],
        bottleneck_dim=bottleneck if bottleneck and bottleneck > 0 else None,
    ).to(device)
    state = ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    return model, targs, ckpt


def main(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    model, targs, ckpt = load_model(args.ckpt, device, use_ema=not args.no_ema)
    diffusion = GaussianDiffusion(
        num_timesteps=targs["num_timesteps"],
        beta_schedule=targs["beta_schedule"],
        learn_sigma=not targs["no_learn_sigma"],
    )

    if args.class_labels:
        labels = args.class_labels
    else:
        labels = torch.randint(
            0, ckpt.get("num_classes", 1000), (args.num_samples,)
        ).tolist()
    y = torch.tensor(labels, device=device, dtype=torch.long)
    n = y.shape[0]
    shape = (n, 3, targs["image_size"], targs["image_size"])

    if args.sampler == "ddim":
        x = diffusion.ddim_sample_loop(
            model, shape, device, y=y, cfg_scale=args.cfg_scale,
            num_steps=args.num_steps, eta=args.eta, progress=True,
        )
    else:
        x = diffusion.p_sample_loop(
            model, shape, device, y=y, cfg_scale=args.cfg_scale, progress=True
        )

    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x.cpu(), args.out, nrow=args.nrow or int(n**0.5) or 1)
    print(f"wrote {args.out} ({n} samples, labels={labels})")


def build_parser():
    p = argparse.ArgumentParser(description="Sample from a DiT checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="samples.png")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--class-labels", type=int, nargs="*", default=None)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--num-steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--nrow", type=int, default=0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
