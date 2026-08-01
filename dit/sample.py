"""Sample images from a trained DiT checkpoint.

Integrates the flow-matching ODE from t=0 (noise) to t=1 (data):

    python -m dit.sample --ckpt results/ckpt_final.pt --class-labels 207 360 88 \
        --cfg-scale 2.0 --num-steps 50 --solver heun --out samples.png
"""

import argparse

import torch
from torchvision.utils import save_image

from .diffusion import FlowMatching
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
    """Rebuild the model and the flow process from a training checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]

    bottleneck = targs.get("bottleneck_dim", 128)
    model = DIT_MODELS[targs["model"]](
        image_size=targs["image_size"],
        in_channels=3,
        num_classes=ckpt.get("num_classes", 1000),
        class_dropout_prob=targs["class_dropout_prob"],
        learn_sigma=False,
        bottleneck_dim=bottleneck if bottleneck and bottleneck > 0 else None,
    ).to(device)
    state = ckpt["ema"] if use_ema and "ema" in ckpt else ckpt["model"]
    model.load_state_dict(state)
    model.eval()

    # noise_scale must match training, or the ODE starts off-distribution.
    noise_scale = targs.get("noise_scale", 0.0) or targs["image_size"] / 256.0
    flow = FlowMatching(
        noise_scale=noise_scale,
        t_mu=targs.get("t_mu", -0.8),
        t_sigma=targs.get("t_sigma", 0.8),
        denom_clip=targs.get("denom_clip", 0.05),
    )
    return model, flow, targs, ckpt


def main(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    model, flow, targs, ckpt = load_model(args.ckpt, device, use_ema=not args.no_ema)
    num_classes = ckpt.get("num_classes", 1000)

    if args.class_labels:
        labels = args.class_labels
    else:
        labels = torch.randint(0, num_classes, (args.num_samples,)).tolist()
    y = torch.tensor(labels, device=device, dtype=torch.long)
    n = y.shape[0]
    shape = (n, 3, targs["image_size"], targs["image_size"])

    cfg_interval = tuple(args.cfg_interval) if args.cfg_interval else None
    x = flow.sample_loop(
        model,
        shape,
        device,
        y=y,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        solver=args.solver,
        cfg_interval=cfg_interval,
        progress=True,
    )

    x = (x.clamp(-1, 1) + 1) / 2
    save_image(x.cpu(), args.out, nrow=args.nrow or int(n**0.5) or 1)
    nfe = 2 * args.num_steps - 1 if args.solver == "heun" else args.num_steps
    print(f"wrote {args.out} ({n} samples, {nfe} NFE, labels={labels})")


def build_parser():
    p = argparse.ArgumentParser(description="Sample from a DiT checkpoint")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="samples.png")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--class-labels", type=int, nargs="*", default=None)
    p.add_argument("--cfg-scale", type=float, default=2.0,
                   help="1.0 disables guidance; the paper sweeps [1.0, 4.0]")
    p.add_argument("--cfg-interval", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="restrict guidance to this t range, e.g. 0.1 1.0")
    p.add_argument("--solver", default="heun", choices=["heun", "euler"])
    p.add_argument("--num-steps", type=int, default=50, help="ODE steps")
    p.add_argument("--nrow", type=int, default=0)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
