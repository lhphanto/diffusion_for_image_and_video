"""Flow matching: the forward interpolant, the training objective, and the
ODE samplers.

Follows "Back to Basics: Let Denoising Generative Models Denoise"
(arXiv:2511.13720): a straight-line interpolant between noise and data, a
network that performs x-prediction with no pre-conditioning, an l2 loss taken
in v-space, and a Heun ODE solver at sampling time.
"""

import torch


def _broadcast_t(t, x):
    """Reshape a per-sample continuous time ``t`` to broadcast against ``x``.

    ``t`` may be a scalar or a 1-D tensor of length ``x.shape[0]``.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=x.device)
    if t.dim() == 0:
        t = t.expand(x.shape[0])
    elif t.dim() != 1:
        raise ValueError(f"t must be scalar or 1-D, got shape {tuple(t.shape)}")
    if t.shape[0] != x.shape[0]:
        raise ValueError(
            f"t has batch size {t.shape[0]} but x has {x.shape[0]}"
        )
    t = t.to(device=x.device, dtype=x.dtype)
    return t.view(-1, *([1] * (x.dim() - 1)))


class FlowMatching:
    """Linear-interpolant (flow matching / rectified flow) forward process.

    Time runs from noise to data on a continuous interval ``t in [0, 1]``:

        t = 0  ->  pure noise
        t = 1  ->  clean data

    which is the convention of arXiv:2511.13720 (Alg. 1). The forward process
    is a straight line between the two endpoints::

        z_t = t * x + (1 - t) * eps

    Note this runs the opposite direction from DDPM-style formulations, where
    ``t`` is a discrete index counting *up* from clean data to noise.

    The network performs **x-prediction**: ``net(z_t, t)`` outputs an estimate
    of the clean data directly, with no pre-conditioning. The loss is applied
    in **v-space**, where ``v = (x - z_t) / (1 - t)``.

    Args:
        noise_scale: standard deviation of the noise endpoint. The paper keeps
            the signal-to-noise ratio fixed across resolutions by scaling the
            noise proportionally, i.e. ``noise_scale = image_size / 256``
            (1.0 at 256x256, 2.0 at 512x512, 4.0 at 1024x1024).
        t_mu, t_sigma: parameters of the logit-normal sampler for ``t``.
            ``logit(t) ~ N(t_mu, t_sigma**2)``. A more negative ``t_mu``
            shifts mass towards small ``t``, i.e. towards higher noise.
        denom_clip: lower bound on ``1 - t`` when dividing, which keeps the
            v-space conversion finite as ``t -> 1``.

    The model is called with ``t`` in [0, 1] unchanged; its timestep embedder
    owns the frequency band appropriate to that range.
    """

    def __init__(self, noise_scale=1.0, t_mu=-0.8, t_sigma=0.8, denom_clip=0.05):
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale}")
        if not 0 < denom_clip <= 1:
            raise ValueError(f"denom_clip must be in (0, 1], got {denom_clip}")
        self.noise_scale = noise_scale
        self.t_mu = t_mu
        self.t_sigma = t_sigma
        self.denom_clip = denom_clip

    def sample_t(self, batch_size, device=None, dtype=torch.float32, generator=None):
        """Sample ``t`` from the logit-normal distribution over [0, 1].

        ``logit(t) ~ N(t_mu, t_sigma**2)``, i.e. ``t = sigmoid(s)``. This
        concentrates training on intermediate noise levels rather than the
        uniform spacing used by discrete-time diffusion.
        """
        s = torch.empty(batch_size, device=device, dtype=dtype).normal_(
            self.t_mu, self.t_sigma, generator=generator
        )
        return torch.sigmoid(s)

    def sample_noise(self, x_start, generator=None):
        """Draw the noise endpoint ``eps ~ N(0, noise_scale**2 I)``."""
        noise = torch.empty_like(x_start).normal_(generator=generator)
        if self.noise_scale != 1.0:
            noise = noise * self.noise_scale
        return noise

    def q_sample(self, x_start, t, noise=None):
        """Interpolate between data and noise: ``z_t = t * x + (1 - t) * eps``.

        Args:
            x_start: (N, C, H, W) clean data.
            t: (N,) or scalar continuous time in [0, 1].
            noise: optional (N, C, H, W) noise endpoint; drawn if omitted.
        Returns:
            (N, C, H, W) the network input ``z_t``.
        """
        if noise is None:
            noise = self.sample_noise(x_start)
        elif noise.shape != x_start.shape:
            raise ValueError(
                f"noise shape {tuple(noise.shape)} != "
                f"x_start shape {tuple(x_start.shape)}"
            )
        t = _broadcast_t(t, x_start)
        return t * x_start + (1.0 - t) * noise

    # -- x <-> v conversion ------------------------------------------------

    def _denom(self, t, x):
        """``1 - t`` clipped away from zero, for use as a divisor."""
        return (1.0 - _broadcast_t(t, x)).clamp(min=self.denom_clip)

    def to_velocity(self, x, z_t, t):
        """Convert an x-space quantity to v-space: ``v = (x - z_t) / (1 - t)``.

        Applied to the ground-truth ``x`` this yields the target velocity;
        applied to the network's prediction it yields the predicted velocity.
        Without clipping this equals ``x - eps`` exactly, the constant
        velocity of the straight path from noise to data.
        """
        return (x - z_t) / self._denom(t, z_t)

    def from_velocity(self, v, z_t, t):
        """Inverse of :meth:`to_velocity`: ``x = z_t + (1 - t) * v``."""
        return z_t + self._denom(t, z_t) * v

    # -- training ----------------------------------------------------------

    def training_losses(self, model, x_start, t=None, model_kwargs=None, noise=None):
        """x-prediction trained with an l2 loss in v-space (Alg. 1 of the paper).

            z_t     = t * x + (1 - t) * eps
            v       = (x - z_t) / (1 - t)
            x_pred  = net(z_t, t)
            v_pred  = (x_pred - z_t) / (1 - t)
            loss    = || v - v_pred ||^2

        Because both velocities share the same denominator, this is identical
        to an x-space loss weighted by ``1 / (1 - t)^2`` -- the weighting the
        paper finds preferable, though it reports the choice is not critical
        so long as the *network* predicts x.

        Args:
            model: callable ``(z_t, t, **model_kwargs) -> x_pred``.
            x_start: (N, C, H, W) clean data.
            t: optional (N,) times in [0, 1]; sampled from the logit-normal
                prior if omitted.
            model_kwargs: extra arguments forwarded to the model, e.g. labels.
            noise: optional noise endpoint; drawn if omitted.
        """
        model_kwargs = model_kwargs or {}
        if noise is None:
            noise = self.sample_noise(x_start)
        if t is None:
            t = self.sample_t(
                x_start.shape[0], device=x_start.device, dtype=x_start.dtype
            )

        z_t = self.q_sample(x_start, t, noise=noise)
        v_target = self.to_velocity(x_start, z_t, t)

        x_pred = model(z_t, _broadcast_t(t, x_start).flatten(), **model_kwargs)
        if x_pred.shape != x_start.shape:
            raise ValueError(
                f"model returned {tuple(x_pred.shape)} but x-prediction expects "
                f"{tuple(x_start.shape)}; build the model with learn_sigma=False"
            )
        v_pred = self.to_velocity(x_pred, z_t, t)

        terms = {}
        terms["loss"] = (v_target - v_pred).pow(2).flatten(1).mean(1)
        # Reported for monitoring only; not optimised directly.
        terms["x_mse"] = (x_start - x_pred).pow(2).flatten(1).mean(1).detach()
        return terms

    # -- sampling ----------------------------------------------------------

    def predict_x(self, model, z, t, y=None, cfg_scale=None):
        """Run the network at time ``t`` and return its x-prediction.

        ``t`` may be a scalar; it is expanded to the batch.
        """
        t_in = _broadcast_t(t, z).flatten()
        if cfg_scale is not None and cfg_scale != 1.0:
            if y is None:
                raise ValueError("classifier-free guidance requires labels")
            # forward_with_cfg guides the first in_channels outputs, which for
            # an x-prediction model is the whole prediction, and returns a
            # doubled batch whose halves are identical.
            return model.forward_with_cfg(z, t_in, y, cfg_scale)[: z.shape[0]]
        return model(z, t_in, y) if y is not None else model(z, t_in)

    def velocity(self, model, z, t, y=None, cfg_scale=None):
        """The ODE right-hand side: the model's x-prediction, in v-space."""
        x_pred = self.predict_x(model, z, t, y=y, cfg_scale=cfg_scale)
        return self.to_velocity(x_pred, z, t)

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape,
        device,
        y=None,
        cfg_scale=None,
        num_steps=50,
        solver="heun",
        cfg_interval=None,
        noise=None,
        progress=False,
    ):
        """Integrate dz/dt = v(z, t) from t=0 (noise) to t=1 (data).

        The time grid is linear in [0, 1], as in Tab. 9 of the paper.

        Args:
            solver: "heun" (2nd-order, 2 network evaluations per step) or
                "euler" (1st-order, 1 evaluation per step). Heun costs
                ``2 * num_steps - 1`` evaluations in total, since the final
                step falls back to Euler.
            cfg_interval: optional ``(lo, hi)`` restricting guidance to times
                in that range; the paper uses ``(0.1, 1.0)``. Outside it the
                unguided conditional prediction is used.
        """
        if solver not in ("heun", "euler"):
            raise ValueError(f"unknown solver: {solver}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        z = self.sample_noise(torch.empty(shape, device=device)) if noise is None else noise
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=z.dtype)

        steps = range(num_steps)
        if progress:
            steps = _tqdm(steps)

        for i in steps:
            t, t_next = times[i], times[i + 1]
            h = t_next - t

            scale = self._guidance_at(cfg_scale, cfg_interval, t)
            v1 = self.velocity(model, z, t, y=y, cfg_scale=scale)
            z_euler = z + h * v1

            # The final step always uses Euler: at t=1 the corrector would
            # divide by the clipped denominator, inflating the velocity by up
            # to 1/denom_clip. This is the same endpoint handling as EDM.
            if solver == "euler" or i == num_steps - 1:
                z = z_euler
                continue

            scale_next = self._guidance_at(cfg_scale, cfg_interval, t_next)
            v2 = self.velocity(model, z_euler, t_next, y=y, cfg_scale=scale_next)
            z = z + 0.5 * h * (v1 + v2)

        return z

    @staticmethod
    def _guidance_at(cfg_scale, cfg_interval, t):
        """Guidance scale at time ``t``, or None outside the CFG interval."""
        if cfg_scale is None or cfg_interval is None:
            return cfg_scale
        lo, hi = cfg_interval
        return cfg_scale if lo <= float(t) <= hi else None


def _tqdm(x):
    try:
        from tqdm.auto import tqdm

        return tqdm(x)
    except ImportError:
        return x
