"""Forward/reverse processes: training objectives and samplers.

Two paradigms live here:

* :class:`FlowMatching` -- a linear interpolant with x-prediction, following
  "Back to Basics: Let Denoising Generative Models Denoise"
  (arXiv:2511.13720). This is the simpler formulation and the one being
  migrated to.
* :class:`GaussianDiffusion` -- discrete-time DDPM (Ho et al. 2020) with the
  Nichol & Dhariwal (2021) improvements that the original DiT relies on:
  a cosine schedule and a learned reverse-process variance.
"""

import math

import numpy as np
import torch


def _extract(arr, t, broadcast_shape):
    """Gather ``arr`` at indices ``t`` and reshape for broadcasting over images."""
    res = arr.to(device=t.device)[t].float()
    while res.dim() < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


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

    Note this is the opposite direction from :class:`GaussianDiffusion`, where
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
        time_scale: factor applied to ``t`` before it reaches the network's
            sinusoidal timestep embedding. That embedding was designed for
            integer timesteps in [0, 1000), so continuous ``t`` in [0, 1] is
            rescaled to the same range.
    """

    def __init__(
        self,
        noise_scale=1.0,
        t_mu=-0.8,
        t_sigma=0.8,
        denom_clip=0.05,
        time_scale=1000.0,
    ):
        if noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {noise_scale}")
        if not 0 < denom_clip <= 1:
            raise ValueError(f"denom_clip must be in (0, 1], got {denom_clip}")
        self.noise_scale = noise_scale
        self.t_mu = t_mu
        self.t_sigma = t_sigma
        self.denom_clip = denom_clip
        self.time_scale = time_scale

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

        x_pred = model(z_t, self.time_scale * _broadcast_t(t, x_start).flatten(),
                       **model_kwargs)
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

        ``t`` may be a scalar; it is expanded to the batch and rescaled by
        ``time_scale`` before reaching the model.
        """
        t_batch = _broadcast_t(t, z).flatten()
        t_in = self.time_scale * t_batch
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


def make_beta_schedule(name, num_timesteps):
    if name == "linear":
        # Scaled so that the schedule is invariant to num_timesteps.
        scale = 1000.0 / num_timesteps
        return np.linspace(
            scale * 1e-4, scale * 0.02, num_timesteps, dtype=np.float64
        )
    if name == "cosine":
        return betas_for_alpha_bar(
            num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise ValueError(f"unknown beta schedule: {name}")


def betas_for_alpha_bar(num_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """KL between two diagonal Gaussians, in nats, elementwise."""
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
    )


def discretized_gaussian_log_likelihood(x, means, log_scales):
    """Log-likelihood of 8-bit ``x`` (in [-1, 1]) under a discretized Gaussian."""
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(cdf_delta.clamp(min=1e-12)),
        ),
    )


class GaussianDiffusion:
    """Discrete-time Gaussian diffusion with an epsilon-predicting model.

    Args:
        num_timesteps: length of the forward process.
        beta_schedule: "cosine" (default, as in DiT) or "linear".
        learn_sigma: if True the model is expected to output 2*C channels,
            the second half being the variance interpolation parameter v in
            [-1, 1] (the "learned range" parameterisation).
    """

    def __init__(self, num_timesteps=1000, beta_schedule="cosine", learn_sigma=True):
        self.num_timesteps = num_timesteps
        self.learn_sigma = learn_sigma

        betas = make_beta_schedule(beta_schedule, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        to_t = lambda x: torch.from_numpy(np.asarray(x, dtype=np.float64)).float()
        self.betas = to_t(betas)
        self.alphas_cumprod = to_t(alphas_cumprod)
        self.alphas_cumprod_prev = to_t(alphas_cumprod_prev)
        self.sqrt_alphas_cumprod = to_t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_t(np.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_t(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_t(np.sqrt(1.0 / alphas_cumprod - 1))

        # q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = to_t(posterior_variance)
        # Clipped because the first posterior variance is 0.
        self.posterior_log_variance_clipped = to_t(
            np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        )
        self.posterior_mean_coef1 = to_t(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = to_t(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    # -- forward process ---------------------------------------------------

    def q_sample(self, x_start, t, noise=None):
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        mean = (
            _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = _extract(self.posterior_variance, t, x_t.shape)
        log_var = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def predict_eps_from_xstart(self, x_t, t, x_start):
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start
        ) / _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # -- reverse process ---------------------------------------------------

    def _split_model_output(self, model_output, x, t):
        """Split into (eps, log_variance), handling the learned-range case."""
        if not self.learn_sigma:
            eps = model_output
            log_var = _extract(self.posterior_log_variance_clipped, t, x.shape)
            return eps, log_var

        c = x.shape[1]
        eps, var_v = model_output[:, :c], model_output[:, c : 2 * c]
        # v in [-1, 1] interpolates in log-space between the posterior
        # variance (lower bound) and beta_t (upper bound).
        min_log = _extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract(torch.log(self.betas), t, x.shape)
        frac = (var_v + 1) / 2
        log_var = frac * max_log + (1 - frac) * min_log
        return eps, log_var

    def p_mean_variance(self, model_output, x, t, clip_denoised=True):
        eps, log_var = self._split_model_output(model_output, x, t)
        x_start = self.predict_xstart_from_eps(x, t, eps)
        if clip_denoised:
            x_start = x_start.clamp(-1, 1)
        mean, _, _ = self.q_posterior_mean_variance(x_start, x, t)
        return {
            "mean": mean,
            "log_variance": log_var,
            "pred_xstart": x_start,
            "eps": eps,
        }

    # -- training ----------------------------------------------------------

    def _vb_terms_bpd(self, model_output, x_start, x_t, t, clip_denoised=False):
        """Variational bound term at timestep t, in bits per dimension."""
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        out = self.p_mean_variance(model_output, x_t, t, clip_denoised=clip_denoised)
        kl = normal_kl(true_mean, true_log_var, out["mean"], out["log_variance"])
        kl = kl.flatten(1).mean(1) / math.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = decoder_nll.flatten(1).mean(1) / math.log(2.0)

        # At t == 0 use the decoder NLL, otherwise the KL.
        return torch.where(t == 0, decoder_nll, kl)

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """Hybrid loss: simple MSE on epsilon plus a VLB term for the variance.

        The VLB term is applied with the mean detached (via a stop-gradient on
        the epsilon output) so that it only trains the variance head, exactly
        as in Nichol & Dhariwal.
        """
        model_kwargs = model_kwargs or {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        model_output = model(x_t, t, **model_kwargs)

        terms = {}
        if self.learn_sigma:
            c = x_start.shape[1]
            eps, var_v = model_output[:, :c], model_output[:, c : 2 * c]
            frozen = torch.cat([eps.detach(), var_v], dim=1)
            terms["vb"] = self._vb_terms_bpd(frozen, x_start, x_t, t)
            # Rescale so the VLB does not overwhelm the simple loss.
            terms["vb"] = terms["vb"] * self.num_timesteps / 1000.0
        else:
            eps = model_output
            terms["vb"] = torch.zeros_like(t, dtype=x_start.dtype)

        terms["mse"] = (noise - eps).pow(2).flatten(1).mean(1)
        terms["loss"] = terms["mse"] + terms["vb"]
        return terms

    # -- sampling ----------------------------------------------------------

    def _model_fn(self, model, x, t, cfg_scale=None, y=None):
        if cfg_scale is not None and cfg_scale != 1.0:
            out = model.forward_with_cfg(x, t, y, cfg_scale)
            # forward_with_cfg returns a doubled batch; both halves are equal
            # in the eps channels, so keep the first.
            return out[: x.shape[0]]
        return model(x, t, y) if y is not None else model(x, t)

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape,
        device,
        y=None,
        cfg_scale=None,
        clip_denoised=True,
        noise=None,
        progress=False,
    ):
        """Ancestral (DDPM) sampling over all ``num_timesteps`` steps."""
        x = torch.randn(*shape, device=device) if noise is None else noise
        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            indices = _tqdm(indices)
        for i in indices:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            model_output = self._model_fn(model, x, t, cfg_scale, y)
            out = self.p_mean_variance(model_output, x, t, clip_denoised)
            noise_t = torch.randn_like(x)
            nonzero = (t != 0).float().view(-1, *([1] * (x.dim() - 1)))
            x = out["mean"] + nonzero * torch.exp(0.5 * out["log_variance"]) * noise_t
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model,
        shape,
        device,
        y=None,
        cfg_scale=None,
        num_steps=50,
        eta=0.0,
        clip_denoised=True,
        noise=None,
        progress=False,
    ):
        """DDIM sampling on a uniformly strided subsequence of timesteps."""
        if num_steps > self.num_timesteps:
            raise ValueError("num_steps cannot exceed num_timesteps")
        step_ratio = self.num_timesteps // num_steps
        timesteps = (np.arange(num_steps) * step_ratio).round().astype(np.int64)[::-1]

        x = torch.randn(*shape, device=device) if noise is None else noise
        iterator = _tqdm(timesteps) if progress else timesteps
        for idx, i in enumerate(iterator):
            t = torch.full((shape[0],), int(i), device=device, dtype=torch.long)
            model_output = self._model_fn(model, x, t, cfg_scale, y)
            out = self.p_mean_variance(model_output, x, t, clip_denoised)
            eps = self.predict_eps_from_xstart(x, t, out["pred_xstart"])

            alpha_bar = _extract(self.alphas_cumprod, t, x.shape)
            prev_i = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
            if prev_i >= 0:
                t_prev = torch.full(
                    (shape[0],), int(prev_i), device=device, dtype=torch.long
                )
                alpha_bar_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar)

            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            dir_xt = torch.sqrt((1 - alpha_bar_prev - sigma**2).clamp(min=0)) * eps
            x = torch.sqrt(alpha_bar_prev) * out["pred_xstart"] + dir_xt
            if eta > 0 and prev_i >= 0:
                x = x + sigma * torch.randn_like(x)
        return x


def _tqdm(x):
    try:
        from tqdm.auto import tqdm

        return tqdm(x)
    except ImportError:
        return x
