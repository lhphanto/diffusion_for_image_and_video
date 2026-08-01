"""Gaussian diffusion: schedules, training objective, and samplers.

Follows DDPM (Ho et al. 2020) with the improvements from Nichol & Dhariwal
(2021) that DiT relies on: a cosine schedule and a learned reverse-process
variance trained with the variational bound.
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
