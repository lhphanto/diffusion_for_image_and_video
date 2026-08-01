import math

import pytest
import torch

from dit.diffusion import GaussianDiffusion
from dit.model import DIT_MODELS, DiT, DiT_B, get_2d_sincos_pos_embed


def tiny_model(**kwargs):
    """A small DiT with the same structure as DiT-B/16 but cheap to run."""
    defaults = dict(
        image_size=32,
        patch_size=16,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    defaults.update(kwargs)
    return DiT(**defaults)


# -- model ---------------------------------------------------------------


def test_forward_shape_learn_sigma():
    model = tiny_model(learn_sigma=True)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    assert out.shape == (2, 6, 32, 32)


def test_forward_shape_no_learn_sigma():
    model = tiny_model(learn_sigma=False)
    out = model(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long),
                torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 3, 32, 32)


def test_unconditional_model_needs_no_labels():
    model = tiny_model(num_classes=0)
    out = model(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))
    assert out.shape == (2, 6, 32, 32)


def test_adaln_zero_init_makes_blocks_identity():
    """With zeroed gates, the residual stream should pass through untouched."""
    model = tiny_model()
    x = torch.randn(2, 4, 64)
    c = torch.randn(2, 64)
    for block in model.blocks:
        assert torch.allclose(block(x, c), x, atol=1e-6)


def test_final_layer_zero_init_gives_zero_output():
    model = tiny_model()
    out = model(torch.randn(2, 3, 32, 32), torch.zeros(2, dtype=torch.long),
                torch.zeros(2, dtype=torch.long))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_unpatchify_inverts_patchify():
    model = tiny_model(patch_size=8, learn_sigma=False)
    x = torch.randn(2, 3, 32, 32)
    tokens = model.x_embedder.proj(x).flatten(2).transpose(1, 2)
    # Rebuild raw patches directly and check unpatchify reassembles them.
    p = 8
    patches = (
        x.reshape(2, 3, 4, p, 4, p)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(2, 16, p * p * 3)
    )
    assert torch.allclose(model.unpatchify(patches), x, atol=1e-6)
    assert tokens.shape == (2, 16, 64)


def test_patch_size_must_divide_image_size():
    with pytest.raises(ValueError):
        tiny_model(image_size=30, patch_size=16)


def test_wrong_input_resolution_raises():
    model = tiny_model()
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 64, 64), torch.zeros(2, dtype=torch.long),
              torch.zeros(2, dtype=torch.long))


def test_pos_embed_shape_and_determinism():
    pe = get_2d_sincos_pos_embed(64, 4)
    assert pe.shape == (16, 64)
    assert (pe == get_2d_sincos_pos_embed(64, 4)).all()


def test_pos_embed_is_not_a_trained_parameter():
    model = tiny_model()
    names = [n for n, _ in model.named_parameters()]
    assert "pos_embed" not in names
    assert model.pos_embed.abs().sum() > 0


def test_cfg_forward_shapes_and_guidance_math():
    model = tiny_model()
    # Break the zero-init so the two branches actually differ.
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    torch.nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))

    with torch.no_grad():
        guided = model.forward_with_cfg(x, t, y, cfg_scale=3.0)
        cond = model(x, t, y)
        null = torch.full_like(y, model.num_classes)
        uncond = model(x, t, null)

    assert guided.shape == (4, 6, 32, 32)
    expected = uncond[:, :3] + 3.0 * (cond[:, :3] - uncond[:, :3])
    assert torch.allclose(guided[:2, :3], expected, atol=1e-5)
    # Both halves of the eps channels are the same guided prediction.
    assert torch.allclose(guided[:2, :3], guided[2:, :3], atol=1e-6)


def test_cfg_scale_one_is_the_conditional_prediction():
    model = tiny_model()
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    model.eval()
    x, t = torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        guided = model.forward_with_cfg(x, t, y, cfg_scale=1.0)
        cond = model(x, t, y)
    assert torch.allclose(guided[:2, :3], cond[:, :3], atol=1e-5)


def test_label_dropout_only_in_training_mode():
    model = tiny_model(class_dropout_prob=1.0)
    y = torch.zeros(8, dtype=torch.long)
    model.train()
    assert (model.y_embedder(y, model.training) ==
            model.y_embedder.embedding_table.weight[10]).all()
    model.eval()
    assert (model.y_embedder(y, model.training) ==
            model.y_embedder.embedding_table.weight[0]).all()


def test_dit_b_16_config_matches_spec():
    model = DiT_B(16, image_size=256, num_classes=1000)
    assert len(model.blocks) == 12
    assert model.blocks[0].attn.qkv.in_features == 768
    assert model.num_heads == 12
    assert model.patch_size == 16
    assert model.x_embedder.num_patches == 256
    n_params = sum(p.numel() for p in model.parameters())
    assert 120e6 < n_params < 145e6, n_params


def test_all_registered_configs_build():
    for name, ctor in DIT_MODELS.items():
        model = ctor(image_size=32, num_classes=4)
        assert isinstance(model, DiT), name


def test_gradients_flow_to_every_block():
    model = tiny_model()
    out = model(torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)),
                torch.randint(0, 10, (2,)))
    out.pow(2).mean().backward()
    for i, block in enumerate(model.blocks):
        g = block.adaLN_modulation[-1].weight.grad
        assert g is not None and torch.isfinite(g).all(), i


# -- diffusion -----------------------------------------------------------


@pytest.mark.parametrize("schedule", ["cosine", "linear"])
def test_schedule_is_valid(schedule):
    d = GaussianDiffusion(num_timesteps=100, beta_schedule=schedule)
    assert (d.betas > 0).all() and (d.betas < 1).all()
    # alphas_cumprod decreases monotonically from near 1 towards 0.
    assert (d.alphas_cumprod[1:] < d.alphas_cumprod[:-1]).all()
    assert d.alphas_cumprod[0] < 1.0
    assert d.alphas_cumprod[-1] < 0.05


def test_q_sample_matches_closed_form_statistics():
    d = GaussianDiffusion(num_timesteps=1000)
    x0 = torch.ones(4096, 1, 4, 4)
    t = torch.full((4096,), 500, dtype=torch.long)
    xt = d.q_sample(x0, t)
    a = d.sqrt_alphas_cumprod[500].item()
    s = d.sqrt_one_minus_alphas_cumprod[500].item()
    assert abs(xt.mean().item() - a) < 0.02
    assert abs(xt.std().item() - s) < 0.02


def test_q_sample_at_t0_is_nearly_clean():
    d = GaussianDiffusion(num_timesteps=1000)
    x0 = torch.randn(8, 3, 8, 8)
    t = torch.zeros(8, dtype=torch.long)
    xt = d.q_sample(x0, t, noise=torch.randn_like(x0))
    assert (xt - x0).abs().mean() < 0.05


def test_predict_xstart_from_eps_is_exact_inverse_of_q_sample():
    d = GaussianDiffusion(num_timesteps=1000)
    x0 = torch.randn(8, 3, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.randint(0, 1000, (8,))
    xt = d.q_sample(x0, t, noise=eps)
    assert torch.allclose(d.predict_xstart_from_eps(xt, t, eps), x0, atol=1e-3)
    assert torch.allclose(d.predict_eps_from_xstart(xt, t, x0), eps, atol=1e-3)


def test_posterior_mean_is_x0_at_t0():
    d = GaussianDiffusion(num_timesteps=1000)
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.zeros(4, dtype=torch.long)
    xt = d.q_sample(x0, t)
    mean, var, _ = d.q_posterior_mean_variance(x0, xt, t)
    assert torch.allclose(mean, x0, atol=1e-3)
    assert var.max() < 1e-4


def test_learned_variance_stays_between_posterior_and_beta():
    d = GaussianDiffusion(num_timesteps=1000, learn_sigma=True)
    x = torch.randn(4, 3, 8, 8)
    t = torch.full((4,), 500, dtype=torch.long)
    for v, expected in [(-1.0, d.posterior_log_variance_clipped[500]),
                        (1.0, torch.log(d.betas[500]))]:
        out = torch.cat([torch.zeros_like(x), torch.full_like(x, v)], dim=1)
        _, log_var = d._split_model_output(out, x, t)
        assert torch.allclose(log_var, expected.expand_as(log_var), atol=1e-5)


def test_training_losses_shapes_and_finiteness():
    model = tiny_model(learn_sigma=True)
    d = GaussianDiffusion(num_timesteps=1000, learn_sigma=True)
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,))
    terms = d.training_losses(model, x, t, model_kwargs={"y": y})
    for key in ("loss", "mse", "vb"):
        assert terms[key].shape == (4,)
        assert torch.isfinite(terms[key]).all(), key
    assert (terms["loss"] >= 0).all()


def test_vb_term_does_not_train_the_epsilon_head():
    """The VLB gradient must reach the variance channels only."""
    model = tiny_model(learn_sigma=True)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    d = GaussianDiffusion(num_timesteps=1000, learn_sigma=True)
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(1, 1000, (4,))
    y = torch.randint(0, 10, (4,))

    terms = d.training_losses(model, x, t, model_kwargs={"y": y})
    terms["vb"].mean().backward()
    grad = model.final_layer.linear.weight.grad
    # Output rows are ordered patch-major: (p*p, out_channels) flattened.
    per_pixel = grad.view(16, 16, 6, -1)
    eps_grad = per_pixel[:, :, :3].abs().sum()
    var_grad = per_pixel[:, :, 3:].abs().sum()
    assert eps_grad.item() == 0.0
    assert var_grad.item() > 0.0


def test_mse_term_equals_zero_for_a_perfect_model():
    d = GaussianDiffusion(num_timesteps=1000, learn_sigma=False)
    x0 = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.randint(0, 1000, (4,))
    oracle = lambda x_t, t_, **kw: noise
    terms = d.training_losses(oracle, x0, t, noise=noise)
    assert terms["mse"].abs().max() < 1e-12


def test_training_losses_without_learn_sigma_has_zero_vb():
    model = tiny_model(learn_sigma=False)
    d = GaussianDiffusion(num_timesteps=1000, learn_sigma=False)
    terms = d.training_losses(
        model, torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,)),
        model_kwargs={"y": torch.randint(0, 10, (2,))},
    )
    assert (terms["vb"] == 0).all()
    assert torch.allclose(terms["loss"], terms["mse"])


def test_normal_kl_is_zero_for_identical_gaussians():
    from dit.diffusion import normal_kl

    mean, logvar = torch.randn(10), torch.randn(10)
    assert torch.allclose(normal_kl(mean, logvar, mean, logvar),
                          torch.zeros(10), atol=1e-6)


def test_normal_kl_matches_analytic_value():
    from dit.diffusion import normal_kl

    # KL(N(0,1) || N(1,1)) = 0.5
    kl = normal_kl(torch.zeros(1), torch.zeros(1), torch.ones(1), torch.zeros(1))
    assert abs(kl.item() - 0.5) < 1e-6


def test_discretized_log_likelihood_is_a_log_probability():
    from dit.diffusion import discretized_gaussian_log_likelihood

    x = torch.linspace(-1, 1, 100).view(-1, 1)
    ll = discretized_gaussian_log_likelihood(
        x, means=torch.zeros_like(x), log_scales=torch.full_like(x, math.log(0.1))
    )
    assert (ll <= 1e-6).all()
    assert torch.isfinite(ll).all()


@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_sampling_loops_produce_finite_images(sampler):
    model = tiny_model(learn_sigma=True).eval()
    d = GaussianDiffusion(num_timesteps=50, learn_sigma=True)
    y = torch.randint(0, 10, (2,))
    shape = (2, 3, 32, 32)
    if sampler == "ddpm":
        out = d.p_sample_loop(model, shape, torch.device("cpu"), y=y)
    else:
        out = d.ddim_sample_loop(model, shape, torch.device("cpu"), y=y, num_steps=10)
    assert out.shape == shape
    assert torch.isfinite(out).all()


def test_sampling_with_cfg_runs():
    model = tiny_model(learn_sigma=True).eval()
    d = GaussianDiffusion(num_timesteps=50, learn_sigma=True)
    out = d.ddim_sample_loop(
        model, (2, 3, 32, 32), torch.device("cpu"),
        y=torch.randint(0, 10, (2,)), cfg_scale=4.0, num_steps=5,
    )
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_ddim_is_deterministic_at_eta_zero():
    model = tiny_model(learn_sigma=True).eval()
    d = GaussianDiffusion(num_timesteps=100, learn_sigma=True)
    noise = torch.randn(2, 3, 32, 32)
    y = torch.zeros(2, dtype=torch.long)
    kw = dict(y=y, num_steps=10, eta=0.0, noise=noise)
    a = d.ddim_sample_loop(model, (2, 3, 32, 32), torch.device("cpu"), **kw)
    b = d.ddim_sample_loop(model, (2, 3, 32, 32), torch.device("cpu"), **kw)
    assert torch.allclose(a, b, atol=1e-6)


def test_ddim_step_count_is_validated():
    d = GaussianDiffusion(num_timesteps=10)
    with pytest.raises(ValueError):
        d.ddim_sample_loop(tiny_model().eval(), (1, 3, 32, 32),
                           torch.device("cpu"), y=torch.zeros(1, dtype=torch.long),
                           num_steps=50)


def test_reverse_process_recovers_data_with_an_oracle_model():
    """An oracle that knows the true noise should reconstruct x0 via DDIM."""
    torch.manual_seed(0)
    d = GaussianDiffusion(num_timesteps=1000, learn_sigma=False)
    x0 = torch.randn(2, 3, 8, 8).clamp(-1, 1)

    def oracle(x_t, t, **kwargs):
        # Invert q_sample analytically for the known x0.
        return d.predict_eps_from_xstart(x_t, t, x0)

    out = d.ddim_sample_loop(
        oracle, x0.shape, torch.device("cpu"), num_steps=100, clip_denoised=False
    )
    assert (out - x0).abs().mean() < 0.05
