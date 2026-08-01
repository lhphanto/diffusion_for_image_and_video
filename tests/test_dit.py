import math

import pytest
import torch

from dit.diffusion import FlowMatching
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


def test_timestep_embedding_is_well_conditioned_on_the_unit_interval():
    """The frequency band must span the unit interval, not collapse onto it.

    If the ladder were confined to well under one cycle, every argument would
    land where sin(x) ~ x and cos(x) ~ 1, and the whole basis would degenerate
    onto a couple of near-collinear directions.
    """
    from dit.model import TimestepEmbedder

    t = torch.linspace(0.001, 0.999, 256)
    emb = TimestepEmbedder.timestep_embedding(t, 256)
    centred = emb - emb.mean(0)
    sv = torch.linalg.svdvals(centred.double())
    effective_rank = int((sv > sv[0] * 1e-3).sum())
    assert effective_rank > 50, effective_rank


def test_timestep_embedding_separates_nearby_times():
    from dit.model import TimestepEmbedder

    t = torch.linspace(0.001, 0.999, 256)
    emb = TimestepEmbedder.timestep_embedding(t, 256)
    step = (emb[1:] - emb[:-1]).norm(dim=1)
    assert step.min() > 0.1


def test_timestep_embedding_frequency_band():
    """Fastest channel ~max_freq cycles over [0,1], slowest ~max_freq/bandwidth."""
    from dit.model import TimestepEmbedder

    te = TimestepEmbedder(64, frequency_embedding_size=256)
    assert te.max_freq == 1000.0 and te.bandwidth == 10000.0

    # sin(freq * t) at t=1 for the fastest and slowest channels. The ladder
    # runs over i = 0..half-1, so the slowest is bandwidth^(-(half-1)/half).
    half = 128
    lo = 1000.0 * 10000.0 ** (-(half - 1) / half)
    emb = TimestepEmbedder.timestep_embedding(torch.tensor([1.0]), 256)
    fastest, slowest = emb[0, 128], emb[0, 255]
    assert torch.allclose(fastest, torch.sin(torch.tensor(1000.0)), atol=1e-3)
    assert torch.allclose(slowest, torch.sin(torch.tensor(lo)), atol=1e-4)
    assert 0.1 < lo < 0.11  # roughly a tenth of a radian across [0,1]


def test_timestep_embedding_shape_and_finiteness():
    from dit.model import TimestepEmbedder

    te = TimestepEmbedder(64, frequency_embedding_size=256)
    out = te(torch.rand(8))
    assert out.shape == (8, 64)
    assert torch.isfinite(out).all()


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


# -- bottleneck patch embedding ------------------------------------------


def test_bottleneck_layer_shapes():
    """16x16x3 = 768-d raw patch -> 128 -> 768 hidden, both layers linear."""
    model = DiT_B(16, image_size=256, num_classes=10, bottleneck_dim=128)
    emb = model.x_embedder
    assert emb.patch_dim == 768
    assert emb.bottleneck_dim == 128
    assert emb.proj.out_channels == 128
    assert emb.proj.bias is None
    assert emb.expand.in_features == 128
    assert emb.expand.out_features == 768


def test_bottleneck_is_default_for_configs():
    assert DiT_B(16, image_size=32, num_classes=4).x_embedder.bottleneck_dim == 128
    assert DIT_MODELS["DiT-XL/2"](image_size=32,
                                  num_classes=4).x_embedder.bottleneck_dim == 256


def test_bottleneck_disabled_matches_plain_embedding():
    model = DiT_B(16, image_size=32, num_classes=4, bottleneck_dim=None)
    assert model.x_embedder.expand is None
    assert model.x_embedder.proj.out_channels == 768
    assert model.x_embedder.proj.bias is not None


def test_bottleneck_forward_shape_and_rank():
    model = tiny_model(bottleneck_dim=8, hidden_size=64)
    x = torch.randn(2, 3, 32, 32)
    tokens = model.x_embedder(x)
    assert tokens.shape == (2, 4, 64)

    # The composed map really is rank <= bottleneck_dim. Count singular values
    # with an explicit relative threshold: the weights are float32, so the
    # numerical noise floor sits well above matrix_rank's float64 default tol.
    w1 = model.x_embedder.proj.weight.reshape(8, -1)  # (8, 768)
    w2 = model.x_embedder.expand.weight  # (64, 8)
    composed = w2 @ w1  # (64, 768)
    assert composed.shape == (64, 768)
    sv = torch.linalg.svdvals(composed.double())
    assert int((sv > sv[0] * 1e-4).sum()) == 8


def test_bottleneck_has_no_nonlinearity_between_layers():
    """A low-rank linear map must be exactly additive over its input."""
    model = tiny_model(bottleneck_dim=8, hidden_size=64)
    emb = model.x_embedder
    a, b = torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        # Subtract the bias to isolate the linear part.
        bias = emb(torch.zeros_like(a))
        lhs = emb(a + b) - bias
        rhs = (emb(a) - bias) + (emb(b) - bias)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_bottleneck_reduces_patch_embedding_parameters():
    with_bn = DiT_B(16, image_size=256, num_classes=10, bottleneck_dim=128)
    without = DiT_B(16, image_size=256, num_classes=10, bottleneck_dim=None)
    n = lambda m: sum(p.numel() for p in m.x_embedder.parameters())
    # 768*128 + 128*768 + 768 vs 768*768 + 768
    assert n(with_bn) == 768 * 128 + 128 * 768 + 768
    assert n(without) == 768 * 768 + 768
    assert n(with_bn) < n(without)


def test_bottleneck_model_trains_end_to_end():
    model = tiny_model(bottleneck_dim=8, learn_sigma=False)
    fm = FlowMatching()
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    fm.training_losses(model, x, model_kwargs={"y": y})["loss"].mean().backward()
    for name in ("proj", "expand"):
        for p in getattr(model.x_embedder, name).parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_bottleneck_survives_state_dict_roundtrip():
    a = tiny_model(bottleneck_dim=8).eval()
    b = tiny_model(bottleneck_dim=8).eval()
    b.load_state_dict(a.state_dict())
    x, t = torch.randn(2, 3, 32, 32), torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        assert torch.allclose(a(x, t, y), b(x, t, y), atol=1e-6)


def test_high_resolution_patch_dim_exceeds_hidden_size():
    """The point of the bottleneck: patch dim can dwarf the transformer width."""
    model = DiT_B(32, image_size=512, num_classes=10, bottleneck_dim=128)
    assert model.x_embedder.patch_dim == 3072  # 32*32*3
    assert model.x_embedder.num_patches == 256  # same sequence length as /16 @ 256
    assert model.x_embedder.expand.out_features == 768


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


def test_flow_q_sample_endpoints():
    """t=1 is exactly the data, t=0 is exactly the noise."""
    fm = FlowMatching()
    x0 = torch.randn(4, 3, 8, 8)
    eps = torch.randn_like(x0)
    ones = torch.ones(4)
    assert torch.allclose(fm.q_sample(x0, ones, noise=eps), x0, atol=1e-6)
    assert torch.allclose(fm.q_sample(x0, ones * 0, noise=eps), eps, atol=1e-6)


def test_flow_q_sample_is_the_linear_interpolant():
    fm = FlowMatching()
    x0 = torch.randn(4, 3, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.rand(4)
    tb = t.view(-1, 1, 1, 1)
    expected = tb * x0 + (1 - tb) * eps
    assert torch.allclose(fm.q_sample(x0, t, noise=eps), expected, atol=1e-6)


def test_flow_q_sample_midpoint_is_the_average():
    fm = FlowMatching()
    x0 = torch.randn(2, 3, 8, 8)
    eps = torch.randn_like(x0)
    got = fm.q_sample(x0, torch.full((2,), 0.5), noise=eps)
    assert torch.allclose(got, 0.5 * (x0 + eps), atol=1e-6)


def test_flow_q_sample_applies_t_per_sample():
    """Each element of the batch must get its own time."""
    fm = FlowMatching()
    x0 = torch.ones(3, 1, 2, 2)
    eps = torch.zeros_like(x0)
    z = fm.q_sample(x0, torch.tensor([0.0, 0.5, 1.0]), noise=eps)
    assert torch.allclose(z[0], torch.zeros(1, 2, 2), atol=1e-6)
    assert torch.allclose(z[1], torch.full((1, 2, 2), 0.5), atol=1e-6)
    assert torch.allclose(z[2], torch.ones(1, 2, 2), atol=1e-6)


def test_flow_q_sample_accepts_scalar_t():
    fm = FlowMatching()
    x0 = torch.randn(4, 3, 8, 8)
    eps = torch.randn_like(x0)
    a = fm.q_sample(x0, torch.tensor(0.3), noise=eps)
    b = fm.q_sample(x0, 0.3, noise=eps)
    c = fm.q_sample(x0, torch.full((4,), 0.3), noise=eps)
    assert torch.allclose(a, c, atol=1e-6)
    assert torch.allclose(b, c, atol=1e-6)


def test_flow_q_sample_draws_noise_when_not_given():
    fm = FlowMatching()
    x0 = torch.zeros(4096, 1, 4, 4)
    z = fm.q_sample(x0, torch.zeros(4096))  # t=0 -> z is exactly the noise
    assert z.shape == x0.shape
    assert abs(z.mean().item()) < 0.02
    assert abs(z.std().item() - 1.0) < 0.02


def test_flow_noise_scale_controls_the_noise_endpoint():
    fm = FlowMatching(noise_scale=2.0)
    x0 = torch.zeros(4096, 1, 4, 4)
    z = fm.q_sample(x0, torch.zeros(4096))
    assert abs(z.std().item() - 2.0) < 0.05


def test_flow_noise_scale_does_not_touch_supplied_noise():
    fm = FlowMatching(noise_scale=2.0)
    x0 = torch.randn(2, 3, 4, 4)
    eps = torch.randn_like(x0)
    assert torch.allclose(fm.q_sample(x0, 0.0, noise=eps), eps, atol=1e-6)


def test_flow_q_sample_variance_matches_closed_form():
    """For independent x ~ N(0,1) and eps ~ N(0,1), Var(z_t) = t^2 + (1-t)^2."""
    fm = FlowMatching()
    x0 = torch.randn(20000, 1, 2, 2)
    for t_val in (0.25, 0.5, 0.75):
        z = fm.q_sample(x0, torch.full((20000,), t_val))
        expected = math.sqrt(t_val**2 + (1 - t_val) ** 2)
        assert abs(z.std().item() - expected) < 0.02, t_val


def test_flow_velocity_is_constant_along_the_path():
    """dz/dt = x - eps everywhere, which is what makes the path straight."""
    fm = FlowMatching()
    x0 = torch.randn(2, 3, 4, 4)
    eps = torch.randn_like(x0)
    z1 = fm.q_sample(x0, 0.2, noise=eps)
    z2 = fm.q_sample(x0, 0.7, noise=eps)
    assert torch.allclose((z2 - z1) / 0.5, x0 - eps, atol=1e-5)


def test_flow_paper_velocity_identity():
    """The paper's v = (x - z_t) / (1 - t) equals x - eps."""
    fm = FlowMatching()
    x0 = torch.randn(2, 3, 4, 4)
    eps = torch.randn_like(x0)
    t = 0.6
    z = fm.q_sample(x0, t, noise=eps)
    assert torch.allclose((x0 - z) / (1 - t), x0 - eps, atol=1e-5)


def test_flow_q_sample_preserves_dtype_and_shape():
    fm = FlowMatching()
    x0 = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    z = fm.q_sample(x0, torch.rand(2))
    assert z.shape == x0.shape and z.dtype == torch.float64


def test_flow_q_sample_rejects_mismatched_batch():
    fm = FlowMatching()
    with pytest.raises(ValueError):
        fm.q_sample(torch.randn(4, 3, 8, 8), torch.rand(3))


def test_flow_q_sample_rejects_mismatched_noise():
    fm = FlowMatching()
    with pytest.raises(ValueError):
        fm.q_sample(torch.randn(4, 3, 8, 8), torch.rand(4),
                    noise=torch.randn(4, 3, 4, 4))


def test_flow_rejects_nonpositive_noise_scale():
    with pytest.raises(ValueError):
        FlowMatching(noise_scale=0.0)


def test_flow_q_sample_feeds_the_model_directly():
    """z_t is the network input, unscaled -- no preconditioning."""
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    x0 = torch.randn(2, 3, 32, 32)
    z = fm.q_sample(x0, torch.rand(2))
    with torch.no_grad():
        out = model(z, torch.rand(2), torch.randint(0, 10, (2,)))
    assert out.shape == x0.shape


# -- flow matching: x-prediction with v-loss ------------------------------


def test_logit_normal_t_is_in_range_and_centred():
    fm = FlowMatching(t_mu=-0.8, t_sigma=0.8)
    t = fm.sample_t(50000)
    assert t.shape == (50000,)
    assert (t > 0).all() and (t < 1).all()
    # logit(t) should recover the prior.
    s = torch.log(t / (1 - t))
    assert abs(s.mean().item() - (-0.8)) < 0.02
    assert abs(s.std().item() - 0.8) < 0.02


def test_more_negative_t_mu_means_more_noise():
    """Smaller t = less signal in z_t = t*x + (1-t)*eps."""
    low = FlowMatching(t_mu=-1.2).sample_t(50000).mean().item()
    high = FlowMatching(t_mu=-0.0).sample_t(50000).mean().item()
    assert low < high


def test_to_velocity_equals_x_minus_eps():
    fm = FlowMatching(denom_clip=1e-8)
    x0 = torch.randn(4, 3, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.rand(4) * 0.9  # stay away from the clip
    z = fm.q_sample(x0, t, noise=eps)
    assert torch.allclose(fm.to_velocity(x0, z, t), x0 - eps, atol=1e-4)


def test_from_velocity_inverts_to_velocity():
    fm = FlowMatching()
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.rand(4) * 0.9
    z = fm.q_sample(x0, t)
    v = fm.to_velocity(x0, z, t)
    assert torch.allclose(fm.from_velocity(v, z, t), x0, atol=1e-5)


def test_denominator_is_clipped_at_t_near_one():
    """Without clipping, v blows up as t -> 1."""
    fm = FlowMatching(denom_clip=0.05)
    x0 = torch.randn(2, 3, 4, 4)
    t = torch.full((2,), 1.0)
    z = fm.q_sample(x0, t)  # z == x0 exactly
    v = fm.to_velocity(x0, z, t)
    assert torch.isfinite(v).all()
    assert (v == 0).all()  # numerator is zero, denominator is 0.05 not 0

    off = fm.to_velocity(x0 + 1.0, z, t)
    assert torch.allclose(off, torch.full_like(off, 1 / 0.05), atol=1e-4)


def test_v_loss_is_zero_for_a_perfect_x_predictor():
    fm = FlowMatching()
    x0 = torch.randn(4, 3, 8, 8)
    oracle = lambda z, t, **kw: x0
    terms = fm.training_losses(oracle, x0)
    assert terms["loss"].abs().max() < 1e-10
    assert terms["x_mse"].abs().max() < 1e-10


def test_v_loss_equals_x_loss_weighted_by_inverse_one_minus_t_squared():
    """Footnote 4 of the paper: v-loss == x-loss with weight 1/(1-t)^2."""
    fm = FlowMatching()
    x0 = torch.randn(4, 3, 8, 8)
    t = torch.rand(4) * 0.9
    noise = torch.randn_like(x0)
    bias = torch.randn_like(x0)
    imperfect = lambda z, t_, **kw: x0 + bias

    terms = fm.training_losses(imperfect, x0, t=t, noise=noise)
    w = (1 - t).clamp(min=fm.denom_clip).pow(-2)
    x_loss = bias.pow(2).flatten(1).mean(1)
    assert torch.allclose(terms["loss"], w * x_loss, atol=1e-5)


def test_training_losses_samples_t_when_omitted():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False)
    terms = fm.training_losses(
        model, torch.randn(4, 3, 32, 32),
        model_kwargs={"y": torch.randint(0, 10, (4,))},
    )
    assert terms["loss"].shape == (4,)
    assert torch.isfinite(terms["loss"]).all()
    assert (terms["loss"] >= 0).all()


def test_training_losses_rejects_a_learn_sigma_model():
    """x-prediction has no variance head; a 2C output is a configuration error."""
    fm = FlowMatching()
    model = tiny_model(learn_sigma=True)
    with pytest.raises(ValueError, match="learn_sigma=False"):
        fm.training_losses(
            model, torch.randn(2, 3, 32, 32),
            model_kwargs={"y": torch.randint(0, 10, (2,))},
        )


def test_model_receives_unscaled_time():
    """The model is handed t in [0,1] directly; the embedder owns the band."""
    fm = FlowMatching()
    seen = {}

    def spy(z, t, **kw):
        seen["t"] = t
        return torch.zeros_like(z)

    t = torch.tensor([0.0, 0.25, 1.0])
    fm.training_losses(spy, torch.randn(3, 3, 8, 8), t=t)
    assert torch.allclose(seen["t"], t, atol=1e-6)
    assert seen["t"].shape == (3,)


def test_gradients_reach_the_model_through_v_loss():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False, bottleneck_dim=8)
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    terms = fm.training_losses(
        model, torch.randn(2, 3, 32, 32),
        model_kwargs={"y": torch.randint(0, 10, (2,))},
    )
    terms["loss"].mean().backward()
    for name in ("proj", "expand"):
        for p in getattr(model.x_embedder, name).parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert torch.isfinite(model.final_layer.linear.weight.grad).all()


def test_x_mse_is_detached():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False)
    terms = fm.training_losses(
        model, torch.randn(2, 3, 32, 32),
        model_kwargs={"y": torch.randint(0, 10, (2,))},
    )
    assert not terms["x_mse"].requires_grad


def test_supplied_t_and_noise_make_the_loss_deterministic():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    x0 = torch.randn(2, 3, 32, 32)
    kw = dict(t=torch.rand(2), noise=torch.randn_like(x0),
              model_kwargs={"y": torch.zeros(2, dtype=torch.long)})
    with torch.no_grad():
        a = fm.training_losses(model, x0, **kw)["loss"]
        b = fm.training_losses(model, x0, **kw)["loss"]
    assert torch.allclose(a, b, atol=1e-7)


# -- flow matching: ODE samplers -----------------------------------------


def quadrature_oracle(fm, u):
    """Build a model whose velocity is exactly ``u(t)``, independent of z.

    Then dz/dt = u(t), so integrating from 0 to 1 is pure quadrature with the
    known answer ``z0 + integral(u)``. This isolates the solver's accuracy
    from any property of the network.
    """

    def oracle(z, t_in, y=None):
        t = t_in  # the model receives t in [0,1] directly
        tb = t.view(-1, *([1] * (z.dim() - 1)))
        denom = (1 - tb).clamp(min=fm.denom_clip)
        return z + denom * u(tb)

    return oracle


def test_sampler_recovers_a_constant_target():
    """With a perfect denoiser the path is straight, so both solvers are exact."""
    fm = FlowMatching()
    target = torch.randn(2, 3, 8, 8)
    z0 = torch.randn_like(target)
    oracle = lambda z, t, y=None: target
    for solver in ("euler", "heun"):
        out = fm.sample_loop(oracle, target.shape, torch.device("cpu"),
                             num_steps=10, solver=solver, noise=z0)
        assert torch.allclose(out, target, atol=1e-4), solver


def test_heun_is_more_accurate_than_euler():
    fm = FlowMatching()
    z0 = torch.zeros(1, 1, 1, 1)
    u = torch.exp  # integral over [0,1] is e - 1
    exact = math.e - 1.0
    oracle = quadrature_oracle(fm, u)

    def err(solver, n):
        out = fm.sample_loop(oracle, z0.shape, torch.device("cpu"),
                             num_steps=n, solver=solver, noise=z0)
        return abs(out.item() - exact)

    assert err("heun", 10) < err("euler", 10) / 3


def test_heun_is_second_order_and_euler_is_first_order():
    """Doubling the steps should quarter Heun's error but only halve Euler's."""
    fm = FlowMatching()
    z0 = torch.zeros(1, 1, 1, 1)
    exact = math.e - 1.0
    oracle = quadrature_oracle(fm, torch.exp)

    def err(solver, n):
        out = fm.sample_loop(oracle, z0.shape, torch.device("cpu"),
                             num_steps=n, solver=solver, noise=z0)
        return abs(out.item() - exact)

    heun_ratio = err("heun", 40) / err("heun", 20)
    euler_ratio = err("euler", 40) / err("euler", 20)
    assert heun_ratio < 0.35, heun_ratio  # ~0.25
    assert 0.4 < euler_ratio < 0.6, euler_ratio  # ~0.5


def test_solver_evaluation_counts():
    """Heun costs 2N-1 evaluations: two per step, minus the Euler final step."""
    fm = FlowMatching()
    calls = []

    def spy(z, t, y=None):
        calls.append(float(t.flatten()[0]))
        return torch.zeros_like(z)

    shape = (1, 1, 2, 2)
    fm.sample_loop(spy, shape, torch.device("cpu"), num_steps=5, solver="euler")
    assert len(calls) == 5

    calls.clear()
    fm.sample_loop(spy, shape, torch.device("cpu"), num_steps=5, solver="heun")
    assert len(calls) == 2 * 5 - 1


def test_time_grid_is_linear_from_zero_to_one():
    fm = FlowMatching()
    seen = []

    def spy(z, t, y=None):
        seen.append(float(t.flatten()[0]))
        return torch.zeros_like(z)

    fm.sample_loop(spy, (1, 1, 2, 2), torch.device("cpu"), num_steps=4,
                   solver="euler")
    assert seen == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_final_step_never_evaluates_at_t_equals_one():
    """t=1 would divide by the clipped denominator and inflate the velocity."""
    fm = FlowMatching()
    seen = []

    def spy(z, t, y=None):
        seen.append(float(t.flatten()[0]))
        return torch.zeros_like(z)

    fm.sample_loop(spy, (1, 1, 2, 2), torch.device("cpu"), num_steps=8,
                   solver="heun")
    assert max(seen) < 1.0


def test_sampler_runs_with_a_real_model():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    y = torch.randint(0, 10, (2,))
    out = fm.sample_loop(model, (2, 3, 32, 32), torch.device("cpu"), y=y,
                         num_steps=4, solver="heun")
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_sampler_runs_with_cfg():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    out = fm.sample_loop(model, (2, 3, 32, 32), torch.device("cpu"),
                         y=torch.randint(0, 10, (2,)), cfg_scale=4.0,
                         num_steps=3, solver="heun")
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_sampler_is_deterministic_given_the_initial_noise():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    z0 = torch.randn(2, 3, 32, 32)
    kw = dict(y=torch.zeros(2, dtype=torch.long), num_steps=4, noise=z0)
    a = fm.sample_loop(model, z0.shape, torch.device("cpu"), **kw)
    b = fm.sample_loop(model, z0.shape, torch.device("cpu"), **kw)
    assert torch.allclose(a, b, atol=1e-6)


def test_initial_noise_uses_the_noise_scale():
    fm = FlowMatching(noise_scale=3.0)
    seen = {}

    def spy(z, t, y=None):
        seen.setdefault("z0_std", z.std().item())
        return torch.zeros_like(z)

    fm.sample_loop(spy, (4096, 1, 2, 2), torch.device("cpu"), num_steps=1)
    assert abs(seen["z0_std"] - 3.0) < 0.1


def test_cfg_interval_gates_guidance_by_time():
    at = FlowMatching._guidance_at
    assert at(4.0, (0.1, 1.0), 0.05) is None
    assert at(4.0, (0.1, 1.0), 0.5) == 4.0
    assert at(4.0, (0.1, 1.0), 1.0) == 4.0
    # No interval means guidance everywhere; no scale means never.
    assert at(4.0, None, 0.05) == 4.0
    assert at(None, (0.1, 1.0), 0.5) is None


def test_cfg_interval_changes_the_result():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    # Break the zero-init so the output actually depends on the label,
    # otherwise the conditional and unconditional branches coincide and
    # guidance is a no-op at any scale.
    torch.nn.init.normal_(model.final_layer.linear.weight, std=0.05)
    torch.nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.05)
    z0 = torch.randn(2, 3, 32, 32)
    kw = dict(y=torch.randint(0, 10, (2,)), cfg_scale=4.0, num_steps=6, noise=z0)
    full = fm.sample_loop(model, z0.shape, torch.device("cpu"), **kw)
    gated = fm.sample_loop(model, z0.shape, torch.device("cpu"),
                           cfg_interval=(0.5, 1.0), **kw)
    assert not torch.allclose(full, gated, atol=1e-5)


def test_cfg_without_labels_is_an_error():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    with pytest.raises(ValueError, match="requires labels"):
        fm.sample_loop(model, (2, 3, 32, 32), torch.device("cpu"),
                       cfg_scale=2.0, num_steps=2)


def test_sampler_validates_arguments():
    fm = FlowMatching()
    model = tiny_model(learn_sigma=False).eval()
    with pytest.raises(ValueError, match="unknown solver"):
        fm.sample_loop(model, (1, 3, 32, 32), torch.device("cpu"), solver="rk4",
                       y=torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="num_steps"):
        fm.sample_loop(model, (1, 3, 32, 32), torch.device("cpu"), num_steps=0,
                       y=torch.zeros(1, dtype=torch.long))


def test_single_step_sampling_returns_the_x_prediction():
    """One Euler step from t=0 to t=1 is exactly the model's x-prediction."""
    fm = FlowMatching()
    target = torch.randn(2, 3, 8, 8)
    z0 = torch.randn_like(target)
    oracle = lambda z, t, y=None: target
    out = fm.sample_loop(oracle, target.shape, torch.device("cpu"),
                         num_steps=1, solver="heun", noise=z0)
    assert torch.allclose(out, target, atol=1e-5)


# -- train / sample wiring ------------------------------------------------


def test_train_and_sample_use_flow_matching_only():
    import dit.sample as sample_mod
    import dit.train as train_mod

    src = (
        __import__("pathlib").Path(train_mod.__file__).read_text()
        + __import__("pathlib").Path(sample_mod.__file__).read_text()
    )
    assert "GaussianDiffusion" not in src
    assert "FlowMatching" in src


def test_train_parser_exposes_flow_matching_options():
    from dit.train import build_parser

    args = build_parser().parse_args([])
    assert args.t_mu == -0.8 and args.t_sigma == 0.8
    assert args.denom_clip == 0.05
    assert args.noise_scale == 0.0  # auto -> image_size / 256
    assert not hasattr(args, "num_timesteps")
    assert not hasattr(args, "no_learn_sigma")


def test_sample_parser_defaults_to_heun():
    from dit.sample import build_parser

    args = build_parser().parse_args(["--ckpt", "x.pt"])
    assert args.solver == "heun"
    assert args.num_steps == 50


def test_train_then_sample_end_to_end(tmp_path):
    """Run the real training step and the real sampling path over a checkpoint."""
    import dit.sample as sample_mod
    from dit.train import build_parser as train_parser

    targs = train_parser().parse_args([])
    targs.image_size, targs.model = 32, "DiT-S/16"
    targs.bottleneck_dim, targs.class_dropout_prob = 8, 0.1

    model = DIT_MODELS[targs.model](
        image_size=targs.image_size, in_channels=3, num_classes=10,
        class_dropout_prob=targs.class_dropout_prob, learn_sigma=False,
        bottleneck_dim=targs.bottleneck_dim,
    )
    flow = FlowMatching(noise_scale=targs.image_size / 256.0, t_mu=targs.t_mu,
                        t_sigma=targs.t_sigma, denom_clip=targs.denom_clip)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    before = [p.detach().clone() for p in model.parameters()]
    loss = flow.training_losses(model, x, model_kwargs={"y": y})["loss"].mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "ema": model.state_dict(),
                "step": 1, "args": vars(targs), "num_classes": 10}, ckpt)

    out = tmp_path / "samples.png"
    sample_mod.main(sample_mod.build_parser().parse_args([
        "--ckpt", str(ckpt), "--out", str(out), "--num-samples", "2",
        "--num-steps", "3", "--solver", "heun", "--cfg-scale", "2.0",
        "--cfg-interval", "0.1", "1.0", "--device", "cpu",
    ]))
    assert out.exists() and out.stat().st_size > 0


def test_sample_reconstructs_the_training_noise_scale(tmp_path):
    """A 512-trained checkpoint must sample with noise_scale 2.0, not 1.0."""
    from dit.sample import load_model
    from dit.train import build_parser as train_parser

    targs = train_parser().parse_args([])
    targs.image_size, targs.model = 512, "DiT-S/16"
    targs.bottleneck_dim, targs.class_dropout_prob = 8, 0.1

    model = DIT_MODELS[targs.model](
        image_size=512, in_channels=3, num_classes=4, class_dropout_prob=0.1,
        learn_sigma=False, bottleneck_dim=8,
    )
    ckpt = tmp_path / "c.pt"
    torch.save({"model": model.state_dict(), "ema": model.state_dict(),
                "args": vars(targs), "num_classes": 4}, ckpt)

    _, flow, _, _ = load_model(str(ckpt), torch.device("cpu"))
    assert flow.noise_scale == 2.0
    assert flow.t_mu == -0.8 and flow.denom_clip == 0.05

