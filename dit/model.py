"""Diffusion Transformer (DiT) for images.

Reference: "Scalable Diffusion Models with Transformers", Peebles & Xie (2023).
https://arxiv.org/abs/2212.09748

The default configuration here is DiT-B/16: depth 12, hidden size 768,
12 heads, 16x16 patches.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    """FiLM-style modulation used by adaLN: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# -----------------------------------------------------------------------------
# Embedders
# -----------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and linearly project them."""

    def __init__(self, image_size=256, patch_size=16, in_channels=3, hidden_size=768):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        _, _, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(
                f"input is {h}x{w}, but model was built for "
                f"{self.image_size}x{self.image_size}"
            )
        x = self.proj(x)  # (N, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (N, T, D)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        freqs = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(freqs.to(self.mlp[0].weight.dtype))


class LabelEmbedder(nn.Module):
    """Class-label embedding table with an extra "null" row for classifier-free guidance.

    During training, labels are randomly replaced by the null class with
    probability ``dropout_prob`` so that the same network learns both the
    conditional and unconditional score.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg_embedding), hidden_size
        )

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# -----------------------------------------------------------------------------
# Transformer
# -----------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention using PyTorch's fused SDPA kernel."""

    def __init__(self, dim, num_heads=12, qkv_bias=True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        n, t, d = x.shape
        qkv = self.qkv(x).reshape(n, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, heads, T, head_dim)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(n, t, d)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.

    The conditioning vector ``c`` produces per-block shift/scale for both
    sub-layers plus a gate that is zero-initialised, so each block starts as
    the identity function.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """adaLN-modulated norm + linear projection back to patch pixels."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# -----------------------------------------------------------------------------
# Positional embeddings (fixed 2D sin-cos, as in the DiT / MAE codebases)
# -----------------------------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Return (grid_size**2, embed_dim) fixed sin-cos positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# -----------------------------------------------------------------------------
# DiT
# -----------------------------------------------------------------------------


class DiT(nn.Module):
    """Diffusion Transformer with adaLN-Zero conditioning.

    Args:
        image_size: side length of the (square) input. For latent-space
            training this is the latent side length, not the pixel one.
        patch_size: side length of each patch. Must divide ``image_size``.
        in_channels: channels of the noisy input (3 for RGB pixels, 4 for a
            typical SD VAE latent).
        hidden_size: transformer width.
        depth: number of transformer blocks.
        num_heads: attention heads.
        mlp_ratio: MLP expansion factor.
        num_classes: number of class labels for conditional generation. Set to
            0 for an unconditional model.
        class_dropout_prob: probability of dropping the label during training,
            which enables classifier-free guidance at sampling time.
        learn_sigma: if True the model outputs 2*in_channels channels, the
            second half parameterising the reverse-process variance.
    """

    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        class_dropout_prob=0.1,
        learn_sigma=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma
        self.num_classes = num_classes
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(image_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = (
            LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            if num_classes > 0
            else None
        )

        num_patches = self.x_embedder.num_patches
        # Fixed (non-learned) positional embedding, registered as a buffer.
        self.register_buffer(
            "pos_embed", torch.zeros(1, num_patches, hidden_size), persistent=False
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.x_embedder.grid_size
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Patch embedding: initialise like nn.Linear.
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # adaLN-Zero: zero out the modulation outputs so every block (and the
        # final layer) starts as an identity / zero map.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """(N, T, patch_size**2 * C) -> (N, C, H, W)."""
        c = self.out_channels
        p = self.patch_size
        h = w = self.x_embedder.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(self, x, t, y=None):
        """
        Args:
            x: (N, C, H, W) noisy inputs.
            t: (N,) diffusion timesteps.
            y: (N,) class labels, or None for an unconditional model.
        Returns:
            (N, out_channels, H, W)
        """
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t)
        if self.y_embedder is not None:
            if y is None:
                raise ValueError("model is class-conditional but no labels were given")
            c = c + self.y_embedder(y, self.training)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """Forward pass with classifier-free guidance.

        Expects ``x`` to be a single batch; it is duplicated internally so the
        conditional and unconditional branches run in one pass. Guidance is
        applied to the first ``in_channels`` channels only (the eps prediction),
        matching the reference implementation.
        """
        half = x
        combined = torch.cat([half, half], dim=0)
        t = torch.cat([t, t], dim=0)
        null = torch.full_like(y, self.num_classes)
        y = torch.cat([y, null], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = eps.chunk(2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


def DiT_B(patch_size=16, **kwargs):
    """DiT-B: depth 12, hidden size 768, 12 heads (~130M params at /16)."""
    return DiT(patch_size=patch_size, depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_S(patch_size=16, **kwargs):
    return DiT(patch_size=patch_size, depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_L(patch_size=16, **kwargs):
    return DiT(patch_size=patch_size, depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_XL(patch_size=16, **kwargs):
    return DiT(patch_size=patch_size, depth=28, hidden_size=1152, num_heads=16, **kwargs)


DIT_MODELS = {
    "DiT-S/16": lambda **kw: DiT_S(16, **kw),
    "DiT-S/8": lambda **kw: DiT_S(8, **kw),
    "DiT-S/4": lambda **kw: DiT_S(4, **kw),
    "DiT-B/16": lambda **kw: DiT_B(16, **kw),
    "DiT-B/8": lambda **kw: DiT_B(8, **kw),
    "DiT-B/4": lambda **kw: DiT_B(4, **kw),
    "DiT-L/16": lambda **kw: DiT_L(16, **kw),
    "DiT-L/8": lambda **kw: DiT_L(8, **kw),
    "DiT-L/4": lambda **kw: DiT_L(4, **kw),
    "DiT-XL/16": lambda **kw: DiT_XL(16, **kw),
    "DiT-XL/8": lambda **kw: DiT_XL(8, **kw),
    "DiT-XL/2": lambda **kw: DiT_XL(2, **kw),
}
