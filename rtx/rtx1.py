from functools import partial
import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from typing import List, Optional, Callable, Tuple

# from beartype import beartype
from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from classifier_free_guidance_pytorch import (
    TextConditioner as FilmTextConditioner,
    AttentionTextConditioner as FilmAttentionTextConditioner,
    classifier_free_guidance,
)


# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


# sinusoidal positions


def posemb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float32):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)


# helper classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_fn=None):
        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate=4,
    shrinkage_rate=0.25,
    dropout=0.0,
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim,
            hidden_dim,
            3,
            stride=stride,
            padding=1,
            groups=hidden_dim,
        ),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out),
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes


class Attention(nn.Module):
    def __init__(self, dim, dim_head=32, dropout=0.0, window_size=7):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        grid = rearrange(grid, "c i j -> (i j) c")
        rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
            grid, "j ... -> 1 j ..."
        )
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        (
            batch,
            height,
            width,
            window_height,
            window_width,
            _,
            device,
            h,
        ) = (
            *x.shape,
            x.device,
            self.heads,
        )

        x = self.norm(x)

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d ) -> b h n d", h=h),
            (q, k, v),
        )

        # scale

        q = q * self.scale

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(
            out,
            "b h (w1 w2) d -> b w1 w2 (h d)",
            w1=window_height,
            w2=window_width,
        )

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class FilmViTConfig:
    """Configuration class to store the configuration of a `FilmMaxVit`."""

    def __init__(
        self,
        num_classes=1000,  # 1000 for ImageNet
        input_channels=3,
        stem_channels_in=64,  # Number of stem channels
        dim_head=32,  # Attention head dimension
        block_channel_ins: List = [
            64,
            128,
            256,
            512,
        ],  # Number of channels for each ViT block
        block_layers=[
            2,
            2,
            5,
            2,
        ],  # Number of layers for each ViT block
        window_size=7,  # Partition size
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,  # MBConv squeeze ratio
        dropout=0.1,
        norm_layer: nn.Module = None,
        activation_layer=nn.GELU,
        stochastic_depth_prob=0.2,
        pretrained=False,
    ):
        """
        Constructs a MaxVit architecture with optional film layers from
        `MaxVit: Multi-Axis Vision Transformer <https://arxiv.org/abs/2204.01697>`_.
            Parameters
            ----------
            num_classes : int
                Number of classes for the classification task
            input_channels : int
                Number of input channels
            stem_channels_in : int
                Number of stem channels
            dim_head : int
                Dimension of the head
            block_channel_ins : List
                Number of channels for each ViT block
            block_layers : List
                Number of layers for each ViT block
            window_size : int
                Partition size
            mbconv_expansion_rate : int
                MBConv expansion rate
            mbconv_shrinkage_rate : float
                MBConv squeeze ratio
            dropout : float
                Dropout probability
            norm_layer : nn.Module
                Normalization layer
            activation_layer : nn.Module
                Activation layer
            stochastic_depth_prob : float
                Stochastic depth probability
        """
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.stem_channels_in = stem_channels_in
        self.block_channel_ins = block_channel_ins
        self.block_layers = block_layers
        self.dim_head = dim_head
        self.stem_channels_in = stem_channels_in
        self.window_size = window_size
        self.mbconv_expansion_rate = mbconv_expansion_rate
        self.mbconv_shrinkage_rate = mbconv_shrinkage_rate
        self.dropout = dropout
        self.norm_layer = norm_layer
        if self.norm_layer is None:
            self.norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99)
        self.activation_layer = activation_layer
        self.pretrained = pretrained
        self.stochastic_depth_prob = stochastic_depth_prob


class FilmMaxVit(nn.Module):
    def __init__(
        self,
        config: FilmViTConfig,
    ):
        super().__init__()
        assert isinstance(config.block_layers, tuple | list), (
            "depth needs to be tuple if integers indicating number of"
            " transformer blocks at that stage"
        )

        # List of number of input and output channels for each ViT block.
        in_channels: List = [config.stem_channels_in] + config.block_channel_ins[:-1]
        out_channels: List = config.block_channel_ins

        # Condition after each layer starting with the input to the stem block.
        self.cond_hidden_dims = [config.stem_channels_in]  # Used by FilmTextConditioner
        for block_in_channels, block_layers in zip(out_channels, config.block_layers):
            for _ in range(block_layers):
                self.cond_hidden_dims.append(block_in_channels)
        self.cond_hidden_dims = self.cond_hidden_dims[
            :-1
        ]  # Don't condition on last embedding.
        self.embed_dim = out_channels[-1]

        if config.pretrained:
            from torchvision.models import maxvit_t, MaxVit_T_Weights

            self._vit = maxvit_t(weights=MaxVit_T_Weights.DEFAULT)
            self.conv_stem = self._vit.stem
            self.mlp_head = self._vit.classifier
            self.layers = nn.ModuleList([])
            for block in self._vit.blocks:
                for layer in block.layers:
                    self.layers.append(layer)
            return

        # convolutional stem
        self.conv_stem = nn.Sequential(
            nn.Conv2d(
                config.input_channels,
                config.stem_channels_in,
                3,
                stride=2,
                padding=1,
            ),
            nn.Conv2d(
                config.stem_channels_in,
                config.stem_channels_in,
                3,
                padding=1,
            ),
        )
        self.layers = nn.ModuleList([])

        for (
            block_channels_in,
            block_channels_out,
            block_num_layers,
        ) in zip(in_channels, out_channels, config.block_layers):
            for i in range(block_num_layers):
                layer_channels_in = block_channels_in if i == 0 else block_channels_out

                layer = nn.Sequential(
                    MBConv(
                        layer_channels_in,
                        block_channels_out,
                        downsample=(i == 0),
                        expansion_rate=config.mbconv_expansion_rate,
                        shrinkage_rate=config.mbconv_shrinkage_rate,
                    ),
                    Rearrange(
                        "b d (x w1) (y w2) -> b x y w1 w2 d",
                        w1=config.window_size,
                        w2=config.window_size,
                    ),  # block-like attention
                    Residual(
                        Attention(
                            dim=block_channels_out,
                            dim_head=config.dim_head,
                            dropout=config.dropout,
                            window_size=config.window_size,
                        )
                    ),
                    Residual(
                        FeedForward(
                            dim=block_channels_out,
                            dropout=config.dropout,
                        )
                    ),
                    Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
                    Rearrange(
                        "b d (w1 x) (w2 y) -> b x y w1 w2 d",
                        w1=config.window_size,
                        w2=config.window_size,
                    ),  # grid-like attention
                    Residual(
                        Attention(
                            dim=block_channels_out,
                            dim_head=config.dim_head,
                            dropout=config.dropout,
                            window_size=config.window_size,
                        )
                    ),
                    Residual(
                        FeedForward(
                            dim=block_channels_out,
                            dropout=config.dropout,
                        )
                    ),
                    Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
                )

                self.layers.append(layer)

        # mlp head out

        self.mlp_head = nn.Sequential(
            Reduce("b d h w -> b d", "mean"),
            LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, config.num_classes, bias=False),
        )

    # @beartype
    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob=0.0,
        return_embeddings=False,
    ):
        x = self.conv_stem(x)

        cond_fns = iter(default(cond_fns, []))

        for stage in self.layers:
            cond_fn = next(cond_fns, None)

            if exists(cond_fn):
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)


# attention


class TransformerAttention(nn.Module):
    def __init__(
        self,
        dim,
        causal=False,
        dim_head=64,
        dim_context=None,
        heads=8,
        norm_context=False,
        dropout=0.1,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_head * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context=None,
        mask=None,
        attn_bias=None,
        attn_mask=None,
        cond_fn: Optional[Callable] = None,
    ):
        x.shape[0]

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        if exists(cond_fn):
            # adaptive layer-norm
            x = cond_fn(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)

        q = q * self.scale

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        if exists(attn_mask):
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=x.device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# @beartype
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        depth=6,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        TransformerAttention(
                            dim=dim, heads=heads, dropout=attn_dropout
                        ),
                        FeedForward(dim=dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask=None,
    ):
        cond_fns = iter(default(cond_fns, []))

        for attn, ff in self.layers:
            x = (
                attn(
                    x,
                    attn_mask=attn_mask,
                    cond_fn=next(cond_fns, None),
                )
                + x
            )
            x = ff(x, cond_fn=next(cond_fns, None)) + x
        return x


# token learner module


class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8, num_layers=2):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(
                dim * num_output_tokens,
                inner_dim,
                1,
                groups=num_output_tokens,
            ),
            nn.GELU(),
            nn.Conv2d(
                inner_dim,
                num_output_tokens,
                1,
                groups=num_output_tokens,
            ),
        )

    def forward(self, x):
        x, ps = pack_one(x, "* c h w")
        x = repeat(x, "b c h w -> b (g c) h w", g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, "b g h w -> b 1 g h w")
        x = rearrange(x, "b (g c) h w -> b c g h w", g=self.num_output_tokens)

        x = reduce(x * attn, "b c g h w -> b c g", "mean")
        x = unpack_one(x, ps, "* c n")
        return x


# Robotic Transformer


class RT1Config:
    def __init__(
        self,
        num_actions=11,
        action_bins=256,
        depth=6,
        heads=8,
        dim_head=64,
        token_learner_ff_mult=2,
        token_learner_num_layers=2,
        token_learner_num_output_tokens=8,
        cond_drop_prob=0.2,
        use_attn_conditioner=False,
    ):
        """Configuration class to store the configuration of a `RT1`.

        Args:
            num_actions (int): Number of actions for the classification task
            action_bins (int): Number of bins for each action
            depth (int): Number of transformer blocks
            heads (int): Number of heads for the transformer
            dim_head (int): Dimension of the head
            token_learner_ff_mult (int): Multiplier for the token learner
            token_learner_num_layers (int): Number of layers for the token learner
            token_learner_num_output_tokens (int): Number of output tokens for the token learner
            cond_drop_prob (float): Dropout probability
            use_attn_conditioner (bool): Whether to use the attention conditioner
        """
        self.num_actions = num_actions
        self.action_bins = action_bins
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.token_learner_ff_mult = token_learner_ff_mult
        self.token_learner_num_layers = token_learner_num_layers
        self.token_learner_num_output_tokens = token_learner_num_output_tokens
        self.cond_drop_prob = cond_drop_prob
        self.use_attn_conditioner = use_attn_conditioner


# @beartype
class RT1(nn.Module):
    def __init__(
        self,
        config: RT1Config,
        vit: FilmMaxVit,
        conditioner_kwargs: dict = dict(),
    ):
        super().__init__()
        self.vit = vit
        self.num_vit_stages = len(vit.cond_hidden_dims)

        film_layer = (
            FilmAttentionTextConditioner
            if config.use_attn_conditioner
            else FilmTextConditioner
        )

        self.conditioner = film_layer(
            hidden_dims=(
                *tuple(vit.cond_hidden_dims),
                *((vit.embed_dim,) * config.depth * 2),
            ),
            hiddens_channel_first=(
                *((True,) * self.num_vit_stages),
                *((False,) * config.depth * 2),
            ),
            cond_drop_prob=config.cond_drop_prob,
            **conditioner_kwargs,
        )

        self.token_learner = TokenLearner(
            dim=vit.embed_dim,
            ff_mult=config.token_learner_ff_mult,
            num_output_tokens=config.token_learner_num_output_tokens,
            num_layers=config.token_learner_num_layers,
        )

        self.num_learned_tokens = config.token_learner_num_output_tokens

        self.transformer_depth = config.depth

        self.transformer = Transformer(
            dim=vit.embed_dim,
            dim_head=config.dim_head,
            heads=config.heads,
            depth=config.depth,
        )

        self.cond_drop_prob = config.cond_drop_prob

        self.to_logits = nn.Sequential(
            LayerNorm(vit.embed_dim),
            nn.Linear(vit.embed_dim, config.num_actions * config.action_bins),
            Rearrange("... (a b) -> ... a b", b=config.action_bins),
        )

    def embed_texts(self, texts: List[str]):
        return self.conditioner.embed_texts(texts)

    @classifier_free_guidance
    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob=0.0,
    ):
        assert exists(texts) ^ exists(text_embeds)
        cond_kwargs = dict(texts=texts, text_embeds=text_embeds)

        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[2], video.device

        cond_fns, _ = self.conditioner(
            **cond_kwargs,
            cond_drop_prob=cond_drop_prob,
            repeat_batch=(
                *((frames,) * self.num_vit_stages),
                *((1,) * self.transformer_depth * 2),
            ),
        )

        vit_cond_fns, transformer_cond_fns = (
            cond_fns[: -(depth * 2)],
            cond_fns[-(depth * 2) :],
        )

        video = rearrange(video, "b c f h w -> b f c h w")
        images, packed_shape = pack_one(video, "* c h w")

        tokens = self.vit(
            images,
            texts=texts,
            cond_fns=vit_cond_fns,
            cond_drop_prob=cond_drop_prob,
            return_embeddings=True,
        )

        tokens = unpack_one(tokens, packed_shape, "* c h w")
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, "b f c n -> b (f n) c")

        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype=torch.bool, device=device).triu(
            1
        )
        attn_mask = repeat(
            attn_mask,
            "i j -> (i r1) (j r2)",
            r1=self.num_learned_tokens,
            r2=self.num_learned_tokens,
        )

        # sinusoidal positional embedding

        pos_emb = posemb_sincos_1d(
            frames,
            learned_tokens.shape[-1],
            dtype=learned_tokens.dtype,
            device=learned_tokens.device,
        )

        learned_tokens = learned_tokens + repeat(
            pos_emb, "n d -> (n r) d", r=self.num_learned_tokens
        )

        # attention

        attended_tokens = self.transformer(
            learned_tokens,
            cond_fns=transformer_cond_fns,
            attn_mask=~attn_mask,
        )

        pooled = reduce(attended_tokens, "b (f n) d -> b f d", "mean", f=frames)

        logits = self.to_logits(pooled)
        return logits


class RTX1(nn.Module):
    """
    A class for real-time video processing using Vision Transformers (ViT) and Reinforcement Learning (RT1) models.

    ...

    Attributes
    ----------
    vit : FilmMaxVit
        a Vision Transformer model
    model : RT1
        a reinforcement learning model

    Methods
    -------
    train(video, instructions):
        Computes the logits for the given video and instructions using the RT1 model in training mode.
    eval(video, instructions, cond_scale=1.0):
        Computes the logits for the given video and instructions using the RT1 model in evaluation mode.
    """

    def __init__(
        self,
        rt1_config: RT1Config = None,
        vit_config: FilmViTConfig = None,
    ):
        """
        Constructs all the necessary attributes for the RTX1 object.

        Parameters
        ----------
        rt1_config : RT1Config, optional
            a configuration object for the RT1 model (default is None)
        vit_config : FilmViTConfig, optional
            a configuration object for the ViT model (default is None)



        Example:

        import torch
        from rtx import RTX1

        model = RTX1()

        video = torch.randn(2, 3, 6, 224, 224)

        instructions = ["bring me that apple sitting on the table", "please pass the butter"]

        # compute the train logits
        train_logits = model.train(video, instructions)

        # set the model to evaluation mode
        model.model.eval()

        # compute the eval logits with a conditional scale of 3
        eval_logits = model.run(video, instructions, cond_scale=3.0)
        print(eval_logits.shape)
        """
        super().__init__()
        if rt1_config is None:
            rt1_config = RT1Config()
        if vit_config is None:
            vit_config = FilmViTConfig()

        self.vit = FilmMaxVit(vit_config)
        self.model = RT1(
            config=rt1_config,
            vit=self.vit,
        )

    def train(self, video, instructions):
        """
        Computes the logits for the given video and instructions using the RT1 model in training mode.

        Parameters
        ----------
        video : torch.Tensor
            a tensor containing the video data
        instructions : torch.Tensor
            a tensor containing the instructions

        Returns
        -------
        torch.Tensor
            a tensor containing the computed logits
        """

        try:
            train_logits = self.model(video, instructions)
            return train_logits
        except Exception as e:
            raise RuntimeError("Error in training: {}".format(e))

    def run(self, video, instructions, cond_scale=1.0):
        """
        Computes the logits for the given video and instructions using the RT1 model in evaluation mode.

        Parameters
        ----------

        video : torch.Tensor
            a tensor containing the video data
        instructions : torch.Tensor
            a tensor containing the instructions
        cond_scale : float, optional
            a scale factor for the conditional scaling (default is 1.0)

        Returns
        -------
        torch.Tensor
            a tensor containing the computed logits
        """

        try:
            self.model.eval()
            # shape => 2, 3, 6, 224, 224
            eval_logits = self.model(video, instructions, cond_scale=cond_scale)
            return eval_logits
        except Exception as e:
            raise RuntimeError("Error in evaluation: {}".format(e))
