""" This file contains Image Encoder definitions for MobileCLIP-S3 & -S4 models. """
import torch
import torch.nn as nn

from timm.models.fastvit import _create_fastvit, RepConditionalPosEnc
from timm.models.fastvit import MobileOneBlock
from functools import partial
from timm.models import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def _cfg(url='', **kwargs):
    return {
        'url': url, 'input_size': (3, 256, 256),
        'crop_pct': .95, 'interpolation': 'bicubic', 'fixed_input_size': False,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        "classifier": "head.fc",
        **kwargs
    }


def convolutional_stem_timm(
        in_chs: int,
        out_chs: int,
        act_layer: nn.Module = nn.GELU,
        inference_mode: bool = False
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_chs: Number of input channels.
        out_chs: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        MobileOneBlock(
            in_chs=in_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=False
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            group_size=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=False
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=1,
            stride=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=False
        ),
    )


class LayerNormChannel(nn.Module):
    """
    LayerNorm for Channel-first format 4D Tensor.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_features, eps=1e-05) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


@register_model
def fastvit_mci3(pretrained=False, **kwargs):
    """Instantiate L model variant."""
    updated_kwargs = {
        'num_classes': kwargs.get('num_classes'),
        'global_pool': kwargs.get('global_pool'),
        'drop_path_rate': kwargs.get('drop_path_rate'),
    }
    model_args = dict(
        layers=(2, 12, 24, 4, 2),
        embed_dims=(96, 192, 384, 768, 1536),
        mlp_ratios=(4, 4, 4, 4, 4),
        se_downsamples=(False, False, False, False, False),
        downsamples=(False, True, True, True, True),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7)), partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
        lkc_use_act=True,
        norm_layer=LayerNormChannel
    )

    stem_modified = convolutional_stem_timm(
        3,
        model_args['embed_dims'][0],
        nn.GELU,
        inference_mode=False,
    )
    model = _create_fastvit('fastvit_mci3', pretrained=pretrained, pretrained_cfg=_cfg(), **dict(model_args, **updated_kwargs))
    model.stem = stem_modified
    return model


@register_model
def fastvit_mci4(pretrained=False, **kwargs):
    """Instantiate XL model variant."""
    updated_kwargs = {
        'num_classes': kwargs.get('num_classes'),
        'global_pool': kwargs.get('global_pool'),
        'drop_path_rate': kwargs.get('drop_path_rate'),
    }
    model_args = dict(
        layers=(2, 12, 24, 4, 4),
        embed_dims=(128, 256, 512, 1024, 2048),
        mlp_ratios=(4, 4, 4, 4, 4),
        se_downsamples=(False, False, False, False, False),
        downsamples=(False, True, True, True, True),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7)), partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention", "attention"),
        lkc_use_act=True,
        norm_layer=LayerNormChannel
    )

    stem_modified = convolutional_stem_timm(
        3,
        model_args['embed_dims'][0],
        nn.GELU,
        inference_mode=False,
    )
    model = _create_fastvit('fastvit_mci4', pretrained=pretrained, pretrained_cfg=_cfg(), **dict(model_args, **updated_kwargs))
    model.stem = stem_modified
    return model
