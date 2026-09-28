"""Vision Transformers trained from scratch (no pretrained weights), built with torchvision's ViT implementation.

vit_s16 = ViT-Small/16 (same size as DeiT-S): 12 blocks, 384-dim tokens, 6 heads, MLP 1536, 16x16 patches.
          ~22M parameters, i.e. about the same as our ResNet-36, for a fair CNN-vs-Transformer comparison.
vit_t16 = ViT-Tiny/16 (DeiT-Ti): 12 blocks, 192-dim, 3 heads (~5.5M params), for quick tests.
"""
from torchvision.models.vision_transformer import VisionTransformer

VIT_CONFIGS = {
    "vit_s16": dict(patch_size=16, num_layers=12, num_heads=6, hidden_dim=384, mlp_dim=1536),
    "vit_t16": dict(patch_size=16, num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768),
}


def build_vit(name, num_classes, image_size=224):
    """Randomly initialized ViT. The image is cut into (image_size/16)^2 patches, each embedded as a token;
    a learnable [class] token + position embeddings go through the Transformer encoder; the head reads the
    [class] token."""
    return VisionTransformer(image_size=image_size, num_classes=num_classes, **VIT_CONFIGS[name])
