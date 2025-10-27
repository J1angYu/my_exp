import torch
import timm


def create_plain_vit_small(model_name="vit_small_patch16_224"):
    # Use JAX initialization scheme to match big_vision
    model = timm.create_model(
        model_name,
        pretrained=False,
        global_pool='avg',
        weight_init='jax',
    )

    model = replace_pos_embed_with_sincos2d(model)

    return model


def posemb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32):
    """
    Implementation of 2D sine-cosine positional embeddings.
    """
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def replace_pos_embed_with_sincos2d(model, img_size=224, patch_size=16):
    """
    Replace the learned positional embeddings in a ViT model with fixed 2D sine-cosine embeddings.

    Args:
        model: The vision transformer model
        img_size: The input image size

    Returns:
        The model with updated positional embeddings
    """

    h = w = img_size // patch_size
    grid_size = h
    num_patches = grid_size ** 2

    # Generate sine-cosine positional embeddings
    pos_embed = posemb_sincos_2d(h, w, model.pos_embed.shape[-1])

    # Handle class token if present
    if model.cls_token is not None:
        pos_embed_with_cls = torch.zeros(1, num_patches + 1, model.pos_embed.shape[-1])
        pos_embed_with_cls[0, 1:] = pos_embed
        pos_embed = pos_embed_with_cls

    # Store as a buffer rather than parameter (following simple_vit approach)
    if hasattr(model, 'pos_embed'):
        del model.pos_embed

    model.register_buffer('pos_embed', pos_embed)

    return model
