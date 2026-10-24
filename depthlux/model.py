"""DepthLux model (Fig. 2): a 3-level U-shaped Transformer."""
import tensorflow as tf
from tensorflow.keras import layers, Model

from .layers import DSC, TransformerBlock


def build_depthlux(input_shape=(None, None, 3), dims=(64, 128, 256),
                   bottleneck_dim=256, activation="gelu", use_dsc=True,
                   name="DepthLux"):
    """Builds DepthLux.

    Args:
      input_shape: (H, W, 3); H and W must be divisible by 2**len(dims) * 4
        (3 downsamplings, plus the x4 contraction inside the POAB).
      dims: embedding sizes of the encoders / decoders, [64, 128, 256].
      bottleneck_dim: embedding size of the bottleneck, 256.
      activation: DSC activation (gelu | relu | prelu | none).
      use_dsc: False swaps every DSC for a standard 3x3 conv (Sec. 4.4).
    """
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(dims[0], 1)(inp)  # lift RGB to the first embedding size

    # Encoder: Transformer block -> 3x3 stride-2 downsample conv
    skips = []
    for i, d in enumerate(dims):
        x = TransformerBlock(d, activation, use_dsc, name=f"enc{i}")(x)
        skips.append(x)
        nxt = dims[i + 1] if i + 1 < len(dims) else bottleneck_dim
        x = layers.Conv2D(nxt, 3, strides=2, padding="same",
                          name=f"down{i}")(x)

    # Bottleneck: POAB + DSIB
    x = TransformerBlock(bottleneck_dim, activation, use_dsc,
                         name="bottleneck")(x)

    # Decoder: 3x3 transposed-conv upsample -> concat skip -> Transformer block
    for i in reversed(range(len(dims))):
        x = layers.Conv2DTranspose(dims[i], 3, strides=2, padding="same",
                                   name=f"up{i}")(x)
        x = layers.Concatenate()([x, skips[i]])
        x = DSC(dims[i], activation, use_dsc, name=f"fuse{i}")(x)
        x = TransformerBlock(dims[i], activation, use_dsc, name=f"dec{i}")(x)

    out = layers.Conv2D(3, 1, name="out_1x1")(x)
    return Model(inp, out, name=name)
