"""Building blocks of DepthLux: DSC, DSIB (Sec. 3.1) and POAB (Sec. 3.2)."""
import tensorflow as tf
from tensorflow.keras import layers


def get_activation(name):
    """Activation used inside the DSC (ablation in Table 2a)."""
    name = (name or "none").lower()
    if name == "gelu":
        return layers.Activation(tf.nn.gelu)
    if name == "relu":
        return layers.ReLU()
    if name == "prelu":
        return layers.PReLU(shared_axes=[1, 2])
    if name in ("none", "linear"):
        return layers.Activation("linear")
    raise ValueError(f"Unknown activation: {name}")


class DSC(layers.Layer):
    """Depthwise separable convolution, Eq. (5):
    DSC(F) = act(PWConv(act(DWConv(F)))), with a 3x3 depthwise and a 1x1
    pointwise kernel.

    If ``use_dsc`` is False the layer is replaced by a standard 3x3
    convolution (+ activation). This is only used for the parameter
    comparison of Sec. 4.4 / Fig. 5.
    """

    def __init__(self, filters, activation="gelu", use_dsc=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_dsc = use_dsc
        if use_dsc:
            self.dw = layers.DepthwiseConv2D(3, padding="same")
            self.act1 = get_activation(activation)
            self.pw = layers.Conv2D(filters, 1)
            self.act2 = get_activation(activation)
        else:
            self.conv = layers.Conv2D(filters, 3, padding="same")
            self.act2 = get_activation(activation)

    def call(self, x):
        if self.use_dsc:
            x = self.act1(self.dw(x))
            return self.act2(self.pw(x))
        return self.act2(self.conv(x))


class DSIB(layers.Layer):
    """Depthwise Separable Illumination Block (feed-forward part).

    Two parallel branches of DSC(DSC(.)) are summed (Eq. 3), refined by a
    further DSC(DSC(.)), and concatenated with the block input (Eq. 4):
        F_out = [F_in, DSC(DSC(F'_in))]  -> 2C channels
    followed by layer normalisation (Fig. 3, left).
    """

    def __init__(self, activation="gelu", use_dsc=True, hidden=None, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.use_dsc = use_dsc
        self.hidden = hidden  # width of the DSC branches (default: input C)

    def build(self, input_shape):
        c = int(input_shape[-1])
        h = self.hidden or c
        mk = lambda: DSC(h, self.activation, self.use_dsc)
        self.branch1 = [mk(), mk()]
        self.branch2 = [mk(), mk()]
        self.refine = [mk(), mk()]
        self.norm = layers.LayerNormalization()
        super().build(input_shape)

    def call(self, x):
        b1 = self.branch1[1](self.branch1[0](x))
        b2 = self.branch2[1](self.branch2[0](x))
        f = b1 + b2                                           # Eq. (3)
        f = self.refine[1](self.refine[0](f))
        return self.norm(tf.concat([x, f], axis=-1))          # Eq. (4)


class POAB(layers.Layer):
    """Performance-Optimized Attention Block.

    Contraction -> self-attention -> expansion (Sec. 3.2, Fig. 3 right):
      F_down = conv3x3(F_in)           H/4 x W/4 x C/2        (stride 4)
      Q,K,V  = dense(F_down)           C channels each
      Attn   = softmax(QK^T / sqrt(dim)) V                    Eq. (7)
      Attn_out = convT3x3(Attn)        H x W x C              Eq. (8)
      F_out  = LN([F_in, Attn_out])    H x W x 2C
    """

    def __init__(self, reduction=4, down_ratio=1.25, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.down_ratio = down_ratio  # paper: C/2; 1.25 fits the 9.75M budget

    def build(self, input_shape):
        c = int(input_shape[-1])
        r = self.reduction
        self.down = layers.Conv2D(max(int(c * self.down_ratio), 1), 3, strides=r, padding="same")
        self.wq = layers.Dense(c)
        self.wk = layers.Dense(c)
        self.wv = layers.Dense(c)
        self.up = layers.Conv2DTranspose(c, 3, strides=r, padding="same")
        self.norm = layers.LayerNormalization()
        self.dim = c
        super().build(input_shape)

    def call(self, x):
        d = self.down(x)
        s = tf.shape(d)
        tokens = tf.reshape(d, [s[0], s[1] * s[2], d.shape[-1]])
        q, k, v = self.wq(tokens), self.wk(tokens), self.wv(tokens)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(
            tf.cast(self.dim, q.dtype))
        attn = tf.matmul(tf.nn.softmax(scores, axis=-1), v)
        attn = tf.reshape(attn, [s[0], s[1], s[2], self.dim])
        out = self.up(attn)
        return self.norm(tf.concat([x, out], axis=-1))


class TransformerBlock(layers.Layer):
    """POAB (attention) followed by DSIB (feed-forward) for ``dim`` channels.

    POAB doubles the channels (concat), the DSIB branches work at ``dim`` and
    its output is concatenated with its input (Eq. 4); a final DSC maps the
    result back to ``dim`` so that widths stay constant through the U-Net.
    """

    def __init__(self, dim, activation="gelu", use_dsc=True, hidden_ratio=0.75,
                 **kwargs):
        super().__init__(**kwargs)
        self.poab = POAB()
        self.dsib = DSIB(activation, use_dsc, hidden=int(dim * hidden_ratio))
        self.proj = DSC(dim, activation, use_dsc)

    def call(self, x):
        return self.proj(self.dsib(self.poab(x)))
