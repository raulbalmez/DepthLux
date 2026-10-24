import numpy as np
import tensorflow as tf

from .losses import psnr_metric, ssim_metric

DIV = 32 * 4  # 3 downsamplings x POAB contraction


def pad_to_multiple(x, m=DIV):
    h, w = x.shape[1], x.shape[2]
    ph, pw = (-h) % m, (-w) % m
    return tf.pad(x, [[0, 0], [0, ph], [0, pw], [0, 0]], "REFLECT"), h, w


def gt_mean_scale(pred, gt):
    """Retinexformer's --gt_mean trick: rescale the prediction so that its
    mean grey level equals that of the ground truth (done per image)."""
    mp = tf.reduce_mean(tf.image.rgb_to_grayscale(pred))
    mg = tf.reduce_mean(tf.image.rgb_to_grayscale(gt))
    return tf.clip_by_value(pred * (mg / mp), 0.0, 1.0)


def default_gt_mean(dataset):
    """All reported PSNR/SSIM are measured on GT-mean corrected outputs."""
    return True


def evaluate_model(model, ds, gt_mean=False):
    ps, ss = [], []
    for low, high in ds:
        x, h, w = pad_to_multiple(low)
        pred = tf.clip_by_value(model(x, training=False)[:, :h, :w], 0., 1.)
        if gt_mean:
            pred = gt_mean_scale(pred, high)
        ps.append(float(psnr_metric(high, pred)))
        ss.append(float(ssim_metric(high, pred)))
    return float(np.mean(ps)), float(np.mean(ss))
