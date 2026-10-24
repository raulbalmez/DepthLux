import tensorflow as tf


def psnr_loss(y_true, y_pred):
    """L_PSNR = 40 - PSNR(pred, gt)  (Eq. 9). Images in [0, 1]."""
    return 40.0 - tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim_loss(y_true, y_pred):
    """L_SSIM = 1 - SSIM(y_target, y_pred)  (Eq. 10)."""
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


LOSSES = {"psnr": psnr_loss, "ssim": ssim_loss, "l1": l1_loss, "l2": l2_loss}


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(
        tf.clip_by_value(y_true, 0, 1), tf.clip_by_value(y_pred, 0, 1), 1.0))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(
        tf.clip_by_value(y_true, 0, 1), tf.clip_by_value(y_pred, 0, 1), 1.0))
