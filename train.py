"""Train DepthLux (Sec. 4.2): Adam(0.9, 0.999), cosine annealing 2e-4 -> 1e-6,
L_PSNR loss, 256x256 random crops + flips."""
import argparse
import tensorflow as tf

from depthlux import build_depthlux
from depthlux.data import make_dataset
from depthlux.evaluation import default_gt_mean, evaluate_model
from depthlux.losses import LOSSES, psnr_metric, ssim_metric


class EvalCallback(tf.keras.callbacks.Callback):
    """Full-image PSNR/SSIM on the test split (with GT-mean on LOL-v2 as in
    Retinexformer); keeps the best weights."""

    def __init__(self, ds, gt_mean, path, every, target=None, tol=0.05):
        super().__init__()
        self.ds, self.gt_mean, self.path, self.every = ds, gt_mean, path, every
        self.best = -1.0
        self.target, self.tol = target, tol

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every:
            return
        psnr, ssim = evaluate_model(self.model, self.ds, self.gt_mean)
        print(f"\n[eval] epoch {epoch + 1}: PSNR {psnr:.2f} SSIM {ssim:.4f}",
              flush=True)
        if psnr > self.best:
            self.best = psnr
            self.model.save_weights(self.path)
        if self.target is not None and psnr >= self.target - self.tol:
            print(f"[eval] reached paper PSNR {self.target} (+/-{self.tol}); "
                  "stopping.", flush=True)
            self.model.stop_training = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="lolv1",
                   choices=["lolv1", "lolv2_real", "lolv2_syn"])
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--loss", default="psnr", choices=list(LOSSES))
    p.add_argument("--activation", default="gelu",
                   choices=["gelu", "relu", "prelu", "none"])
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--target_psnr", type=float, default=None,
                   help="stop once test PSNR >= target - tol (paper value)")
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    train_ds, n_train = make_dataset(args.data_root, args.dataset, "train",
                                     args.batch_size, args.patch_size)
    val_ds, _ = make_dataset(args.data_root, args.dataset, "test")
    steps = max(n_train // args.batch_size, 1)

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        args.lr, decay_steps=steps * args.epochs, alpha=args.lr_min / args.lr)
    opt = tf.keras.optimizers.Adam(schedule, beta_1=0.9, beta_2=0.999)

    model = build_depthlux(activation=args.activation)
    model.compile(opt, loss=LOSSES[args.loss],
                  metrics=[psnr_metric])
    model.summary()

    import os
    os.makedirs(args.out, exist_ok=True)
    cb = EvalCallback(val_ds, default_gt_mean(args.dataset),
                      f"{args.out}/{args.dataset}_best.weights.h5",
                      args.eval_every, args.target_psnr, args.tol)
    model.fit(train_ds, epochs=args.epochs, steps_per_epoch=steps,
              callbacks=[cb], verbose=2)
    model.save_weights(f"{args.out}/{args.dataset}_last.weights.h5")


if __name__ == "__main__":
    main()
