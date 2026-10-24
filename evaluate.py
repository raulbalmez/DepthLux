"""Evaluate a checkpoint with PSNR / SSIM on a LOL test split."""
import argparse

from depthlux import build_depthlux
from depthlux.data import make_dataset
from depthlux.evaluation import default_gt_mean, evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--dataset", default="lolv1",
                   choices=["lolv1", "lolv2_real", "lolv2_syn"])
    p.add_argument("--weights", required=True)
    p.add_argument("--activation", default="gelu")
    p.add_argument("--gt_mean", choices=["auto", "on", "off"], default="auto",
                   help="auto = on (metrics on GT-mean corrected images); "
                        "uncorrected numbers are printed as well")
    args = p.parse_args()
    use = (args.gt_mean == "on" or
           (args.gt_mean == "auto" and default_gt_mean(args.dataset)))

    model = build_depthlux(activation=args.activation)
    model.load_weights(args.weights)
    ds, _ = make_dataset(args.data_root, args.dataset, "test")
    for gm in sorted({False, use}):
        psnr, ssim = evaluate_model(model, ds, gm)
        print(f"{args.dataset} (gt_mean={gm}): PSNR {psnr:.2f}  SSIM {ssim:.4f}")


if __name__ == "__main__":
    main()
