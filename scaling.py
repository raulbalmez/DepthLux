"""Sec. 4.4 / Fig. 5: parameter count of DSC vs standard-convolution models as
the layer-dimension scaling factor grows."""
import argparse
import matplotlib.pyplot as plt

from depthlux import build_depthlux


def count(scale, use_dsc):
    # scale = base embedding size; DepthLux default is 64 -> dims (64,128,256)
    dims = (scale, scale * 2, scale * 4)
    m = build_depthlux((128, 128, 3), dims, dims[-1], use_dsc=use_dsc)
    return m.count_params()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", type=int, nargs="+", default=[32, 64, 80, 96, 128])
    p.add_argument("--out", default="scaling.png")
    args = p.parse_args()

    dsc = [count(s, True) / 1e6 for s in args.scales]
    std = [count(s, False) / 1e6 for s in args.scales]
    for s, a, b in zip(args.scales, dsc, std):
        print(f"size {s:4d}: DSC {a:7.2f}M  standard {b:7.2f}M  ({a / b:.0%})")
    plt.plot(args.scales, dsc, "o-", label="Depthwise Separable Convolutions")
    plt.plot(args.scales, std, "s--", label="Standard Convolutions")
    plt.xlabel("Model Size"); plt.ylabel("Parameters (Millions)")
    plt.title("Model Size vs. Number of Parameters"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
