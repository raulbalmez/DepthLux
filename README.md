<div align="center">

# DepthLux

### Employing Depthwise Separable Convolutions for Low-Light Image Enhancement

[![Paper](https://img.shields.io/badge/Sensors-10.3390%2Fs25051530-blue.svg?style=flat-square)](https://doi.org/10.3390/s25051530)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00.svg?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square)](LICENSE)

[**Paper**](https://doi.org/10.3390/s25051530) · [**Results**](#-results) · [**Citation**](#-citation)

</div>

---

**DepthLux** is a lightweight U-shaped Vision Transformer for low-light image enhancement (LLIE). Its feed-forward network is built exclusively from depthwise separable convolutions, and its self-attention is computed on a contracted feature space, which keeps the model small without giving up competitive quality.

## 📌 Highlights

- **Depthwise Separable Illumination Block (DSIB)** — a feed-forward network made only of depthwise separable convolutions with GELU, using two parallel multi-scale branches.
- **Performance-Optimized Attention Block (POAB)** — features are contracted before the attention map is computed and expanded afterwards, reducing the cost of self-attention.
- **Efficient** — **9.75M parameters**, about **37%** of the same network built with standard convolutions at every model size.
- **Competitive on LOL** — **26.06 / 26.16 / 28.69 dB PSNR** on LOL-v1 / LOL-v2-real / LOL-v2-synthetic with GT-mean correction.

## 📊 Results

### Low-Light Image Enhancement

| Method | Params (M) ↓ | LOL-v1 PSNR / SSIM ↑ | LOL-v2-real PSNR / SSIM ↑ | LOL-v2-syn PSNR / SSIM ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Restormer | 26.13 | 22.43 / 0.823 | 19.94 / 0.827 | 21.41 / 0.830 |
| SNR-Net | 4.01 | 24.61 / 0.842 | 21.48 / 0.849 | 24.14 / 0.928 |
| LLFlow | 37.68 | 25.13 / 0.872 | 26.20 / 0.888 | 24.81 / 0.919 |
| Retinexformer | **1.61** | 25.16 / 0.845 | 22.80 / 0.840 | 25.67 / 0.930 |
| LLFormer | 24.55 | 25.76 / 0.823 | 26.20 / 0.819 | 28.01 / 0.927 |
| LL-SKF | 39.91 | **26.80 / 0.879** | **28.45 / 0.905** | **29.11 / 0.953** |
| **DepthLux (ours)** | 9.75 | 26.06 / 0.793 | 26.16 / 0.794 | 28.69 / 0.920 |

Numbers are taken from Table 1 of the paper and are measured on full-resolution test images with the ground-truth-mean correction of Retinexformer (the output is rescaled so that its mean grey level matches the ground truth).

## ⚙️ Installation

**Prerequisite:** [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/).

**1. Clone the repository**

```bash
git clone https://github.com/raulbalmez/DepthLux.git
cd DepthLux
```

**2. Create the environment** (Python 3.13 was used for development and is recommended)

```bash
conda create -n depthlux python=3.13 -y
conda activate depthlux
```

**3. Install the requirements**

```bash
pip install -r requirements.txt
```

## 📂 Data

Download LOL-v1, LOL-v2-real and LOL-v2-synthetic with:

```bash
python scripts/download_data.py --out data
```

The datasets are placed under the `data/` folder.

## 🏋️ Training

```bash
python train.py --data_root <path/to/dataset> --dataset <lolv1|lolv2_real|lolv2_syn>
```

Training uses random 256×256 crops with flips, Adam, a cosine learning rate from 2e-4 to 1e-6 and the PSNR loss. To train and test on all three datasets in sequence, stopping each run at the paper PSNR:

```bash
./run_all.sh
```

## 🔍 Inference

```bash
python evaluate.py \
    --data_root <path/to/dataset> \
    --dataset <lolv1|lolv2_real|lolv2_syn> \
    --weights <path/to/weights.h5>
```

PSNR and SSIM are reported with and without the GT-mean correction.

## 🔧 Configuration

The ablations from the paper are available through command-line flags:

```bash
python train.py ... --activation {none,relu,prelu,gelu}
python train.py ... --loss {l1,l2,ssim,psnr}
```

The width of the DSIB branches and of the POAB contraction can be changed with `hidden_ratio` in `TransformerBlock` and `down_ratio` in `POAB` ([`depthlux/layers.py`](depthlux/layers.py)).

## 📄 Citation

If you find DepthLux useful in your research, please consider citing:

```bibtex
@article{balmez2025depthlux,
  title   = {DepthLux: Employing Depthwise Separable Convolutions for Low-Light Image Enhancement},
  author  = {Balmez, Raul and Brateanu, Alexandru and Orhei, Ciprian and Ancuti, Codruta O. and Ancuti, Cosmin},
  journal = {Sensors},
  volume  = {25},
  number  = {5},
  pages   = {1530},
  year    = {2025},
  doi     = {10.3390/s25051530}
}
```

## Acknowledgements

Thanks to the authors of [LOL](https://daooshee.github.io/BMVC2018website/) and [LOL-v2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) for making their datasets publicly available, and to the authors of [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) for the evaluation protocol.

<div align="center">

If you find this work useful, please consider starring the repository.

</div>
