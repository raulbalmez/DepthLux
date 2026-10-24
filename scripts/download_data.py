"""Download LOL-v1, LOL-v2-real and LOL-v2-synthetic from the Hugging Face Hub
mirrors into <out>/ (default: ./data).

    python scripts/download_data.py --out data
"""
import argparse
import os
import zipfile

from huggingface_hub import snapshot_download

MIRRORS = {
    "lolv1": "geekyrakshit/LoL-Dataset",
    "lolv2_real": "okhater/lolv2-real",
    "lolv2_syn": "okhater/lolv2-synthetic",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    args = p.parse_args()
    for name, repo in MIRRORS.items():
        path = os.path.join(args.out, name)
        snapshot_download(repo, repo_type="dataset", local_dir=path)
        zips = [f for f in os.listdir(path) if f.endswith(".zip")]
        for z in zips:  # LOL-v1 ships as a single archive
            with zipfile.ZipFile(os.path.join(path, z)) as f:
                f.extractall(path)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
