# -*- coding: utf-8 -*-
"""人声分离：任意音频 → vocals.wav（GPU，htdemucs）。

用法：<装了 demucs 的 python> _separate.py 输入 [输出目录]
默认输出到 视频解读/<同名>/ 或 输入同目录下的 _vocals/。

🔴 2026-11-02 实测：
  - torch 升级到 2.5.1+cu121 后 GPU 可用，30 秒音频分离只要 2.0 秒（RTF 0.07）
  - 但 demucs 启动时会去 huggingface.co 做 HEAD 检查，国内连不通会卡 2~3 分钟
    （第一版跑出 185 秒就是这么来的，不是 GPU 慢）。所以这里强制离线，
    只用本地缓存的 htdemucs 权重。
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 必须在 import demucs 之前
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
FFMPEG = config.FFMPEG if config.ok(config.FFMPEG) else "ffmpeg"


def load_audio(path):
    """非 wav 先用 ffmpeg 转成 44.1k 立体声 wav，避开 av/解码器版本问题。"""
    if path.lower().endswith(".wav"):
        return sf.read(path, always_2d=True)
    import subprocess
    import tempfile

    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", path,
                    "-ar", "44100", "-ac", "2", wav], check=True)
    data = sf.read(wav, always_2d=True)
    os.remove(wav)
    return data


def main():
    src = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(src)), "_vocals")
    os.makedirs(out_dir, exist_ok=True)

    wav, sr = load_audio(src)
    # 有 CUDA 就用，没有就退回 CPU（慢一些，但不用装显卡驱动）
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("没有可用 GPU，用 CPU 分离（会慢很多）", flush=True)
    model = get_model("htdemucs")
    model.to(dev).eval()
    x = torch.from_numpy(wav.T).float().to(dev)
    with torch.no_grad():
        out = apply_model(model, x[None], device=dev, progress=True)[0]
    idx = model.sources.index("vocals")
    vocals = out[idx].cpu().numpy().T
    dst = os.path.join(out_dir, "vocals.wav")
    sf.write(dst, vocals, sr)
    print(f"{len(wav) / sr:.1f}s 音频  {sr}Hz  ->  {dst}")


if __name__ == "__main__":
    main()
