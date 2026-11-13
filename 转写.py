# -*- coding: utf-8 -*-
"""快速转写：先用 ffmpeg 解成 16k 单声道浮点，再喂给 faster-whisper。
绕开 PyAV 版本不兼容（av 19 去掉了 metadata_errors，faster-whisper 1.2.1 还在传）。"""
import os
import subprocess
import sys

# 单独跑有时会报 "cublas64_12.dll is not found"——那个 DLL 在 torch\lib 里，
# 以前只有主脚本会往 PATH 塞，直接调用就崩。填了 config.TORCH_LIB 就补上。
# 放在 import faster_whisper 之前：ctranslate2 一加载就要找它。
# PATH 给 CTranslate2 内部的 LoadLibrary 用；add_dll_directory 给 Python 3.8+
# 的扩展加载用——两个都加，少一个都可能还是找不到。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

_TORCH_LIB = config.TORCH_LIB
if _TORCH_LIB and os.path.isdir(_TORCH_LIB):
    os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_TORCH_LIB)
    except (AttributeError, OSError):
        pass

import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel

# 参数：<音频> <模型目录> [热词] [--batch N]
# --batch 0 = 退回逐段模式（要逐句时间轴时用，慢 3.9 倍）
args = list(sys.argv[1:])
batch = 8
if "--batch" in args:
    i = args.index("--batch")
    batch = int(args[i + 1])
    del args[i:i + 2]
src, model_dir = args[0], args[1]
# 2026-11-05：热词。专有名词靠"更大的模型"是治不好的——large-v3 也会把
# 「福瑞」听成「芙蕊」。它缺的是词典，不是耳朵。faster-whisper 1.2.1 的
# transcribe() 原生支持 hotwords（比 initial_prompt 更对口：后者是"上一段
# 上下文"，前者是"接下去会出现的词"）。
hotwords = args[2] if len(args) > 2 and args[2].strip() else None
# ⚠️ 2026-11-05 实测：热词里带英文（如 "Claude"）时，**批处理模式**会把整段
#    输出成繁体/英文混杂（逐一对比过：同样的热词，逐段模式只漂第一句）。
#    纯中文热词（"福瑞 furry 兽圈"）在批处理下完全正常，且热词确实生效。
#    真遇到这种，就把热词换成中文，或者 --batch 0 退回逐段。
_FFMPEG = config.FFMPEG if config.ok(config.FFMPEG) else "ffmpeg"
pcm = subprocess.run(
    [_FFMPEG, "-v", "error", "-i", src,
     "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
    capture_output=True, check=True).stdout
audio = np.frombuffer(pcm, dtype=np.float32)

# 有 CUDA 用 float16，没有就用 CPU + int8（慢，但跑得起来）
import torch  # noqa: E402
if torch.cuda.is_available():
    _DEV, _CT = "cuda", "float16"
else:
    _DEV, _CT = "cpu", "int8"
m = WhisperModel(model_dir, device=_DEV, compute_type=_CT)

# 2026-11-05：批处理提速。large-v3 在 15.4 分钟音频上：
#   逐段 158.5s → batch=4 52.0s → batch=8 40.5s → batch=16 218.2s（爆显存，反而更慢）
# 8 GB 卡上 batch=8 是最优点，还比 turbo 逐段（44.4s）更快、更准。
# 代价是时间戳变粗（771 段→40 段），要逐句字幕就 --batch 0。
# 显存被别的任务占着时 batch=8 可能 OOM，所以留一条降级链。
def _run(b):
    if b > 0:
        pipe = BatchedInferencePipeline(model=m)
        return pipe.transcribe(audio, language="zh", beam_size=5, vad_filter=True,
                               hotwords=hotwords, batch_size=b)
    return m.transcribe(audio, language="zh", beam_size=5, vad_filter=True,
                        hotwords=hotwords)


tried = []
for b in ([batch, 4, 1] if batch > 1 else [batch]):
    if b in tried:
        continue
    tried.append(b)
    try:
        segs, info = _run(b)
        if b != batch:
            print("  （显存不够，batch 降到 %d）" % b, flush=True)
        break
    except RuntimeError as e:
        if "out of memory" not in str(e).lower() and "cublas" not in str(e).lower():
            raise
        if b == tried[-1] and b == 1:
            raise

for s in segs:
    print(f"[{s.start:6.1f} -> {s.end:6.1f}] {s.text}")
