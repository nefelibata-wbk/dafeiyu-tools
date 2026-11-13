# -*- coding: utf-8 -*-
"""所有跟「这台电脑」有关的路径都收在这里。

第一次用，**先把下面标着「待填写」的几行改掉**（或者不改文件，用环境变量覆盖，
变量名写在每行注释里）。只有前两个是必须的，后面几个不填就不走那部分功能。

    FFMPEG       必须   ffmpeg.exe 的完整路径；填 "" 则从 PATH 里找
    OUT          必须   产物目录（建议 ./data）
    EDGE         可选   不填就取不到高清直链（会报错，见 README「已知限制」）
    EDGE_PROFILE 可选   一个**登录过 B站**的 Edge 用户目录，填了才有 1080P
    DEMUCS_PY    可选   人声分离用的 python，不填就跳过分离
    ASR_PY       可选   转写用的 python，不填就跳过转写
"""
import os

UNSET = "待填写"

# 环境变量优先，其次是这里写的值
def _p(key, default=UNSET):
    v = os.environ.get(key)
    return v if v else default


def ok(path):
    """这个路径是真的填了、而且存在吗。"""
    return bool(path) and path != UNSET and os.path.exists(path)


# ── 必填 ──────────────────────────────────────────────────────────────
FFMPEG = _p("FFMPEG")                 # 例：r"C:\ffmpeg\bin\ffmpeg.exe"
OUT = _p("VIDEO_OUT")                 # 例：r"D:\video-kit\data"

# ── 取高清直链用（不填的话只能拿到 360P，见 README）───────────────
EDGE = _p("EDGE")                     # 例：r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
EDGE_PROFILE = _p("EDGE_PROFILE")     # 例：r"D:\video-kit\edge-profile"
EDGE_PORT = int(_p("EDGE_PORT", "9333"))

# ── 可选的「声音」部分（不填就只出分镜图，不报错）──────────────────
DEMUCS_PY = _p("DEMUCS_PY", "")       # 装了 demucs 的那个 python
ASR_PY = _p("ASR_PY", "")             # 装了 faster-whisper 的那个 python
ASR_SCRIPT = _p("ASR_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "转写.py"))
MODELS_ROOT = _p("MODELS_ROOT", "")   # faster-whisper-<模型名> 的上级目录
TORCH_LIB = _p("TORCH_LIB", "")       # 只有 Windows + CUDA 才需要，填 torch\lib


def summary():
    """启动时打印一下现状，省得「为什么没字幕」靠猜。"""
    print("── 配置检查 ──")
    print("  ffmpeg    ", FFMPEG if ok(FFMPEG) else "未填/不存在（会退回 PATH）")
    print("  输出目录   ", OUT if OUT != UNSET else "未填")
    print("  Edge      ", EDGE if ok(EDGE) else "未填（取不到高清直链）")
    print("  登录档案   ", EDGE_PROFILE if ok(EDGE_PROFILE) else "未填（画质会被压）")
    print("  人声分离   ", "开" if ok(DEMUCS_PY) else "关（不填 DEMUCS_PY）")
    print("  转写      ", "开" if ok(ASR_PY) else "关（不填 ASR_PY）")
    print()
