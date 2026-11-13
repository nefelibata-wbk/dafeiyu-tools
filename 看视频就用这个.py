# -*- coding: utf-8 -*-
# ════════════════════════════════════════════════════════════════════════
#
#   🔴🔴🔴  改这个文件之前，先把全文注释读一遍  🔴🔴🔴
#
# ════════════════════════════════════════════════════════════════════════
#
#   这是"看视频"的唯一入口，也是踩坑最多的一份代码——每一条注释背后都是
#   一次真实的翻车（漏段、静默降级、元数据撒谎、模型听错……）。
#
#   用户 2026-11-05 特意交代：
#     「你的看视频就用这个可是你呕心沥血写出来的，改之前一定要仔细看看哦」
#
#   同一天的教训（值得反复看）：
#     我本来打算"片尾取不到就退一秒钟重试"直接改，跑了五组诊断才发现
#     真正原因是 **B站 duration 比视频流实际末尾长 1 秒**——
#     照第一直觉改，会改到错误的地方。
#
#   改之前的三条硬规矩：
#     1. 先 Grep 关键词，把相关注释读全，别只盯着要改的那个函数
#     2. 先做诊断（取一帧、量一次、把错误信息读完），不许猜原因
#     3. 改完用**真视频**跑一遍，别只看 py_compile 通过
#
# ════════════════════════════════════════════════════════════════════════
#
#   📌 一条使用口径（比代码更重要）：
#
#      别人特意推给你的视频，默认按"细看"处理，不按普通流程扫一遍就下结论。
#
#      落到这个脚本上就是：
#        · **不加 --frames**（该参数已废弃，帧数跟画面变化走、没有硬顶）
#        · 分镜_分段_NN.jpg **每一张都要读**，别只看第一张
#        · 帧与帧之间接不上时，单独再取一帧，别靠猜
#        · 结论里凡是自己推出来的，明说是推的；看不清就说看不清
#
#      为什么要有这条：把抽帧留下的空白用推理填上，是最不容易被发现、
#      也最容易连续犯的错。宁可说"这里我没看到"，也别补一个像样的解释。
#
#   📎 顺带说清 28 这个数的来历：
#      默认的 **28 帧不是调出来的**，是早期把画面丢给一个描述模型时的上限——
#      超过 28 张它就不干了。后来改成自己看图 + 转写，"一次喂多少张"这个约束
#      就不存在了，所以帧数才改成跟画面变化走、没有硬顶。
#      —— 换句话说：28 是旧瓶颈留下的疤，不是设计。
#
# ════════════════════════════════════════════════════════════════════════
"""看视频：一条命令跑完"画面 + 声音"，不用再挑工具、记参数。

2026-11-02 建。用户说："设置为默认路径还会用其他呀，那不是还不够懒嘛"。
所以把整条链收成一个入口：

    下视频 → 找镜头硬切 → 按镜头选帧 → 出分镜图
           → 抽音轨 → 分离人声 → 转写
           → 打印"该读哪几张图 + 说了什么"

用法：
    python 看视频.py <BV号> [--frames 28] [--no-audio] [--again] [--asr medium|small]

    --again：这个 BV 之前跑过就别重复下载了——入口会自动查 视频解读/<BV>/，
             有字幕或分镜就拦下并告诉你什么时候看的。确实想重看才加 --again。

产物都在 视频解读/<BV>/ 下：
    分镜_切点.jpg        按镜头选好的帧（一张图看完）
    分镜_分段_NN.jpg     帧太多时自动分段，按提示分批读
    字幕.txt             转写（带时间戳）
    _vocals/vocals.wav   分离后的人声（补听细节用）

为什么音频要起子进程：分离和转写各自需要一套依赖，装在一起版本会打架，
   在同一个解释器里 import 不了。用 subprocess 调各自的 python 最省心。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import config  # noqa: E402
import frame_pick as af  # noqa: E402
import bilibili as vp  # noqa: E402

# 路径全在 config.py 里；没填的保持空，对应功能自己跳过（见 audio_chain）
FFMPEG = config.FFMPEG if config.ok(config.FFMPEG) else "ffmpeg"
OUT = config.OUT if config.OUT != config.UNSET else os.path.join(HERE, "data")
DEMUCS_PY = config.DEMUCS_PY
GPU_PY = config.ASR_PY
TORCH_LIB = config.TORCH_LIB


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def render(imgs, dst, cell_w=640, cols=3):
    thumbs = []
    for t, p in imgs:
        im = Image.open(p).convert("RGB")
        h = int(im.height * cell_w / im.width)
        thumbs.append((t, im.resize((cell_w, h), Image.LANCZOS)))
    cell_h = max(im.height for _, im in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (t, im) in enumerate(thumbs):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(im, (x, y))
        draw.rectangle([x, y, x + 78, y + 22], fill="black")
        draw.text((x + 4, y + 4), f"{t:.1f}s", fill="white")
    sheet.save(dst, quality=88)
    return dst, sheet.size


def grab(src, times, folder):
    """按时间点取帧。**取不到就往前退，别直接跳过。**

    🔴 2026-11-05 修，起因：看 BV16V4G6KEAh 时漏了结局。
       B站给的 duration 元数据常比**视频流实际末尾**长——这条 duration=23s，
       但实测 22.0s 能取到、22.1s 起就没有帧了。
       而 frame_pick 的片尾保帧是 dur-0.3=22.7s → 必然越界 → 原来只 print 一句
       "取帧失败，跳过"，于是整条视频看完了，**恰好漏掉最后一秒的结论**。

       现在的做法：仍从原时间点试起，失败就以 0.5 秒为步长往后退，最多退 4 次
       （最远 2 秒）。退成功要**报出来**，不能让"退了一步"变成新的静默降级。
       判据是"宁可比片尾早半秒，也不能没有片尾"。
    """
    os.makedirs(folder, exist_ok=True)
    imgs = []
    for i, t in enumerate(times, 1):
        got = None
        for back in range(5):                     # 0 / 0.5 / 1.0 / 1.5 / 2.0 秒
            tt = round(t - 0.5 * back, 2)
            if tt < 0:
                break
            p = os.path.join(folder, f"{i:02d}_{tt:06.1f}.jpg")
            if os.path.exists(p):                 # 上次跑过，直接用
                got = (tt, p)
                break
            try:
                subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", str(tt), "-i", src,
                                "-frames:v", "1", "-q:v", "3", p], check=True)
            except subprocess.CalledProcessError:
                continue
            if os.path.exists(p):
                got = (tt, p)
                if back:
                    print(f"  {t:.1f}s 无帧（元数据比实际长），退到 {tt:.1f}s")
                break
        if got:
            imgs.append(got)
        else:
            print(f"  取帧失败，退到 {(t - 2.0):.1f}s 仍无帧，跳过 {t:.1f}s")
    return imgs


def audio_chain(src, base, bv, page=1, asr="medium", hotwords="", skip_asr=False):
    """抽音轨 → 分离 → 转写。任何一步失败都不影响画面那部分。

    skip_asr=True 时跳过转写（已经拿到 B站 CC 字幕了，不用再花几分钟转一遍）。

    🔴 2026-11-02 坑：B站直链给的 mp4 是 **DASH 纯视频流，里面根本没有音轨**，
       ffmpeg 直接报 "Output file does not contain any stream"。所以抽不出来时
       要回头单独下音频流。
    """
    wav = os.path.join(base, "_audio.wav")
    vocals = os.path.join(base, "_vocals", "vocals.wav")
    txt = os.path.join(base, "字幕.txt")
    ok = subprocess.run([FFMPEG, "-y", "-v", "error", "-i", src, "-vn",
                         "-ar", "44100", "-ac", "2", wav],
                        capture_output=True).returncode == 0
    if not ok or not os.path.exists(wav) or os.path.getsize(wav) < 1000:
        print("  视频流里没音轨，单独下音频…", flush=True)
        m4a = os.path.join(base, "_audio.m4a")
        # 🔴 2026-11-02：下到一半超时会留下 0 字节的 _audio.m4a，
        #    下次看到"文件存在"就不再下，ffmpeg 直接报错。必须先看大小。
        if not os.path.exists(m4a) or os.path.getsize(m4a) < 1000:
            import urllib.request
            # 2026-11-04：一次超时就直接放弃，代价太大——没有音轨就没有字幕，
            # 整条流水线从"读一分钟文字"退化成"一张张贴分镜图"。BV17fnT6BEex
            # 就是这么慢下来的（TimeoutError）。
            # 重试要点：**每次重新 fetch_playurl**。B站直链带时效和 CDN 参数，
            # 拿旧链接重试等于白试。
            got = False
            for attempt in range(1, 4):
                _v, _r, audio_url, _q = vp.fetch_playurl(bv, page=page)
                if not audio_url:
                    print("  拿不到音频直链，跳过声音部分")
                    return None
                req = urllib.request.Request(audio_url, headers={
                    "User-Agent": "Mozilla/5.0",
                    "Referer": f"https://www.bilibili.com/video/{bv}/"})
                try:
                    with urllib.request.urlopen(req, timeout=300) as r, open(m4a, "wb") as f:
                        f.write(r.read())
                    got = True
                    break
                except Exception as e:
                    print("  音频下载失败（第 %d 次）：%s" % (attempt, type(e).__name__))
                    if os.path.exists(m4a):
                        os.remove(m4a)      # 别留半截文件害下一次
            if not got:
                print("  ⚠️ 音频三次都失败——这次只能看图，读起来会慢", flush=True)
                return None
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", m4a,
                        "-ar", "44100", "-ac", "2", wav], check=True)

    if not os.path.exists(vocals):
        if not config.ok(DEMUCS_PY):
            print("  没配 DEMUCS_PY，跳过人声分离（不影响分镜图）", flush=True)
            return None
        print("  分离人声…", flush=True)
        # 必须显式 utf-8：Windows 默认 GBK，子进程一输出中文就炸
        r = subprocess.run([DEMUCS_PY, os.path.join(os.path.dirname(__file__), "_separate.py"),
                            wav, os.path.join(base, "_vocals")],
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print("  分离失败:", (r.stderr or "")[-300:])
            return None

    if skip_asr:
        print("  已有 CC 字幕，跳过转写", flush=True)
        return vocals
    if not config.ok(GPU_PY):
        print("  没配 ASR_PY，跳过转写（人声已经分离好了）", flush=True)
        return vocals
    snap, used = _whisper_snapshot(asr)
    if not snap:
        print("  找不到本地 whisper 模型，跳过转写")
        return vocals
    if used != asr:
        # 2026-11-04：以前这里静默退回 small，日志却写着"模型 medium"——
        # 用户看到的是假的，我也以为换了。现在退回必须说出来。
        print("  ⚠️ 要的是 %s，但它缺 model.bin；实际用 %s" % (asr, used), flush=True)
    print("  转写（模型 %s）…" % used, flush=True)
    env = dict(os.environ)
    if TORCH_LIB:
        env["PATH"] = TORCH_LIB + os.pathsep + env.get("PATH", "")
    cmd = [GPU_PY, os.path.join(os.path.dirname(__file__), "转写.py"), vocals, snap]
    if hotwords.strip():
        cmd.append(hotwords)
        print("  热词：%s" % hotwords, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    if r.returncode == 0:
        with open(txt, "w", encoding="utf-8") as f:
            f.write(r.stdout)
        print(f"  转写完成：{txt}")
    else:
        print("  转写失败:", (r.stderr or "")[-300:])
    return vocals


def _whisper_snapshot(prefer="large-v3"):
    """找一个能用的 whisper 模型目录，返回 (路径, 模型名)。

    两处都找：
      · 先看 config.MODELS_ROOT 下的 faster-whisper-<名>\\（自己下好放这儿的）
        ——国内直连 huggingface 基本下不动，走 ModelScope 镜像会快很多。
      · 再看 HF 缓存 ~/.cache/huggingface/hub。
    找不到 prefer 就按 large-v3 → medium → small 退。
    注意：**退回必须由调用方报出来**（见 audio_chain 里那句 ⚠️）。
    之前静默退过一次，日志写着 medium、实际用的 small，白折腾一场。
    """
    order = [prefer] + [m for m in ("large-v3-turbo", "large-v3", "medium", "small") if m != prefer]

    local_root = config.MODELS_ROOT
    for name in order:
        if not local_root:
            break
        d = os.path.join(local_root, "faster-whisper-%s" % name)
        if os.path.exists(os.path.join(d, "model.bin")):
            return (d, name)

    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    for name in order:
        base = os.path.join(hub, "models--Systran--faster-whisper-%s" % name, "snapshots")
        if not os.path.isdir(base):
            continue
        subs = [os.path.join(base, d) for d in os.listdir(base)]
        subs = [d for d in subs if os.path.exists(os.path.join(d, "model.bin"))]
        if subs:
            return (subs[0], name)
    return (None, None)


def already_seen(bv, page):
    """这个 BV 之前跑过没有。返回 (证据文件, mtime) 或 None。

    判据只看磁盘，不看记忆：视频解读/<BV>/[P<n>/] 下有非空的 字幕.txt，
    或者有 分镜*.jpg —— 有就说明跑过。

    2026-11-04 加。起因：11-03 晚上看过 BV1xn4w6DE5z，11-04 又下了一遍，
    因为"忘了"。不是缺记录（字幕.txt 就在那儿躺着），是**没查**。
    所以把检查挪到入口，默认就查，省得靠记性。
    """
    d = os.path.join(OUT, bv)
    if page > 1:
        d = os.path.join(d, "P%d" % page)
    if not os.path.isdir(d):
        return None
    sub = os.path.join(d, "字幕.txt")
    if os.path.exists(sub) and os.path.getsize(sub) > 0:
        return (sub, os.path.getmtime(sub))
    for name in sorted(os.listdir(d)):
        if name.startswith("分镜") and name.endswith(".jpg"):
            p = os.path.join(d, name)
            return (p, os.path.getmtime(p))
    return None


def main():
    bv = [a for a in sys.argv[1:] if a.upper().startswith("BV")][0]
    max_frames = int(arg("--frames", 28, int))
    # 🔴 2026-11-06：--frames 已废弃。帧数不再由预算决定，而是「慢的合并、快的补」。
    #    仍然解析它只是为了老命令不报错；给了就提醒一句。
    if "--frames" in sys.argv:
        print("⚠️ --frames 已废弃（帧数不再有硬顶），这次忽略。")
    # 2026-11-06 新参数（用户定的方案 A）
    min_gap = arg("--min-gap", 0.5)            # 二分补帧的最小间隔（秒）
    merge_thr = arg("--merge-thr", 0.06)       # 分块最大差低于它就算"没变"，合并
    refine_thr = arg("--refine-thr", 0.20)     # 分块最大差高于它就劈开补帧（标定见 frame_pick.bisect_refine）
    mode = arg("--mode", "auto", str)          # auto|ppt|normal（PPT 走"回退一秒取旧页"）
    # 2026-11-06：PPT 分支的两个阈值（16 格中位数）。幻灯片没占满画面时（录屏改 PPT 那种），
    # 整页换掉也只变动几格，0.10 会漏翻页 —— 这类片子把 --ppt-thr 降到 0.04。
    ppt_thr = arg("--ppt-thr", 0.10)
    ppt_merge = arg("--ppt-merge", 0.04)
    page = int(arg("--page", 1, int))          # 2026-11-03：多 P 视频看第几 P
    asr = arg("--asr", "large-v3", str)        # 2026-11-05：默认改回 large-v3（turbo 快但"视频/诅咒/各异"这类会错）；要快加 --asr large-v3-turbo
    # 2026-11-05：热词。专有名词（"福瑞"→"芙蕊"这种）换多大的模型都治不好，
    # 得直接告诉它"这条里有这些词"。空格分隔，例如：
    #   --hotwords "福瑞 furry 兽圈 毛茸茸"
    hotwords = arg("--hotwords", "", str)
    want_audio = "--no-audio" not in sys.argv

    # 看过就别再下一遍——先看磁盘，再决定要不要花那几分钟。
    hit = already_seen(bv, page)
    if hit and "--again" not in sys.argv:
        import datetime
        p, t = hit
        print("这个看过了：" + p)
        print("  生成时间 " + datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M"))
        print("  真要重看一遍，加 --again")
        return 0

    meta = vp.fetch_meta(bv, page=page)
    dur = float(meta["duration"])
    pc = meta.get("page_count", 1)
    tag = "" if pc <= 1 else f"  [P{page}/{pc}]"
    print(f"《{meta['title']}》 {dur:.0f}s{tag}")
    src = af.get_source(bv, page)
    base = os.path.dirname(os.path.abspath(src))

    # 2026-11-06：改成「基准铺底 → 合并相似 → 二分补帧」，不再用 28 帧硬顶。
    info = af.pick_frames_smart(
        src, dur, min_gap=min_gap, merge_thr=merge_thr,
        refine_thr=refine_thr, mode=mode,
        ppt_thr=ppt_thr, ppt_merge=ppt_merge)
    picks = info["picks"]
    mode_txt = "PPT/黑板（回退取旧页）" if info["mode"] == "ppt" else "普通（二分补帧）"
    print(f"类型 {mode_txt}；基准 {len(info['base'])} 帧（每 {info['step']:.1f} 秒）"
          f" → 合并 {info['n_merged']} → 补 {info['n_added']}"
          f" → 共 {len(picks)} 帧")

    imgs = grab(src, picks, os.path.join(base, "_frames"))
    # 🔴 2026-11-06：先清掉上一次的分镜图。帧数会变（这次 11 帧走"单张"，
    #    上次 28 帧留下 4 张分段图），不清就会留着旧图 —— 我今天就误读了
    #    一张 11-05 的旧图，还以为新算法没生效。
    for old in os.listdir(base):
        if old.startswith("分镜_") and old.endswith(".jpg"):
            try:
                os.remove(os.path.join(base, old))
            except OSError:
                pass
    sheets = []
    if len(imgs) <= 12:
        dst = os.path.join(base, "分镜_切点.jpg")
        _, size = render(imgs, dst)
        sheets.append(dst)
        print(f"分镜图: {dst}  {size[0]}x{size[1]}")
    else:
        # 一张图塞不下就每 9 帧一张，方便分批读
        for i in range(0, len(imgs), 9):
            dst = os.path.join(base, f"分镜_分段_{i // 9 + 1:02d}.jpg")
            _, size = render(imgs[i:i + 9], dst)
            sheets.append(dst)
        print(f"分镜图 {len(sheets)} 张（每张 9 帧）")

    got_cc = False
    if want_audio:
        # 先试 B站自带的 CC 字幕：不装任何模型，准确率还比转写高
        print("  找 B站 CC 字幕…", flush=True)
        cues = vp.fetch_subtitle(bv, page)
        if cues:
            with open(os.path.join(base, "字幕.txt"), "w", encoding="utf-8") as f:
                f.write(vp.fmt_subtitle(cues))
            print("  拿到 CC 字幕 %d 条 → 字幕.txt（不用转写了）" % len(cues))
            got_cc = True
        audio_chain(src, base, bv, page, asr, hotwords, skip_asr=got_cc)

    print("\n=== 现在该读这些 ===")
    for s in sheets:
        print(" ", os.path.basename(s))
    if os.path.exists(os.path.join(base, "字幕.txt")):
        print(" ", "字幕.txt")

    ledger(base, dur, info)


def count_sentences(path):
    """数转写里有多少句（空行不算）。没有字幕文件返回 0。"""
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def ledger(base, dur, info):
    """本次取点账（2026-11-03 用户定的方案 B；2026-11-06 改成真数字）。

    目的不是好看，是**让没走的那条路自己显形**。
    2026-11-03 早上就是因为看不出用了什么，才"只跑了一路"还没被察觉。
    2026-11-06 起三条路都是真数，不再有"未启用"这种占位。
    """
    sent = count_sentences(os.path.join(base, "字幕.txt"))

    print("\n─── 本次取点账 ───")
    print(f"  时长        {dur:,.0f} s")
    mtxt = "PPT/黑板" if info["mode"] == "ppt" else "普通"
    wtxt = "" if info["white"] is None else "（白底占比 %.2f）" % info["white"]
    print(f"  类型        {mtxt}{wtxt}")
    print(f"  基准铺底    {len(info['base'])} 帧（每 {info['step']:.1f} 秒一帧）")
    if info["mode"] == "ppt":
        print(f"  合并相似    -{info['n_merged']} → {len(info['merged'])} 帧")
        print(f"  回退取帧    +{info['n_added']}")
    else:
        print(f"  二分补帧    +{info['n_added']}")
        print(f"  合并相似    -{info['n_merged']}（补帧后才合，省下的不会被补回来）")
    tip = "   ← 转写无句子，没有语音锚点" if sent == 0 else "   ← 未校验（可能是 BGM 歌词）"
    print(f"  换句(音频)  {sent}" + tip)
    print(f"  最终选帧    {len(info['picks'])}")
    print(f"  帧间隔      最小 {info['min_gap']:.1f}s / 最大 {info['max_gap']:.1f}s")

    last = info["picks"][-1] if info["picks"] else 0
    print(f"  结尾帧      {last:.1f}s / 片长 {dur:.1f}s")

    # 只陈述事实，不替数据下"有效/无效"的结论：
    # 换句那一路分不清台词和 BGM 歌词（Demucs 把歌声归到人声轨）。
    if info["n_added"] == 0:
        print("  ⚠️ 二分一帧都没补 —— 要么片子真没变化，要么阈值太高")
    if dur - last > 2.0:
        print(f"  ⚠️ 结尾空了 {dur - last:.1f} 秒")


if __name__ == "__main__":
    main()
