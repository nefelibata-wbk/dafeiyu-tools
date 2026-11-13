# -*- coding: utf-8 -*-
"""自适应抽帧：二分找"画面变化点"，而不是按固定间隔硬抽。

思路（2026-10-31 用户提出）：
   取[起, 止]两端，如果两端差别小，就认为中间没什么变化 → 不再取帧；
   如果差别大，就取中点，分别和左右两端比，继续二分。

⚠️ 关键补丁（我加的）：**两端相同 ≠ 中间相同**。
   第 1 秒和第 6 秒可能一模一样，但第 3 秒闪过一行字。
   所以加了「保底」：任何相邻两帧的间隔不超过 --max-gap 秒，强制留帧。

两种模式：
  **A. 切点模式（默认，2026-11-02 定）**：扫描**相邻帧**找镜头边界，再按镜头分配帧。
     125 秒的片子 1.9 秒扫出 27 个镜头硬切，28 帧覆盖到结尾。
  B. 二分（`--bisect`，退路）：比较一段的**两端**。只适合实拍剪辑。
     AI 动画每帧都在动，B 模式会退化成均匀网格——实测 5 秒间隔差值 100% 超阈值、
     中位 0.178、最大 0.237（只差 1.3 倍没有断层）；而 0.2 秒相邻帧中位 0.020、
     尖峰 0.21~0.25（**相差 12 倍**）。这就是必须默认 A 的原因。

用法：
   python frame_pick.py <BV> [--cut-thr 0.12] [--max-frames 28]
   --cut-thr    相邻帧差值达到多少算切镜头
   --scan-fps   扫描帧率，越大越细、越慢
   --bisect     退回旧的二分法
   --thr        差异阈值（二分用）
   --max-gap    保底间隔（二分用，秒）
   --min-gap    二分的最小粒度（秒），小于它就停
   --max-frames 帧数上限
   --step       对照用：固定间隔的秒数，用来和旧方案比帧数

对比的核心指标只有两个：**取了几帧、有没有漏掉变化**。
"""
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config                  # noqa: E402
import bilibili as vp           # noqa: E402

FFMPEG = config.FFMPEG if config.ok(config.FFMPEG) else "ffmpeg"
OUT = config.OUT if config.OUT != config.UNSET else os.path.join(HERE, "data")
PROBE_W, PROBE_H = 64, 36          # 比较用的极小灰度图，只要"变没变"，不要细节


def arg(name, default, cast=float):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def get_source(bv, page=1):
    """下载源片（已经下过就复用），返回 mp4 路径。

    2026-11-03：多 P 视频从第 2 P 起放 <BV>/P{n}/_src.mp4，别覆盖 P1。
    """
    page = int(page or 1)
    base = os.path.join(OUT, bv)
    if page > 1:
        base = os.path.join(base, "P%d" % page)
    os.makedirs(base, exist_ok=True)
    src = os.path.join(base, "_src.mp4")
    if os.path.exists(src) and os.path.getsize(src) > 10 * 1024:
        return src
    import urllib.request
    import time
    # 2026-11-04：连着下几个视频后，B站 CDN 会开始回 503。
    # 以前一遇就整个崩掉、一个视频都拿不到。现在重试 3 次，而且
    # **每次都重新 fetch_playurl** —— 直链带时效和 CDN 参数，
    # 拿旧链接重试大概率还打同一个坏节点。间隔也递增，别把限流撞得更死。
    last = None
    for attempt in range(1, 4):
        url, _wh, _audio, _q = vp.fetch_playurl(bv, page=page)
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"})
            with urllib.request.urlopen(req, timeout=180) as r, open(src, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            return src
        except Exception as e:
            last = e
            print("  下载失败（第 %d 次）：%s" % (attempt, type(e).__name__), flush=True)
            if os.path.exists(src):
                os.remove(src)      # 别留半截文件害下一次
            if attempt < 3:
                time.sleep(2 * attempt)
    raise last


def frame_at(src, t):
    """取某一时刻的 64×36 **RGB** 探针图（不是灰度）。

    ⚠️ 2026-11-02 改：原来直接取 gray。问题是**颜色变了、亮度没变时灰度差为 0**，
       "红按钮变绿按钮""血条变色"这类切换会被判成"没变化"。
       实测（_change_score.py）：红→绿同亮度，灰度差 0.0000，HSV 差 0.2505。
    现在统一取 RGB，比较时再决定用哪个指标。
    """
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.3f" % t, "-i", src,
                        "-frames:v", "1", "-vf", "scale=%d:%d" % (PROBE_W, PROBE_H),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, timeout=60)
    buf = p.stdout
    need = PROBE_W * PROBE_H * 3
    if len(buf) < need:
        return None
    return np.frombuffer(buf[:need], dtype=np.uint8).astype(np.float32).reshape(PROBE_H, PROBE_W, 3)


def _rgb_to_hsv(a):
    """向量化 RGB(0~1) → HSV。不依赖 colorsys，逐像素 Python 环太慢。"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    nz = d > 1e-6
    # 三段分别算，最后都归一到 0~1
    r_is = nz & (mx == r)
    g_is = nz & (mx == g) & ~r_is
    b_is = nz & ~r_is & ~g_is
    h[r_is] = ((g[r_is] - b[r_is]) / d[r_is]) % 6
    h[g_is] = ((b[g_is] - r[g_is]) / d[g_is]) + 2
    h[b_is] = ((r[b_is] - g[b_is]) / d[b_is]) + 4
    h = h / 6.0
    return np.stack([h, s, mx], axis=-1)


def diff(a, b, metric="hsv"):
    """0~1 的平均差。metric="hsv"（默认）或 "gray"。"""
    if a is None or b is None:
        return 1.0
    if metric == "gray":
        wa = a @ np.array([0.299, 0.587, 0.114], np.float32)
        wb = b @ np.array([0.299, 0.587, 0.114], np.float32)
        return float(np.abs(wa - wb).mean()) / 255.0
    ha, hb = _rgb_to_hsv(a / 255.0), _rgb_to_hsv(b / 255.0)
    dh = np.abs(ha[..., 0] - hb[..., 0])
    dh = np.minimum(dh, 1.0 - dh)      # 色相是圆环：0.99 和 0.01 其实只差 0.02
    return float((2 * dh + np.abs(ha[..., 1] - hb[..., 1])
                  + np.abs(ha[..., 2] - hb[..., 2])).mean() / 4)


def adaptive(src, dur, thr, max_gap, min_gap, max_frames, verbose=True, metric="hsv"):
    """选帧，返回时间点列表（升序）。

    ⚠️ 2026-10-31 改过一版，原因是踩了坑：
       第一版是「深度优先递归二分」，结果**帧数上限一到，右边还没轮到就停了**——
       175 秒的视频，时间点变成 0,5,11,...,88,**175**，最后一段空了 87 秒。
       测试时拿熟悉的视频一对就露馅了（我以为抽了 12 帧，其实是"前 88 秒 11 帧 + 结尾 1 帧"）。

    现在的做法是「**先均匀铺底，再把余下的预算花在差异最大的区间**」：
      ① 按 max_gap 先把整条时间轴均匀铺满（保底，保证不漏整段）
      ② 剩下的帧数预算，每次挑当前「两端差异最大」的那一段从中间劈开
      ③ 差异低于 thr 就不再劈了（静止片段不再浪费帧）
    这样帧永远不会集中在开头。
    """
    cache = {}

    def f(t):
        t = round(t, 2)
        if t not in cache:
            cache[t] = frame_at(src, t)
        return cache[t]

    # ① 均匀铺底：只花一半预算
    #    为什么不是"每 max_gap 秒一张"？因为那样长视频会直接把预算撑爆
    #    （175 秒 + 保底 8 秒 = 23 帧，上限 12 根本管不住）。
    #    所以改成：拿一半预算把整条时间轴均匀铺开，剩下一半留给"差异大的地方"。
    base_n = max(2, min(max_frames, max_frames // 2 + 1))
    picks = sorted({round(dur * i / (base_n - 1), 2) for i in range(base_n)})

    # ② 一轮一轮地"均匀劈"：所有超阈值的段一起劈，而不是盯着最热的那段猛劈
    #    ⚠️ 2026-10-31 又踩一坑：第一版是"每次挑差异最大的一段劈"，
    #       结果预算全砸在片尾动画上（160,168,171,173,174,175），正文反而没帧。
    #       改成按轮次劈，并优先劈"更长的段"，分布才稳。
    while len(picks) < max_frames:
        cands = []
        for a, b in zip(picks, picks[1:]):
            if b - a <= 2 * min_gap:
                continue
            if diff(f(a), f(b), metric) >= thr:
                cands.append((b - a, a, b))      # 按段长排，长的优先
        if not cands:
            break
        cands.sort(reverse=True)
        room = max_frames - len(picks)
        for _len, a, b in cands[:room]:
            picks.append(round((a + b) / 2, 2))
        picks.sort()
    # ③ 去重：把"和上一张几乎一样"的帧删掉，腾出预算再补
    #    ⚠️ 2026-10-31 实测遇到的问题：片尾卡静止不动，算法却在 2:47/2:51/2:53
    #       连放三帧同样的画面，12 帧里白白浪费了 3 帧。
    def dedup(ps, eps=0.0):
        # 🔴 默认 eps=0.0 = **不删任何帧**（只裁掉首尾重复）。
        #    教训（2026-10-31 一天内两次）：
        #      0.02 → 把聊天滚动当成"同一张"，12 秒视频删到只剩 4 帧；
        #      0.005 → 仍误伤，175 秒视频被砍到 9 帧、最大空档 44 秒。
        #    多留 2~3 帧重复画面的代价，远小于"漏掉一段内容"。
        #    真要开，用 --dedup 0.02 显式指定。
        if not ps or eps <= 0:
            return ps
        keep = [ps[0]]
        for t in ps[1:-1]:
            if diff(f(keep[-1]), f(t), metric) >= eps:
                keep.append(t)
        if ps[-1] != keep[-1]:
            keep.append(ps[-1])
        return keep

    if len(picks) > 3:
        before = len(picks)
        picks = dedup(picks)
        if verbose and len(picks) < before:
            print("  去重: %d -> %d 帧" % (before, len(picks)))
        # 🔴 2026-10-31 死循环事故（卡了 8 分钟）：这里原本还有一段"去重后把
        #    名额补回去"的循环——新劈的中点若被 dedup 判为重复就删掉，帧数不涨，
        #    下一轮又劈同一个点，无限转。**整段删掉**，不做"删了再补"，
        #    宁可少几帧也不要再挂住。去重只做一次，做在最后。

    if verbose and len(picks) >= max_frames:
        print("  （已用满 %d 帧上限）" % max_frames)

    return picks, len(cache)


def frames_stream(src, fps=5):
    """一条 ffmpeg 命令把全片按 fps 解成 64×36 探针帧，边解边吐。

    为什么不用 frame_at() 逐点取：每次都要起一个 ffmpeg 进程，
    125 秒 × 5fps = 625 次，光启动开销就几分钟。这里只解一遍。
    """
    p = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", src, "-vf", "fps=%g,scale=%d:%d" % (fps, PROBE_W, PROBE_H),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    need = PROBE_W * PROBE_H * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(need)
            if not buf or len(buf) < need:
                break
            yield i / fps, np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(PROBE_H, PROBE_W, 3)
            i += 1
    finally:
        p.stdout.close()
        p.wait()


def find_cuts(src, thr=0.12, fps=5, merge=0.4, metric="hsv"):
    # 🔴 2026-11-06 阈值标定（详见 _归档/20261106_PPT抽帧标定/_thr_calib_1106.py）：
    #    拿 scenedetect 的 ContentDetector（默认 27/255）当参照系，扫了 0.08~0.30。
    #    三支有切点的片子合并：F1 在 0.14~0.16 是平台（0.85~0.87），0.15 本来就在平台里。
    #    但两边性格相反：0.12 召回 0.94 / 精确 0.78，0.18 精确 0.98 / 召回 0.55。
    #    **选 0.12 不是 F1 最高，是代价不对等**：
    #       漏一刀 → 整个镜头可能一帧都没有（内容没了）
    #       多一刀 → 多一张图（多花点 token）
    #    按「宁可多留不可漏段」的老规矩，往召回那边压。
    #    ⚠️ 参照系是另一个实现、不是真值；这组数只说明两个实现的吻合度。
    """按**相邻帧**差异找镜头边界。

    🔴 2026-11-02 起因：《三万碗不过岗》AI 动画每一帧都在动，原来"拿一段的开头帧
       和结尾帧比"的做法在任何阈值下都判成"全都在变"（5 秒间隔差值 100% 超阈值，
       中位 0.178、最大 0.237，只差 1.3 倍，没有断层）。
       改成相邻帧比：0.2 秒间隔的中位差 0.020、尖峰 0.21~0.25，**相差 12 倍**，
       镜头硬切一下就跳出来。实测 10.0/12.0/30.0/45.0s 四处尖峰全是真切镜。
    """
    cuts, prev = [], None
    for t, fr in frames_stream(src, fps):
        if prev is not None:
            d = diff(prev, fr, metric)
            if d >= thr:
                # 连续好几帧都高，说明是一次切换，合并成一个点
                if not cuts or t - cuts[-1] > merge:
                    cuts.append(round(t, 2))
        prev = fr
    return cuts


def pick_by_shots(src, dur, cuts, max_frames=28, min_shot=0.8):
    """按镜头取帧：每个镜头保底 1 帧（取中点），预算有剩就劈最长的镜头。

    🔴 2026-11-02 修：第一版在"镜头数 > 帧预算"时**直接丢掉最短的几个镜头**，
       结果 105–125 秒结尾"被踩在脚下"那 20 秒一帧都没剩（只在 115 秒留了一张）。
       改成**把短镜头并进邻居**，宁可粗略也不丢段；并且首尾各强制保一帧。
    """
    bounds = [0.0] + [c for c in cuts if min_shot < c < dur - min_shot] + [float(dur)]
    shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a > 0.3]
    if not shots:
        return []

    # 镜头比预算还多：把最短的并进邻居，而不是扔掉
    # （留 2 个名额给首尾强制帧，否则最后会被 picks[:max_frames] 截掉结尾）
    while len(shots) > max_frames - 2:
        i = min(range(len(shots)), key=lambda k: shots[k][1] - shots[k][0])
        if i == 0:
            a, b = shots[0], shots[1]
            shots = [(a[0], b[1])] + shots[2:]
        elif i == len(shots) - 1:
            a, b = shots[-2], shots[-1]
            shots = shots[:-2] + [(a[0], b[1])]
        else:
            if (shots[i - 1][1] - shots[i - 1][0]) <= (shots[i + 1][1] - shots[i + 1][0]):
                a, b = shots[i - 1], shots[i]
                shots = shots[:i - 1] + [(a[0], b[1])] + shots[i + 1:]
            else:
                a, b = shots[i], shots[i + 1]
                shots = shots[:i] + [(a[0], b[1])] + shots[i + 2:]

    picks = [round((a + b) / 2, 2) for a, b in shots]
    # 首尾强制保帧：结尾最容易被"预算用完"吞掉
    picks = sorted(set(picks) | {0.5, round(max(0.5, dur - 0.3), 2)})

    # 还有预算：反复劈当前最长的镜头
    while len(picks) < max_frames:
        i = max(range(len(shots)), key=lambda k: shots[k][1] - shots[k][0])
        a, b = shots[i]
        if b - a < 2 * min_shot:
            break
        mid = round((a + b) / 2, 2)
        picks = sorted(set(picks) | {mid})
        shots = shots[:i] + [(a, (a + b) / 2), ((a + b) / 2, b)] + shots[i + 1:]
    return sorted(picks)[:max_frames]


# ════════════════════════════════════════════════════════════════════════
# 2026-11-06 新增：基准铺底 → 合并相似 → 二分补帧（用户定的方案 A）
# ════════════════════════════════════════════════════════════════════════
# 起因：旧的 pick_by_shots 有 28 帧硬顶。实测 BV1Nc4c61EWF（123 秒、51 个
#   切点）被压成 26 个窗口，相邻帧间隔全落在 4~5 秒 —— 等于均匀网格，
#   而且被并掉的镜头一帧都没有。用户原话：
#     「28 对于短视频还好，超过一分钟的你基本没法看了」
# 现在：帧数不跟预算走，跟画面变化走 —— 慢的地方合并、变化大的地方补。


BLOCK_R, BLOCK_C = 4, 4


def diff_block(a, b, rows=BLOCK_R, cols=BLOCK_C):
    """分块最大差：把探针图切成 4×4 块，取「变化最大那块」的平均差（0~1）。

    🔴 2026-11-06 为什么不用全帧平均（diff）：**主体小的时候会被背景冲淡**。
       实测 BV16V4G6KEAh（23 秒桌面片），12.5–15.5s 那段正在盖章：
           全帧平均差 0.102（低于任何合理阈值，判成"没变化"）
           分块最大差 0.416（一眼就跳出来）
       所以新流程的合并/二分都用这个指标：**任何一块变化大，就算变了**。
    """
    if a is None or b is None:
        return 1.0
    d = np.abs(a - b).mean(axis=-1) / 255.0
    h, w = d.shape
    bh, bw = h // rows, w // cols
    if bh < 1 or bw < 1:
        return float(d.mean())
    best = 0.0
    for i in range(rows):
        for j in range(cols):
            v = float(d[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean())
            if v > best:
                best = v
    return best


def diff_med(a, b, rows=BLOCK_R, cols=BLOCK_C):
    """分块**中位数**差：4×4 块的平均差里取中间那个（0~1）。

    🔴 2026-11-06 为什么 PPT 分支从「全帧平均」换到它（_归档/20261106_PPT抽帧标定/_ppt_median_full_1106.py）：
       PPT 是整屏投影，讲者站在画面中右，走动时**只有他占的 2~6 格在变**。
       全帧平均会把这份"人动"算进去，中位数不会（16 格里 10 格以上没变 = 没翻页）。
       采矿课 P45（19.5 分钟，10 秒间隔 116 对）：
           指标         ≥0.10 的对数
           全帧平均        24        ← 大量讲者误报
           16 格中位       12        ← 0.086 和 0.108 之间正好断档
       挨帧看过的证据：7.0/9.5、112/114.5、173/174.5、447/449.5、515/517、
       688/689.5、736/738.2 这些"全帧平均"报出来的变化，幻灯片根本没动，是人走过；
       而中位数抓到的 706→709.5（新图出现）、962→966（逐条出字）、1016→1020（翻到空白）
       都是真的 —— 这几处在旧写法里反而漏了。
    """
    if a is None or b is None:
        return 1.0
    d = np.abs(a - b).mean(axis=-1) / 255.0
    h, w = d.shape
    bh, bw = h // rows, w // cols
    if bh < 1 or bw < 1:
        return float(d.mean())
    vals = [float(d[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean())
            for i in range(rows) for j in range(cols)]
    return float(np.median(vals))


def base_step(dur):
    """基准铺底间隔（秒）：短片 3 秒起，长片最宽 10 秒，基准点数控制在 ~80。"""
    return min(10.0, max(3.0, dur / 80.0))


def grid_points(dur, step):
    """从 0.5 秒开始按 step 铺底，末尾补一帧。

    末尾用 dur-0.3：B站 duration 比视频流实际末尾长 1 秒左右（2026-11-05 实测），
    直接用 dur 会取不到帧。
    """
    pts, t = [], 0.5
    while t < dur - 0.3:
        pts.append(round(t, 2))
        t += step
    tail = round(max(0.5, dur - 0.3), 2)
    if not pts or tail - pts[-1] > step * 0.5:
        pts.append(tail)
    return pts


def _probe(src):
    """带缓存的取帧探针：同一时刻只起一次 ffmpeg。二分是递归的，不缓存会重复取。"""
    cache = {}

    def f(t):
        t = round(t, 2)
        if t not in cache:
            cache[t] = frame_at(src, t)
        return cache[t]

    return f


def _diff_by(metric, a, b):
    """按名字取差异指标：block=分块最大 / med=分块中位 / 其余=全帧平均。"""
    if metric == "block":
        return diff_block(a, b)
    if metric == "med":
        return diff_med(a, b)
    return diff(a, b)


def static_cells(a, b, quiet=0.02, rows=BLOCK_R, cols=BLOCK_C):
    """两帧里「几乎没动」的格子数（0~16）。用来分辨"加了一条字"和"换了一页"。

    🔴 2026-11-06 标定（_归档/20261106_PPT抽帧标定/_unchanged_cells_1106.py，勾股定理微课 58 对）：
        逐条出字（同一页加内容）：静格 11~16 个，绝大多数 12~16
        真换页：                    静格 0~3 个
        边界样本：279s 换到暗色动画那一页，静格 10（所以要卡在 11 以上）
    中位数看不见"只动一两格"的小变化，静格数看得见。

    ⚠️ 2026-11-06 试过接进 merge（静格 ≥ 11 就合并），**两支片子帧数一个没变**——
       中位数阈值早就把这些对合掉了，真正卡住的是"每帧最多合一次"。
       又试过"同一页一长串只留首尾"，结果压过头：勾股定理 133→244 的拼图动画
       被当成一页，出现 124 秒空档，推导过程全丢。所以**没有接进流水线**，
       只留作诊断工具（_归档/20261106_PPT抽帧标定/_run_endpoints_1106.py 会用到）。
    """
    if a is None or b is None:
        return 0
    d = np.abs(a - b).mean(axis=-1) / 255.0
    h, w = d.shape
    bh, bw = h // rows, w // cols
    if bh < 1 or bw < 1:
        return 0
    return int(sum(1 for i in range(rows) for j in range(cols)
                   if float(d[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean()) < quiet))


def merge_similar(src, pts, thr=0.06, max_gap=None, probe=None, metric="block"):
    """相邻两帧差异 < thr 就合并：**删前保后**，每帧最多参与一次。

    🔴 「每帧最多参与一次」是硬规矩（用户 2026-11-06 定的）：
       否则 A≈B、B≈C 会连锁合并一路删下去 —— 2026-10-31 就砍出过
       「175 秒视频只剩 9 帧、最大空档 44 秒」的事故（见 dedup 的注释）。
       每帧只合一次 ⇒ 相邻两次合并不重叠 ⇒ 最大空档天然 ≤ 2×基准间隔。

    max_gap 是兜底校验：合并后若空档超过它，就不合并这一对（正常不会触发）。
    """
    f = probe or _probe(src)
    out, merged = [], 0
    i = 0
    n = len(pts)
    while i < n:
        # 🔴 2026-11-06 用户定的规则 B：**首尾两帧永不删，中间照常合并**。
        #    起因：BV16V4G6KEAh 把 0.5s 和 3.5s 判成"一样"（桌面几乎没变），
        #    按"删前保后"合并 → 开头 3.5 秒一帧都没有。
        #    现在第一对改成"留前删后"，首帧保得住；其余仍是"删前保后"，
        #    因为最后一对留的是后一帧，尾帧本来就安全。
        #    点太少（≤2）时一对都不合，否则首尾必有一个保不住。
        if i + 1 < n and n > 2:
            a, b = pts[i], pts[i + 1]
            prev = out[-1] if out else None
            gap_ok = (max_gap is None) or (prev is None) or (b - prev <= max_gap)
            dv = _diff_by(metric, f(a), f(b))
            if gap_ok and dv < thr:
                out.append(a if i == 0 else b)
                merged += 1
                i += 2
                continue
        out.append(pts[i])
        i += 1
    return out, merged


def bisect_refine(src, pts, thr=0.20, min_gap=0.5, probe=None):
    # 🔴 2026-11-06 标定（_归档/20261106_PPT抽帧标定/_shot_cover_1106.py）：
    #    严指标 = 「参照切点切出的每个镜头里，有没有至少一帧」。
    #    0.25 时：跳舞片 59 个镜头里有 2 个是空的（5.2–6.2s、76.4–78.3s），另两支 0 空。
    #    0.20 时：三支片子 0 空镜头，代价是帧数 +11%~+37%。
    #    ⚠️ 先前的「漏 7%」是按「切点 ±0.5s 内有没有帧」算的，那个指标高估了——
    #       帧落在镜头中间也算覆盖，所以别拿那个数下结论。
    """相邻两帧差异 ≥ thr 就在中点补一帧，递归；间隔 ≤ 2×min_gap 就停。

    ⚠️ 为什么不是「每对相邻帧都劈」：铺底的点本来就跨镜头，差异大是正常的。
       阈值用分块最大差实测标定（3 秒间隔）：
           静态/慢速段   0.04 ~ 0.15
           有动作/切镜头 0.19 ~ 0.60
       0.20 大致能把两类分开；快剪片几乎全超，会一路劈到 min_gap，
       这正是「画面变得快就多给帧」要的效果。
    """
    f = probe or _probe(src)
    seen = set(pts)
    picks = sorted(seen)
    added = 0
    while True:
        new = []
        for a, b in zip(picks, picks[1:]):
            if b - a <= 2 * min_gap:
                continue
            mid = round((a + b) / 2, 2)
            if mid in seen:
                continue
            if diff_block(f(a), f(b)) >= thr:
                new.append(mid)
        if not new:
            break
        seen |= set(new)
        picks = sorted(seen)
        added += len(new)
    return picks, added


def frame_rgb_big(src, t, w=320, h=180):
    """取 320×180 RGB 图（判定用，比 64×36 探针看得清）。"""
    p = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.2f" % t, "-i", src,
                        "-frames:v", "1", "-vf", "scale=%d:%d" % (w, h),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, timeout=60)
    buf = p.stdout
    need = w * h * 3
    if len(buf) < need:
        return None
    return (np.frombuffer(buf[:need], dtype=np.uint8)
            .astype(np.float32).reshape(h, w, 3) / 255.0)


def classify_mode(src, dur, samples=10, white_thr=0.15):
    """粗判片子类型：**亮且低饱和**的像素占比中位数 > 0.15 → 'ppt'，否则 'normal'。

    🔴 2026-11-06 两版判据：
      ① 只看「亮度 > 0.85」——**会误判**。广播体操片子背景是明亮的天空，
         白占比 0.49，比采矿课 PPT 的 0.36 还高，被当成 PPT 了。
      ② 改成「亮度 > 0.85 **且** 饱和度 < 0.15」（白/灰底，彩色天空不算）：
             桌面实拍  0.009      孔乙己实拍 0.029
             跳舞实拍  0.016      广播体操   0.094
             采矿课PPT 0.347
         差 3.7 倍，阈值取 0.15。
      ⚠️ 黑板（深底）认不出来，会落回 normal；要强制走 PPT 分支用 --mode ppt。
    """
    ws = []
    for i in range(samples):
        t = max(0.5, dur * (i + 0.5) / samples)
        a = frame_rgb_big(src, t)
        if a is None:
            continue
        mx = a.max(axis=-1)
        mn = a.min(axis=-1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        ws.append(float(((mx > 0.85) & (sat < 0.15)).mean()))
    if not ws:
        return "normal", 0.0
    ws.sort()
    med = ws[len(ws) // 2]
    return ("ppt" if med > white_thr else "normal"), med


def locate_change(src, a, b, thr, tol=0.3, probe=None, metric="block"):
    """在 [a, b] 里二分，找出"变化发生在哪一刻"，返回变化后的第一个时刻。

    ⚠️ 这是二分法的老本行——不是拿它补帧，是拿它**定位**。
    """
    f = probe or _probe(src)

    def d(x, y):
        return _diff_by(metric, f(x), f(y))

    if d(a, b) < thr:
        return None
    lo, hi = a, b
    while hi - lo > tol:
        mid = round((lo + hi) / 2, 2)
        if mid <= lo or mid >= hi:
            break
        if d(lo, mid) >= thr:
            hi = mid
        else:
            lo = mid
    return hi


def refine_ppt(src, pts, thr=0.10, back=1.0, probe=None, metric="mean"):
    """PPT/黑板分支（用户 2026-11-06 定的）：变化点定位后**回退一秒**取旧页。

    为什么回退：变化那一刻截到的是过渡（半透明、动画、黑屏）。
    往前退到「翻页前的稳定态」，才是上一页完整的样子；新页后面的铺底点自然会拍到。
    退的距离取 min(back, 间隔的一半)，最多退到 a 之后，不会越过上一个点。
    """
    f = probe or _probe(src)

    def d(x, y):
        return _diff_by(metric, f(x), f(y))

    out = list(pts)
    for a, b in zip(pts, pts[1:]):
        if d(a, b) < thr:
            continue
        ct = locate_change(src, a, b, thr, probe=f, metric=metric)
        if ct is None:
            continue
        pick = round(max(a + 0.1, ct - min(back, (b - a) / 2.0)), 2)
        # 离已有点太近就别加了（实测出现过 0.1 秒的孪生帧）
        if min(abs(pick - x) for x in out) < 0.4:
            continue
        out.append(pick)
    return sorted(set(out))


def pick_frames_smart(src, dur, min_gap=0.5, merge_thr=0.06,
                      refine_thr=0.20, mode="auto", ppt_thr=0.10, ppt_merge=0.04):
    """基准铺底 → 合并相似 →（按类型）二分补帧 / PPT 回退取帧。"""
    f = _probe(src)
    step = base_step(dur)
    base = grid_points(dur, step)

    white = None
    if mode == "auto":
        mode, white = classify_mode(src, dur)

    # ⚠️ 两种模式用**不同的差异指标**，阈值不能混用：
    #    normal → 分块最大差（块 0.06 / 劈 0.25），局部动作也算变化
    #    ppt    → 全帧平均差（合 0.05 / 变 0.10），讲师走动会被平均掉，
    #             只有整页换掉才算变化（实测采矿课 10 秒间隔：同页 0.02~0.08，
    #             翻页/大幅移动 0.10~0.26；分块最大差在这里完全分不开）
    if mode == "ppt":
        # 🔴 2026-11-06 换成 16 格中位数（原来全帧平均把讲者走动当成了翻页）。
        #    采矿课 10 秒间隔实测：同页中位 0.020 上下，翻页的那批从 0.108 起，
        #    中间 0.086→0.108 是断档。合并 0.04 只吃掉几乎没动的静态页，
        #    补帧 0.10 正好卡在断档里（全片报 12 次，旧写法报 24 次）。
        # 🔴 2026-11-06 顺序也跟普通分支统一：**先补帧再合并**。
        #    先合并会删掉铺底点，10 秒变 20 秒一对，页与页之间的小差别就查不出来了
        #    （实测 416.1s 那次翻页没被抓到，前后选帧跨了整整 20 秒）。
        # ⚠️ 已知短板：幻灯片**不占满画面**时（例如"教你做 PPT"的录屏，画面中间才是
        #    画布，四周全是软件界面），整页换掉只变动 4~6 格，中位数看得见但偏小，
        #    0.10 会漏。这类片子把 --ppt-thr 降到 **0.04**（实测 0.05 还不够，0.04 才行；
        #    帧数几乎不变，67→68，因为合并会把多出来的重复帧吃掉）。
        picks_raw = refine_ppt(src, base, ppt_thr, probe=f, metric="med")
        n_added = len(picks_raw) - len(base)
        picks, n_merged = merge_similar(src, picks_raw, ppt_merge, max_gap=2 * step,
                                        probe=f, metric="med")
        merged = base
    else:
        # 🔴 2026-11-06 顺序修正：**先补帧，再合并**。
        #    原来「先合并再补帧」是白费——合并留下的 6 秒空档，二分立刻又填回来：
        #      孔乙己 合并 -11 帧 → 最终只少 1 帧（130→129）。
        #    倒过来之后：
        #      孔乙己 130→90（-31%）  跳舞 156→139（-11%）  广播体操 66→47（-29%）
        #      空镜头全是 0，最大空档 ≤ 6 秒（= 2×基准间隔）。
        #    指标必须用**全帧平均差**：分块最大差对局部微动太敏感，
        #    3 秒间隔的静止镜头也有 0.04~0.19，永远碰不到 0.06 的阈值。
        merged = base
        picks_raw, n_added = bisect_refine(src, base, refine_thr, min_gap, probe=f)
        picks, n_merged = merge_similar(src, picks_raw, 0.10, max_gap=2 * step,
                                        probe=f, metric="mean")

    gaps = [b - a for a, b in zip(picks, picks[1:])] or [0.0]
    return {
        "step": step,
        "base": base,
        "merged": merged,
        "n_merged": n_merged,
        "picks": picks,
        "n_added": n_added,
        "max_gap": max(gaps),
        "min_gap": min(gaps),
        "mode": mode,
        "white": white,
    }


def auto_min_gap(src, sample_sec=60.0):
    """探开头一小段，按镜头节奏决定二分的最小粒度。

    2026-11-04 加。起因：用户问「要不要把最小取帧区间从 1 秒改小」。
    结论：**不该一刀切改默认值**——讲解类片子 1 秒已经够，改小只会让探针翻倍；
    但快切片（平均镜头 <2 秒）1 秒粒度会整段漏掉。
    所以改成「先量再定」：
        平均镜头 ≥ 2 秒 → 1.0 秒（跟原来一样）
        平均镜头 < 2 秒 → 0.3 秒
    只探前 sample_sec 秒，几秒出结果，长片子也不拖慢。
    判据来源：ffmpeg 场景检测（实测比 OpenCV 直方图可靠，见 _frame_stats.py 注释）。
    """
    try:
        out = subprocess.run(
            [FFMPEG, "-t", "%.0f" % sample_sec, "-i", src,
             "-vf", "select='gt(scene,0.3)',showinfo",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore")
        times = [float(x) for x in re.findall(r"pts_time:([\d.]+)", out.stderr)]
    except Exception:
        return 1.0
    if len(times) < 2:
        return 1.0
    avg = min(times[-1], sample_sec) / len(times)
    return 0.3 if avg < 2.0 else 1.0


def main():
    bv = [a for a in sys.argv[1:] if a.startswith("BV")][0]
    thr = arg("--thr", 0.06)
    max_gap = arg("--max-gap", 5.0)
    # 最小分组 = 旧流水线的抽帧间隔（视频流水线.py --step 默认 1.0 秒）。
    # 2026-11-04 改：默认不再写死 1.0，而是 **0 = 自动**——
    # 讲解类片子 1 秒已够（再细只是同一句话的中间状态），
    # 但快切片平均 1 秒一个镜头，写死 1.0 会整段漏。见 auto_min_gap()。
    min_gap = arg("--min-gap", 0, float)   # 0 = 自动判定
    max_frames = int(arg("--max-frames", 28, int))
    step = arg("--step", 1.5)
    metric = "gray" if ("--metric" in sys.argv and "gray" in sys.argv) else "hsv"

    t0 = time.time()
    meta = vp.fetch_meta(bv)
    dur = float(meta["duration"])
    print("《%s》 %ds" % (meta["title"], dur))
    src = get_source(bv)
    print("源片:", src, os.path.getsize(src) // 1024, "KB", "(%.1fs)" % (time.time() - t0))

    if min_gap <= 0:
        min_gap = auto_min_gap(src)
        print("二分最小粒度：自动判定 %.2f 秒" % min_gap)

    # 2026-11-02：切点模式改成**默认**。用户说过"可别不用这个"，
    # 靠记性不靠谱，索性让旧二分法变成要显式指定的退路（--bisect）。
    if "--bisect" not in sys.argv:
        # 镜头边界模式（2026-11-02 加）：先找硬切，再按镜头分配帧
        cut_thr = arg("--cut-thr", 0.12)
        scan_fps = arg("--scan-fps", 5.0)
        t1 = time.time()
        cuts = find_cuts(src, cut_thr, scan_fps, metric=metric)
        print("镜头边界 %d 处 (%0.1fs 扫描): %s"
              % (len(cuts), time.time() - t1, ", ".join("%.1f" % c for c in cuts)))
        t2 = time.time()
        picks = pick_by_shots(src, dur, cuts, max_frames)
        print("按镜头选帧: %d 帧, %.1fs" % (len(picks), time.time() - t2))
        print("时间点:", ", ".join("%.1f" % x for x in picks))
        fixed = []
        t = 0.5
        while t < dur - 0.3:
            fixed.append(t)
            t += step
        print("固定间隔(%.1fs): %d 帧" % (step, len(fixed)))
        return

    t1 = time.time()
    picks, probes = adaptive(src, dur, thr, max_gap, min_gap, max_frames, metric=metric)
    print("二分选帧: %d 帧, 探针 %d 次, %.1fs, 指标=%s" % (len(picks), probes, time.time() - t1, metric))
    print("时间点:", ", ".join("%.1f" % x for x in picks))

    fixed = []
    t = 0.5
    while t < dur - 0.3:
        fixed.append(t)
        t += step
    print("固定间隔(%.1fs): %d 帧" % (step, len(fixed)))
    print("节省: %.0f%%" % (100 * (1 - len(picks) / max(1, len(fixed)))))


if __name__ == "__main__":
    main()
