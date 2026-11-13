# -*- coding: utf-8 -*-
"""B站那两件事：查元数据、取播放直链。

只做两件事：查元数据、取播放直链。别的（截图、拼图、字幕接口）跟抽帧无关，没带过来。

为什么要借浏览器：B站 playurl 接口不带 wbi 签名会被降级到 480P，
带签名但不带登录态，多数片子也只有 720P。所以这里用 Edge 的调试端口
开一个**你登录过的用户目录**，在页面上下文里请求——拿到的是登录态的高清直链。

依赖：requests 之外的 websocket-client（只有 CDP 用得到，延迟导入）。
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"


# ---------------------------------------------------------------- CDP 基础

def http_json(port, path, method="GET"):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), method=method)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8-sig"))


def port_alive(port):
    try:
        http_json(port, "/json/version")
        return True
    except Exception:
        return False


class CDP:
    def __init__(self, ws_url, timeout=60):
        try:
            import websocket
        except ImportError:
            raise RuntimeError("缺 websocket-client：pip install websocket-client")
        self.ws = websocket.create_connection(ws_url, timeout=timeout, suppress_origin=True)
        self.seq = 0

    def call(self, method, **params):
        self.seq += 1
        mid = self.seq
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError("%s: %s" % (method, msg["error"]))
                return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr,
                      returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def start_edge(profile, port=None):
    port = port or config.EDGE_PORT
    if port_alive(port):
        return None
    if not config.ok(config.EDGE):
        raise RuntimeError("config.EDGE 没填（或路径不存在），取不到高清直链")
    if not config.ok(profile):
        raise RuntimeError("config.EDGE_PROFILE 没填（或目录不存在）——先手动开一次 Edge 登录 B站")
    subprocess.Popen([config.EDGE, "--headless=new", "--remote-debugging-port=%d" % port,
                      "--user-data-dir=%s" % profile, "--no-first-run",
                      "--disable-gpu", "about:blank"])
    for _ in range(30):
        if port_alive(port):
            return "started"
        time.sleep(0.5)
    raise RuntimeError(
        "Edge 调试端口 %d 起不来。常见原因：这个 EDGE_PROFILE 已经有一个 Edge 在跑"
        "（同一个用户目录不能开两个），或者 Edge 路径填错。\n"
        "  → 先把用这个目录的 Edge 全部关掉；或者把它已经在用的端口填进 EDGE_PORT。" % port)


def stop_edge(profile):
    # Windows 上按命令行里的 profile 目录名找进程；别的系统直接跳过
    if os.name != "nt":
        return
    import pathlib
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
          "Where-Object { $_.CommandLine -like '*%s*' } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
          % pathlib.Path(profile).name)
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, timeout=60)
    time.sleep(1)


# ---------------------------------------------------------------- 元数据

def resolve_bv(arg):
    m = re.search(r"(BV[0-9A-Za-z]{10})", arg)
    if m:
        return m.group(1)
    if "b23.tv" in arg or arg.startswith("http"):
        try:
            req = urllib.request.Request(arg, headers={"User-Agent": UA})
            final = urllib.request.urlopen(req, timeout=15).geturl()
            m = re.search(r"(BV[0-9A-Za-z]{10})", final)
            if m:
                return m.group(1)
        except Exception as e:
            print("短链解析失败:", e)
    return None


def fetch_meta(bvid, page=1):
    url = "https://api.bilibili.com/x/web-interface/view?bvid=%s" % bvid
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Referer": "https://www.bilibili.com/"})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read().decode("utf-8"))["data"]
    # 多 P 视频的 d["duration"] 是**所有 P 加起来**的总时长。只取 P1 时按总时长抽帧，
    # 时间点会全部落到 P1 之外。所以有 pages 时用当前 P 的时长。
    pages = d.get("pages") or []
    if pages:
        if page < 1 or page > len(pages):
            page = 1
        dur = pages[page - 1].get("duration", d["duration"])
    else:
        dur = d["duration"]
    return {
        "bvid": bvid, "title": d["title"], "up": d["owner"]["name"],
        "duration": dur, "view": d["stat"]["view"],
        "like": d["stat"]["like"], "reply": d["stat"]["reply"],
        "desc": d.get("desc", ""), "page": page, "page_count": len(pages) or 1,
    }


# ---------------------------------------------------------------- 播放直链

# wbi 签名混排表（B站前端常量，2026-10 有效；哪天失效了，签出来的 w_rid 会被拒）
_WBI_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
            33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61,
            26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36,
            20, 34, 44, 52]


def _wbi_sign(params, img_key, sub_key):
    orig = img_key + sub_key          # 64 位：先拼接，再按表重排
    raw = "".join(orig[i] for i in _WBI_TAB)[:32]
    items = sorted(params.items())
    # 值里的 !'()* 要去掉（官方规则）
    query = urllib.parse.urlencode([(k, "".join(c for c in str(v) if c not in "!'()*"))
                                    for k, v in items])
    w_rid = hashlib.md5((query + raw).encode()).hexdigest()
    return query + "&w_rid=" + w_rid


def fetch_playurl(bvid, qn=80, page=1):
    """返回 (视频直链, (宽, 高), 音频直链, 画质编号)。

    三个坑都在这里踩过：
      ① 裸接口会被降级到 480P —— 必须走 /x/player/wbi/playurl 并算 w_rid。
      ② 刚起浏览器的头几秒是以匿名身份请求的（直链里 mid=0），也是 480P ——
         所以先轮询 nav 接口，等到 isLogin 才算数。
      ③ 多 P 视频必须指定 page，否则永远只有第 1 P。
    """
    page_index = max(1, int(page or 1))
    profile, port = config.EDGE_PROFILE, config.EDGE_PORT
    started = start_edge(profile, port)
    # 开在**首页**而不是视频页：实测在视频页上下文里请求，画质会被压到 480P
    url = "https://www.bilibili.com"
    tgt = http_json(port, "/json/new?" + urllib.parse.quote(url, safe=":/?=&"), method="PUT")
    page_obj = CDP(tgt["webSocketDebuggerUrl"])
    try:
        page_obj.call("Page.enable")
        page_obj.call("Runtime.enable")
        js0 = """(async () => {
          const API = 'https://api.bilibili.com';
          const nav = await (await fetch(API + '/x/web-interface/nav', {credentials:'include'})).json();
          const v = await (await fetch(API + '/x/web-interface/view?bvid=%s', {credentials:'include'})).json();
          const w = nav.data && nav.data.wbi_img || {};
          const pages = (v.data && v.data.pages) || [];
          const cid = pages.length >= %d ? pages[%d].cid : (v.data && v.data.cid);
          return JSON.stringify({login: !!(nav.data && nav.data.isLogin),
                                 img: w.img_url || '', sub: w.sub_url || '', cid: cid,
                                 page_count: pages.length});
        })()""" % (bvid, page_index, page_index - 1)
        info0 = {}
        for _ in range(15):
            raw0 = page_obj.js(js0)
            info0 = json.loads(raw0) if isinstance(raw0, (str, bytes)) else (raw0 or {})
            if info0.get("login") and info0.get("cid"):
                break
            time.sleep(1)
        if not info0.get("cid"):
            raise RuntimeError("拿不到 cid")
        if not info0.get("login"):
            print("  ⚠️ 这个 Edge 用户目录不是登录态，画质会被压到 720P 以下", flush=True)
        img_key = info0["img"].rsplit("/", 1)[-1].split(".")[0]
        sub_key = info0["sub"].rsplit("/", 1)[-1].split(".")[0]

        params = {"bvid": bvid, "cid": info0["cid"], "qn": qn,
                  "fnval": 4048, "fnver": 0, "fourk": 1, "wts": int(time.time())}
        query = _wbi_sign(params, img_key, sub_key)

        js = """(async () => {
          const p = await (await fetch('https://api.bilibili.com/x/player/wbi/playurl?%s',
                                       {credentials:'include'})).json();
          const d = p.data || {}, dash = d.dash || {};
          const vs = (dash.video || []).slice().sort((a, b) => b.id - a.id || b.bandwidth - a.bandwidth);
          const aus = (dash.audio || []).slice().sort((a, b) => b.bandwidth - a.bandwidth);
          if (!vs.length) return JSON.stringify({err: (p.code || 'no dash') + ' ' + (p.message || '')});
          return JSON.stringify({
            url: vs[0].baseUrl, w: vs[0].width, h: vs[0].height,
            audio: aus.length ? aus[0].baseUrl : '',
            quality: d.quality
          });
        })()""" % query
        raw = page_obj.js(js)
        info = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    finally:
        # 只在还剩别的标签页时才关——headless Edge 关掉最后一个页面会整个退出
        try:
            pages = [t for t in http_json(port, "/json/list") if t.get("type") == "page"]
            if len(pages) > 1:
                page_obj.close()
        except Exception:
            pass
        if started:
            stop_edge(profile)
    if info.get("err"):
        raise RuntimeError("取 playurl 失败：%s" % info["err"])
    return info["url"], (info["w"], info["h"]), info.get("audio", ""), info.get("quality")
