#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model-arena · 模型擂台盲测榜
同一个问题问多个模型，盲测投票，本地 Elo 排行榜
零依赖单文件
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__version__ = "1.0.0"

DATA_DIR = Path.home() / ".model-arena"
DB_PATH = DATA_DIR / "arena.db"
CONFIG_PATH = DATA_DIR / "config.json"
REMOTE_URL = "https://raw.githubusercontent.com/dafeiai918/oneclickrelay-remote/main/remote.json"
FALLBACK_RECOMMEND = {"name": "好同根", "url": "https://haotogen.com/"}

K_FACTOR = 32
DEFAULT_RATING = 1000.0

DEFAULT_CONFIG = {
    "base_url": "",
    "api_key": "",
    "models": [],
    "timeout": 90,
    "max_tokens": 800,
}


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; CYAN = "\033[36m"; GRAY = "\033[90m"


def _c(code: str, text: str) -> str:
    return f"{code}{text}{C.RESET}"


def enable_ansi() -> None:
    if os.name == "nt":
        os.system("")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------- 配置与存储 ----------------

def load_config() -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        try:
            stored = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                cfg.update(stored)
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ratings (
            model TEXT PRIMARY KEY,
            rating REAL DEFAULT 1000,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS battles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, prompt TEXT, model_a TEXT, model_b TEXT, result TEXT
        );
    """)
    conn.commit()
    return conn


# ---------------- Elo ----------------

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def apply_result(rating_a: float, rating_b: float, score_a: float,
                 k: float = K_FACTOR) -> tuple[float, float]:
    expect_a = expected_score(rating_a, rating_b)
    expect_b = 1.0 - expect_a
    score_b = 1.0 - score_a
    return rating_a + k * (score_a - expect_a), rating_b + k * (score_b - expect_b)


def get_rating(conn: sqlite3.Connection, model: str) -> float:
    row = conn.execute("SELECT rating FROM ratings WHERE model=?", (model,)).fetchone()
    if row:
        return float(row[0])
    conn.execute("INSERT INTO ratings (model, rating) VALUES (?, ?)", (model, DEFAULT_RATING))
    conn.commit()
    return DEFAULT_RATING


def record_battle(conn: sqlite3.Connection, prompt: str, model_a: str, model_b: str,
                  result: str) -> tuple[float, float]:
    """result: 'a' / 'b' / 'draw'"""
    ra = get_rating(conn, model_a)
    rb = get_rating(conn, model_b)
    score_a = {"a": 1.0, "b": 0.0, "draw": 0.5}[result]
    new_a, new_b = apply_result(ra, rb, score_a)
    conn.execute("UPDATE ratings SET rating=? WHERE model=?", (new_a, model_a))
    conn.execute("UPDATE ratings SET rating=? WHERE model=?", (new_b, model_b))
    if result == "a":
        conn.execute("UPDATE ratings SET wins=wins+1 WHERE model=?", (model_a,))
        conn.execute("UPDATE ratings SET losses=losses+1 WHERE model=?", (model_b,))
    elif result == "b":
        conn.execute("UPDATE ratings SET wins=wins+1 WHERE model=?", (model_b,))
        conn.execute("UPDATE ratings SET losses=losses+1 WHERE model=?", (model_a,))
    else:
        conn.execute("UPDATE ratings SET draws=draws+1 WHERE model=?", (model_a,))
        conn.execute("UPDATE ratings SET draws=draws+1 WHERE model=?", (model_b,))
    conn.execute("INSERT INTO battles (ts, prompt, model_a, model_b, result) VALUES (?,?,?,?,?)",
                 (time.time(), prompt[:200], model_a, model_b, result))
    conn.commit()
    return new_a, new_b


# ---------------- 调用 ----------------

def ask_model(cfg: dict, model: str, prompt: str) -> tuple[str, str]:
    """返回 (答案, 错误信息)"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.get("max_tokens", 800),
    }
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/").removesuffix("/v1") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {cfg['api_key']}",
                 "Content-Type": "application/json",
                 "User-Agent": f"model-arena/{__version__}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=cfg.get("timeout", 90)) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return (data["choices"][0]["message"].get("content") or "").strip(), ""
    except urllib.error.HTTPError as exc:
        exc.close()
        return "", f"HTTP {exc.code}"
    except Exception as exc:
        return "", type(exc).__name__


def ask_both(cfg: dict, model_a: str, model_b: str, prompt: str) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(ask_model, cfg, model_a, prompt)
        fut_b = pool.submit(ask_model, cfg, model_b, prompt)
        ans_a, err_a = fut_a.result()
        ans_b, err_b = fut_b.result()
    return {"a": (model_a, ans_a, err_a), "b": (model_b, ans_b, err_b)}


# ---------------- 命令 ----------------

def cmd_setup(args) -> int:
    cfg = load_config()
    if args.url:
        cfg["base_url"] = args.url
    if args.key:
        cfg["api_key"] = args.key
    if args.models:
        cfg["models"] = list(dict.fromkeys(args.models))
    save_config(cfg)
    print("配置已保存")
    print(_c(C.GRAY, f"文件：{CONFIG_PATH}"))
    if cfg["base_url"]:
        print(f"  地址  {cfg['base_url']}")
    if cfg["models"]:
        print(f"  模型  {', '.join(cfg['models'])}")
    return 0


def cmd_models(args) -> int:
    cfg = load_config()
    if args.action == "add":
        if not args.names:
            print(_c(C.RED, "请给出模型名，如 models add gpt-4o-mini deepseek-chat"))
            return 1
        cfg["models"] = list(dict.fromkeys(cfg["models"] + args.names))
        save_config(cfg)
    elif args.action == "remove":
        for name in args.names:
            if name in cfg["models"]:
                cfg["models"].remove(name)
        save_config(cfg)
    elif args.action == "clear":
        cfg["models"] = []
        save_config(cfg)
    for model in cfg["models"]:
        print(f"  {model}")
    if not cfg["models"]:
        print(_c(C.GRAY, "  （还没有模型，用 models add 添加）"))
    return 0


def cmd_fetch_models(args) -> int:
    cfg = load_config()
    if not cfg["base_url"] or not cfg["api_key"]:
        print(_c(C.RED, "先 setup 配置地址和密钥"))
        return 1
    req = urllib.request.Request(cfg["base_url"].rstrip("/").removesuffix("/v1") + "/v1/models",
                                 headers={"Authorization": f"Bearer {cfg['api_key']}",
                                          "User-Agent": f"model-arena/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        print(_c(C.RED, f"获取失败：{type(exc).__name__}"))
        return 1
    names = [m.get("id", "") for m in (data.get("data") or []) if isinstance(m, dict)]
    names = [n for n in names if n]
    print(f"站点共 {len(names)} 个模型：")
    for name in names[:60]:
        print(f"  {name}")
    if len(names) > 60:
        print(_c(C.GRAY, f"  …还有 {len(names) - 60} 个"))
    print()
    print(_c(C.GRAY, "挑选几个加入擂台：model-arena models add " + " ".join(names[:2])))
    return 0


def cmd_battle(args) -> int:
    cfg = load_config()
    if not cfg["base_url"] or not cfg["api_key"]:
        print(_c(C.RED, "先运行：model-arena setup -u 地址 -k 密钥"))
        return 1
    if len(cfg["models"]) < 2:
        print(_c(C.RED, "至少需要 2 个模型。model-arena models add 模型A 模型B"))
        return 1

    prompt = args.prompt
    if not prompt:
        try:
            prompt = input("本轮问题：").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    if not prompt:
        print(_c(C.RED, "问题不能为空"))
        return 1

    model_a, model_b = random.sample(cfg["models"], 2)
    print(_c(C.GRAY, f"\n正在问两个模型（已隐藏身份）…"))
    started = time.perf_counter()
    outcome = ask_both(cfg, model_a, model_b, prompt)
    used = time.perf_counter() - started

    # 左右位置随机打乱，防止位置偏见
    left_key, right_key = ("a", "b") if random.random() < 0.5 else ("b", "a")
    left_model, left_ans, left_err = outcome[left_key]
    right_model, right_ans, right_err = outcome[right_key]

    if left_err and right_err:
        print(_c(C.RED, f"两个模型都失败了：{left_err} / {right_err}"))
        return 1
    if left_err:
        print(_c(C.YELLOW, f"左侧模型失败（{left_err}），本轮自动判右侧胜"))
        conn = connect()
        record_battle(conn, prompt, left_model, right_model, "b")
        conn.close()
        return 0
    if right_err:
        print(_c(C.YELLOW, f"右侧模型失败（{right_err}），本轮自动判左侧胜"))
        conn = connect()
        record_battle(conn, prompt, left_model, right_model, "a")
        conn.close()
        return 0

    width = 66
    print()
    print(_c(C.GRAY, "═" * width))
    print(_c(C.BOLD, f"  问题：{prompt[:56]}") + _c(C.GRAY, f"   {used:.1f}s"))
    print(_c(C.GRAY, "═" * width))
    print(_c(C.CYAN + C.BOLD, "  ── 答案 A ──"))
    print(left_ans[:1500] + ("…" if len(left_ans) > 1500 else ""))
    print()
    print(_c(C.GREEN + C.BOLD, "  ── 答案 B ──"))
    print(right_ans[:1500] + ("…" if len(right_ans) > 1500 else ""))
    print()
    print(_c(C.GRAY, "═" * width))

    while True:
        try:
            choice = input("  哪个更好？[a]左 / [b]右 / [t]平局 / [s]跳过：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if choice in ("a", "b", "t", "s"):
            break
        print(_c(C.YELLOW, "  请输入 a / b / t / s"))

    if choice == "s":
        print(_c(C.GRAY, "  已跳过，不计入排名"))
        return 0

    conn = connect()
    if choice == "t":
        result = "draw"
    else:
        loser_side = "b" if choice == "a" else "a"
        result = "a" if (choice == "a") == (left_key == "a") else "b"
    new_a, new_b = record_battle(conn, prompt, model_a, model_b, result)
    conn.close()

    print()
    print(_c(C.GRAY, "  揭晓："))
    print(f"    左  {left_model}   {_c(C.GRAY, f'{new_a if left_model == model_a else new_b:.0f} 分')}")
    print(f"    右  {right_model}   {_c(C.GRAY, f'{new_b if right_model == model_b else new_a:.0f} 分')}")
    print(_c(C.GRAY, "\n  看排行：model-arena board"))
    return 0


def cmd_board(args) -> int:
    conn = connect()
    rows = conn.execute("""
        SELECT model, rating, wins, losses, draws FROM ratings
        ORDER BY rating DESC
    """).fetchall()
    if not rows:
        print(_c(C.YELLOW, "还没有对战记录。model-arena battle 开始第一轮"))
        conn.close()
        return 1
    print()
    print(_c(C.GRAY, "═" * 72))
    print(_c(C.BOLD, "  model-arena · 本地模型排行榜"))
    print(_c(C.GRAY, "═" * 72))
    print(f"  {'#':<5}{'模型':<32}{'Elo':<10}{'胜':<6}{'负':<6}{'平':<6}")
    print(_c(C.GRAY, "─" * 72))
    for idx, (model, rating, wins, losses, draws) in enumerate(rows, 1):
        mark = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "  "
        total = (wins or 0) + (losses or 0) + (draws or 0)
        color = C.GREEN if idx == 1 else C.RESET
        print(f"  {mark}{idx:<3}{model[:30]:<32}{_c(color, f'{rating:.0f}'):<19}"
              f"{wins or 0:<6}{losses or 0:<6}{draws or 0:<6}")
    total_battles = conn.execute("SELECT COUNT(*) FROM battles").fetchone()[0]
    print(_c(C.GRAY, "═" * 72))
    print(_c(C.GRAY, f"  共 {total_battles} 场对战 · Elo K={K_FACTOR} · 初始分 {DEFAULT_RATING:.0f}"))
    conn.close()
    return 0


def cmd_reset(args) -> int:
    conn = connect()
    conn.execute("DELETE FROM ratings")
    conn.execute("DELETE FROM battles")
    conn.commit()
    conn.close()
    print("排行榜已重置")
    return 0


def fetch_recommend(timeout: float = 5.0) -> dict | None:
    try:
        req = urllib.request.Request(REMOTE_URL, headers={"User-Agent": f"model-arena/{__version__}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        rec = data.get("recommend")
        if isinstance(rec, dict) and isinstance(rec.get("url"), str):
            return rec
    except Exception:
        pass
    return FALLBACK_RECOMMEND


def main(argv: list[str] | None = None) -> int:
    enable_ansi()
    parser = argparse.ArgumentParser(
        prog="model-arena", description="模型擂台盲测：同题多模型对比投票，本地 Elo 排行")
    parser.add_argument("-V", "--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_setup = sub.add_parser("setup", help="配置站点")
    p_setup.add_argument("-u", "--url")
    p_setup.add_argument("-k", "--key")
    p_setup.add_argument("-m", "--models", nargs="*")
    p_setup.set_defaults(func=cmd_setup)

    p_models = sub.add_parser("models", help="管理参赛模型")
    p_models.add_argument("action", choices=["list", "add", "remove", "clear"])
    p_models.add_argument("names", nargs="*")
    p_models.set_defaults(func=cmd_models)

    sub.add_parser("fetch", help="从站点拉取可用模型").set_defaults(func=cmd_fetch_models)

    p_battle = sub.add_parser("battle", help="开始一轮盲测")
    p_battle.add_argument("prompt", nargs="?", default="")
    p_battle.set_defaults(func=cmd_battle)

    sub.add_parser("board", help="查看排行榜").set_defaults(func=cmd_board)

    p_reset = sub.add_parser("reset", help="重置排行榜")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        rec = fetch_recommend()
        if rec:
            print()
            print(_c(C.CYAN, f"  推荐中转站  {rec.get('name', '')}  {rec.get('url', '')}"))
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
