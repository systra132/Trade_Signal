#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Discord Webhook + GitHub Actions で動く 1時間足 通知Bot（毎回ユニバース再取得）

要件（確定ルール）:
- 実行は毎時05分（未確定足回避）
- 対象: Bybit 上場の USDT 無期限(Linear) 全銘柄
- データ取得: Binance に上場していれば Binance から、なければ Bybit から
- 足: 1時間足。未確定は常に除外
- 欠落判定: 直近30時間（T−29h〜T）グリッドで欠落 ≤5 本なら判定続行、≥6 本はスキップ
- MA算出: T までの30スロットで「終値は LOCF 補完（先頭欠落は初回実値で後方埋め）」し、
           T で MA5 > MA10 > MA30 を評価（比較は T の 1 本のみ）
- 補助条件: 直近5本（いずれも T に向かう確定バー）で「実 Close > 各バー時点の 5MA（LOCF系列で算出）」が 3 本以上
- 陽線数: 直近30本で close > open をカウント（欠落は 0 とし、分母は常に 30）
- 並べ替え: 陽線数（30本中）降順で上位 10 件のみ通知
- スキップ条件: ①直近確定バーが取得できない ②APIが空/エラー ③欠落 ≥6
- 通知: 一致銘柄は「<SYMBOL> — 陽線 x/30 ｜ 欠落 y/30 ｜ CoinGlass: <URL>」
        スキップ銘柄は銘柄名のみ列挙（理由はログにのみ記録）
- 色分け: なし（単一メッセージ）

Secrets:
- DISCORD_WEBHOOK_URL: Discord の Webhook URL
"""

import os
import asyncio
import aiohttp
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ====== Discord ======
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

# ====== Exchange endpoints ======
BINANCE_FAPI_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"        # interval=1h
BYBIT_TICKERS_LINEAR = "https://api.bybit.com/v5/market/tickers"        # category=linear
BYBIT_KLINES_LINEAR  = "https://api.bybit.com/v5/market/kline"          # category=linear, interval=60

# ====== Controls ======
SEMAPHORE = asyncio.Semaphore(10)                 # 並列制御（必要なら下げる）
TIMEOUT = aiohttp.ClientTimeout(total=40)         # HTTPタイムアウト
RETRY_TOTAL_WINDOW_SEC = 300                      # 5分リトライ窓
RETRY_INTERVAL_SEC = 20                           # 20秒間隔で再試行

TOP_N = 10                                        # 上位通知件数

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def floor_to_hour_utc(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

def expected_slots(T_end: datetime, n: int = 30) -> List[datetime]:
    """T_end（直近確定バーの終了時刻=開始時刻+1hの'開始時刻'相当）を「開始時刻」とみなし、
    T_end - 29h ... T_end の 30 本分の '開始時刻(UTC, 毎時00分)' を昇順で返す。"""
    start = T_end - timedelta(hours=n-1)
    return [start + timedelta(hours=i) for i in range(n)]

def locf_fill_30(expected: List[datetime], close_map: Dict[datetime, float]) -> List[float]:
    """LOCF 補完（先頭欠落は最初に出現する実値で後方埋め）で 30 点の close 配列を作る"""
    vals = []
    first_real: Optional[float] = None
    last_seen: Optional[float] = None
    # 先にFIRST実値を探す（先頭欠落バックフィル用）
    for t in expected:
        if t in close_map and close_map[t] is not None:
            first_real = close_map[t]
            break
    # 構築
    for t in expected:
        if t in close_map and close_map[t] is not None:
            last_seen = close_map[t]
            vals.append(last_seen)
        else:
            if last_seen is not None:
                vals.append(last_seen)          # 前方の値を持ち越し
            else:
                vals.append(first_real if first_real is not None else None)  # 先頭欠落は最初の実値で後方埋め
    # 念のため None が残った場合（全欠損など）は False 扱いに備えて平均可能値で補完
    if any(v is None for v in vals):
        # あり得ないが安全策として0埋め
        vals = [0.0 if v is None else v for v in vals]
    return vals

def count_bull_30(expected: List[datetime], oc_map: Dict[datetime, Tuple[Optional[float], Optional[float]]]) -> int:
    """直近30スロットで close>open の本数（欠落は0カウント）"""
    cnt = 0
    for t in expected:
        o, c = oc_map.get(t, (None, None))
        if o is not None and c is not None and c > o:
            cnt += 1
    return cnt

def count_missing(expected: List[datetime], have_keys: set) -> int:
    return sum(1 for t in expected if t not in have_keys)

def last_5_above_ma5(expected: List[datetime], oc_map: Dict[datetime, Tuple[Optional[float], Optional[float]]], ma5_series: List[float]) -> int:
    """直近5本で「実 Close > 5MA（LOCF系列起点）」の本数"""
    # expected と ma5_series は同じ順序で30点
    assert len(expected) == len(ma5_series) == 30
    wins = 0
    for i in range(25, 30):  # 最後の5本
        t = expected[i]
        close = oc_map.get(t, (None, None))[1]
        ma5 = ma5_series[i]
        if close is not None and ma5 is not None and close > ma5:
            wins += 1
    return wins

def coinglass_url(symbol: str, prefer_binance: bool) -> str:
    # 例: https://www.coinglass.com/tv/Binance_BTCUSDT  /  https://www.coinglass.com/tv/Bybit_BTCUSDT
    return f"https://www.coinglass.com/tv/{'Binance' if prefer_binance else 'Bybit'}_{symbol}"

# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------

async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict | list:
    async with SEMAPHORE:
        async with session.get(url, params=params) as r:
            r.raise_for_status()
            # Binance webhook等で204が返る可能性はここでは無い（REST GET）ため json期待
            return await r.json()

async def post_discord(session: aiohttp.ClientSession, content: str):
    if not DISCORD_WEBHOOK_URL:
        print("WARNING: DISCORD_WEBHOOK_URL 未設定。通知は送られません。")
        return
    payload = {"content": content}
    async with session.post(DISCORD_WEBHOOK_URL, json=payload) as r:
        # Discord Webhook は通常 204 No Content
        if r.status not in (200, 204):
            body = await r.text()
            print(f"Discord webhook error: {r.status} {body}")

# ------------------------------------------------------------
# Symbol universe
# ------------------------------------------------------------

async def fetch_binance_symbols(session: aiohttp.ClientSession) -> set:
    js = await fetch_json(session, BINANCE_FAPI_EXCHANGE_INFO)
    out = set()
    for s in js.get("symbols", []):
        if s.get("contractType") == "PERPETUAL" and s.get("status") == "TRADING" and s.get("quoteAsset") == "USDT":
            out.add(s["symbol"])
    return out

async def fetch_bybit_symbols(session: aiohttp.ClientSession) -> set:
    js = await fetch_json(session, BYBIT_TICKERS_LINEAR, params={"category": "linear"})
    out = set()
    for item in js.get("result", {}).get("list", []) or []:
        sym = item.get("symbol", "")
        if sym.endswith("USDT"):
            out.add(sym)
    return out

# ------------------------------------------------------------
# Klines
# ------------------------------------------------------------

async def fetch_klines_binance(session: aiohttp.ClientSession, symbol: str, limit: int = 100) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": "1h", "limit": limit}
    data = await fetch_json(session, BINANCE_FAPI_KLINES, params=params)
    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"
    ])
    # 末尾の未確定を除外
    if len(df) >= 1:
        df = df.iloc[:-1]
    if df.empty:
        return df
    df = df[["open_time","open","close"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(timezone.utc)
    df["open"]  = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    # 正規化（必ず毎時00分に揃える）
    df["slot"] = df["open_time"].dt.floor("H")
    df = df.dropna(subset=["open","close"])
    return df[["slot","open","close"]]

async def fetch_klines_bybit(session: aiohttp.ClientSession, symbol: str, limit: int = 100) -> pd.DataFrame:
    params = {"symbol": symbol, "category": "linear", "interval": 60, "limit": limit}
    js = await fetch_json(session, BYBIT_KLINES_LINEAR, params=params)
    arr = js.get("result", {}).get("list", []) or []
    if not arr:
        return pd.DataFrame()
    # 昇順にして未確定を除外（Bybitは末尾が進行中のことがある）
    arr = sorted(arr, key=lambda x: int(x[0]))
    if len(arr) >= 1:
        arr = arr[:-1]
    if not arr:
        return pd.DataFrame()
    df = pd.DataFrame(arr, columns=["start","open","high","low","close","volume","turnover"])
    df = df[["start","open","close"]].copy()
    df["slot"]  = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True).dt.tz_convert(timezone.utc).dt.floor("H")
    df["open"]  = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["open","close"])
    return df[["slot","open","close"]]

# ------------------------------------------------------------
# Screening per symbol (single pass)
# ------------------------------------------------------------

class EvalResult:
    __slots__ = ("symbol","status","passed","bull30","miss30","prefer_binance","link")
    def __init__(self, symbol: str, status: str, passed: bool, bull30: int, miss30: int, prefer_binance: bool, link: str):
        self.symbol = symbol
        self.status = status                 # "ok" | "skip_no_latest" | "skip_error" | "skip_missing"
        self.passed = passed
        self.bull30 = bull30
        self.miss30 = miss30
        self.prefer_binance = prefer_binance
        self.link = link

async def evaluate_once(session: aiohttp.ClientSession, symbol: str, prefer_binance: bool, T_end: datetime) -> EvalResult:
    """
    1回分の評価:
      - データ取得（Binance優先 or Bybit）
      - 30スロットの欠落数 / 陽線数
      - MA と 補助条件の評価
      - ステータス判定
    """
    df = pd.DataFrame()
    try:
        if prefer_binance:
            df = await fetch_klines_binance(session, symbol)
            if df.empty:
                df = await fetch_klines_bybit(session, symbol)
        else:
            df = await fetch_klines_bybit(session, symbol)
    except Exception as e:
        # APIエラー
        return EvalResult(symbol, "skip_error", False, 0, 30, prefer_binance, coinglass_url(symbol, prefer_binance))

    if df.empty:
        return EvalResult(symbol, "skip_error", False, 0, 30, prefer_binance, coinglass_url(symbol, prefer_binance))

    # 期待30スロット
    slots = expected_slots(T_end, 30)  # 昇順 30点
    slot_set = set(df["slot"])
    miss = count_missing(slots, slot_set)

    # T が存在しない場合は「未反映」扱い（リトライ対象）
    if slots[-1] not in slot_set:
        return EvalResult(symbol, "skip_no_latest", False, 0, miss, prefer_binance, coinglass_url(symbol, prefer_binance))

    if miss >= 6:
        return EvalResult(symbol, "skip_missing", False, 0, miss, prefer_binance, coinglass_url(symbol, prefer_binance))

    # oc_map / close_map
    oc_map: Dict[datetime, Tuple[Optional[float], Optional[float]]] = {r.slot: (float(r.open), float(r.close)) for r in df.itertuples(index=False)}
    close_map: Dict[datetime, float] = {t: oc_map[t][1] for t in oc_map}

    # 陽線数（30本）
    bull30 = count_bull_30(slots, oc_map)

    # LOCF で 30 点の終値系列
    closes_30 = locf_fill_30(slots, close_map)
    s = pd.Series(closes_30, dtype="float64")

    # MA（T 1点のみ比較）
    ma5  = s.rolling(5).mean()
    ma10 = s.rolling(10).mean()
    ma30 = s.rolling(30).mean()

    # いずれかが NaN の場合（先頭補完失敗など）は安全側で不一致
    if any(math.isnan(x) for x in (ma5.iat[-1], ma10.iat[-1], ma30.iat[-1])):
        return EvalResult(symbol, "ok", False, bull30, miss, prefer_binance, coinglass_url(symbol, prefer_binance))

    cond_ma = (ma5.iat[-1] > ma10.iat[-1] > ma30.iat[-1])

    # 直近5本の「実 Close > 5MA」本数
    win5 = count_last5_above_ma5(slots, oc_map, ma5.tolist())

    passed = bool(cond_ma and win5 >= 3)
    return EvalResult(symbol, "ok", passed, bull30, miss, prefer_binance, coinglass_url(symbol, prefer_binance))

def count_last5_above_ma5(slots: List[datetime], oc_map: Dict[datetime, Tuple[Optional[float], Optional[float]]], ma5_series: List[float]) -> int:
    assert len(slots) == 30 and len(ma5_series) == 30
    wins = 0
    for i in range(25, 30):
        close = oc_map.get(slots[i], (None, None))[1]
        ma5 = ma5_series[i]
        if close is not None and ma5 is not None and close > ma5:
            wins += 1
    return wins

# ------------------------------------------------------------
# Retry orchestration (up to 5 minutes for no-latest / api error)
# ------------------------------------------------------------

async def evaluate_with_retry(session: aiohttp.ClientSession, symbol: str, prefer_binance: bool, T_end: datetime) -> EvalResult:
    deadline = datetime.now(timezone.utc) + timedelta(seconds=RETRY_TOTAL_WINDOW_SEC)
    attempt = 0
    last_res: Optional[EvalResult] = None
    while True:
        attempt += 1
        res = await evaluate_once(session, symbol, prefer_binance, T_end)
        last_res = res
        # リトライ対象: 未反映 / API空・エラー
        if res.status in ("skip_no_latest", "skip_error"):
            if datetime.now(timezone.utc) < deadline:
                await asyncio.sleep(RETRY_INTERVAL_SEC)
                continue
        # それ以外は打ち切り（skip_missing / ok）
        return res

# ------------------------------------------------------------
# Message build & send
# ------------------------------------------------------------

def build_messages(top_rows: List[EvalResult], skipped_syms: List[str], total_syms: int) -> List[str]:
    """Discord本文（text）を 2000 文字以内に分割して返す"""
    # 1) ヘッダー + 上位10件
    header = f"[通知Bot] 1時間足フィルター通過（上位{TOP_N}件、陽線数でソート）: {len(top_rows)}/{total_syms}\n"
    lines = []
    for r in top_rows:
        lines.append(f"• {r.symbol} — 陽線 {r.bull30}/30 ｜ 欠落 {r.miss30}/30 ｜ CoinGlass: {r.link}")

    # 2) スキップ銘柄（銘柄名のみ）
    skip_header = f"\nスキップ（未取得/エラー/欠落≥6）: {len(skipped_syms)}件\n"
    if skipped_syms:
        skip_body = ", ".join(skipped_syms)
    else:
        skip_body = "なし"

    full_text = header + "\n".join(lines) + skip_header + skip_body

    # 2000文字で分割
    if len(full_text) <= 1900:
        return [full_text]

    # シンプルにスキップ一覧を分割
    msgs = []
    base = header + "\n".join(lines) + skip_header
    cur = base
    for name in skipped_syms:
        piece = (name + ", ")
        if len(cur) + len(piece) > 1900:
            msgs.append(cur.rstrip(", "))
            cur = base + piece
        else:
            cur += piece
    msgs.append(cur.rstrip(", "))
    return msgs

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

async def main():
    utc_now = datetime.now(timezone.utc)
    T_end = floor_to_hour_utc(utc_now)  # 直近確定 1h の「開始時刻」
    print(f"[INFO] UTC now: {utc_now.isoformat()}  |  T_end: {T_end.isoformat()}")

    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        # ユニバース
        bybit_syms = await fetch_bybit_symbols(session)
        binance_syms = await fetch_binance_symbols(session)
        prefer_map: Dict[str, bool] = {sym: (sym in binance_syms) for sym in bybit_syms}

        # まず全シンボルを並列評価（リトライ込み）
        tasks = [evaluate_with_retry(session, sym, prefer_map[sym], T_end) for sym in bybit_syms]
        results: List[EvalResult] = await asyncio.gather(*tasks)

        # 合成
        passed = [r for r in results if r.status == "ok" and r.passed]
        skipped = [r.symbol for r in results if r.status in ("skip_no_latest", "skip_error", "skip_missing")]

        # 上位10件（陽線数で降順）
        passed_sorted = sorted(passed, key=lambda r: r.bull30, reverse=True)
        top = passed_sorted[:TOP_N]

        # メッセージ
        messages = build_messages(top, skipped, total_syms=len(bybit_syms))
        for m in messages:
            await post_discord(session, m)

        # ログ要約
        print(f"[INFO] universe={len(bybit_syms)}, passed={len(passed)}, skipped={len(skipped)}")
        if skipped:
            print(f"[DEBUG] skipped sample: {', '.join(skipped[:20])}{' ...' if len(skipped)>20 else ''}")

if __name__ == "__main__":
    asyncio.run(main())
