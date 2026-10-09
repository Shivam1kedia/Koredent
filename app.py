"""
Kordent
========================
Quantitative Multi-Agent Investment Committee.
Operating on Graham, Greenblatt, Dorsey, Trajectory, and Lynch frameworks.
Portfolio management formalized on Reilly & Brown: IPS, correlation-based
construction, systematic risk minimization, and macro-aware monitoring.

Streamlit web app with Gemini LLM, ChromaDB RAG, and yfinance tools.
"""

import sys

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import faulthandler
faulthandler.enable()  # SIGSEGV → dump the Python stack at the crash site to stderr (Cloud logs)

from supabase import create_client
import datetime
import streamlit as st
from google import genai
from google.genai import types
import re
import pymupdf
import pymupdf
import yfinance as yf
import json
import re
import requests
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import verdict_engine
import deep_metrics
import selector
import economics
import costs
 
def fmt_inr(value, decimals=0, symbol="₹"):
    """Indian-system digit grouping for money: 12,34,567 not 1,234,567.
    symbol defaults to the rupee sign; pass symbol='' to prefix your own."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{symbol}0"
    neg = v < 0
    v = abs(v)
    if decimals:
        whole = int(v); frac = f"{v - whole:.{decimals}f}"[2:]
    else:
        whole = int(round(v)); frac = ""
    _s = str(whole)
    if len(_s) > 3:
        last3, rest, parts = _s[-3:], _s[:-3], []
        while len(rest) > 2:
            parts.insert(0, rest[-2:]); rest = rest[:-2]
        if rest: parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    else:
        grouped = _s
    out = grouped + (f".{frac}" if frac else "")
    return f"{symbol}{'-' if neg else ''}{out}"

# One-time environment fingerprint → stderr (Cloud reliably captures stderr at
# boot, unlike stdout). Reveals the exact resolved native stack so an ABI
# mismatch (e.g. a pyarrow built against a different numpy) is visible without
# needing to capture a post-build crash.
try:
    import pyarrow as _pa
    _pa_v = _pa.__version__
except Exception as _e:
    _pa_v = f"MISSING({_e})"
try:
    import curl_cffi as _cc
    _cc_v = _cc.__version__
except Exception as _e:
    _cc_v = f"MISSING({_e})"
print(f"[ENV] python={sys.version.split()[0]} pandas={pd.__version__} "
      f"numpy={np.__version__} pyarrow={_pa_v} curl_cffi={_cc_v} "
      f"streamlit={st.__version__}", file=sys.stderr, flush=True)

SECTOR_INDEX_MAP = {
    "Technology": "^CNXIT",
    "Financial Services": "^NSEBANK",
    "Industrials": "^CNXINFRA",
    "Basic Materials": "^CNXMETAL",
    "Consumer Cyclical": "^CNXAUTO",
    "Consumer Defensive": "^CNXFMCG",
    "Healthcare": "^CNXPHARMA",
    "Energy": "^CNXENERGY",
    "Real Estate": "^CNXREALTY",
    "Communication Services": "^CNXMEDIA",
}

try:
    KITE_ENABLED = bool(st.secrets.get("KITE_PUBLISHER_KEY", ""))
except Exception:
    KITE_ENABLED = False

@st.cache_data(ttl=3600, show_spinner=False)

def get_sector_momentum(_sectors_tuple):
    """Fetch 1-month returns for Nifty sectoral indices. Cached for 1 hour."""
    results = {}
    for sector in _sectors_tuple:
        idx_ticker = SECTOR_INDEX_MAP.get(sector)
        if not idx_ticker:
            continue
        try:
            hist = yf.Ticker(idx_ticker).history(period="1mo")
            if len(hist) >= 2:
                start = float(hist["Close"].iloc[0])
                end = float(hist["Close"].iloc[-1])
                results[sector] = round((end - start) / start * 100, 1)
        except Exception:
            continue
    return results

# ──────────────────────────────────────────────
# FREE MODEL FALLBACK LIST
# ──────────────────────────────────────────────
FREE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]

def get_supabase():
    client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    if st.session_state.get("sb_access_token"):
        try:
            resp = client.auth.set_session(
                st.session_state.sb_access_token,
                st.session_state.sb_refresh_token
            )
            # Update tokens in case set_session refreshed them
            st.session_state.sb_access_token = resp.session.access_token
            st.session_state.sb_refresh_token = resp.session.refresh_token
        except Exception:
            # Refresh token expired — force re-login
            st.session_state.sb_access_token = None
            st.session_state.sb_refresh_token = None
            st.session_state.sb_user_email = None
            st.session_state.sb_user_id = None
    return client

def allocate_shares(stocks, sip_amount, existing_shares=None):
    """Breadth-then-depth share allocation (Reilly & Brown Ch 6).

    existing_shares: {ticker: portfolio_shares} for deploy/SIP cycles.
                     None for initial portfolio creation (all start at 0).

    One invariant governs every rupee: a security with 0 shares is never
    skipped so another security can buy a 2nd share. Breadth dominates depth
    until every security holds at least one share.

    TARGET BASIS — mark-to-market, portfolio-level. A stock's target is its
    allocation_pct of the WHOLE portfolio's current value (existing holdings
    at current price + this cycle's new money). "On target" means
    current_holding_value / total_portfolio_value == allocation_pct. This is
    the only basis consistent with minimising *present* unsystematic risk:
    concentration is about today's exposure, not cost basis.

    NOTE: allocation_pct must be the TRUE target weight, NOT a this-cycle
    deficit weight. Callers must pass raw holdings with real targets and let
    this function own all deficit/gap logic.

    Each cycle resolves to one of two states:

    STATE BROAD — at least one security still has 0 total shares.
      Step 1: buy 1 share of each unfunded security, cheapest first, until the
              next-cheapest unfunded one is unaffordable.
      Step 2: mop leftover into ALREADY-FUNDED, still-under-target names.
              Cheapest-first; buy one share at a time until that name's
              portfolio value meets-or-crosses its target (stop at the first
              crossing share), then advance. 0-share names are untouchable
              here — their claim rolls to next cycle's larger pooled budget.

    STATE DEEP — every security holds >= 1 share.
      Pure gap-fill: largest (target_value - current_value) first, as many
      shares as budget allows, descending.
    """
    _existing = existing_shares or {}
    result = []
    for s in stocks:
        result.append({**s, "shares": 0, "actual_amount": 0,
                       "_portfolio_shares": _existing.get(s.get("ticker", ""), 0)})

    remaining = sip_amount

    def _total(s):
        return s["_portfolio_shares"] + s["shares"]

    def _value(s):
        # Current mark-to-market value: existing holding + shares bought this cycle
        return _total(s) * s["price"]

    def _portfolio_after():
        # Total portfolio value once this cycle's money is deployed
        return sum(_value(x) for x in result) + remaining

    def _target_value(s):
        return _portfolio_after() * s.get("allocation_pct", 0) / 100.0

    def _buy(s):
        nonlocal remaining
        s["shares"] += 1
        s["actual_amount"] = s["shares"] * s["price"]
        remaining = round(sip_amount - sum(x["actual_amount"] for x in result), 2)

    # ── Breadth Step 1: one share of each unfunded security, cheapest first ──
    for s in sorted(result, key=lambda x: x["price"]):
        if _total(s) == 0 and 0 < s["price"] <= remaining:
            _buy(s)

    # Re-evaluate AFTER Step 1 — if Step 1 funded the last name, go DEEP.
    _any_unfunded = any(_total(s) == 0 and s["price"] > 0 for s in result)

    if _any_unfunded:
        # ── Breadth Step 2: mop leftover into funded, under-target names ──
        # Cheapest-first; fill each until its portfolio value meets-or-crosses
        # its portfolio-level target, then move on. Never touch a 0-share name.
        for s in sorted(result, key=lambda x: x["price"]):
            if _total(s) < 1:
                continue
            while (_value(s) < _target_value(s)
                   and 0 < s["price"] <= remaining):
                _buy(s)
        return result, round(remaining, 2)

    # ── STATE DEEP: every security funded — gap-fill by largest value gap ──
    _max_iter = len(result) * 200
    _iter = 0
    while remaining > 0 and _iter < _max_iter:
        _iter += 1
        best = None
        best_gap = -float("inf")
        for s in result:
            gap = _target_value(s) - _value(s)
            if s["price"] > 0 and s["price"] <= remaining and gap > best_gap:
                best = s
                best_gap = gap
        if best is None:
            break
        _buy(best)

    return result, round(remaining, 2)

def record_transaction(sb, portfolio_id, user_id, ticker, shares, price, amount_inr, txn_type="buy", nifty_cache=None,
                       benchmark_ticker=None, raise_on_error=False):
    """Record a buy/sell in sip_transactions with benchmark shadow data.

    benchmark_ticker: the PORTFOLIO's own benchmark. Passing None keeps the old
    NIFTYBEES default, but any portfolio benchmarked elsewhere had its shadow
    silently mispriced by every app-side transaction before this was added.

    raise_on_error: buys may stay non-blocking (the holding row is the record of
    truth and the tracker's genesis bootstrap can reconstruct them). SELLS MUST
    NOT. A swallowed sell insert permanently loses the proceeds, and nothing can
    reconstruct them. Sell callers pass True and write the ledger BEFORE
    mutating holdings.
 
    cost_inr is computed HERE, at write time, and stored on the row. Not at read
    time: Zerodha's rates change, and a cost recomputed years later at today's
    rates is a different number from the one actually paid. Storing it freezes
    the rate that applied on the day, which is the same point-in-time discipline
    the score archive is built on.
    """
    _bt = benchmark_ticker or "NIFTYBEES.NS"
    nifty_px = nifty_cache
    if nifty_px is None:
        try:
            nifty_px = yf.Ticker(_bt).fast_info.last_price
        except Exception:
            nifty_px = None
    nifty_u = None
    if nifty_px and nifty_px > 0:
        raw = float(amount_inr) / nifty_px
        nifty_u = round(raw, 6) if txn_type == "buy" else round(-raw, 6)
    # Sell-side carries the flat DP charge and is ~40x the buy-side cost at a
    # Rs 333 position. costs.py is the single source of truth for the rates.
    _ex = costs.exchange_for(ticker)
    _cost = (costs.buy_cost(amount_inr, _ex) if txn_type == "buy"
             else costs.sell_cost(amount_inr, _ex))
    try:
        sb.table("sip_transactions").insert({
            "portfolio_id": str(portfolio_id),
            "user_id": str(user_id),
            "ticker": ticker,
            "shares": float(shares),
            "price": round(float(price), 2),
            "amount_inr": round(float(amount_inr), 2),
            "cost_inr": _cost,
            "transaction_type": txn_type,
            "transaction_date": datetime.date.today().isoformat(),
            "nifty_price": round(nifty_px, 2) if nifty_px else None,
            "nifty_units": nifty_u,
            "benchmark_ticker": _bt,
        }).execute()
    except Exception as e:
        if raise_on_error:
            raise
        print(f"Txn log failed (non-blocking): {e}")
    return nifty_px


def live_price(ticker, fallback=0.0):
    """Best-effort last traded price, falling back to a caller-supplied number."""
    try:
        _p = yf.Ticker(ticker).fast_info.last_price
        return float(_p) if _p and _p > 0 else float(fallback or 0.0)
    except Exception:
        return float(fallback or 0.0)

def record_withdrawal(sb, portfolio, user_id, amount_inr):
    """Record money LEAVING the portfolio for the user's own bank.

    Always raises on failure. A swallowed withdrawal is the worst row to lose:
    the cash stays on the books, the next buy is funded from money that is not
    there, external capital is understated, and every return figure after that
    point is overstated - compounding with each cycle.

    ticker/shares/price are NOT NULL in the schema and meaningless here, so the
    row is tagged CASH / 0 / 0. economics.replay_ledger reads amount_inr only
    for this type.
    """
    _bt = portfolio.get("benchmark_ticker") or "NIFTYBEES.NS"
    _bp = live_price(_bt, 0.0)
    _amt = round(float(amount_inr), 2)
    sb.table("sip_transactions").insert({
        "portfolio_id": str(portfolio["id"]),
        "user_id": str(user_id),
        "ticker": "CASH",
        "shares": 0,
        "price": 0,
        "amount_inr": _amt,
        # 0.0, never NULL. Moving cash to your own bank costs nothing, and that
        # is a KNOWN zero — NULL means "cost unknown" and would wrongly count
        # this row into cost_rows_missing.
        "cost_inr": 0.0,
        "transaction_type": "withdrawal",
        "transaction_date": datetime.date.today().isoformat(),
        "nifty_price": round(_bp, 2) if _bp > 0 else None,
        "nifty_units": round(-_amt / _bp, 6) if _bp > 0 else None,
        "benchmark_ticker": _bt,
    }).execute()


KITE_RELAY_URL = "https://shivam1kedia.github.io/Koredent/kite-basket.html"

def kite_buy_url(ticker, quantity=1, order_type="MARKET"):
    """Single stock Kite Publisher URL via GitHub Pages relay (POST required by Kite)."""
    import urllib.parse
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    exchange = "NSE" if ".NS" in ticker else "BSE"
    data = json.dumps([{"exchange": exchange, "tradingsymbol": symbol,
             "transaction_type": "BUY", "quantity": int(quantity),
             "order_type": order_type}])
    key = st.secrets["KITE_PUBLISHER_KEY"]
    return f"{KITE_RELAY_URL}?api_key={urllib.parse.quote(key)}&data={urllib.parse.quote(data)}"


def kite_sell_url(ticker, quantity=1, order_type="MARKET"):
    """Single-stock Kite Publisher SELL URL via the GitHub Pages relay (mirror of
    kite_buy_url with transaction_type=SELL). Used by the portfolio review's
    per-holding 'Sell on Kite' button."""
    import urllib.parse
    symbol = ticker.replace(".NS", "").replace(".BO", "")
    exchange = "NSE" if ".NS" in ticker else "BSE"
    data = json.dumps([{"exchange": exchange, "tradingsymbol": symbol,
             "transaction_type": "SELL", "quantity": int(quantity),
             "order_type": order_type}])
    key = st.secrets["KITE_PUBLISHER_KEY"]
    return f"{KITE_RELAY_URL}?api_key={urllib.parse.quote(key)}&data={urllib.parse.quote(data)}"


def kite_basket_url(stocks):
    """Multiple stocks in one Kite session via GitHub Pages relay (POST required by Kite).
    stocks: list of dicts with 'ticker' and 'quantity' keys. Max 10 per Kite limit."""
    import urllib.parse
    data = []
    for s in stocks[:10]:
        symbol = s["ticker"].replace(".NS", "").replace(".BO", "")
        exchange = "NSE" if ".NS" in s["ticker"] else "BSE"
        data.append({"exchange": exchange, "tradingsymbol": symbol,
                     "transaction_type": "BUY", "quantity": int(s.get("quantity", 1)),
                     "order_type": "MARKET"})
    key = st.secrets["KITE_PUBLISHER_KEY"]
    return f"{KITE_RELAY_URL}?api_key={urllib.parse.quote(key)}&data={urllib.parse.quote(json.dumps(data))}"


def load_txns(sb, portfolio_id):
    """Raw ledger rows for one portfolio. Every column economics.replay_ledger
    needs, including created_at (the same-day tiebreak that makes a sell-then-
    rebuy fund itself from cash instead of drawing fresh external capital)."""
    try:
        return sb.table("sip_transactions").select(
            "id, created_at, ticker, shares, price, amount_inr, cost_inr, "
            "transaction_type, transaction_date, nifty_price"
        ).eq("portfolio_id", str(portfolio_id)).execute().data or []
    except Exception:
        return []


def portfolio_money(sb, portfolio_id, enriched_holdings, benchmark_ticker=None):
    """THE money computation for app.py. Decide once, display everywhere.

    Every site that shows invested / value / P&L / return calls this and nothing
    else. enriched_holdings must have come through enrich_holdings_live so
    current_value is a live market value.

    benchmark_ticker: pass the portfolio's own benchmark to get shadow_value
    back. The shadow is derived from EXTERNAL flows inside the replay, not from
    summing the stored nifty_units column - that column treats a sale as a
    withdrawal, which it is not.
    """
    mv = sum((h.get("current_value") or 0) for h in (enriched_holdings or []))
    _bp = live_price(benchmark_ticker, 0.0) if benchmark_ticker else None
    return economics.portfolio_economics(load_txns(sb, portfolio_id), mv, _bp)


def compute_portfolio_xirr(econ, current_nifty_shadow=None):
    """XIRR from EXTERNAL cash flows only (economics model (a)).

    Buys and sells are internal transfers between cash and securities, not money
    entering or leaving the investor's pocket. The previous version signed every
    buy negative and every sell positive, which reports a return on gross
    turnover and turns a same-day rotation into a fictitious round trip.

    econ: output of portfolio_money(). Terminal value is total_assets.
    """
    try:
        from pyxirr import xirr
    except ImportError:
        return None, None

    dates, amounts = economics.xirr_flows(econ)
    if not dates or len(dates) < 2:
        return None, None

    # XIRR meaningless under 90 days
    if (datetime.date.today() - dates[0]).days < 90:
        return None, None

    try:
        port_xirr = xirr(dates, amounts)
    except Exception:
        port_xirr = None

    port_xirr_pct = round(port_xirr * 100, 2) if port_xirr is not None else None

    nifty_xirr_pct = None
    if current_nifty_shadow and current_nifty_shadow > 0:
        # The shadow holds securities only; the portfolio's uninvested cash is
        # added so both sides are measured on the same total-assets basis.
        nifty_amounts = amounts[:-1] + [float(current_nifty_shadow) + float(econ.get("cash_balance") or 0.0)]
        try:
            n_xirr = xirr(dates, nifty_amounts)
            nifty_xirr_pct = round(n_xirr * 100, 2) if n_xirr is not None else None
        except Exception:
            pass

    return port_xirr_pct, nifty_xirr_pct


def _add_months(d, months):
    """Add months to a date, clamping day to valid range."""
    import calendar
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    max_day = calendar.monthrange(y, m)[1]
    return d.replace(year=y, month=m, day=min(d.day, max_day))


def compute_goal_projection(current_value, sip_monthly, target_amount, target_date_str, actual_cagr=None):
    """Compute goal trajectory projection.
    Returns dict with status, gap, projected_value, chart points, sip_increase suggestion.
    actual_cagr: decimal (0.12 = 12%). None → use 12% Nifty default."""
    today = datetime.date.today()
    try:
        target_date = datetime.date.fromisoformat(str(target_date_str))
    except (ValueError, TypeError):
        return None

    months_remaining = max(0, (target_date.year - today.year) * 12 + (target_date.month - today.month))
    if months_remaining <= 0:
        gap = target_amount - current_value
        return {"status": "achieved" if gap <= 0 else "missed", "gap": gap,
                "months_remaining": 0, "using_default": actual_cagr is None}

    using_default = actual_cagr is None
    cagr = actual_cagr if actual_cagr is not None else 0.12
    actual_monthly = (1 + cagr) ** (1 / 12) - 1

    def fv(pv, r, pmt, n):
        if r == 0 or n == 0:
            return pv + pmt * n
        g = (1 + r) ** n
        return pv * g + pmt * (g - 1) / r

    projected_value = fv(current_value, actual_monthly, sip_monthly, months_remaining)
    gap = target_amount - projected_value

    if gap <= 0:
        status = "ahead"
    elif abs(gap) <= target_amount * 0.05:
        status = "on_track"
    else:
        status = "behind"

    # Needed monthly rate via bisection
    needed_monthly = None
    needed_cagr = None
    def objective(r):
        return fv(current_value, r, sip_monthly, months_remaining) - target_amount
    lo, hi = -0.01, 0.08
    try:
        if objective(lo) * objective(hi) < 0:
            for _ in range(60):
                mid = (lo + hi) / 2
                if objective(mid) > 0:
                    hi = mid
                else:
                    lo = mid
            needed_monthly = (lo + hi) / 2
            needed_cagr = (1 + needed_monthly) ** 12 - 1
    except Exception:
        pass

    # SIP increase suggestion (if behind)
    sip_increase = None
    if status == "behind" and actual_monthly > 0 and months_remaining > 0:
        try:
            g = (1 + actual_monthly) ** months_remaining
            if g > 1:
                needed_pmt = (target_amount - current_value * g) * actual_monthly / (g - 1)
                sip_increase = max(0, round(needed_pmt - sip_monthly, -2))  # round to nearest 100
        except Exception:
            pass

    # Generate projection points (monthly)
    current_points = []
    needed_points = []
    for m in range(months_remaining + 1):
        pt_date = _add_months(today, m)
        current_points.append({"date": pt_date, "value": fv(current_value, actual_monthly, sip_monthly, m)})
        if needed_monthly is not None:
            needed_points.append({"date": pt_date, "value": fv(current_value, needed_monthly, sip_monthly, m)})

    return {
        "status": status, "gap": gap, "projected_value": projected_value,
        "months_remaining": months_remaining,
        "actual_cagr": cagr, "needed_cagr": needed_cagr,
        "sip_increase": sip_increase,
        "current_points": current_points, "needed_points": needed_points,
        "using_default": using_default,
    }


def render_score_history_chart(sb, ticker, stock_name=None, chart_key=None):
    """Query score_history for a ticker and render a Plotly score trend chart.
    Returns True if chart was rendered, False otherwise."""
    try:
        resp = sb.table("score_history").select(
            "date, score, graham_pass, greenblatt_pass, dorsey_pass, trajectory_pass, quality_pass"
        ).eq("ticker", ticker).order("date").execute()
        rows = resp.data or []
    except Exception:
        return False

    if len(rows) < 2:
        if len(rows) == 1:
            st.caption(f"Only 1 data point so far — chart appears after 2+ days of tracking.")
        return False

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    # Score line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["score"],
        line=dict(color="#1D4ED8", width=2),
        fill="tozeroy", fillcolor="rgba(29, 78, 216, 0.06)",
        name="Score",
        # No denominator: score_history rows carry the raw integer and no
        # applicability columns, so the denominator for a PAST date is not
        # reconstructable. Historical points also span the v3->v4 schema break.
        # An unlabelled number is honest; a fabricated "/5" is not.
        hovertemplate="Score %{y}<extra></extra>",
    ))

    # Quality-fail markers
    if "quality_pass" in df.columns:
        q_fail = df[df["quality_pass"] == False]
        if not q_fail.empty:
            fig.add_trace(go.Scatter(
                x=q_fail["date"], y=q_fail["score"],
                mode="markers",
                marker=dict(color="#EF4444", size=7, symbol="x"),
                name="Quality Fail",
                hovertemplate="%{y}/4 (quality fail)<extra></extra>",
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=200,
        yaxis=dict(range=[-0.3, 4.5], dtick=1, gridcolor="rgba(0,0,0,0.05)"),
        xaxis=dict(showgrid=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    label = stock_name or ticker
    st.caption(f"**{label}** — Score Trend")
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # Framework summary from latest row
    latest = rows[-1]
    parts = []
    for fw, key in [("Graham", "graham_pass"), ("Greenblatt", "greenblatt_pass"),
                     ("Dorsey", "dorsey_pass"), ("Trajectory", "trajectory_pass")]:
        parts.append(f"{fw} {'✓' if latest.get(key) else '✗'}")
    q_tag = " · Quality ✓" if latest.get("quality_pass") else " · Quality ✗"
    st.caption(f"{' · '.join(parts)}{q_tag}")
    return True


def enrich_holdings_live(holdings, cache_key=None):
    """Add live market data to holdings.

    Computes actual_allocation_pct from current market prices:
        (shares × current_price) / total_portfolio_value × 100
    Preserves the stored allocation_pct (target allocation) from the database.
    Adds 'current_price', 'current_value', and 'actual_allocation_pct' to each holding dict.
    Caches prices in session state for 5 min to avoid re-fetching on Streamlit reruns.
    """
    import time as _time

    price_cache = {}
    if cache_key:
        cached = st.session_state.get(f"_price_cache_{cache_key}")
        if cached and _time.time() - cached.get("_ts", 0) < 300:
            price_cache = dict(cached)
            price_cache.pop("_ts", None)

    for h in holdings:
        t = h.get("ticker", "")
        if t and t not in price_cache:
            try:
                price_cache[t] = yf.Ticker(t).fast_info.last_price or h.get("price_at_entry", 0)
            except Exception:
                price_cache[t] = h.get("price_at_entry", 0)

    if cache_key:
        st.session_state[f"_price_cache_{cache_key}"] = {**price_cache, "_ts": _time.time()}

    enriched = []
    for h in holdings:
        h = dict(h)  # shallow copy — don't mutate originals
        t = h.get("ticker", "")
        h["current_price"] = round(price_cache.get(t, h.get("price_at_entry", 0)), 2)
        h["current_value"] = round(h.get("shares", 0) * h["current_price"], 2)
        enriched.append(h)

    total_val = sum(h["current_value"] for h in enriched)
    for h in enriched:
        h["actual_allocation_pct"] = round(h["current_value"] / total_val * 100, 1) if total_val > 0 else 0

    return enriched

def generate_portfolio_pdf(portfolio, holdings, history_data=None, alerts=None,
                           chart_buf=None, narrative=None,
                           xirr_data=None, goal_data=None, goal_chart_buf=None,
                           sector_data=None, score_data=None, user_name=None,
                           redact_holdings=False, econ=None):
    """Generate a premium Alpha Report PDF for a portfolio."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, PageBreak, HRFlowable, KeepTogether)
    from reportlab.platypus import Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT, TA_JUSTIFY
    import io
 
    buffer = io.BytesIO()
    W = 170 * mm  # content width (A4 - 40mm margins)
 
    # ── Colors ──
    NAVY = colors.HexColor("#0F172A")
    BLUE = colors.HexColor("#1E3A5F")
    ACCENT = colors.HexColor("#1D4ED8")
    LIGHT_BG = colors.HexColor("#F8FAFC")
    BORDER = colors.HexColor("#E2E8F0")
    MUTED = colors.HexColor("#64748B")
    DARK = colors.HexColor("#0F172A")
    GREEN = colors.HexColor("#16A34A")
    RED = colors.HexColor("#DC2626")
    WHITE = colors.white
 
    # ── Page footer ──
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(20*mm, 10*mm,
            "Generated by Kordent  ·  Not financial advice  ·  Past performance does not guarantee future results")
        canvas.drawRightString(A4[0] - 20*mm, 10*mm, f"Page {doc.page}")
        # Thin top rule on footer
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(20*mm, 14*mm, A4[0] - 20*mm, 14*mm)
        canvas.restoreState()
 
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            topMargin=20*mm, bottomMargin=20*mm,
                            leftMargin=20*mm, rightMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []
 
    # ── Typography ──
    s_title = ParagraphStyle("RTitle", fontName="Helvetica-Bold", fontSize=32,
                              textColor=NAVY, alignment=TA_CENTER, spaceAfter=2)
    s_subtitle = ParagraphStyle("RSub", fontName="Helvetica", fontSize=14,
                                 textColor=MUTED, alignment=TA_CENTER, spaceAfter=0)
    s_cover_name = ParagraphStyle("RCoverName", fontName="Helvetica-Bold", fontSize=14,
                                   textColor=DARK, alignment=TA_CENTER)
    s_cover_user = ParagraphStyle("RCoverUser", fontName="Helvetica", fontSize=11,
                                   textColor=MUTED, alignment=TA_CENTER)
    s_cover_date = ParagraphStyle("RCoverDate", fontName="Helvetica", fontSize=10,
                                   textColor=MUTED, alignment=TA_CENTER)
    s_heading = ParagraphStyle("RHeading", fontName="Helvetica-Bold", fontSize=13,
                                textColor=ACCENT, spaceBefore=12, spaceAfter=6)
    s_body = ParagraphStyle("RBody", fontName="Helvetica", fontSize=10,
                             textColor=DARK, leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
    s_small = ParagraphStyle("RSmall", fontName="Helvetica", fontSize=8,
                              textColor=MUTED, leading=10)
    s_cell = ParagraphStyle("RCell", fontName="Helvetica", fontSize=8,
                             textColor=DARK, leading=10)
    s_cell_bold = ParagraphStyle("RCellBold", fontName="Helvetica-Bold", fontSize=8,
                                  textColor=DARK, leading=10)
    s_cell_green = ParagraphStyle("RCellGreen", fontName="Helvetica-Bold", fontSize=8,
                                   textColor=GREEN, leading=10)
    s_cell_red = ParagraphStyle("RCellRed", fontName="Helvetica-Bold", fontSize=8,
                                 textColor=RED, leading=10)
 
    # Common table style base
    def _table_style(extra=None):
        cmds = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("TEXTCOLOR", (0, 1), (-1, -1), DARK),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, BORDER),
        ]
        if extra:
            cmds.extend(extra)
        return TableStyle(cmds)
 
    today_str = datetime.date.today().strftime("%B %d, %Y")
 
    # ══════════════════════════════════════
    # COVER PAGE
    # ══════════════════════════════════════
    story.append(Spacer(1, 55*mm))
    story.append(Paragraph("KORDENT", s_title))
    story.append(Spacer(1, 8*mm))
    story.append(HRFlowable(width="25%", thickness=2, color=ACCENT,
                            spaceAfter=8, spaceBefore=0, hAlign="CENTER"))
    story.append(Paragraph("Alpha Report", s_subtitle))
    story.append(Spacer(1, 25*mm))
    story.append(Paragraph(portfolio.get("name", "Portfolio"), s_cover_name))
    if user_name:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(user_name, s_cover_user))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(today_str, s_cover_date))
    story.append(PageBreak())
 
    # ══════════════════════════════════════
    # EXECUTIVE SUMMARY — KPI Cards
    # ══════════════════════════════════════
    story.append(Paragraph("Executive Summary", s_heading))
 
    sip = portfolio.get("sip_amount", 0)
    # One basis for all three KPIs. Previously pnl came from live holdings while
    # return_pct came from the cached portfolios column, so this row could print
    # a loss in rupees next to a gain in percent.
    if econ is None:
        # Degraded path: fall back to the portfolio's cached economics columns,
        # which the tracker writes on ONE consistent basis. Do not blend these
        # with live holdings values — that blend is what printed a rupee loss
        # beside a percentage gain.
        _ta = float(portfolio.get("current_value") or 0)
        _cash = float(portfolio.get("cash_balance") or 0)
        _wd = float(portfolio.get("withdrawn") or 0)
        _rp = portfolio.get("current_return_pct")
        # return_pct = (assets + withdrawn - ext) / ext, so ext = (assets + withdrawn) / (1 + r)
        _ext = round((_ta + _wd) / (1 + _rp / 100.0), 2) if (_rp is not None and _rp != -100) else (_ta + _wd)
        econ = {
            "external_capital": _ext,
            "cash_balance": _cash,
            "market_value": max(0.0, _ta - _cash),
            "total_assets": _ta,
            "realized_pnl": float(portfolio.get("realized_pnl") or 0),
            "unrealized_pnl": 0.0,
            "withdrawn": _wd,
            "total_pnl": round(_ta + _wd - _ext, 2),
            "return_pct": _rp,
            # None, not 0.0. This degraded path reconstructs economics from the
            # portfolios table, which stores no cost total. It does not know what
            # costs were paid, and a 0.0 here would assert that none were.
            "total_costs_paid": None,
            "cost_rows_missing": None,
        }
    current_val = econ["total_assets"]
    total_invested = econ["external_capital"]
    pnl = econ["total_pnl"]
    return_pct = econ["return_pct"] if econ["return_pct"] is not None else 0
    cash_bal = econ.get("cash_balance", 0) or 0
    withdrawn_amt = econ.get("withdrawn", 0) or 0
 
    port_xirr = xirr_data[0] if xirr_data else None
    nifty_xirr = xirr_data[1] if xirr_data else None
    alpha_xirr = round(port_xirr - nifty_xirr, 2) if port_xirr is not None and nifty_xirr is not None else None
 
    pnl_color = GREEN if pnl >= 0 else RED
    pnl_hex = "#16A34A" if pnl >= 0 else "#DC2626"
 
    def _kpi(label, value, color_hex="#0F172A"):
        return Paragraph(
            f'<font size="8" color="#64748B">{label}</font><br/>'
            f'<font size="16" color="{color_hex}"><b>{value}</b></font>',
            ParagraphStyle("KPI", fontName="Helvetica", fontSize=16, leading=20,
                           alignment=TA_CENTER, spaceAfter=0)
        )
 
    # Row 1: Value / Invested / P&L
    kpi_row1 = [
        _kpi("Total Assets" if cash_bal > 0 else "Current Value",
             f"Rs. {fmt_inr(current_val, symbol='')}"),
        _kpi("Capital Invested", f"Rs. {fmt_inr(total_invested, symbol='')}"),
        _kpi("P&L", f"Rs. {pnl:+,.0f} ({return_pct:+.1f}%)", pnl_hex),
    ]
 
    # Row 2: XIRR / Nifty / Alpha (only if data exists)
    kpi_row2 = None
    if port_xirr is not None:
        xirr_hex = "#16A34A" if port_xirr >= 0 else "#DC2626"
        alpha_hex = "#16A34A" if alpha_xirr and alpha_xirr >= 0 else "#DC2626"
        kpi_row2 = [
            _kpi("Portfolio XIRR", f"{port_xirr:+.1f}%", xirr_hex),
            _kpi("Nifty XIRR", f"{nifty_xirr:+.1f}%" if nifty_xirr is not None else "—"),
            _kpi("Alpha", f"{alpha_xirr:+.1f}%" if alpha_xirr is not None else "—", alpha_hex),
        ]
 
    # Withdrawals are value returned to the investor. They sit in the P&L
    # numerator, never netted off invested capital, so the KPI row alone would
    # not explain why assets are below capital on a winding-down portfolio.
    kpi_note = None
    if withdrawn_amt > 0:
        kpi_note = (f"Includes Rs. {fmt_inr(withdrawn_amt, symbol='')} already withdrawn "
                    f"to your bank. Capital Invested is what you paid in and does not "
                    f"fall when money is taken back out.")

    kpi_col = W / 3
    kpi_data = [kpi_row1]
    if kpi_row2:
        kpi_data.append(kpi_row2)

    # Sprint 11: Row 3 — Risk metrics from Reilly & Brown
    _beta = portfolio.get("portfolio_beta")
    _sharpe = portfolio.get("sharpe_ratio")
    _div_score = portfolio.get("diversification_score")
    if _beta is not None or _sharpe is not None or _div_score is not None:
        _div_hex = "#16A34A" if _div_score and _div_score >= 70 else "#F59E0B" if _div_score and _div_score >= 40 else "#DC2626"
        _sharpe_hex = "#16A34A" if _sharpe and _sharpe > 0 else "#DC2626"
        kpi_row3 = [
            _kpi("Portfolio Beta (β)", f"{_beta:.2f}" if _beta is not None else "—"),
            _kpi("Sharpe Ratio", f"{_sharpe:.2f}" if _sharpe is not None else "—", _sharpe_hex if _sharpe is not None else "#0F172A"),
            _kpi("Diversification", f"{_div_score}/100" if _div_score is not None else "—", _div_hex if _div_score is not None else "#0F172A"),
        ]
        kpi_data.append(kpi_row3)
 
    kpi_table = Table(kpi_data, colWidths=[kpi_col] * 3)
    kpi_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BORDER),
        ("LINEBEFORE", (1, 0), (1, -1), 0.5, BORDER),
        ("LINEBEFORE", (2, 0), (2, -1), 0.5, BORDER),
    ]))
    story.append(kpi_table)
    if kpi_note:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            f'<font size="8" color="#64748B">{kpi_note}</font>',
            ParagraphStyle("KPINote", fontName="Helvetica", fontSize=8, leading=11)))
    story.append(Spacer(1, 4*mm))

    # Sprint 11: IPS Policy Summary
    _ips = (portfolio.get("portfolio_profile") or {}).get("ips_policy")
    if _ips:
        story.append(Paragraph("Investment Policy Statement", s_heading))
        _alloc = _ips.get("allocation_policy", {})
        _sizing = _ips.get("portfolio_sizing", {})
        _ips_text = (
            f"Return objective: {_ips.get('return_objective', '—').replace('_', ' ').title()}. "
            f"Risk tolerance: {_ips.get('risk_tolerance', '—').title()}. "
            f"Life cycle: {_ips.get('life_cycle_phase', '—').replace('_', ' ').title()} (age {_ips.get('age', '—')}). "
            f"Benchmark: {_ips.get('benchmark', '—')}. "
            f"Target stocks: {_sizing.get('actual', '—')} (budget supports up to {_sizing.get('affordable', '—')}, book minimum: {_sizing.get('book_minimum', 12)}). "
            f"Diversification status: {_sizing.get('diversification_status', '—').replace('_', ' ').title()}. "
            f"Constraints — max single stock: {_alloc.get('max_single_stock_pct', '—')}%, "
            f"max sector: {_alloc.get('max_sector_pct', '—')}%, "
            f"min sectors: {_alloc.get('min_sectors', '—')}, "
            f"large-cap floor: {_alloc.get('large_cap_min_pct', '—')}%."
        )
        story.append(Paragraph(_ips_text, s_body))
        story.append(Spacer(1, 4*mm))
 
    # Meta row: SIP / Type / Horizon / Review
    _inv_type = str(portfolio.get("investor_type", "—")).title()
    _horizon = str(portfolio.get("time_horizon", "—")).title()
    _review = f"Every {portfolio.get('review_freq', 90)} days"
    _next_rev = str(portfolio.get("next_review_date", "—"))
 
    meta_parts = [f"SIP: Rs. {fmt_inr(sip, symbol='')}/mo", f"Type: {_inv_type}",
                  f"Horizon: {_horizon}", f"Review: {_review}", f"Next: {_next_rev}"]
    # Goal status
    if goal_data and goal_data.get("status"):
        _gs = goal_data
        _target = portfolio.get("target_amount", 0)
        _status_map = {
            "on_track": f"Goal: On track — Rs. {fmt_inr(_target, symbol='')}",
            "behind": f"Goal: Behind by Rs. {fmt_inr(abs(_gs.get('gap', 0)), symbol='')}",
            "ahead": f"Goal: Ahead by Rs. {fmt_inr(abs(_gs.get('gap', 0)), symbol='')}",
            "achieved": f"Goal: Achieved!",
        }
        meta_parts.append(_status_map.get(_gs["status"], ""))
 
    story.append(Paragraph("  ·  ".join(meta_parts), s_small))
    story.append(Spacer(1, 8*mm))
 
    # ══════════════════════════════════════
    # HOLDINGS TABLE
    # ══════════════════════════════════════
    if holdings:
        story.append(Paragraph("Holdings", s_heading))
 
        header = ["Stock", "Shares", "CMP", "Invested", "Value", "P&L", "Alloc %"]
        rows = [header]
 
        _tot_inv = 0
        _tot_val = 0
        _tot_pnl = 0
        for idx, h in enumerate(holdings):
            entry = h.get("price_at_entry", 0)
            cmp = h.get("current_price", entry)
            shares = h.get("shares", 0)
            invested = h.get("sip_amount_inr", 0)
            value = h.get("current_value", 0)
            h_pnl = value - invested
            h_ret = ((cmp - entry) / entry * 100) if entry > 0 else 0
            _tot_inv += invested
            _tot_val += value
            _tot_pnl += h_pnl
 
            stock_name = h.get("name", "—")
            if redact_holdings:
                stock_name = f"Holding {idx + 1}"
 
            pnl_text = f"Rs. {h_pnl:+,.0f} ({h_ret:+.1f}%)"
            pnl_para = Paragraph(pnl_text, s_cell_green if h_pnl >= 0 else s_cell_red)
 
            rows.append([
                Paragraph(stock_name, s_cell),
                str(int(shares)),
                f"Rs. {fmt_inr(cmp, symbol='')}",
                f"Rs. {fmt_inr(invested, symbol='')}",
                f"Rs. {fmt_inr(value, symbol='')}",
                pnl_para,
                f"{h.get('actual_allocation_pct', h.get('allocation_pct', 0))}%",
            ])
 
        _tot_pnl_para = Paragraph(f"Rs. {_tot_pnl:+,.0f}", s_cell_green if _tot_pnl >= 0 else s_cell_red)
        rows.append([
            Paragraph("Total", s_cell_bold), "", "",
            f"Rs. {fmt_inr(_tot_inv, symbol='')}", f"Rs. {fmt_inr(_tot_val, symbol='')}", _tot_pnl_para, ""
        ])
 
        # 170mm total: Stock(48) + Shares(14) + CMP(20) + Invested(22) + Value(22) + P&L(28) + Alloc(16)
        h_widths = [48*mm, 14*mm, 20*mm, 22*mm, 22*mm, 28*mm, 16*mm]
        h_table = Table(rows, colWidths=h_widths)
        h_table.setStyle(_table_style([
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, NAVY),
        ]))
        story.append(h_table)
        story.append(Spacer(1, 6*mm))
 
    # ══════════════════════════════════════
    # PORTFOLIO GROWTH CHART
    # ══════════════════════════════════════
    if chart_buf:
        chart_section = []
        chart_section.append(Paragraph("Portfolio Growth", s_heading))
        chart_buf.seek(0)
        chart_section.append(RLImage(chart_buf, width=W, height=65*mm))
        chart_section.append(Spacer(1, 4*mm))
        story.append(KeepTogether(chart_section))
 
    # ══════════════════════════════════════
    # GOAL PROJECTION CHART
    # ══════════════════════════════════════
    if goal_chart_buf:
        goal_section = []
        goal_section.append(Paragraph("Goal Projection", s_heading))
        goal_chart_buf.seek(0)
        goal_section.append(RLImage(goal_chart_buf, width=W, height=55*mm))
        goal_section.append(Spacer(1, 4*mm))
        story.append(KeepTogether(goal_section))
 
    # ══════════════════════════════════════
    # SECTOR EXPOSURE + SCORE SUMMARY (same page)
    # ══════════════════════════════════════
    combined_section = []
 
    if holdings:
        total_val_sec = sum(h.get("current_value", 0) for h in holdings)
        if total_val_sec > 0:
            sector_weights = {}
            for h in holdings:
                sec = h.get("sector", "Unknown")
                sector_weights[sec] = sector_weights.get(sec, 0) + h.get("current_value", 0)
 
            if sector_weights:
                combined_section.append(Paragraph("Sector Exposure", s_heading))
                sec_header = ["Sector", "Weight", "1M Momentum"]
                sec_rows = [sec_header]
                sec_style_extra = []
 
                for row_i, (sec, val) in enumerate(sorted(sector_weights.items(), key=lambda x: -x[1])):
                    weight_pct = val / total_val_sec * 100
                    momentum = sector_data.get(sec) if sector_data else None
                    mom_str = f"{momentum:+.1f}%" if momentum is not None else "—"
                    sec_rows.append([sec, f"{weight_pct:.1f}%", mom_str])
                    if momentum is not None:
                        c = GREEN if momentum >= 0 else RED
                        sec_style_extra.append(("TEXTCOLOR", (2, row_i + 1), (2, row_i + 1), c))
 
                # 170mm: Sector(80) + Weight(45) + Momentum(45)
                sec_table = Table(sec_rows, colWidths=[80*mm, 45*mm, 45*mm])
                sec_table.setStyle(_table_style(sec_style_extra))
                combined_section.append(sec_table)
                combined_section.append(Spacer(1, 8*mm))
 
    # Score Summary
    if score_data and holdings:
        combined_section.append(Paragraph("Framework Scores", s_heading))
        sc_header = ["Stock", "Score", "Graham", "Greenblatt", "Dorsey", "Trajectory", "Quality"]
        sc_rows = [sc_header]
        sc_style_extra = []
 
        for row_i, h in enumerate(holdings):
            ticker = h.get("ticker", "")
            sd = score_data.get(ticker, {})
            sc = sd.get("score", "—")
 
            stock_name = h.get("name", ticker)
            if redact_holdings:
                stock_name = f"Holding {row_i + 1}"
 
            pass_cells = []
            for col_i, key in enumerate(["graham_pass", "greenblatt_pass", "dorsey_pass", "trajectory_pass", "quality_pass"]):
                passed = sd.get(key, False)
                pass_cells.append("Pass" if passed else "Fail")
                c = GREEN if passed else RED
                sc_style_extra.append(("TEXTCOLOR", (col_i + 2, row_i + 1), (col_i + 2, row_i + 1), c))
                sc_style_extra.append(("FONTNAME", (col_i + 2, row_i + 1), (col_i + 2, row_i + 1),
                                       "Helvetica-Bold"))
 
            sc_rows.append([Paragraph(stock_name, s_cell), str(sc)] + pass_cells)
 
        # 170mm: Stock(44) + Score(16) + Gra(22) + Grn(22) + Dor(22) + Tra(22) + Qua(22)
        sc_table = Table(sc_rows, colWidths=[44*mm, 16*mm, 22*mm, 22*mm, 22*mm, 22*mm, 22*mm])
        sc_table.setStyle(_table_style(sc_style_extra + [
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        combined_section.append(sc_table)
        combined_section.append(Spacer(1, 6*mm))
 
    if combined_section:
        story.append(KeepTogether(combined_section))
 
    # ══════════════════════════════════════
    # ACTIVE ALERTS
    # ══════════════════════════════════════
    if alerts:
        alert_section = [Paragraph("Active Alerts", s_heading)]
        for a in alerts:
            # Type picks a distinctive glyph where one exists; severity is the
            # fallback. This map used to key on alert_type when alert_type held
            # SEVERITY words — so the moment types became real, every score_drop,
            # quality_fail and price_crash would have silently fallen through to
            # the bullet.
            icon = ({"review_due": "⏰", "goal_drift": "🎯", "new_entry": "🆕"}
                    .get(a.get("alert_type", ""))
                    or {"danger": "⚠", "warning": "!", "info": "★"}
                    .get(a.get("severity") or "", "•"))
            safe_hl = str(a.get("headline", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            alert_section.append(Paragraph(f"{icon}  {safe_hl}", s_body))
        alert_section.append(Spacer(1, 4*mm))
        story.append(KeepTogether(alert_section))

    # Sprint 11: Risk & Performance Metrics (Reilly & Brown Ch 7, 18)
    _sortino = portfolio.get("sortino_ratio")
    _treynor = portfolio.get("treynor_ratio")
    _jensen = portfolio.get("jensen_alpha")
    _ir = portfolio.get("information_ratio")
    _drawdown = portfolio.get("max_drawdown")
    _capm = portfolio.get("capm_expected_return")

    # Sprint 13 §3: two a retail investor acts on as plain-English headline
    # (Jensen as prose; drawdown with its short-history caveat), the rest as a
    # compact methodology line — the PDF equivalent of "behind the expander".
    _dd_prov = portfolio.get("max_drawdown_provisional")
    _semidev = portfolio.get("semi_deviation")
    _rfr_used = portfolio.get("rfr_used")
    _hist_days = portfolio.get("metrics_history_days")

    if any(v is not None for v in [_sortino, _treynor, _jensen, _ir, _drawdown, _capm, _semidev]):
        story.append(Paragraph("Risk & Performance Metrics", s_heading))

        if _jensen is not None:
            _jp = _jensen * 100
            _dir = "ahead of" if _jp >= 0 else "behind"
            _tail = "a selection edge, if it holds up" if _jp >= 0 else "the picks have lagged so far"
            story.append(Paragraph(
                f"<b>Selection alpha:</b> about {_jp:+.1f} points {_dir} what this portfolio's risk "
                f"level alone would predict &mdash; the part attributable to which stocks were picked, "
                f"after stripping out how the market moved and how much risk was taken ({_tail}).",
                s_body))
        if _drawdown is not None:
            if _dd_prov:
                story.append(Paragraph(
                    f"<b>Worst fall so far:</b> {_drawdown*100:.1f}% &mdash; on short history "
                    f"(under ~6 months) this is not yet a reliable risk estimate; the true worst "
                    f"fall is likely deeper.", s_body))
            else:
                story.append(Paragraph(
                    f"<b>Maximum drawdown:</b> {_drawdown*100:.1f}% peak-to-trough.", s_body))

        _method = []
        if _sortino is not None: _method.append(f"Sortino {_sortino:.2f}")
        if _treynor is not None: _method.append(f"Treynor {_treynor:.4f}")
        if _ir is not None:      _method.append(f"Information ratio {_ir:.2f}")
        if _capm is not None:    _method.append(f"CAPM expected {_capm*100:.1f}% p.a.")
        if _semidev is not None: _method.append(f"Semi-deviation {_semidev*100:.1f}%")
        if _method:
            story.append(Paragraph("Methodology: " + " &middot; ".join(_method) + ".", s_small))
 
        _stamp = []
        if _rfr_used is not None:
            # Sprint 16: the rate is live (macro_read), so the stamp must say
            # whether it was a reading or the fallback. Before this the column
            # was never written and the stamp never rendered at all.
            _rs = portfolio.get("rfr_status")
            _src = ("" if _rs in (None, "ok")
                    else f", fallback — live series {str(_rs).lower()}")
            _stamp.append(f"risk-free rate {_rfr_used*100:.2f}%{_src}")
        if _hist_days is not None: _stamp.append(f"computed on {_hist_days} trading days of history")
        if _stamp:
            story.append(Paragraph("(" + "; ".join(_stamp) + ".)", s_small))

        story.append(Spacer(1, 4*mm))
 
    # ══════════════════════════════════════
    # INVESTMENT ANALYSIS (narrative)
    # ══════════════════════════════════════
    if narrative:
        story.append(PageBreak())
        story.append(Paragraph("Investment Analysis", s_heading))
        story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT,
                                spaceAfter=8, spaceBefore=2))
 
        s_stock_head = ParagraphStyle("RStockHead", fontName="Helvetica-Bold", fontSize=10,
                                       textColor=ACCENT, spaceBefore=10, spaceAfter=3)
        s_narrative = ParagraphStyle("RNarrative", fontName="Helvetica", fontSize=9.5,
                                      textColor=DARK, leading=13.5, spaceAfter=6,
                                      leftIndent=0, rightIndent=0, alignment=TA_JUSTIFY)
        s_section_head = ParagraphStyle("RSectionHead", fontName="Helvetica-Bold", fontSize=11,
                                         textColor=NAVY, spaceBefore=14, spaceAfter=6)
 
        for line in narrative.split("\n"):
            line = line.strip()
            if not line:
                continue
 
            safe_line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
 
            # Section headers (PORTFOLIO THESIS, PORTFOLIO ASSESSMENT, etc.)
            if line.isupper() and len(line) > 5 and not line.startswith("("):
                story.append(HRFlowable(width="30%", thickness=0.5, color=BORDER,
                                        spaceAfter=4, spaceBefore=8, hAlign="LEFT"))
                story.append(Paragraph(safe_line, s_section_head))
            # Stock headers — "Company Name (Sector · Score X/5 · XX%)"
            elif "·" in line and ("/" in line or "%" in line):
                story.append(Paragraph(safe_line, s_stock_head))
            else:
                story.append(Paragraph(safe_line, s_narrative))
 
        story.append(Spacer(1, 6*mm))
 
    # ── Build ──
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buffer.seek(0)
    return buffer.getvalue()
 
 
def generate_portfolio_chart(history_data):
    """Render stacked absolute chart (Invested + Portfolio + Nifty Shadow) as PNG bytes."""
    if not history_data or len(history_data) < 2:
        return None
 
    df = pd.DataFrame(history_data)
    df["date"] = pd.to_datetime(df["date"])
 
    has_invested = "cumulative_invested" in df.columns and df["cumulative_invested"].notna().sum() >= 2
    has_shadow = "nifty_shadow_value" in df.columns and df["nifty_shadow_value"].notna().sum() >= 2
 
    fig = go.Figure()
 
    if has_invested:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["cumulative_invested"],
            fill="tozeroy", fillcolor="rgba(29, 78, 216, 0.08)",
            line=dict(color="rgba(29, 78, 216, 0.25)", width=1),
            name="Invested",
        ))
 
    if has_shadow:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["nifty_shadow_value"],
            line=dict(color="#9CA3AF", width=1.5, dash="dash"),
            name="Nifty Shadow",
        ))
 
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["total_value"],
        line=dict(color="#1D4ED8", width=2.5),
        name="Portfolio",
    ))
 
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=350, width=800,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(tickprefix="Rs.", tickformat=","),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
 
    # Fallback: if kaleido unavailable, try matplotlib
    buf = _plotly_to_png(fig)
    if buf:
        return buf
 
    # Matplotlib fallback (old behavior)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import io
 
        port_base = df["total_value"].iloc[0]
        df["portfolio_pct"] = ((df["total_value"] / port_base) - 1) * 100 if port_base > 0 else 0
        has_nifty_old = "nifty_value" in df.columns and df["nifty_value"].notna().sum() >= 2
        if has_nifty_old:
            nifty_base = df["nifty_value"].dropna().iloc[0]
            df["nifty_pct"] = ((df["nifty_value"] / nifty_base) - 1) * 100 if nifty_base > 0 else 0
 
        mpl_fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=150)
        ax.plot(df["date"], df["portfolio_pct"], color="#1D4ED8", linewidth=2, label="Portfolio")
        if has_nifty_old:
            ax.plot(df["date"], df["nifty_pct"], color="#9CA3AF", linewidth=1.5, linestyle="--", label="Nifty 50")
        ax.set_ylabel("Return %", fontsize=9, color="#374151")
        ax.tick_params(labelsize=8, colors="#6B7280")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.axhline(y=0, color="#E5E7EB", linewidth=0.8)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("white")
        mpl_fig.patch.set_facecolor("white")
        plt.tight_layout()
        fb = io.BytesIO()
        mpl_fig.savefig(fb, format="png", bbox_inches="tight", facecolor="white")
        plt.close(mpl_fig)
        fb.seek(0)
        return fb
    except Exception:
        return None


def generate_portfolio_narrative(portfolio, holdings, collection, score_data=None):
    """Generate LLM-written investment analysis — concise, layman-friendly."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
 
    portfolio_type = portfolio.get("investor_type", "balanced")
 
    # ── Build per-stock context with actual score data ──
    stocks_block = ""
    for idx, h in enumerate(holdings):
        ticker = h.get("ticker", "")
        name = h.get("name", ticker)
        sector = h.get("sector", "unknown")
        entry = h.get("price_at_entry", 0)
        shares = h.get("shares", 0)
        alloc = h.get("actual_allocation_pct", h.get("allocation_pct", 0))
        invested = h.get("sip_amount_inr", 0)
        current_price = h.get("current_price", entry)
        pnl_pct = ((current_price - entry) / entry * 100) if entry > 0 else 0
 
        # Score data from universe
        sd = score_data.get(ticker, {}) if score_data else {}
        score = sd.get("score", h.get("score_at_entry", 0))
        graham = "Pass" if sd.get("graham_pass") else "Fail"
        greenblatt = "Pass" if sd.get("greenblatt_pass") else "Fail"
        dorsey = "Pass" if sd.get("dorsey_pass") else "Fail"
        trajectory = "Pass" if sd.get("trajectory_pass") else "Fail"
        quality = "Pass" if sd.get("quality_pass", True) else "Fail"
 
        # Book passages — query with specific frameworks that PASSED
        passing = []
        if sd.get("graham_pass"): passing.append("Graham value margin of safety")
        if sd.get("greenblatt_pass"): passing.append("Greenblatt magic formula return on capital")
        if sd.get("dorsey_pass"): passing.append("Dorsey economic moat competitive advantage")
        if sd.get("trajectory_pass"): passing.append("growth momentum earnings trajectory")
        query = f"{' '.join(passing)} {sector}" if passing else f"{sector} stock investment"
 
        try:
            passages = search_book_passages(collection, query, 2)
            book_text = "\n".join(p["text"] for p in passages[:2])
        except Exception:
            book_text = ""
 
        stocks_block += f"""
--- {name} ({ticker}) ---
Sector: {sector} | Allocation: {alloc}% | Score: {score} | P&L: {pnl_pct:+.1f}%
Frameworks: Graham={graham}, Greenblatt={greenblatt}, Dorsey={dorsey}, Trajectory={trajectory}, Quality={quality}
Book context (use sparingly, in your own words): {book_text[:500]}
"""
 
    # ── Sector distribution ──
    from collections import Counter
    sector_dist = Counter(h.get("sector", "Unknown") for h in holdings)
    sector_summary = ", ".join(f"{s}: {c}" for s, c in sector_dist.most_common())
 
    prompt = f'''You are Kordent's Chief Investment Analyst writing a portfolio report for a retail investor who does NOT know financial jargon.
 
Portfolio: {portfolio.get('name')} | Type: {portfolio_type} | Horizon: {portfolio.get('time_horizon')} | SIP: Rs.{portfolio.get('sip_amount', 0):,}/month
Sectors: {sector_summary}
Holdings count: {len(holdings)}
 
{stocks_block}
 
Write the analysis with these sections:
 
PORTFOLIO THESIS
2-3 sentences. What is this portfolio designed to do? Who is it for? Be specific to THIS portfolio, not generic.
 
Then for EACH stock, write a header line and ONE paragraph (4-6 sentences):
 
Header format: Company Name (Sector · Score N of M · XX% allocation)
 
The paragraph must cover:
1. WHY we hold this — connect to the specific frameworks that PASS (e.g. "Greenblatt and Trajectory pass, meaning the company is capital-efficient and on an upward trend")
2. What the FAILING frameworks mean — be honest (e.g. "Graham fails because the stock is not cheap by classic value metrics")
3. The KEY RISK in plain English — what could go wrong, specific to this company or sector
 
Do NOT use labels like "Selection:", "Strength:", "Risk:". Write it as flowing prose.
Do NOT use terms like "the book says" or "according to the framework". Translate into plain English.
Do NOT use markdown, bullets, asterisks, or dashes. Plain text only.
 
PORTFOLIO ASSESSMENT
3-4 sentences covering: Is the portfolio well-diversified? Is it concentrated in any one sector? What is the one thing the investor should watch or improve? Be specific and actionable.'''
 
    try:
        last_good = st.session_state.get("last_working_model")
        if last_good and last_good in FREE_MODELS:
            models_to_try = [last_good] + [m for m in FREE_MODELS if m != last_good]
        else:
            models_to_try = FREE_MODELS
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                st.session_state["last_working_model"] = model_name
                return response.text
            except Exception as e:
                error_msg = str(e).upper()
                if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                    continue
                raise e
        return "Analysis unavailable — all models rate-limited."
    except Exception as e:
        return f"Analysis unavailable: {e}"

def upload_shared_report(sb, user_id, pdf_bytes):
    """Upload PDF to Supabase storage and return a 24h signed URL.
    
    Args:
        sb: Supabase client
        user_id: User's UUID
        pdf_bytes: Raw PDF bytes
    
    Returns:
        dict with 'url' (signed URL) and 'expires_in' (seconds), or 'error'
    """
    import uuid as _uuid
 
    report_id = str(_uuid.uuid4())
    storage_path = f"{user_id}/{report_id}.pdf"
 
    try:
        # Upload to 'reports' bucket
        sb.storage.from_("reports").upload(
            path=storage_path,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
 
        # Generate 24h signed URL
        expires_in = 86400  # 24 hours
        signed = sb.storage.from_("reports").create_signed_url(storage_path, expires_in)
 
        if signed and signed.get("signedURL"):
            return {"url": signed["signedURL"], "expires_in": expires_in}
        elif isinstance(signed, dict) and "signedUrl" in signed:
            return {"url": signed["signedUrl"], "expires_in": expires_in}
        else:
            return {"url": str(signed), "expires_in": expires_in}
 
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

def generate_health_check(portfolio, holdings, universe_df, collection):
    """Portfolio-level diagnostic: diversification, risk, valuation, book-grounded assessment."""
    if not holdings:
        return None

    # ── Look up each holding in universe CSV ──
    enriched = []
    sectors = []
    betas = []
    pe_vs_avgs = []
    pct_from_highs = []
    scores = []

    for h in holdings:
        ticker = h.get("ticker", "")
        alloc = h.get("actual_allocation_pct", h.get("allocation_pct", 0))
        row = universe_df[universe_df["ticker"] == ticker]

        sector = h.get("sector", "Unknown")
        beta = None
        pe_vs_avg = None
        pct_from_high = None
        score = h.get("score_at_entry", 0)

        if not row.empty:
            r = row.iloc[0]
            sector = r.get("sector", sector) if pd.notna(r.get("sector")) else sector
            beta = round(float(r["beta"]), 2) if pd.notna(r.get("beta")) else None
            pe_vs_avg = round(float(r["pe_vs_avg"]), 2) if pd.notna(r.get("pe_vs_avg")) else None
            pct_from_high = round(float(r["pct_from_high"]), 2) if pd.notna(r.get("pct_from_high")) else None
            score = int(r["score"]) if pd.notna(r.get("score")) else score

        sectors.append(sector)
        if beta is not None:
            betas.append((beta, alloc))
        if pe_vs_avg is not None:
            pe_vs_avgs.append(pe_vs_avg)
        if pct_from_high is not None:
            pct_from_highs.append(pct_from_high)
        scores.append(score)

        enriched.append({
            "name": h.get("name", ticker), "ticker": ticker, "sector": sector,
            "alloc": alloc, "beta": beta, "pe_vs_avg": pe_vs_avg,
            "pct_from_high": pct_from_high, "score": score,
            "score_label": _score_label(ticker, score),
        })

    # ── Sector concentration (HHI) ──
    from collections import Counter
    sector_counts = Counter(sectors)
    total = len(sectors)
    sector_weights = {s: c / total for s, c in sector_counts.items()}
    hhi = sum(w ** 2 for w in sector_weights.values())
    diversification_score = round((1 - hhi) * 100)

    # ── Weighted average beta ──
    if betas:
        total_alloc = sum(a for _, a in betas)
        avg_beta = round(sum(b * a for b, a in betas) / total_alloc, 2) if total_alloc > 0 else None
    else:
        avg_beta = None

    # ── Valuation positioning ──
    avg_pe_vs_avg = round(sum(pe_vs_avgs) / len(pe_vs_avgs), 1) if pe_vs_avgs else None
    avg_pct_from_high = round(sum(pct_from_highs) / len(pct_from_highs), 1) if pct_from_highs else None

    # ── Quality distribution ──
    score_dist = Counter(scores)

    # ── Concentration warnings ──
    warnings = []
    for sector, weight in sector_weights.items():
        if weight > 0.3:
            warnings.append(f"{sector} is {weight*100:.0f}% of portfolio (>30%)")
    if avg_beta and avg_beta > 1.3:
        warnings.append(f"High portfolio beta ({avg_beta}) — amplifies market swings")
    if avg_pe_vs_avg and avg_pe_vs_avg > 20:
        warnings.append(f"Holdings trading {avg_pe_vs_avg}% above their historical PE — possible overvaluation")

    metrics = {
        "diversification_score": diversification_score,
        "sector_distribution": dict(sector_counts),
        "avg_beta": avg_beta,
        "avg_pe_vs_historical": avg_pe_vs_avg,
        "avg_pct_from_52w_high": avg_pct_from_high,
        "score_distribution": dict(score_dist),
        "warnings": warnings,
        "holdings_detail": enriched,
    }

    # ── LLM narrative ──
    investor_type = portfolio.get("investor_type", "balanced")
    time_horizon = portfolio.get("time_horizon", "medium")

    # Book passages for portfolio construction
    try:
        passages = search_book_passages(
            collection,
            f"{investor_type} portfolio construction sector diversification "
            f"concentration risk margin of safety number of holdings",
            3,
        )
        book_text = "\n".join(p["text"] for p in passages[:2])
    except Exception:
        book_text = ""

    holdings_summary = "\n".join(
        f"  {e['name']} ({e['ticker']}) — Sector: {e['sector']}, Alloc: {e['alloc']}%, "
        f"Beta: {e['beta'] or 'N/A'}, PE vs Avg: {e['pe_vs_avg'] or 'N/A'}%, "
        f"From 52w High: {e['pct_from_high'] or 'N/A'}%, Score: {e.get('score_label') or e['score']}"
        for e in enriched
    )
    # ── Find complementary stocks from universe ──
    complement_candidates = []
    try:
        overweight_sectors = [s for s, w in sector_weights.items() if w > 0.25]
        candidates = universe_df[
            selector.meets_score_mask(universe_df, 3) &
            (universe_df["quality_pass"] == True) &
            (~universe_df["ticker"].isin([h.get("ticker") for h in holdings])) &
            (universe_df["pe"] > 0) &
            (pd.notna(universe_df["pe"]))
        ].copy()

        # Prefer sectors not already overweight
        if overweight_sectors:
            underweight = candidates[~candidates["sector"].isin(overweight_sectors)]
            if len(underweight) >= 5:
                candidates = underweight

        # Top 5 by score then lowest PE
        candidates = candidates.sort_values(["score", "pe"], ascending=[False, True]).head(5)

        for _, r in candidates.iterrows():
            complement_candidates.append({
                "ticker": r["ticker"],
                "name": str(r.get("name", r["ticker"])),
                "sector": str(r.get("sector", "N/A")),
                "score": int(r["score"]),
                "score_label": selector.score_label(r),
                "pe": round(float(r["pe"]), 2) if pd.notna(r.get("pe")) else None,
                "roe_pct": round(float(r["roe_pct"]), 2) if pd.notna(r.get("roe_pct")) else None,
                "pct_from_high": round(float(r["pct_from_high"]), 2) if pd.notna(r.get("pct_from_high")) else None,
                "price": round(float(r["price"]), 2) if pd.notna(r.get("price")) else None,
            })
    except Exception:
        pass

    # ── User Decision Context ──
    user_context = ""
    _profile = portfolio.get("portfolio_profile") or {}
    if isinstance(_profile, str):
        try: _profile = json.loads(_profile)
        except: _profile = {}
    if _profile.get("decision_context"):
        user_context = f"\nUSER DECISION CONTEXT:\n{_profile.get('decision_context')}\n(CRITICAL: Do not penalize the portfolio for risks or sector concentrations that the user explicitly accepted during portfolio creation.)\n"

    prompt = f"""You are Kordent's Chief Risk Officer diagnosing a portfolio's health.

Portfolio: {portfolio.get('name')} | Type: {investor_type} | Horizon: {time_horizon}
Holdings: {total} stocks
{user_context}
PORTFOLIO METRICS:
Diversification Score: {diversification_score}/100 (based on sector HHI)
Sector Distribution: {dict(sector_counts)}
Average Beta: {avg_beta or 'N/A'}
Average PE vs Historical Average: {avg_pe_vs_avg or 'N/A'}% (negative = discount, positive = premium)
Average Distance from 52-Week High: {avg_pct_from_high or 'N/A'}%
Score Distribution: {dict(score_dist)}
Warnings: {warnings if warnings else 'None'}

HOLDINGS:
{holdings_summary}

BOOK CONTEXT:
{book_text[:800]}

Write a diagnostic with these sections:
1. VERDICT (one line: is this portfolio healthy, needs attention, or at risk?)
2. STRENGTHS (what's working well — cite book principles)
3. RISKS (what could go wrong — cite book warnings, be specific about which holdings)
4. ACTION ITEMS — ONLY if the portfolio has real problems. If diversification score is above 80, no sector exceeds 30%, and quality scores are 3+, then state "Portfolio is well-constructed. No changes recommended." and output ACTIONS_JSON: []. Do NOT recommend changes just to have something to say. A good portfolio deserves acknowledgment, not perpetual tinkering. Graham explicitly warns against excessive trading and over-optimization.

Be direct and specific. Reference actual holdings by name. Under 300 words total.

COMPLEMENTARY CANDIDATES (stocks not in portfolio that could improve diversification):
{chr(10).join(f"  {c['ticker']} — {c['name']} | Sector: {c['sector']} | Score: {c.get('score_label') or c['score']} | PE: {c['pe']} | ROE: {c['roe_pct']}%" for c in complement_candidates) if complement_candidates else "None available"}
If the portfolio needs more holdings or sector diversity, recommend specific stocks from the candidates above in your ACTION ITEMS.

After the narrative, on a new line, output a JSON block starting with ACTIONS_JSON: followed by a JSON array.
Each action object must have:
- "type": one of "sell", "reduce", "investigate"
- "ticker": the stock ticker
- "reason": one line explanation
- For "reduce": include "target_alloc_pct" (the new target allocation percentage)
- For "sell": include "shares" (number of shares to sell, 0 means all)
- For "add": include "name", "sector", "score", "pe", and "suggested_alloc_pct"

Example: ACTIONS_JSON: [{{"type": "reduce", "ticker": "CHENNPETRO.NS", "target_alloc_pct": 10, "reason": "Trim cyclical concentration"}}, {{"type": "add", "ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Financial Services", "score": 4, "pe": 18.5, "suggested_alloc_pct": 10, "reason": "Adds financial sector exposure, improves diversification"}}]

Only include actions for holdings that need changes. Do not include "investigate" for more than 2 stocks."""

    narrative = None
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        last_good = st.session_state.get("last_working_model")
        models = [last_good] + [m for m in FREE_MODELS if m != last_good] if last_good else FREE_MODELS
        for model in models:
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                raw_text = response.text
                # Parse structured actions from LLM response
                actions = []
                narrative = raw_text
                if "ACTIONS_JSON:" in raw_text:
                    parts = raw_text.split("ACTIONS_JSON:", 1)
                    narrative = parts[0].strip()
                    try:
                        actions_str = parts[1].strip()
                        # Handle markdown code fences
                        actions_str = actions_str.replace("```json", "").replace("```", "").strip()
                        actions = json.loads(actions_str)
                    except Exception:
                        actions = []
                st.session_state.last_working_model = model
                break
            except Exception as e:
                error_msg = str(e).upper()
                if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                    continue
                break
    except Exception:
        pass

    metrics["narrative"] = narrative
    metrics["actions"] = actions if 'actions' in dir() else []
    metrics["complement_candidates"] = complement_candidates
    return metrics


def find_replacement_candidates(investor_type, time_horizon, exclude_tickers, current_sectors):
    """Find replacement stocks when review flags sells."""
    df = universe_df.copy()

    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] != False]
    df = df[df["years_of_data"] >= 2]
    df = df[pd.notna(df["pe"]) & pd.notna(df["roe_pct"]) & pd.notna(df["de"])]
    df = df[df["pe"] > 0]

    # Same profile filtering as get_sip_candidates
    if investor_type == "defensive":
        df = df[selector.meets_score_mask(df, 3)]
        mask = df["graham_pass"] == True
        if mask.sum() >= 5:
            df = df[mask]
    elif investor_type == "enterprising":
        df = df[selector.meets_score_mask(df, 2)]
        mask = df["trajectory_pass"] == True
        if mask.sum() >= 5:
            df = df[mask]
    else:
        df = df[selector.meets_score_mask(df, 2)]
        mask = (df["greenblatt_pass"] == True) | (df["dorsey_pass"] == True)
        if mask.sum() >= 5:
            df = df[mask]

    if time_horizon == "short":
        high_score = df[selector.meets_score_mask(df, 3)]
        if len(high_score) >= 5:
            df = high_score

    # Exclude stocks already in portfolio
    df = df[~df["ticker"].isin(exclude_tickers)]

    # Exclude sectors at the 2-stock cap
    from collections import Counter
    sector_counts = Counter(current_sectors)
    full_sectors = [s for s, c in sector_counts.items() if c >= 2]
    if full_sectors:
        df = df[~df["sector"].isin(full_sectors)]

    # Sort
    df = df.copy()
    df["_sort_score"] = -df["score"]
    df["_sort_pe"] = df["pe"].apply(lambda x: x if pd.notna(x) else 9999)
    df["_sort_roe"] = df["roe_pct"].apply(lambda x: -x if pd.notna(x) else 9999)
    df = df.sort_values(["_sort_score", "_sort_pe", "_sort_roe"])

    candidates = []
    for _, row in df.head(5).iterrows():
        candidates.append({
            "ticker": row["ticker"],
            "name": row.get("name", "N/A") if pd.notna(row.get("name")) else "N/A",
            "sector": row.get("sector", "N/A") if pd.notna(row.get("sector")) else "N/A",
            "price": round(row["price"], 2) if pd.notna(row.get("price")) else 0,
            "score": int(row["score"]),
            "pe": round(row["pe"], 2) if pd.notna(row.get("pe")) else "N/A",
            "roe_pct": round(row["roe_pct"], 2) if pd.notna(row.get("roe_pct")) else "N/A",
        })
    return candidates

def get_nifty_return(days):
    """Get Nifty 50 return over a given number of days."""
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period=f"{max(days + 10, 30)}d")
        if len(hist) < 2:
            return None
        end_price = float(hist["Close"].iloc[-1])
        start_idx = max(0, len(hist) - days)
        start_price = float(hist["Close"].iloc[start_idx])
        return round(((end_price - start_price) / start_price) * 100, 2)
    except Exception:
        return None

_DRIFT_FRAMEWORK_LABEL = {
    "graham": "Graham",
    "greenblatt": "Greenblatt",
    "dorsey_buffett": "Dorsey/Buffett",
    "trajectory": "Trajectory",
    "lynch": "Lynch",
}

# Display strings for selector's closed label vocabulary. DESCRIPTIVE ONLY:
# they say what moved, never what to do about it. Any key selector emits that
# is missing here renders as a bare framework name rather than crashing or
# inventing a story — a new label must be added deliberately, not guessed.
_DRIFT_REASON_TEXT = {
    "valuation":      "price moved, the business held",
    "fundamental":    "the business moved",
    "relative_rank":  "its own numbers held — other stocks moved past it",
    "mixed":          "price and business both moved",
    "unclear":        "cause not identifiable from what we track",
    "unknown_inputs": "too little recorded at entry to say why",
}

# Why this alert is the colour it is. Shown only where the answer is not
# obvious from the flip line above it: an amber score-drop needs to say what
# was established, and a red one that we could NOT establish anything needs to
# say that too — otherwise "we don't know" and "the business broke" look
# identical to the reader. fundamental/mixed get nothing; the flip line
# already says it.
_DRIFT_ALERT_NOTE = {
    "valuation": "The business metrics held; the price moved. That is why this "
                 "is amber rather than red.",
    "relative_rank": "This stock's own numbers held — other stocks moved past "
                     "it. That is why this is amber rather than red.",
    "unclear": "We could not establish a cause from what we track, so this "
               "stays at full severity.",
    "unknown_inputs": "Too little was recorded at entry to establish a cause, "
                      "so this stays at full severity.",
}


def _drift_flip_line(verb, names, reasons):
    """'No longer passes: Graham, Lynch (price moved, the business held).'

    Groups by reason so a shared cause is stated once instead of repeated per
    framework. Insertion order is deterministic because `names` arrives sorted."""
    reasons = reasons or {}
    by_reason = {}
    for n in names:
        by_reason.setdefault(reasons.get(n), []).append(
            _DRIFT_FRAMEWORK_LABEL.get(n, n))
    parts = []
    for reason, fws in by_reason.items():
        txt = _DRIFT_REASON_TEXT.get(reason)
        parts.append(", ".join(fws) + (f" ({txt})" if txt else ""))
    return f"{verb}: {'; '.join(parts)}."


def _format_thesis_drift(diff):
    """Render a diff_thesis result as a deterministic drift line — the diff,
    NOT a fresh explanation. Returns (badge, markdown) or None to skip.

    DISPLAY ONLY. The badge deliberately ignores the W1 reason labels: a
    fundamental break is graver than a valuation one, but escalating severity
    is a DECISION, and decisions belong to portfolio_tracker/watchlist_reasons.
    Two places judging severity would eventually judge differently."""
    if not diff:
        return None
    d = diff.get("drift"); entry = diff.get("entry") or {}; curr = diff.get("current") or {}
    changes = diff.get("changes") or []

    def _sr(f):
        r, depth, sec = f.get("rank_in_sector"), f.get("sector_depth"), f.get("sector") or "its sector"
        return f"#{r} of {depth} in {sec}" if r and depth else None

    if d == "no_trace":
        return ("neutral", "_No entry thesis on record — bought before drift tracking or added manually._")
    if d == "no_longer_investable":
        er = _sr(entry)
        return ("broken", "**Thesis broken.** Fallen out of the investable pool (turnover/quality floor) — "
                          "would **not** be bought today." + (f" Entered as {er}." if er else ""))
    if d == "outranked":
        er = _sr(entry)
        return ("weak", "**Outranked.** Still investable, but other names now rank above it — "
                        "not in today's portfolio." + (f" Entered as {er}." if er else ""))

    # Did anything actually deteriorate? Consulted by the still_selected line
    # below, which otherwise asserts "intact" on top of a list of failures.
    _lost = any(ch["field"] == "newly_failing" and ch.get("to") for ch in changes)
    _fell = any(ch["field"] == "continuous_drift"
                and (ch.get("detail") or {}).get("delta", 0) < 0
                for ch in changes)

    lines = []; cr = _sr(curr)
    if d == "now_merit":
        lines.append("**Thesis strengthened.** Entered on conviction (merit had left it behind); "
                     "today it clears the gate on merit.")
    elif d == "now_conviction":
        lines.append("**Thesis weakened.** Entered on merit; today it survives only via the conviction sleeve.")
    else:
        # "Intact" is a CLAIM, and it is false the moment a framework has
        # dropped or the continuous score has fallen. `drift` tracks only the
        # selection SLOT (merit vs conviction), so still_selected means "same
        # seat", never "nothing changed". Reassuring the user immediately above
        # the evidence contradicting the reassurance is worse than saying less.
        if _lost or _fell:
            lines.append("**Same selection basis**, but the thesis has moved "
                         "since entry.")
        else:
            lines.append("**Thesis intact.** Same basis as at entry.")
    if cr:
        lines.append(f"Now {cr}.")
    for ch in changes:
        f = ch["field"]
        if f == "rank_in_sector":
            (er, ed), (crk, cd) = ch["from"], ch["to"]
            if er and crk and er != crk:
                lines.append(f"Sector rank {er}→{crk} (of {cd}).")
        elif f == "score_applicable":
            lines.append(f"Passes {ch['from']}→{ch['to']} of its applicable frameworks.")
        elif f == "newly_passing" and ch["to"]:
            lines.append(_drift_flip_line("Now also passes", ch["to"],
                                          ch.get("reasons")))
        elif f == "newly_failing" and ch["to"]:
            lines.append(_drift_flip_line("No longer passes", ch["to"],
                                          ch.get("reasons")))
        elif f == "continuous_drift":
            det = ch.get("detail") or {}
            delta = det.get("delta")
            if delta is not None:
                _frm = det.get("from", ch.get("from"))
                _to = det.get("to", ch.get("to"))
                s = (f"Continuous score {_frm}→{_to} "
                     f"({'up' if delta > 0 else 'down'} {abs(delta):.2f}).")
                lg = det.get("largest_move")
                if lg and lg in (det.get("by_framework") or {}):
                    s += (f" Largest measured move: "
                          f"{_DRIFT_FRAMEWORK_LABEL.get(lg, lg)} "
                          f"({det['by_framework'][lg]:+.2f}).")
                # The caveat travels WITH the number, never in a footnote. An
                # unattributed chunk means a framework became scoreable or
                # stopped being scoreable — arithmetic, not the business. Left
                # unsaid, that reads as a thesis crack that never happened.
                unatt = det.get("unattributed") or 0.0
                if abs(unatt) >= 0.05:
                    _who = ", ".join(_DRIFT_FRAMEWORK_LABEL.get(u, u)
                                     for u in (det.get("unmeasured") or []))
                    s += (f" {unatt:+.2f} of that is **not** attributable to any "
                          f"measured framework — {_who or 'a framework'} could not "
                          f"be scored on one side, so that portion is arithmetic, "
                          f"not a change in the business.")
                lines.append(s)
        elif f == "conviction_rank":
            lines.append(f"Conviction rank {ch['from']}→{ch['to']}.")
    badge = {"now_merit": "strong", "now_conviction": "weak",
             "still_selected": "neutral" if (_lost or _fell) else "intact"}.get(d, "neutral")
    return (badge, " ".join(lines))


def build_review_context(holdings, port):
    """Gather enriched data per holding: market context, earnings quality, ROE trend, book passage."""
    today = datetime.date.today()
    try:
        created = datetime.date.fromisoformat(str(port["created_at"])[:10])
        holding_days = (today - created).days
    except Exception:
        holding_days = 30

    nifty_return = get_nifty_return(holding_days)

    enriched = []
    for h in holdings:
        ticker = h["ticker"]
        entry_price = h.get("price_at_entry") or 0
        entry_score = h.get("score_at_entry") or 0
        shares = h.get("shares") or 0

        try:
            cinfo = yf.Ticker(ticker).info
            now_price = cinfo.get("currentPrice") or cinfo.get("regularMarketPrice") or 0
        except Exception:
            now_price = 0

        urow = universe_df[universe_df["ticker"] == ticker]
        now_score = int(urow["score"].iloc[0]) if len(urow) and pd.notna(urow["score"].iloc[0]) else 0

        roe_trend = []
        for y in ["roe_y0", "roe_y1", "roe_y2", "roe_y3"]:
            if len(urow) and y in urow.columns and pd.notna(urow[y].iloc[0]):
                roe_trend.append(round(float(urow[y].iloc[0]), 2))

        quality = get_earnings_quality_metrics(ticker)
        if "error" not in quality:
            quality_flags = quality.get("anomaly_flags", ["Unable to check"])
            cash_conversion = quality.get("cash_conversion_ratio", "N/A")
        else:
            quality_flags = ["Unable to check"]
            cash_conversion = "N/A"

        stock_return = ((now_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        pnl = (now_price - entry_price) * shares if entry_price > 0 else 0
        score_change = now_score - entry_score
        market_relative = round(stock_return - nifty_return, 2) if nifty_return is not None else None
        roe_declining = len(roe_trend) >= 3 and roe_trend[0] < roe_trend[-1]
        live_red = any("RED FLAG" in f for f in quality_flags) if isinstance(quality_flags, list) else False
        
        # CSV quality_pass is deterministic — computed monthly, doesn't shift between calls
        csv_red = False
        if len(urow) and "quality_pass" in urow.columns:
            qp_val = urow["quality_pass"].iloc[0]
            if pd.notna(qp_val) and qp_val == False:
                csv_red = True
                if not live_red:
                    quality_flags.append(
                        "RED FLAG: Monthly pre-screen flagged quality failure "
                        "(live check returned clean — yfinance data inconsistency)."
                    )
        
        has_red_flags = live_red or csv_red

        # Pattern-specific book query
        if has_red_flags:
            book_query = "Graham warnings about earnings quality non-recurring income value traps"
        elif score_change <= -2 and market_relative is not None and market_relative > -5:
            book_query = "Dorsey signs of eroding economic moat competitive advantage deterioration"
        elif stock_return < -10 and nifty_return is not None and nifty_return < -5:
            book_query = "Graham holding through market declines Mr Market temporary price drops"
        elif roe_declining:
            book_query = "Dorsey declining return on equity moat erosion when to sell"
        elif score_change >= 1:
            book_query = "Graham margin of safety increases buying more undervalued stocks"
        elif score_change == 0 and stock_return > 20:
            book_query = "Greenblatt when to take profits selling appreciated stocks"
        else:
            book_query = "Graham intelligent investor patience holding quality companies"

        book_result = search_book(book_query)
        book_passage = ""
        if "error" not in book_result:
            passages = book_result["passages"].split("\n\n")
            book_passage = passages[0][:500] if passages else ""

        enriched.append({
            "ticker": ticker, "name": h.get("name") or ticker, "sector": h.get("sector", ""),
            "shares": shares, "entry_price": entry_price, "now_price": now_price,
            "entry_score": entry_score, "now_score": now_score, "score_change": score_change,
            "stock_return": round(stock_return, 2), "pnl": round(pnl, 0),
            "nifty_return": nifty_return, "market_relative": market_relative,
            "roe_trend": roe_trend, "roe_declining": roe_declining,
            "quality_flags": quality_flags, "cash_conversion": cash_conversion,
            "has_red_flags": has_red_flags, "book_query": book_query,
            "book_passage": book_passage, "holding_days": holding_days,
            "holding_id": h.get("id"),
        })

    return enriched


def generate_review_recommendations(enriched_holdings, investor_type, time_horizon, portfolio):
    """LLM-powered review recommendations grounded in book philosophy."""
    holdings_text = ""
    for i, h in enumerate(enriched_holdings):
        holdings_text += (
            f"\nStock {i+1}: {h['name']} ({h['ticker']})\n"
            f"- Shares: {h['shares']}, Entry: INR {h['entry_price']:.2f}, Now: INR {h['now_price']:.2f}\n"
            f"- Stock return: {h['stock_return']:+.1f}%, Nifty return: {h['nifty_return']}%, Market-relative: {h['market_relative']}%\n"
            f"- Score: {h['entry_score']} to {h['now_score']} (change: {h['score_change']:+d})\n"
            f"- ROE trend (recent to oldest): {h['roe_trend']}\n"
            f"- Earnings quality: {', '.join(h['quality_flags']) if isinstance(h['quality_flags'], list) else h['quality_flags']}\n"
            f"- Cash conversion ratio: {h['cash_conversion']}\n"
            f"- Held for: {h['holding_days']} days\n"
            f"- Relevant book passage: {h['book_passage']}\n"
        )

    # ── User Decision Context ──
    user_context = ""
    _profile = portfolio.get("portfolio_profile") or {}
    if isinstance(_profile, str):
        try: _profile = json.loads(_profile)
        except: _profile = {}
    if _profile.get("decision_context"):
        user_context = f"\nUSER DECISION CONTEXT:\n{_profile.get('decision_context')}\n(CRITICAL: Honor these preferences. Do not recommend selling a stock solely for a trait the user explicitly accepted, such as sector volatility.)\n"

    # ── Macro shift context (review-diff) — enters as a CONSTRAINED, asymmetric input ──
    macro_context = ""
    _mdiff = _profile.get("_pending_macro_diff") or {}
    if _mdiff:
        _mtext = _render_macro_diff(_mdiff)
        if _mtext and "No material macro shift" not in _mtext:
            macro_context = (
                f"\nMACRO SHIFT SINCE LAST REVIEW:\n{_mtext}\n"
                f"(RULES for using this — asymmetric, buy-only philosophy:\n"
                f" - Inflation projection UP raises the real-return hurdle: it may DEMOTE a "
                f"marginal BUY MORE to HOLD. It must NEVER promote anything toward SELL — you "
                f"do not sell a quality compounder because forward inflation rose; you starve it "
                f"of new money instead.\n"
                f" - LTCG/STCG UP raises the cost of realizing gains: it should make you MORE "
                f"reluctant on DISCRETIONARY sells (moat-erosion SELL HALF). It does NOT change "
                f"conviction exits — a red flag or score=0 is still SELL ALL regardless of tax.\n"
                f" - Sector deterioration is CONTEXT to weigh, not a sell trigger. Per Marks, "
                f"sector pessimism is often already priced in; selling into it locks the loss.\n"
                f" The macro shift may cool BUY MORE and cool discretionary SELLs. It must NEVER "
                f"CREATE sell pressure that a broken thesis did not already justify.)\n"
            )

    # Sprint 12: universe statistics as CALIBRATION context. The LLM uses these
    # to set confidence LANGUAGE — it must NOT quote raw percentages at the user
    # (two-consumer rule). The load-bearing fact is the correlation cluster:
    # Dorsey/Trajectory/Lynch travel together (phi ~0.4), so a 3/5 passing those
    # three is a WEAKER signal than a 3/5 spread across independent frameworks.
    stats_context = ""
    try:
        import stats as _kstats
        _udf = globals().get("universe_df")
        if _udf is not None and len(_udf):
            _us = _kstats.compute_universe_stats(_udf)
            _br = _us.get("base_rates", {})
            _cl = _us.get("least_orthogonal_pair", {})
            if _br:
                _rare = _us.get("base_rate_spread", {}).get("rarest", "")
                _rare_lbl = _br.get(_rare, {}).get("label", "Graham")
                _rate_lines = ", ".join(
                    f"{v['label']} {v['pass_rate']*100:.0f}%" for v in _br.values())
                stats_context = (
                    "\nUNIVERSE CALIBRATION (context for your CONFIDENCE LANGUAGE only "
                    "— do NOT quote these numbers to the user):\n"
                    f" - Flag pass-rates across the whole market: {_rate_lines}.\n"
                    f" - {_rare_lbl} is the RAREST and hardest test; a stock passing it "
                    "deserves genuinely higher confidence in your wording.\n")
                if _cl and _cl.get("labels"):
                    _a, _b = _cl["labels"]
                    stats_context += (
                        f" - {_a} and {_b} are CORRELATED (they tend to pass together). "
                        f"When a stock passes both, treat them as ~1.5 signals, not 2 — "
                        f"be MORE measured in your confidence, not less. A score built on "
                        f"independent frameworks (e.g. {_rare_lbl} + one other) is stronger "
                        f"than the same score built on the correlated cluster.\n")
    except Exception as _se:
        print(f"Stats calibration skipped (non-blocking): {type(_se).__name__}: {_se}")

    review_prompt = (
        f"You are the Kordent Investment Committee reviewing a {investor_type} investor's "
        f"portfolio with a {time_horizon}-term horizon.\n\n"
        f"For each stock below, provide a recommendation.\n\n"
        f"DECISION FRAMEWORK (apply in order):\n"
        f"1. RED FLAGS OVERRIDE: If earnings quality has RED FLAGS, recommend SELL ALL. Cite Graham on value traps.\n"
        f"2. MOAT EROSION: If ROE declined for 3+ years AND stock underperformed market, recommend SELL HALF. Cite Dorsey.\n"
        f"3. MARKET EFFECT: If stock dropped BUT Nifty also dropped similarly (within 5%), recommend HOLD. "
        f"Cite Graham on Mr. Market. The business hasn't changed.\n"
        f"4. NO THESIS: If current score = 0 (no framework passes), recommend SELL ALL regardless of "
        f"other signals. A score of 0 means no investment thesis exists.\n"
        f"5. THESIS INTACT: If score >= 1 AND score stable or improved AND no red flags AND cash conversion > 0.5, "
        f"recommend HOLD or BUY MORE. Cite the relevant framework.\n"
        f"6. OVERVALUATION: If stock gained >30% and score dropped, recommend HOLD but note reduced margin of safety.\n"
        f"7. INVESTOR PROFILE: "
        f"{'Be conservative. Prefer HOLD over BUY MORE, SELL sooner on red flags.' if investor_type == 'defensive' else 'Balance risk and reward.' if investor_type == 'balanced' else 'Tolerate volatility. HOLD through short-term drops if moat is intact.'}\n\n"
        f"{user_context}"
        f"{macro_context}"
        f"{stats_context}"
        f"{holdings_text}\n\n"
        f"Respond ONLY with a JSON array (no markdown, no backticks, no preamble). Each element:\n"
        f'{{"ticker": "TICKER.NS", "action": "HOLD", "sell_qty": 0, "reasoning": "2-3 sentences grounded in Graham/Greenblatt/Dorsey.", "confidence": "high"}}\n'
        f"action must be one of: SELL ALL, SELL HALF, HOLD, BUY MORE\n"
        f"sell_qty: number of shares to sell (0 for HOLD/BUY MORE, all shares for SELL ALL, half for SELL HALF)\n"
    )

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    for model_name in FREE_MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=review_prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            if text.startswith("json"):
                text = text[4:].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            continue
        except Exception as e:
            error_msg = str(e).upper()
            if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                continue
            break
    return None

def validate_portfolio_ips(stocks: list, ips_policy: dict) -> dict:
    """Deterministic IPS guardrail check. Runs on every portfolio mutation.
    Returns pass/fail with specific violations. Book is the standard."""
    if not ips_policy:
        return {"valid": True, "violations": [], "warnings": []}

    alloc = ips_policy.get("allocation_policy", {})
    sizing = ips_policy.get("portfolio_sizing", {})
    violations = []
    warnings = []

    # Every constraint below is "<= max" or ">= min", not strict. A portfolio
    # sitting EXACTLY at its cap is compliant — and the deterministic selector
    # puts it there on purpose: per_sector = floor(max_sector_pct * n / 100), so
    # a full sector is exactly at the limit by construction.
    #
    # Without tolerance, n=12 fails: allocation_pct rounds to 8.3, the validator
    # re-normalizes 8.3 / (12*8.3) * 100 = 8.333..., three of them sum to
    # 25.000000000000004, and 25.000000000000004 > 25.0. The error message then
    # prints "25.0% exceeds 25.0% max", which is not a sentence any user should
    # ever see.
    #
    # generate_ips already hacked around exactly this once, with
    # `max_single_stock_pct = min(10.0, ...) + 0.5`. One constraint got a fudge
    # factor and three didn't. Handle it here, once, for all of them.
    _EPS = 1e-6
    def _over(value, limit):
        return value - limit > _EPS

    def _under(value, limit):
        return limit - value > _EPS

    if not stocks:
        return {"valid": False, "violations": ["No stocks in portfolio"], "warnings": []}

    # Normalize allocation_pct
    total_alloc = sum(s.get("allocation_pct", 0) for s in stocks)
    if total_alloc <= 0:
        total_alloc = 100.0  # assume equal weight if not set

    # ── 1. Single stock concentration ──
    max_stock_pct = alloc.get("max_single_stock_pct", 10.0)
    for s in stocks:
        pct = (s.get("allocation_pct", 0) / total_alloc) * 100 if total_alloc else 0
        if _over(pct, max_stock_pct):
            violations.append(
                f"{s.get('name', s.get('ticker', '?'))} is {pct:.1f}% — exceeds {max_stock_pct}% max per stock (SEBI/Reilly & Brown)")

    # ── 2. Sector concentration ──
    max_sector_pct = alloc.get("max_sector_pct", 25.0)
    max_same_sector = alloc.get("max_same_sector", 2)
    min_sectors = alloc.get("min_sectors", 3)

    sector_weights = {}
    sector_counts = {}
    for s in stocks:
        sec = s.get("sector", "Unknown")
        pct = (s.get("allocation_pct", 0) / total_alloc) * 100 if total_alloc else 0
        sector_weights[sec] = sector_weights.get(sec, 0) + pct
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    for sec, weight in sector_weights.items():
        if _over(weight, max_sector_pct):
            violations.append(f"Sector '{sec}' is {weight:.1f}% — exceeds {max_sector_pct}% max")

    for sec, count in sector_counts.items():
        if count > max_same_sector:
            violations.append(f"Sector '{sec}' has {count} stocks — exceeds {max_same_sector} max same-sector")

    distinct_sectors = len([s for s in sector_counts if s != "Unknown"])
    if distinct_sectors < min_sectors:
        violations.append(f"Only {distinct_sectors} sectors — need at least {min_sectors} (Reilly & Brown Ch 6)")

    # ── 3. Cap distribution (needs risk_tier from universe CSV) ──
    large_min = alloc.get("large_cap_min_pct", 0)
    small_max = alloc.get("small_cap_max_pct", 100)
    try:
        large_weight = 0
        small_weight = 0
        for s in stocks:
            ticker = s.get("ticker", "")
            pct = (s.get("allocation_pct", 0) / total_alloc) * 100 if total_alloc else 0
            row = universe_df[universe_df["ticker"] == ticker]
            if not row.empty:
                tier = row.iloc[0].get("risk_tier", "Unknown")
                if tier == "Large":
                    large_weight += pct
                elif tier == "Small":
                    small_weight += pct

        if large_min > 0 and _under(large_weight, large_min):
            violations.append(f"Large-cap is {large_weight:.1f}% — below {large_min}% minimum")
        if small_weight > small_max:
            violations.append(f"Small-cap is {small_weight:.1f}% — exceeds {small_max}% maximum")
    except Exception:
        pass  # universe_df not available in all contexts

    # ── 4. Holdings count ──
    # 4a. IPS target (hard — user paid for this many stocks)
    # DEMOTED from violation to warning. With the pool widened to 200, a
    # shortfall is no longer a truncation artifact — it means the universe
    # genuinely cannot supply `ips_target` sector-legal, tier-1 names. Per
    # Reilly & Brown the count is a PROXY for "diversified enough"; when the
    # sector cap binds before the count, the constraint-limited maximum IS the
    # book-compliant answer. Blocking would force padding-with-garbage (raises
    # unsystematic risk) or breaching a sector cap (breaks diversification) —
    # both violate the book to satisfy a proxy. So a genuinely-scarce short
    # count SAVES, but is DISCLOSED via the warning below.
    ips_target = sizing.get("actual", sizing.get("ips_target", 0))
    if ips_target and len(stocks) < ips_target:
        warnings.append(
            f"Portfolio has {len(stocks)} stocks vs IPS target of {ips_target}. "
            f"Your universe cannot supply {ips_target} sector-legal, quality-passing "
            f"names under these constraints — {len(stocks)} is the diversified maximum. "
            f"Adding more would require breaching a sector cap or admitting "
            f"illiquid/low-quality stocks, both of which raise unsystematic risk.")
    # 4b. Book minimum (warning — honest about under-diversification)
    book_min = sizing.get("book_minimum", 12)
    if len(stocks) < book_min:
        warnings.append(
            f"Portfolio has {len(stocks)} stocks — below book minimum of {book_min}. "
            f"Carrying meaningful unsystematic risk (Reilly & Brown Ch 6).")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }


# ── Demand tilt: builder answers -> per-axis lean (Q/G/P/S) ──
# Per-axis amplitude differs by NEGOTIABILITY. Safety is a floor the user may raise
# but never lower (the quality gate stays absolute regardless of tilt). Quality is a
# soft floor — deep-value may de-emphasise it, never zero it. Price and Growth are dials.
AXIS_TILT_RANGE = {
    "safety":  (1.0, 1.6),   # up only
    "quality": (0.5, 1.5),   # soft floor
    "price":   (0.4, 1.8),   # wide both ways
    "growth":  (0.0, 1.8),   # free
}


def derive_demand_tilt(profile: dict) -> dict:
    """Builder answers -> multiplicative per-axis tilt, clamped then normalised to sum 4.0.

    A TILT, not a filter: it re-ranks candidates that ALREADY passed the absolute layer
    (quality gate + framework scores). It can never admit a stock the gate rejected.
    """
    t = {"quality": 1.0, "growth": 1.0, "price": 1.0, "safety": 1.0}

    ph = profile.get("philosophy")                      # Q7 identity -> primary axis
    if ph == "growth_at_fair_price":  t["growth"] *= 1.6; t["price"] *= 0.8
    elif ph == "deep_value":          t["price"] *= 1.6;  t["growth"] *= 0.8
    elif ph == "quality_compounder":  t["quality"] *= 1.4; t["price"] *= 0.8

    risk = profile.get("risk")                          # Q4 behavioural stress test
    if risk == "moderate":            t["safety"] *= 1.15
    elif risk == "conservative":      t["safety"] *= 1.4; t["growth"] *= 0.7

    if profile.get("preference") == "income":           # Q6 income need
        t["growth"] *= 0.6; t["safety"] *= 1.15
    else:
        t["growth"] *= 1.2

    tr = profile.get("acceptable_tradeoff")             # Q9 concession (dials only)
    if tr == "ok_fail_graham":             t["price"] *= 0.7
    elif tr == "ok_fail_trajectory_lynch": t["growth"] *= 0.7
    elif tr == "ok_fail_dorsey_buffett":   t["quality"] *= 0.7

    for k, (lo, hi) in AXIS_TILT_RANGE.items():
        t[k] = max(lo, min(hi, t[k]))
    s = sum(t.values()) or 1.0
    return {k: round(v * 4.0 / s, 3) for k, v in t.items()}


def detect_demand_contradiction(profile):
    """Aspirational vs behavioural mismatch. Surface it — never silently average."""
    ph, risk = profile.get("philosophy"), profile.get("risk")
    if ph == "growth_at_fair_price" and risk == "conservative":
        return ("You picked growth-focused, but you'd sell in a 20% drop. We've leaned "
                "your portfolio toward steadier growers — you can lean further either way.")
    if ph == "deep_value" and risk == "conservative":
        return ("Bargain-hunting usually means holding through volatility, but you'd sell "
                "in a 20% drop. We've kept the safety bar high in your picks.")
    return None


def generate_ips(profile: dict, age: int = 30) -> dict:
    """Derive a formal Investment Policy Statement from builder profile inputs.
    Based on Reilly & Brown — Investment Analysis & Portfolio Management, Ch 2 & 6.
    The book is the definitive standard. The system adapts to the book, not the reverse.
    """
    investor_type = profile.get("investor_type", "balanced")
    time_horizon = profile.get("time_horizon", "medium")
    preference = profile.get("preference", "growth")
    sip_amount = profile.get("sip_amount", 5000)
    philosophy = profile.get("philosophy", "growth_at_fair_price")

    # ── Life cycle phase from age (Ch 2, Exhibit 2.1) ──
    if age < 35:
        life_cycle = "accumulation"
    elif age < 55:
        life_cycle = "consolidation"
    elif age < 70:
        life_cycle = "spending"
    else:
        life_cycle = "gifting"

    # ── Return objective (Ch 2, Section 2.4.1) ──
    if preference == "income" or investor_type == "defensive":
        if time_horizon == "short":
            return_objective = "capital_preservation"
        else:
            return_objective = "current_income"
    elif investor_type == "enterprising":
        return_objective = "capital_appreciation"
    else:
        return_objective = "total_return"

    # ── Benchmark ──
    if investor_type == "defensive":
        benchmark = "NIFTY50"
    elif investor_type == "enterprising":
        benchmark = "NIFTY500"
    else:
        benchmark = "NIFTY200"

    # ══════════════════════════════════════════════════════════
    # PORTFOLIO SIZING — from the book, not from our system
    # Evans & Archer (1968): 12-18 stocks for ~90% benefit
    # Statman (1987): 30-40 optimal with costs
    # Campbell et al. (2001): ~50 for full diversification
    # ══════════════════════════════════════════════════════════
    BOOK_MIN_STOCKS = 12
    BOOK_RECOMMENDED_STOCKS = 20
    BOOK_OPTIMAL_STOCKS = 40

    # What the SIP can actually support (affordability)
    # Progressive diversification buys cheapest first (1 share each),
    # so the floor is ~₹250/stock (median cheap large-cap), not ₹500.
    affordable_stocks = max(3, sip_amount // 250)

    # IPS target follows the book
    if investor_type == "defensive":
        ips_target_stocks = max(BOOK_MIN_STOCKS, 15)
    elif investor_type == "enterprising":
        ips_target_stocks = max(BOOK_MIN_STOCKS, 12)
    else:
        ips_target_stocks = max(BOOK_MIN_STOCKS, 15)

    # Diversification gap — honest assessment
    if affordable_stocks >= ips_target_stocks:
        diversification_status = "adequate"
        sip_needed_for_target = sip_amount
    elif affordable_stocks >= BOOK_MIN_STOCKS:
        diversification_status = "acceptable"
        sip_needed_for_target = ips_target_stocks * 500
    else:
        diversification_status = "under_diversified"
        sip_needed_for_target = BOOK_MIN_STOCKS * 500

    actual_stocks = min(affordable_stocks, ips_target_stocks)

    # ══════════════════════════════════════════════════════════
    # CONCENTRATION LIMITS — book + SEBI norms
    # Book Ch 2: mutual funds limited to 5% per stock
    # SEBI: 10% max in single stock for mutual funds
    # Book Ch 6: same-sector = high correlation = poor diversification
    # ══════════════════════════════════════════════════════════
    # Per-stock cap must not be derived from a ROUNDED equal weight — the
    # validator computes actual % from normalized weights, and round(100/n,1)
    # re-normalizes back ABOVE the rounded cap (e.g. 100/12→8.3, but 8.3/99.6
    # ×100 = 8.33 > 8.3 → false violation on every stock). Use the exact
    # equal weight plus a small tolerance so an equal-weighted portfolio can
    # never violate its own per-stock ceiling by float rounding alone.
    _equal_weight = 100.0 / max(actual_stocks, 5)
    max_single_stock_pct = round(min(10.0, max(_equal_weight, 5.0)) + 0.5, 1)
    max_sector_pct = 25.0

    if actual_stocks <= 10:
        max_same_sector = 2
    elif actual_stocks <= 20:
        max_same_sector = 3
    else:
        max_same_sector = 4

    min_sectors = max(3, actual_stocks // 3)

    # Cap distribution — book Ch 7 Fama-French size factor
    cap_policy = {
        "defensive":    {"large_cap_min_pct": 50, "mid_cap_min_pct": 20, "small_cap_max_pct": 10},
        "balanced":     {"large_cap_min_pct": 30, "mid_cap_min_pct": 20, "small_cap_max_pct": 25},
        "enterprising": {"large_cap_min_pct": 20, "mid_cap_min_pct": 20, "small_cap_max_pct": 35},
    }.get(investor_type, {"large_cap_min_pct": 30, "mid_cap_min_pct": 20, "small_cap_max_pct": 25})

    return {
        "return_objective": return_objective,
        "risk_tolerance": investor_type,
        "life_cycle_phase": life_cycle,
        "age": age,
        "time_horizon": time_horizon,
        "benchmark": benchmark,
        "constraints": {
            "liquidity": "monthly_sip",
            "tax_awareness": "dynamic",
            "sector_exclusions": profile.get("avoid_sectors", []),
            "unique_preferences": profile.get("acceptable_tradeoff", "any"),
        },
        "portfolio_sizing": {
            "book_minimum": BOOK_MIN_STOCKS,
            "book_recommended": BOOK_RECOMMENDED_STOCKS,
            "book_optimal": BOOK_OPTIMAL_STOCKS,
            "ips_target": ips_target_stocks,
            "affordable": affordable_stocks,
            "actual": actual_stocks,
            "diversification_status": diversification_status,
            "sip_for_book_minimum": BOOK_MIN_STOCKS * 500,
            "sip_for_ips_target": sip_needed_for_target,
        },
        "allocation_policy": {
            "max_single_stock_pct": max_single_stock_pct,
            "max_sector_pct": max_sector_pct,
            "max_same_sector": max_same_sector,
            "min_sectors": min_sectors,
            **cap_policy,
        },
        "framework_weights": profile.get("framework_weights", {}),
        "philosophy": philosophy,
    }


def register_portfolio(portfolio_name: str, investor_type: str, sip_amount: int, time_horizon: str, review_days: int = 90, stocks_json: str = "[]", portfolio_profile: str = "{}", target_amount: float = 0, target_date: str = "", decision_context: str = "") -> dict:
    """Register a finalized SIP portfolio so the user can save it to their account.
    Call this ONLY after you have presented the final portfolio table with all stocks and allocations.

    Args:
        portfolio_name: Short descriptive name, e.g. 'Conservative Growth SIP - June 2026'
        investor_type: The investor profile - defensive, balanced, or enterprising
        sip_amount: Monthly SIP amount in INR
        time_horizon: Investment time horizon from the builder profile
        review_days: Number of days between portfolio reviews from the builder profile.
        stocks_json: A JSON string representing a list of stock objects. Each object must have keys: ticker (str), name (str), sector (str), allocation_pct (number).
        portfolio_profile: JSON string of the full investor profile from the builder form. Pass through from the [BUILDER_PROFILE] message.
        target_amount: Savings goal in INR. 0 if no goal set.
        target_date: Goal deadline as ISO date string (YYYY-MM-DD). Empty string if no goal.
        decision_context: A brief summary of any specific preferences, trade-offs, or choices the user made during the Phase 1 clarification questions. Pass an empty string if no questions were asked.
    """
    try:
        stocks = json.loads(stocks_json) if isinstance(stocks_json, str) else stocks_json
    except json.JSONDecodeError:
        return {"error": f"Could not parse stocks_json: {stocks_json[:200]}"}

    if not stocks:
        return {"error": "No stocks provided."}

    _profile = st.session_state.get("builder_profile") or {}
    
    if decision_context:
        _profile["decision_context"] = decision_context
        
    final_target = _profile.get("target_amount") or (target_amount if target_amount > 0 else None)
    final_date = _profile.get("target_date") or (target_date if target_date else None)

    # ── IPS validation: PURE TRIPWIRE, no mutation ──
    # The deterministic selector builds compliance in by construction (sector
    # count cap → transitively guarantees min_sectors; large-cap floor enforced
    # incl. distress-backfill tier-matching; per-stock cap derived from exact
    # equal weight with tolerance). So on selector output this ALWAYS passes.
    # If it ever fails, that is a SELECTOR bug to surface loudly — NOT something
    # to silently auto-fix, which only masks the bug and drifts the two
    # definitions of "compliant" apart (the old auto-fix could not even repair
    # a broken large-cap floor — removing stocks can't raise large%). A single
    # source of truth (the selector) beats two disagreeing validators.
    _ips = _profile.get("ips_policy", {})
    if _ips and stocks:
        _validation = validate_portfolio_ips(stocks, _ips)
        if not _validation["valid"]:
            return {
                "error": "IPS validation failed on deterministic output — this is a "
                         "selector bug, not user error. Violations: "
                         + "; ".join(_validation["violations"]),
                "violations": _validation["violations"],
                "instruction": "Do not retry blindly — the portfolio selector produced "
                               "non-compliant output and must be fixed at the source.",
            }
        # Non-blocking warnings (e.g. genuine count scarcity) — stash for disclosure.
        if _validation.get("warnings"):
            st.session_state._ips_save_warnings = _validation["warnings"]

    # Weld the macro snapshot (built during get_sip_candidates) onto the profile
    # via bounded-append so a history accumulates for later review-diffing.
    _snap = st.session_state.get("_macro_snapshot")
    if _snap:
        _profile = _append_macro_snapshot(_profile, _snap, n=6)

    st.session_state.pending_portfolio = {
        "name": portfolio_name,
        "investor_type": investor_type,
        "sip_amount": sip_amount,
        "time_horizon": time_horizon,
        "review_days": int(review_days),
        "stocks": stocks,
        "portfolio_profile": _profile if _profile else None,
        "target_amount": final_target,
        "target_date": final_date,
        "is_paper": _profile.get("is_paper", False),
    }
    # Include IPS warnings (under-diversified etc) in status
    _ips = _profile.get("ips_policy", {})
    _warnings = []
    if _ips and stocks:
        _val = validate_portfolio_ips(stocks, _ips)
        _warnings = _val.get("warnings", [])
    _warn_str = (" ⚠️ " + " | ".join(_warnings)) if _warnings else ""
    return {"status": f"Portfolio '{portfolio_name}' registered with {len(stocks)} stocks. Review every {review_days} days.{_warn_str}"}


def _commit_portfolio(portfolio: dict) -> dict:
    """Single source of truth for committing a staged portfolio to the DB.
    Called by BOTH the chat save button and the deterministic build_result view.
    Fixes the row-before-assignment price bug; returns a structured result
    instead of doing UI, so callers render as they wish.

    Returns {ok, portfolio_id, invested, unallocated, allocated, stale_priced, error}.
    """
    try:
        sb = get_supabase()
        review_days = portfolio.get("review_days", 90)
        next_review = (datetime.date.today() + datetime.timedelta(days=review_days)).isoformat()
        next_sip = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()

        stocks_for_alloc = []
        _stale = []
        for stock in portfolio["stocks"]:
            ticker = stock["ticker"]
            # Universe row FIRST (fixes row-before-assignment bug)
            row = universe_df[universe_df["ticker"] == ticker]
            _uni_close = None
            if len(row) and "price" in row.columns and pd.notna(row["price"].iloc[0]):
                _uni_close = float(row["price"].iloc[0])

            price = 0
            _max_tries = 2 if _uni_close is None else 1
            for _attempt in range(_max_tries):
                try:
                    info = yf.Ticker(ticker).info
                    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                except Exception:
                    price = 0
                if price and price > 0:
                    break
                if _attempt < _max_tries - 1:
                    import time as _t
                    _t.sleep(1.5)

            if price <= 0 and _uni_close and _uni_close > 0:
                price = _uni_close
                _stale.append(ticker)

            pe = float(row["pe"].iloc[0]) if len(row) and pd.notna(row["pe"].iloc[0]) else None
            roe = float(row["roe_y0"].iloc[0]) if len(row) and "roe_y0" in row.columns and pd.notna(row["roe_y0"].iloc[0]) else None
            score = int(row["score"].iloc[0]) if len(row) and pd.notna(row["score"].iloc[0]) else None
            sector = stock.get("sector", "") or (str(row["sector"].iloc[0]) if len(row) and "sector" in row.columns and pd.notna(row["sector"].iloc[0]) else "")

            if price <= 0:
                # unpriceable (delisted/renamed) — skip rather than store a phantom
                continue

            stocks_for_alloc.append({
                "ticker": ticker, "name": stock.get("name", ""), "sector": sector,
                "allocation_pct": stock.get("allocation_pct", 0), "price": price,
                "pe": pe, "roe": roe, "score": score,
            })

        if not stocks_for_alloc:
            return {"ok": False, "error": "No stocks could be priced — portfolio not saved."}

        # Benchmark chosen ONCE from the IPS mandate and frozen here. Never
        # recomputed at review time (that is benchmark shopping), and the
        # transactions below buy units of THIS specific ETF at THIS price.
        _bench = selector.choose_benchmark(
            (portfolio.get("portfolio_profile") or {}).get("ips_policy"))

        port_resp = sb.table("portfolios").insert({
            "user_id": st.session_state.sb_user_id,
            "name": portfolio["name"],
            "investor_type": portfolio["investor_type"],
            "sip_amount": portfolio["sip_amount"],
            "time_horizon": portfolio["time_horizon"],
            "review_freq": str(review_days),
            "next_review_date": next_review,
            "next_sip_date": next_sip,
            "is_paper": portfolio.get("is_paper", False),
            "portfolio_profile": portfolio.get("portfolio_profile", {}),
            "benchmark_ticker": _bench["ticker"],
        }).execute()
        portfolio_id = port_resp.data[0]["id"]

        # 4. ALLOCATE AND INSERT CHILDREN
        allocated, unallocated = allocate_shares(stocks_for_alloc, portfolio["sip_amount"])
        
        # BULK INSERT HOLDINGS
        # Recover the deterministic entry thesis. select_portfolio stashed the full
        # holdings (each carrying _trace) at _last_candidates; join by ticker so the
        # recorded REASON a stock was bought is persisted verbatim and out of the
        # LLM's hands. A manually-added ticker absent from the last selection gets
        # null — diff_thesis treats that as "no_trace" and skips it, never crashes.
        _trace_by_ticker = {
            h.get("ticker"): h.get("_trace")
            for h in (st.session_state.get("_last_candidates") or [])
            if h.get("ticker")
        }
        holdings_data = []
        for s in allocated:
            holdings_data.append({
                "portfolio_id": portfolio_id, "ticker": s["ticker"], "name": s["name"],
                "sector": s["sector"], "allocation_pct": s["allocation_pct"], "shares": s["shares"],
                "sip_amount_inr": s["actual_amount"], "price_at_entry": s["price"],
                "pe_at_entry": s["pe"], "roe_at_entry": s["roe"], "score_at_entry": s["score"],
                "entry_trace": _sanitize_for_json(_trace_by_ticker.get(s["ticker"])),
            })
            
        if holdings_data:
            sb.table("holdings").insert(holdings_data).execute()

            # A stock that's now IN a portfolio shouldn't linger on the watchlist —
            # the user acted on it. Remove those tickers from this user's watchlist.
            try:
                _uid = portfolio.get("user_id") or st.session_state.get("sb_user_id")
                _new_tickers = [h["ticker"] for h in holdings_data if h.get("ticker")]
                if _uid and _new_tickers:
                    sb.table("watchlist").delete().eq("user_id", str(_uid)).in_(
                        "ticker", _new_tickers).execute()
            except Exception as _wle:
                print(f"Watchlist cleanup after commit failed (non-blocking): {_wle}")

        # BULK INSERT TRANSACTIONS
        # Shadow priced against the CHOSEN benchmark ETF, not always Nifty 50.
        # nifty_price/nifty_units are legacy column names; here they hold units
        # of _bench["ticker"], and benchmark_ticker (below) disambiguates them.
        nifty_px = None
        try:
            nifty_px = yf.Ticker(_bench["ticker"]).fast_info.last_price
        except Exception:
            pass
            
        txns_data = []
        today_iso = datetime.date.today().isoformat()
        
        for s in allocated:
            amt = s["actual_amount"]
            # allocate_shares returns EVERY candidate, with shares=0 for the ones
            # this cycle could not fund. A zero-share holdings row is intentional
            # (it carries allocation_pct to the next cycle), but a zero-rupee
            # "buy" is not a transaction and has no business in the ledger.
            if not (float(s.get("shares") or 0) > 0 and float(amt or 0) > 0):
                continue
            nifty_u = round(float(amt) / nifty_px, 6) if (nifty_px and nifty_px > 0) else None
            
            txns_data.append({
                "portfolio_id": str(portfolio_id),
                "user_id": str(st.session_state.sb_user_id),
                "ticker": s["ticker"],
                "shares": float(s["shares"]),
                "price": round(float(s["price"]), 2),
                "amount_inr": round(float(amt), 2),
                "transaction_type": "buy",
                "transaction_date": today_iso,
                "nifty_price": round(nifty_px, 2) if nifty_px else None,
                "nifty_units": nifty_u,
                "benchmark_ticker": _bench["ticker"],
            })
            
        if txns_data:
            try:
                sb.table("sip_transactions").insert(txns_data).execute()
            except Exception as e:
                print(f"Bulk txn log failed (non-blocking): {e}")

        st.session_state.pending_portfolio = None
        return {"ok": True, "portfolio_id": portfolio_id,
                "invested": portfolio["sip_amount"] - unallocated,
                "unallocated": unallocated, "allocated": allocated,
                "stale_priced": _stale}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ──────────────────────────────────────────────
# TICKER ALIAS MAP
# ──────────────────────────────────────────────
TICKER_ALIASES = {
    # ── Nifty 50 & common Indian abbreviations ──
    "RIL": "RELIANCE.NS",
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "TATA CONSULTANCY SERVICES": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "HDFC": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "STATE BANK": "SBIN.NS",
    "STATE BANK OF INDIA": "SBIN.NS",
    "WIPRO": "WIPRO.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "L&T": "LT.NS",
    "LARSEN": "LT.NS",
    "LARSEN AND TOUBRO": "LT.NS",
    "M&M": "M&M.NS",
    "MAHINDRA": "M&M.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "BAJAJ FINSERV": "BAJAJFINSV.NS",
    "KOTAK": "KOTAKBANK.NS",
    "KOTAK BANK": "KOTAKBANK.NS",
    "KOTAK MAHINDRA": "KOTAKBANK.NS",
    "MARUTI": "MARUTI.NS",
    "MARUTI SUZUKI": "MARUTI.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "TATA STEEL": "TATASTEEL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "BHARTI AIRTEL": "BHARTIARTL.NS",
    "HUL": "HINDUNILVR.NS",
    "HINDUSTAN UNILEVER": "HINDUNILVR.NS",
    "ASIAN PAINTS": "ASIANPAINT.NS",
    "SUN PHARMA": "SUNPHARMA.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "ADANI": "ADANIENT.NS",
    "ADANI ENTERPRISES": "ADANIENT.NS",
    "ADANI PORTS": "ADANIPORTS.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "NYKAA": "NYKAA.NS",
    "DMART": "DMART.NS",
    "AVENUE SUPERMARTS": "DMART.NS",
    "TITAN": "TITAN.NS",
    "NESTLE": "NESTLEIND.NS",
    "NESTLE INDIA": "NESTLEIND.NS",
    "POWER GRID": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "COAL INDIA": "COALINDIA.NS",
    "ONGC": "ONGC.NS",
    "AXIS": "AXISBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
    "TECH MAHINDRA": "TECHM.NS",
    "HCL": "HCLTECH.NS",
    "HCLTECH": "HCLTECH.NS",
    "HCL TECH": "HCLTECH.NS",
    "ULTRATECH": "ULTRACEMCO.NS",
    "ULTRATECH CEMENT": "ULTRACEMCO.NS",
    "BAJAJ AUTO": "BAJAJ-AUTO.NS",
    "HERO": "HEROMOTOCO.NS",
    "HERO MOTOCORP": "HEROMOTOCO.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "DR REDDY": "DRREDDY.NS",
    "DR REDDYS": "DRREDDY.NS",
    "EICHER": "EICHERMOT.NS",
    "EICHER MOTORS": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS",
    "HINDALCO": "HINDALCO.NS",
    "INDUSIND": "INDUSINDBK.NS",
    "INDUSIND BANK": "INDUSINDBK.NS",
    "JSW STEEL": "JSWSTEEL.NS",
    "TATA CONSUMER": "TATACONSUM.NS",
    "UPL": "UPL.NS",
    "DIVIS": "DIVISLAB.NS",
    "DIVIS LAB": "DIVISLAB.NS",
    "SHREE CEMENT": "SHREECEM.NS",
    "SBI LIFE": "SBILIFE.NS",
    "SBILIFE": "SBILIFE.NS",
    "HDFC LIFE": "HDFCLIFE.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "TATA POWER": "TATAPOWER.NS",
    "TATA ELXSI": "TATAELXSI.NS",
    "HAL": "HAL.NS",
    "BEL": "BEL.NS",
    "IRCTC": "IRCTC.NS",
    "VEDANTA": "VEDL.NS",
    "VEDL": "VEDL.NS",
    "SAIL": "SAIL.NS",
    "IOC": "IOC.NS",
    "INDIAN OIL": "IOC.NS",
    "BPCL": "BPCL.NS",
    "HPCL": "HINDPETRO.NS",
    "PNB": "PNB.NS",
    "BANK OF BARODA": "BANKBARODA.NS",
    "BOB": "BANKBARODA.NS",
    "CANARA BANK": "CANBK.NS",
    # ── Major US stocks ──
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "GOOGLE": "GOOGL",
    "ALPHABET": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "TESLA": "TSLA",
    "NVIDIA": "NVDA",
    "NETFLIX": "NFLX",
    "BERKSHIRE": "BRK-B",
    "JPMORGAN": "JPM",
    "JP MORGAN": "JPM",
    "GOLDMAN": "GS",
    "GOLDMAN SACHS": "GS",
    "DISNEY": "DIS",
    "COCA COLA": "KO",
    "PEPSI": "PEP",
    "JOHNSON AND JOHNSON": "JNJ",
    "WALMART": "WMT",
    "VISA": "V",
    "MASTERCARD": "MA",
}




# ──────────────────────────────────────────────
# TICKER RESOLUTION HELPERS
# ──────────────────────────────────────────────
def _search_yahoo(query):
    """Search Yahoo Finance for ticker matches."""
    try:
        search_result = yf.Search(query)
        quotes = getattr(search_result, "quotes", None)
        if quotes:
            return [
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchange"),
                    "type": q.get("quoteType"),
                }
                for q in quotes[:5]
            ]
    except Exception:
        pass

    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=5)
        data = resp.json()
        if "quotes" in data and data["quotes"]:
            return [
                {
                    "symbol": q.get("symbol"),
                    "name": q.get("longname") or q.get("shortname"),
                    "exchange": q.get("exchange"),
                    "type": q.get("quoteType"),
                }
                for q in data["quotes"][:5]
            ]
    except Exception:
        pass

    return None


def _resolve_ticker(query):
    """Central ticker resolution: alias map -> yf.Search -> raw fallback."""
    key = query.strip().upper()

    if key in TICKER_ALIASES:
        return TICKER_ALIASES[key]

    if ".NS" in key or ".BO" in key:
        return key

    results = _search_yahoo(query)
    if results:
        indian = next(
            (q for q in results if q.get("exchange") in ("NSI", "BSE", "NSE")),
            None,
        )
        if indian and indian.get("symbol"):
            return indian["symbol"]
        if results[0].get("symbol"):
            return results[0]["symbol"]

    return key

def fuzzy_search_universe(query: str, df, max_results: int = 6):
    """
    Fuzzy-match a user query against universe_df name + ticker columns.
    Returns list of {ticker, name, match_score, score, quality_pass} sorted desc.
    """
    from difflib import SequenceMatcher

    if df is None or df.empty:
        return []

    q = query.lower().strip()
    if not q:
        return []

    noise = {
        # pronouns / determiners
        "i", "me", "my", "you", "your", "we", "our", "it", "its", "a", "an",
        "the", "this", "that", "these", "those", "any", "some", "all", "each",
        # verbs / auxiliaries
        "is", "are", "was", "were", "be", "been", "am", "do", "does", "did",
        "will", "would", "could", "should", "shall", "may", "might", "can",
        "have", "has", "had", "get", "got", "make", "let", "go", "going",
        # common action verbs in finance queries
        "buy", "sell", "hold", "invest", "investing", "invested", "analyse",
        "analyze", "analysis", "review", "check", "show", "tell", "give",
        "find", "look", "looking", "think", "want", "need", "know", "see",
        "compare", "pick", "choose", "suggest", "recommend", "evaluate",
        # prepositions / conjunctions / adverbs
        "in", "on", "at", "to", "of", "for", "with", "from", "by", "about",
        "into", "between", "through", "after", "before", "up", "down", "out",
        "and", "or", "but", "not", "nor", "so", "if", "then", "than", "also",
        "just", "only", "very", "really", "how", "what", "why", "when", "where",
        "which", "who", "whom", "whose", "whether", "now", "still", "yet",
        # finance generic words
        "stock", "stocks", "share", "shares", "company", "companies", "price",
        "market", "investment", "portfolio", "sector", "industry", "worth",
        "good", "bad", "best", "worst", "top", "right", "safe", "risky",
        "value", "valued", "undervalued", "overvalued", "growth", "income",
        "dividend", "return", "returns", "profit", "loss", "money", "rupee",
        "long", "short", "term", "today", "currently", "recent", "recently",
        # filler
        "please", "thanks", "hey", "hi", "hello", "ok", "okay",
    }
    q_words = [w for w in q.split() if w not in noise]
    q_clean = " ".join(q_words).strip()

    if not q_clean:
        return []

    candidates = []
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        name = str(row.get("name", ""))
        t_bare = ticker.lower().replace(".ns", "").replace(".bo", "")
        n_lower = name.lower()

        score = 0.0

        # 1. Exact ticker match
        if q_clean == t_bare or q_clean == ticker.lower():
            score = 1.0
        # 2. Ticker appears as a word in cleaned query
        elif t_bare in q_words:
            score = 0.95
        # 3. Full company name is substring of cleaned query
        elif n_lower in q_clean:
            score = 0.92
        # 4. Cleaned query is substring of company name (min 3 chars)
        elif len(q_clean) >= 3 and q_clean in n_lower:
            score = 0.85
        # 5. All cleaned query words appear in company name
        elif len(q_words) >= 2 and all(w in n_lower for w in q_words):
            score = 0.80
        else:
            # 6. Fuzzy match via SequenceMatcher
            if len(q_clean) >= 3:
                best = max(
                    SequenceMatcher(None, q_clean, n_lower).ratio(),
                    SequenceMatcher(None, q_clean, t_bare).ratio()
                )
                if best > 0.55:
                    score = best * 0.75

        if score > 0.4:
            candidates.append({
                "ticker": ticker,
                "name": name,
                "match_score": round(score, 3),
                "score": int(row["score"]) if pd.notna(row.get("score")) else 0,
                "quality_pass": bool(row["quality_pass"]) if pd.notna(row.get("quality_pass")) else False,
            })

    candidates.sort(key=lambda x: x["match_score"], reverse=True)
    return candidates[:max_results]


# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Kordent",
    page_icon="logo.svg",
    layout="centered",
)


if "sb_view_mode" not in st.session_state:
    st.session_state.sb_view_mode = "chat"

# ══════════════════════════════════════════════
# CRITICAL INITIALIZATION (Prevents Crashes)
# ══════════════════════════════════════════════
default_state = {
    "sb_view_mode": "chat",
    "messages": [],
    "chat_history": [],
    "sb_access_token": None,
    "sb_refresh_token": None,
    "sb_user_email": None,
    "sb_user_id": None,
    "pending_portfolio": None,
    "pending_retry": None,
    "pending_disambiguation": None,
    "pending_watch_tickers": None,
    "builder_profile": None,
    "_screen_open": False,        # is the deterministic screener table showing?
    "_screen_pending": None,      # tickers picked BEFORE sign-in, flushed after
}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value



# ══════════════════════════════════════════════
# PRESET PROMPTS — reduced to essentials
# ══════════════════════════════════════════════
STOCK_PRESETS = [
    ("📊 Full Analysis",
     "Give me a complete investment analysis of {company} — valuation, financials, growth, and recommendation using all frameworks."),
    ("💰 Graham Value",
     "Calculate the Graham intrinsic value for {company}. Is it undervalued or overvalued? What is the margin of safety?"),
    ("📈 Performance & Chart",
     "How has {company} stock performed over the last 1 year? Show me returns, highs/lows, volatility, and a price chart."),
    ("🎯 Analyst View",
     "What do analysts recommend for {company}? What are the price targets?"),
    ("💸 Dividends",
     "Does {company} pay dividends? Show me the full dividend track record, growth rate, and current yield."),
    ("⚖️ Compare",
     "Compare {company} as investments — valuation, growth, profitability, and which is the better buy."),
]

def _add_to_watchlist(rows) -> int:
    """Insert selected tickers, deduped. Same row shape as the single-stock path."""
    if not st.session_state.get("sb_user_id"):
        return 0
    _sb = get_supabase()
    try:
        _have = {w["ticker"] for w in (_sb.table("watchlist").select("ticker")
                 .eq("user_id", st.session_state.sb_user_id).execute().data or [])}
    except Exception:
        _have = set()

    _new = [{
        "user_id": st.session_state.sb_user_id,
        "ticker": r["ticker"],
        "name": str(r["name"]),
        # RAW 5-denominator `score`, matching the single-stock path. This is what
        # portfolio_tracker compares against to fire watchlist_score_up. Storing
        # score_applicable here would make every financial look like it improved
        # the moment it was added. The CARD shows "3 of 4"; the DATABASE stores 3.
        "score_when_added": int(r["score"]) if pd.notna(r["score"]) else None,
        "quality_when_added": bool(r["quality_pass"]) if pd.notna(r["quality_pass"]) else None,
        # The entry thesis. Without it a watchlist alert can only say the score
        # moved, never why — classify_score_change would return unknown_inputs
        # forever. Captured at add time because add-time facts cannot be
        # reconstructed once the universe is rescored.
        "entry_trace": selector.build_watch_trace(r),
    } for _, r in rows.iterrows() if r["ticker"] not in _have]

    if not _new:
        st.info("Already watching all of those.")
        return 0
    try:
        _sb.table("watchlist").insert(_new).execute()
        st.success(f"Watching {len(_new)}. You'll hear from us when their scores move.")
        return len(_new)
    except Exception as e:
        st.error(f"Failed: {e}")
        return 0


# The list is DETERMINISTIC and rendered by Streamlit, never by the model.
# Asking an LLM to produce a table of tickers and then parsing them back out to
# build checkboxes is how hallucinated tickers get into a user's watchlist. The
# model gets the stocks it was GIVEN, and explains them. It never chooses them.
SCREENER_EXPLAIN_PROMPT = (
    "Explain these stocks, which the system selected deterministically:\n{tickers}\n\n"
    "Each passes 3 or 4 of the frameworks that apply to it, plus a three-year "
    "trajectory test on revenue growth, earnings growth, margin expansion, and "
    "whether that growth was debt-funded.\n\n"
    "For EACH stock, name the specific framework(s) it fails and what that means "
    "in plain language. graham_pass fires on only 5.6% of the market, so 'fails "
    "Graham' means 'not statistically cheap' — a common and often acceptable "
    "trade-off. 'Fails Dorsey' means no durable competitive moat, which is a "
    "much more serious objection. These are different stocks for different "
    "investors.\n\n"
    "Where a stock shows 'X of 4', a framework ABSTAINED rather than failed — "
    "Greenblatt's formula uses ROIC and earnings yield, meaningless for a "
    "levered balance sheet, and he instructs that it not be applied to "
    "financials or utilities. Say so; it is a strength of the method.\n\n"
    "Do NOT recommend a portfolio and do NOT add stocks. This is a screen."
)

# ══════════════════════════════════════════════
# CSS — INSTITUTIONAL LIGHT THEME
# ══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@700;900&family=Inter:wght@400;500;600&display=swap');

/* ── Fix 1: Nuke the dark chat container ── */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.05) !important;
}

[data-testid="stChatInput"] > div {
    background-color: transparent !important;
}

/* Ensure the send arrow icon is visible */
[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
}

/* ── Fix 2: Institutionalize the st.info alerts ── */
[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-left: 4px solid #1D4ED8 !important; /* Trust Blue Accent */
    color: #374151 !important;
    border-radius: 4px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}


/* ── Fix 5: Force All Standard Buttons to Light Theme ── */
.stButton > button, 
div[data-testid="stButton"] > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.05) !important;
    font-weight: 600 !important;
    transition: all 0.1s ease !important;
}

.stButton > button:hover,
div[data-testid="stButton"] > button:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background-color: #F8F9FA !important;
    border-color: #1D4ED8 !important; /* Trust Blue on hover */
    color: #1D4ED8 !important;
}

/* Ensure text/markdown inside the button inherits the correct color */
.stButton > button * {
    color: inherit !important; 
}

/* ── Fix 6: Eradicate the dotted outline inside the chat text area ── */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
    outline-style: none !important; /* Kills the browser default dotted line */
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"] textarea:focus {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
    outline-style: none !important;
}

/* ── Fix 7: Force the Info Box background to solid white ── */
div[data-testid="stAlert"] {
    background-color: transparent !important;
}
div[data-testid="stAlert"] > div {
    background-color: #FFFFFF !important;
}

/* ── Fix 8: Portfolio Boundary Cards ── */
/* Targets the st.container(border=True) wrappers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    padding: 1.5rem !important; /* Gives the text and tables breathing room */
    box-shadow: 2px 2px 0px rgba(0,0,0,0.03) !important; /* Subtle institutional weight */
    margin-bottom: 2rem !important; /* Space between different portfolios */
}

/* Make sure the portfolio title stands out inside the card */
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
    border-bottom: 1px solid #F3F4F6 !important;
    padding-bottom: 10px !important;
    margin-bottom: 15px !important;
}

/* ── Fix 9: Fallback Table Styling ── */
/* Forces any native HTML/Markdown tables into the light theme */
.stTable {
    background-color: #FFFFFF !important;
}
.stTable > div > table {
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important;
}
.stTable th {
    background-color: #F9FAFB !important;
    color: #374151 !important;
    border-bottom: 2px solid #D1D5DB !important;
    font-weight: 600 !important;
}
.stTable td {
    color: #111827 !important;
    border-bottom: 1px solid #E5E7EB !important;
}

/* ── Fix 10: Force Symmetric Rounded Corners on DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 6px !important;
    overflow: hidden !important; /* This acts like a cookie-cutter, clipping sharp inner corners */
    border: 1px solid #E5E7EB !important;
}

[data-testid="stDataFrame"] > div {
    border-radius: 6px !important;
}

[data-testid="stDataFrame"] [data-baseweb="table"] {
    border-radius: 6px !important;
}

/* ── Fix 11: File Uploader Emoji Replacement (Nukes the Overlap) ── */
[data-testid="stFileUploaderDropzone"] button {
    position: relative !important;
    min-height: 38px !important;
    min-width: 120px !important;
}
[data-testid="stFileUploaderDropzone"] button * {
    font-size: 0 !important; /* Eliminates the overlapping broken text & icons */
    color: transparent !important;
}
[data-testid="stFileUploaderDropzone"] button::after {
    content: "📁 Upload";
    color: #111827 !important;
    font-size: 14px !important; /* Restores normal text size for our replacement */
    font-weight: 600 !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    pointer-events: none !important;
    white-space: nowrap !important;
}


/* ── Base ── */
.stApp {
    background-color: #F8F9FA !important; /* Concrete Off-White */
}

.stApp, .stApp * {
    font-family: 'Inter', sans-serif !important;
    color: #374151 !important; /* Dark Slate */
}

/* Exclude icons and code blocks from the universal font override */
.stApp *:not(code):not(.material-symbols-rounded):not(i):not(svg) {
    font-family: 'Inter', sans-serif !important;
}

/* Explicitly protect the expander toggle icons */
[data-testid="stExpanderToggleIcon"], 
.material-symbols-rounded {
    font-family: "Material Symbols Rounded" !important;
    color: #6B7280 !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdown"] p {
    color: #6B7280 !important;
    font-size: 0.85rem !important;
}

/* ── Heavy, Carved Headers ── */
[data-testid="stSidebar"] h1, .stApp h1 {
    font-family: 'Roboto Slab', serif !important;
    color: #111827 !important;
    font-weight: 900 !important;
    letter-spacing: -0.5px !important;
    /* 3D Debossed 'stamped concrete' effect */
    text-shadow: 1px 1px 0px #ffffff, 2px 2px 0px rgba(0,0,0,0.08) !important;
}

[data-testid="stSidebar"] h1 {
    font-size: 1.3rem !important;
}

.stApp h1 {
    font-size: 2.2rem !important;
    padding-bottom: 2px;
    text-transform: none !important;
}

[data-testid="stSidebar"] h3 {
    font-family: 'Roboto Slab', serif !important;
    color: #4B5563 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    margin-top: 1.5rem !important;
}

[data-testid="stSidebar"] hr, .stApp hr {
    border-color: #E5E7EB !important;
}

.stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {
    color: #6b7280 !important;
    font-size: 0.88rem !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important; /* Sharper, institutional corners */
    padding: 1rem 1.2rem !important;
    margin-bottom: 10px !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #1F2937 !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatMessage"] strong {
    color: #1D4ED8 !important; /* Trust Blue */
}

[data-testid="stChatMessage"] code {
    background: #F3F4F6 !important;
    color: #1D4ED8 !important;
    border-radius: 2px !important;
    padding: 2px 6px !important;
    border: 1px solid #E5E7EB !important;
}

[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important; /* Square avatar */
    background: #F8F9FA !important;
}

/* ── Chat input: Institutional Single-Box Design ── */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

/* 1. Nuke the outer Streamlit wrapper and its red focus ring */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background-color: transparent !important;
}

/* 2. Style the Base Web input container to act as the main box */
[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.05) !important;
    padding: 4px !important; /* Space for the button */
    transition: all 0.2s ease;
}

/* Trust Blue focus state for the whole container */
[data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within {
    border-color: #1D4ED8 !important;
    box-shadow: 0 0 0 1px rgba(29, 78, 216, 0.2) !important;
    outline: none !important;
}

/* 3. Strip all borders from the raw textarea so it blends in */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    border: none !important; /* Removes the inner blue box */
    box-shadow: none !important;
    outline: none !important;
    color: #111827 !important;
    font-size: 0.95rem !important;
    padding: 8px 12px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9CA3AF !important;
}

/* 4. Button styling to match */
[data-testid="stChatInput"] button {
    background: #111827 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
    margin-top: 4px !important;
}

[data-testid="stChatInput"] button:hover {
    background: #374151 !important;
}

[data-testid="stChatInput"] button svg {
    fill: #FFFFFF !important;
}

/* Kill generic focus outlines */
*:focus, *:active, *:focus-visible { outline: none !important; }
div[data-baseweb] [aria-invalid] { box-shadow: none !important; }


/* ── Text input ── */
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 4px !important;
    color: #111827 !important;
    font-family: 'Roboto Slab', serif !important;
    font-size: 0.95rem !important;
    padding: 10px 14px !important;
    text-align: center !important;
}

.stTextInput > div > div > input::placeholder {
    color: #9CA3AF !important;
}

.stTextInput > div > div > input:focus {
    border-color: #1D4ED8 !important;
    box-shadow: 0 0 0 1px rgba(29, 78, 216, 0.2) !important;
    outline: none !important;
}

.stTextInput label {
    color: #4B5563 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
}

/* ── Bottom dock ── */
[data-testid="stBottom"] {
    background: #F8F9FA !important;
    background-color: #F8F9FA !important;
    border-top: 1px solid #E5E7EB !important;
}

[data-testid="stBottom"] > div {
    background: transparent !important;
    background-color: transparent !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1D4ED8 !important; }
[data-testid="stSpinnerContainer"] { color: #6B7280 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }

/* ── Tables ── */
.stDataFrame, .stTable {
    max-width: 100% !important;
    overflow-x: auto !important;
}

[data-testid="stChatMessage"] table {
    display: block !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    max-width: 100% !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important;
}

[data-testid="stChatMessage"] th {
    background-color: #F3F4F6 !important;
    color: #111827 !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] td {
    border-top: 1px solid #E5E7EB !important;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .stApp h1 { font-size: 1.8rem !important; }
}

</style>
""", unsafe_allow_html=True)

# ── Google OAuth callback handler ──
_oa_access = st.query_params.get("access_token")
_oa_refresh = st.query_params.get("refresh_token")
_oa_picks = st.query_params.get("picks")   # picks carried through the OAuth redirect
if _oa_access and not st.session_state.sb_user_email:
    try:
        sb = get_supabase()
        resp = sb.auth.set_session(_oa_access, _oa_refresh)
        st.session_state.sb_access_token = _oa_access
        st.session_state.sb_refresh_token = _oa_refresh
        st.session_state.sb_user_email = resp.user.email
        st.session_state.sb_user_id = str(resp.user.id)
        meta = resp.user.user_metadata or {}
        st.session_state["_profile_name"] = meta.get("full_name") or meta.get("name") or ""
        st.session_state["_profile_name_checked"] = True
        try:
            sb.table("profiles").upsert({
                "id": st.session_state.sb_user_id,
                "full_name": st.session_state["_profile_name"],
                "email": resp.user.email
            }, on_conflict="id").execute()
        except Exception:
            pass
    except Exception as e:
        st.error(f"Google login failed: {e}")
    if _oa_picks:
        # Restore the pre-login picks so the post-login flush (the _screen_pending
        # block near line 6191) adds them to the watchlist — the whole point.
        st.session_state._screen_pending = [t for t in _oa_picks.split(",") if t]
    st.query_params.clear()
    st.rerun()



# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    # ── Auth ──
    if st.session_state.sb_user_email is None:
        import urllib.parse as _url
        _cb = "https://shivam1kedia.github.io/Koredent/auth-callback.html"
        _sp = st.session_state.get("_screen_pending")
        if _sp:
            _cb += "?picks=" + _url.quote(",".join(_sp))
        _goog_url = (f"{st.secrets['SUPABASE_URL']}/auth/v1/authorize?provider=google"
                     f"&redirect_to={_url.quote(_cb, safe='')}")
        st.link_button("🔵 Sign in with Google", _goog_url, use_container_width=True)
        st.divider()
        auth_mode = st.radio(
            "Account", ["Login", "Sign Up"],
            horizontal=True, label_visibility="collapsed"
        )
        if auth_mode == "Sign Up":
            auth_full_name = st.text_input("Full Name", key="auth_name_input")
        auth_email = st.text_input("Email", key="auth_email_input")
        auth_password = st.text_input("Password", type="password", key="auth_password_input")

        if auth_mode == "Login":
            if st.button("Log In", width="stretch"):
                if not auth_email or not auth_password:
                    st.warning("Enter email and password.")
                else:
                    try:
                        sb = get_supabase()
                        resp = sb.auth.sign_in_with_password({
                            "email": auth_email,
                            "password": auth_password
                        })
                        st.session_state.sb_access_token = resp.session.access_token
                        st.session_state.sb_refresh_token = resp.session.refresh_token
                        st.session_state.sb_user_email = resp.user.email
                        st.session_state.sb_user_id = str(resp.user.id)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {e}")
        else:
            if st.button("Sign Up", width="stretch"):
                if not auth_full_name or not auth_full_name.strip():
                    st.warning("Enter your full name.")
                elif not auth_email or not auth_password:
                    st.warning("Enter email and password.")
                elif len(auth_password) < 6:
                    st.warning("Password must be at least 6 characters.")
                else:
                    try:
                        sb = get_supabase()
                        resp = sb.auth.sign_up({
                            "email": auth_email,
                            "password": auth_password,
                            "options": {"data": {"full_name": auth_full_name.strip()}}
                        })
                        st.session_state.sb_access_token = resp.session.access_token
                        st.session_state.sb_refresh_token = resp.session.refresh_token
                        st.session_state.sb_user_email = resp.user.email
                        st.session_state.sb_user_id = str(resp.user.id)
                        try:
                            sb.table("profiles").upsert({
                                "id": st.session_state.sb_user_id,
                                "full_name": auth_full_name.strip()
                            }, on_conflict="id").execute()
                        except Exception:
                            pass  # non-blocking — name also lives in user_metadata
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sign up failed: {e}")

    else:
        _display_name = st.session_state.get("_profile_name") or st.session_state.sb_user_email
        st.caption(f"Logged in as {_display_name}")
        sb = get_supabase()

        # ── One-time name collection for existing users ──
        if not st.session_state.get("_profile_name_checked"):
            try:
                _prof = sb.table("profiles").select("full_name").eq(
                    "id", st.session_state.sb_user_id
                ).execute()
                _existing_name = (_prof.data[0].get("full_name") or "") if _prof.data else ""
                st.session_state["_profile_name_checked"] = True
                st.session_state["_profile_name"] = _existing_name
            except Exception:
                _existing_name = ""
                st.session_state["_profile_name_checked"] = True
                st.session_state["_profile_name"] = ""
        
        if st.session_state.get("_profile_name_checked") and not st.session_state.get("_profile_name"):
            with st.container(border=True):
                st.caption("👋 Add your name for personalized reports & emails")
                _name_input = st.text_input("Full Name", key="profile_name_fill")
                if st.button("Save", key="save_profile_name") and _name_input and _name_input.strip():
                    try:
                        sb.table("profiles").upsert({
                            "id": st.session_state.sb_user_id,
                            "full_name": _name_input.strip()
                        }, on_conflict="id").execute()
                        st.session_state["_profile_name"] = _name_input.strip()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        
        # ── New Chat (top) ──
        if st.button("🔄 New Chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.sb_view_mode = "chat"
            st.session_state.pending_disambiguation = None
            st.session_state.pop("_pending_navigate", None)
            # Every key the screener sets must be cleared here. A stale
            # _screen_pending would silently add stocks to the watchlist after
            # an unrelated sign-in, hours later.
            st.session_state._screen_open = False
            st.session_state._screen_pending = None
            st.session_state.pop("_screen_table", None)
            if "pending_portfolio" in st.session_state:
                st.session_state.pending_portfolio = None
                st.session_state.pop("pending_watch_tickers", None)
            st.rerun()

        if st.session_state.sb_view_mode != "chat":
            if st.button("💬 Back to Chat", width="stretch"):
                st.session_state.sb_view_mode = "chat"
                st.rerun()

        # ── Navigation ──
        if st.session_state.sb_view_mode != "builder":
            if st.button("🏗️ Build Portfolio", width="stretch"):
                st.session_state.sb_view_mode = "builder"
                st.rerun()

        if st.session_state.sb_view_mode != "import":
            if st.button("📥 Import Existing Portfolio", width="stretch"):
                st.session_state.sb_view_mode = "import"
                st.rerun()

        if st.session_state.sb_view_mode != "portfolios":
            try:
                _all_ports = sb.table("portfolios").select("id, is_paper").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute().data or []
                _port_count = len([p for p in _all_ports if not p.get("is_paper")])
            except Exception:
                _port_count = 0
            _port_label = f"📁 My Portfolios ({_port_count})" if _port_count else "📁 My Portfolios"
            if st.button(_port_label, width="stretch"):
                st.session_state.sb_view_mode = "portfolios"
                st.rerun()

        if st.session_state.sb_view_mode != "watchlist":
            try:
                _wl_stocks = len((sb.table("watchlist").select("id").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute()).data or [])
                _all_ports_wl = sb.table("portfolios").select("id, is_paper").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute().data or []
                _wl_paper_ports = len([p for p in _all_ports_wl if p.get("is_paper")])
                _wl_count = _wl_stocks + _wl_paper_ports
            except Exception:
                _wl_count = 0
            _wl_label = f"👁 My Watchlist ({_wl_count})" if _wl_count else "👁 My Watchlist"
            if st.button(_wl_label, width="stretch"):
                st.session_state.sb_view_mode = "watchlist"
                st.rerun()

        if st.session_state.sb_view_mode != "backtest":
            if st.button("📊 Does It Work?", width="stretch"):
                st.session_state.sb_view_mode = "backtest"
                st.rerun()

        st.divider()

        # ── Settings + Logout (icon buttons, bottom) ──
        _sb_c1, _sb_c2 = st.columns(2)
        with _sb_c1:
            if st.button("⚙️", key="settings_btn", use_container_width=True, help="Settings"):
                st.session_state.sb_view_mode = "settings"
                st.rerun()
        with _sb_c2:
            if st.button("🚪", key="logout_btn", use_container_width=True, help="Log Out"):
                try:
                    sb = get_supabase()
                    sb.auth.sign_out()
                except Exception:
                    pass
                st.session_state.sb_access_token = None
                st.session_state.sb_refresh_token = None
                st.session_state.sb_user_email = None
                st.session_state.sb_user_id = None
                st.session_state.sb_view_mode = "chat"
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color: #4b5563; font-size: 0.75rem; text-align: center;'>"
        "Not financial advice. For educational and informational purposes only."
        "</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
# Lock the logo and title into a tight horizontal grid
h_col1, h_col2 = st.columns([1, 11])

with h_col1:
    st.image("logo.svg", width=54) # Precise, discrete sizing

with h_col2:
    st.markdown("<h1 style='margin-top: -15px; padding-bottom: 0px;'>Kordent</h1>", unsafe_allow_html=True)

st.markdown("---")
# ──────────────────────────────────────────────
# PUBLIC LEADERBOARD (Landing Page Only)
# ──────────────────────────────────────────────
if st.session_state.sb_view_mode == "chat" and not st.session_state.messages:
    try:
        sb = get_supabase()
        # Fetch the top 3 public portfolios by current return
        leaderboard_resp = sb.table("portfolios").select(
            "name, investor_type, time_horizon, current_return_pct, xirr_pct, "
            "nifty_xirr_pct, is_paper"
        ).not_.is_("current_return_pct", "null").order(
            "current_return_pct", desc=True).limit(20).execute()

        # is_paper filtered in Python: .eq(False) drops NULLs in Postgres, and a
        # simulated portfolio topping a PUBLIC board is a truthfulness failure.
        top_portfolios = [p for p in (leaderboard_resp.data or [])
                          if not p.get("is_paper")][:3]
        
        if top_portfolios:
            st.markdown("### 🏆 Top Performing Portfolios")
            l_cols = st.columns(3)
            for i, port in enumerate(top_portfolios):
                with l_cols[i]:
                    with st.container(border=True):
                        _lb_xirr = port.get("xirr_pct")
                        _lb_val = f"{_lb_xirr:+.1f}% XIRR" if _lb_xirr is not None else f"{port.get('current_return_pct', 0):+.2f}%"
                        st.metric(
                            label=port["name"], 
                            value=_lb_val, 
                            delta=str(port.get("investor_type", "balanced")).title()
                        )
                        st.caption(f"Horizon: {str(port.get('time_horizon', 'medium')).title()}")
            st.markdown("---")
    except Exception as e:
        pass # Fail silently if database is unreachable or empty



# ──────────────────────────────────────────────
# LOAD BOOKS — keyword search (replaces ChromaDB)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_books():
    books = {
        "Graham": "The Intelligent Investor.pdf",
        "Greenblatt": "The Little Book That Still Beats the Market.pdf",
        "Dorsey": "The Five Rules for Successful Stock Investing.pdf",
    }
    chunks = []
    for author, filename in books.items():
        path = os.path.join(BASE_DIR, filename) if "BASE_DIR" in globals() else filename
        if not os.path.exists(path):
            print(f"Warning: {filename} not found.")
            continue
        try:
            doc = pymupdf.open(path)
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
            current = ""
            for para in full_text.split("\n\n"):
                para = para.strip()
                if not para or len(para) < 50:
                    continue
                if len(current) + len(para) < 1200:
                    current = (current + "\n" + para) if current else para
                else:
                    if len(current) >= 100:
                        chunks.append({"author": author, "text": current})
                    current = para
            if current and len(current) >= 100:
                chunks.append({"author": author, "text": current})
        except Exception as e:
            print(f"Warning: could not load {filename}: {e}")
    print(f"Loaded {len(chunks)} book passages from {len(books)} books.")
    return chunks


_STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
         "has", "do", "does", "did", "will", "would", "could", "should", "may",
         "might", "shall", "can", "to", "of", "in", "for", "on", "with", "at",
         "by", "from", "and", "or", "not", "but", "if", "that", "this", "it",
         "its", "as", "about"}


def search_book_passages(chunks, query, n=3):
    """Keyword search. Returns a list of {"author","text"} dicts, best first."""
    kws = [w.lower() for w in re.split(r"\W+", query or "")
           if w.lower() not in _STOP and len(w) > 2]
    if not kws:
        return []
    scored = []
    for c in chunks:
        low = c["text"].lower()
        s = sum(low.count(k) for k in kws)
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:n]]


collection = load_books()

import os
import pandas as pd
import streamlit as st

# Anchor the path absolutely relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "universe_scored.csv")

@st.cache_data(show_spinner=False)
def load_universe(file_path: str):
    """
    Passing the file_path as an argument allows Streamlit to hash the 
    file metadata. If the CSV is updated, the cache invalidates automatically.
    """
    if not os.path.exists(file_path):
        # Fallback empty dataframe to prevent fatal app crashes if file is missing
        st.error(f"Critical System Error: {file_path} not found.")
        return pd.DataFrame()
        
    return pd.read_csv(file_path)

# Initialize the global dataframe safely
universe_df = load_universe(CSV_PATH)


# Display name -> FRAMEWORKS key. The COLUMN comes from selector.PASS_FLAG, not
# from the key: "dorsey_buffett" maps to column "dorsey_pass", so building the
# name by concatenation is wrong. The old list here was four hardcoded labels
# lower()'d into column names — which silently dropped Lynch (no entry at all)
# and would have dropped Dorsey the moment anyone used the real key.
_FW_COLS = (("Graham", "graham"), ("Greenblatt", "greenblatt"),
            ("Dorsey", "dorsey_buffett"), ("Trajectory", "trajectory"),
            ("Lynch", "lynch"))


def _score_label(ticker, score=None) -> str:
    """selector.score_label for a surface that holds a TICKER and a stored
    score but not the universe row — the rebalance table and the action-item
    card, both rendering from persisted JSON.

    Applicability is resolved against TODAY's universe row, not the row as it
    stood when the JSON was written. A user reading the card today should see
    today's denominator; the stored numerator is untouched.

    Falls back to the bare number, never to a fabricated "/5": if the ticker has
    left the universe we do not know what applies to it, and inventing a
    denominator is the exact error this function exists to remove.
    """
    if score is None:
        return "—"
    try:
        _r = universe_df[universe_df["ticker"] == ticker]
        if not _r.empty:
            return selector.score_label(_r.iloc[0], score=score)
    except Exception:
        pass
    return str(score)


# ──────────────────────────────────────────────
# TOOL FUNCTIONS
# ──────────────────────────────────────────────

def get_earnings_quality_metrics(ticker: str) -> dict:
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        inc = t.financials
        cf = t.cashflow

        if inc.empty or cf.empty:
            return {"error": "Financial statements unavailable."}

        def get_latest(df, row_names):
            for name in row_names:
                if name in df.index:
                    val = df.loc[name].dropna()
                    if not val.empty:
                        return float(val.iloc[0])
            return 0.0

        def get_series(df, row_names, n=3):
            """Get up to n years of a metric."""
            for name in row_names:
                if name in df.index:
                    vals = df.loc[name].dropna().tolist()
                    return [float(v) for v in vals[:n]]
            return []

        net_income = get_latest(inc, ['Net Income', 'Net Income Common Stockholders'])
        ocf = get_latest(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        operating_income = get_latest(inc, ['Operating Income', 'EBIT'])
        total_revenue = get_latest(inc, ['Total Revenue'])

        if net_income == 0:
            return {"error": "Net income is 0 or missing."}

        flags = []

        # CHECK 1: Cash conversion
        cash_conversion = ocf / net_income if net_income > 0 else 0
        if cash_conversion < 0.5 and net_income > 0:
            flags.append(
                f"RED FLAG: Cash conversion is {round(cash_conversion, 2)}. "
                f"Only {round(cash_conversion * 100)}% of reported profit is real cash."
            )

        # CHECK 2: Earnings spike (net income vs prior years)
        ni_series = get_series(inc, ['Net Income', 'Net Income Common Stockholders'], n=4)
        if len(ni_series) >= 3:
            prior_avg = sum(ni_series[1:]) / len(ni_series[1:])
            current = ni_series[0]
            if prior_avg > 0 and current > 3 * prior_avg:
                spike_multiple = round(current / prior_avg, 1)
                flags.append(
                    f"RED FLAG: Net income is {spike_multiple}x the prior-year average. "
                    f"Current: {current:,.0f}, Prior avg: {prior_avg:,.0f}. "
                    f"Likely driven by non-recurring event."
                )

        # CHECK 3: Non-operating income gap
        if operating_income > 0 and net_income > 0:
            non_op_gap = (net_income - operating_income) / net_income
            if non_op_gap > 0.4:
                flags.append(
                    f"RED FLAG: {round(non_op_gap * 100)}% of net income comes from "
                    f"below the operating line (non-operational sources). "
                    f"Operating income: {operating_income:,.0f}, Net income: {net_income:,.0f}."
                )

        # ALSO check the legacy unusual items field (catch it if available)
        unusual_items = get_latest(inc, ['Unusual Items', 'Extraordinary Items',
                                         'Special Items', 'Other Special Charges'])
        unusual_pct = abs(unusual_items / net_income) * 100 if net_income != 0 else 0

        if unusual_pct > 20:
            flags.append(
                f"RED FLAG: Tagged non-recurring items are {round(unusual_pct, 1)}% of net income."
            )

        return {
            "ticker": resolved,
            "net_income_reported": net_income,
            "operating_income": operating_income,
            "operating_cash_flow": ocf,
            "cash_conversion_ratio": round(cash_conversion, 2),
            "unusual_items_pct_of_income": round(unusual_pct, 2),
            "anomaly_flags": flags if flags else ["CLEAN: No major anomalies detected."],
            "directive": "If ANY RED FLAG is present, reject positive framework scores."
        }
    except Exception as e:
        return {"error": f"Failed anomaly check: {str(e)}"}


def show_stock_chart(ticker: str) -> dict:
    """Render a 13-month closing price chart for a stock directly in the terminal UI."""
    try:
        import pandas as pd
        import yfinance as yf
        import streamlit as st
        import altair as alt

        resolved = _resolve_ticker(ticker)
        resolved_upper = str(resolved).strip().upper()

        data_feed = yf.Ticker(resolved_upper).history(period="2y")
        if data_feed.empty and not resolved_upper.endswith((".NS", ".BSE")):
            data_feed = yf.Ticker(f"{resolved_upper}.NS").history(period="2y")
            if not data_feed.empty:
                resolved_upper = f"{resolved_upper}.NS"

        if not data_feed.empty:
            df = data_feed.tail(275).reset_index()
            df["Close"] = pd.to_numeric(df["Close"])

            y_min = float(df["Close"].min()) * 0.98
            y_max = float(df["Close"].max()) * 1.02

            st.write(f"### 📈 13-Month Trend: {resolved_upper}")

            chart = alt.Chart(df).mark_line(color="#00f5d4").encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Close:Q', title='Price', scale=alt.Scale(domain=[y_min, y_max])),
                tooltip=['Date', 'Close']
            ).properties(height=400)

            st.altair_chart(chart, width="stretch")

            return {"success": f"Chart successfully rendered for {resolved_upper}."}
        else:
            return {"error": "Failed to fetch chart data."}

    except Exception as e:
        st.error(f"Chart Error: {str(e)}")
        return {"error": str(e)}


def search_book(query: str) -> dict:
    """Search the combined knowledge base of Graham, Greenblatt, and Dorsey.
    Use this when you need specific philosophical frameworks, formulas, or rules
    from any of the three investment authors.

    Args:
        query: What to search for, e.g. "magic formula return on capital" or "economic moat"
    """
    passages = search_book_passages(collection, query, 5)

    if not passages:
        return {"error": "No relevant passages found."}

    formatted = []
    for p in passages:
        author = p.get("author", "Unknown")
        formatted.append(f"[Source: {author}]:\n{p['text']}")

    return {"passages": "\n\n".join(formatted)}


def get_stock_data(company_query: str) -> dict:
    """Get real financial data for a stock using a ticker symbol OR company name.
    Use this when the user asks about a specific company financials.

    Args:
        company_query: Stock ticker or company name, e.g. "AAPL", "RELIANCE.NS",
                       "TCS", "Mahindra", "Groww". Indian tickers should end in .NS
                       (NSE) or .BO (BSE). Common names like RIL, HDFC, SBI are
                       resolved automatically.
    """
    resolved_ticker = _resolve_ticker(company_query)

    try:
        stock = yf.Ticker(resolved_ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return {"error": f"No quantitative data found for '{company_query}'. "
                    f"Resolved to ticker [{resolved_ticker}] but it may be a "
                    f"private entity, mutual fund, or invalid."}

        result = {
            "symbol": info.get("symbol"),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "currency": info.get("currency"),
            "current_price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "book_value": info.get("bookValue"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "profit_margin": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
        }

        # Auto-inject earnings quality — LLM sees flags whether it asks or not
        quality = get_earnings_quality_metrics(resolved_ticker)
        if "error" not in quality:
            result["earnings_quality"] = {
                "cash_conversion_ratio": quality["cash_conversion_ratio"],
                "unusual_items_pct": quality["unusual_items_pct_of_income"],
                "anomaly_flags": quality["anomaly_flags"],
            }

        return result
    except Exception as e:
        return {"error": f"Data retrieval failed for [{resolved_ticker}]: {str(e)}"}


def calculator(expression: str) -> dict:
    """Evaluate a math expression. Use for any calculation:
    ratios, percentages, comparisons, margin of safety computations, etc.

    Args:
        expression: A Python math expression, e.g. "45000 / 1200" or "(52.3 - 41.8) / 52.3 * 100"
    """
    try:
        result = eval(expression)
        return {"expression": expression, "result": round(result, 4)}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {str(e)}"}


def get_historical_trends(company_query: str) -> dict:
    """Get 1-year historical trends (Year-over-Year) for Revenue, Net Income, and Debt.
    Use this when evaluating the immediate recent trajectory of a company.

    Args:
        company_query: Stock ticker or company name. Common names like TCS, Reliance,
                       Mahindra are resolved automatically.
    """
    resolved_ticker = _resolve_ticker(company_query)

    try:
        stock = yf.Ticker(resolved_ticker)
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet

        if income_stmt.empty or balance_sheet.empty:
            return {"error": "Historical financial statements not available."}

        recent_cols = sorted(income_stmt.columns, reverse=True)[:2]
        cols = sorted(recent_cols)

        if len(cols) < 2:
            return {"error": "Not enough historical data to establish a 1-year trend."}

        trends = {}

        def extract_metric(df, row_name):
            try:
                return [df.loc[row_name, col] for col in cols if pd.notna(df.loc[row_name, col])]
            except KeyError:
                return []

        rev_history = extract_metric(income_stmt, "Total Revenue")
        ni_history = extract_metric(income_stmt, "Net Income")
        debt_history = extract_metric(balance_sheet, "Total Debt")

        if len(rev_history) == 2:
            rev_growth = (rev_history[1] / rev_history[0]) - 1
            trends["1Y_Revenue_Growth"] = round(rev_growth * 100, 2)

        if len(ni_history) == 2:
            ni_growth = (ni_history[1] / ni_history[0]) - 1
            trends["1Y_NetIncome_Growth"] = round(ni_growth * 100, 2)

        if len(debt_history) == 2:
            debt_variance = ((debt_history[1] - debt_history[0]) / debt_history[0]) * 100
            trends["Debt_Growth_Trend"] = round(debt_variance, 2)

        return {
            "symbol": resolved_ticker,
            "data_years_analyzed": len(cols),
            "trends": trends
        }
    except Exception as e:
        return {"error": f"Trend data retrieval failed for [{resolved_ticker}]: {str(e)}"}


def get_financial_statements(ticker: str, statement: str) -> dict:
    """Get annual financial statements for a stock.
    Use this to answer questions about revenue, profits, expenses, assets,
    liabilities, debt levels, cash flow, margins, or multi-year growth trends.

    Args:
        ticker: Stock ticker symbol in Yahoo Finance format.
                Indian stocks need .NS suffix (e.g. RELIANCE.NS, TCS.NS).
                US stocks use plain symbol (e.g. AAPL, MSFT).
                Common names like Reliance, TCS, Infosys are also accepted.
        statement: Which financial statement to retrieve. Must be one of:
                   income   - Revenue, EBITDA, net income, operating expenses
                   balance  - Total assets, total debt, shareholder equity, cash
                   cashflow - Operating cash flow, capital expenditure, free cash flow
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)

        if statement == "income":
            df = t.financials
        elif statement == "balance":
            df = t.balance_sheet
        elif statement == "cashflow":
            df = t.cashflow
        else:
            return {"error": f"Invalid statement type: '{statement}'. Use 'income', 'balance', or 'cashflow'."}

        if df is None or df.empty:
            return {"error": f"No {statement} statement data available for {resolved}"}

        data = {}
        for col in df.columns[:4]:
            year_key = str(col.date()) if hasattr(col, "date") else str(col)
            year_data = {}
            for idx in df.index:
                val = df.at[idx, col]
                if val is not None and val == val:
                    year_data[str(idx)] = round(float(val), 2)
            data[year_key] = year_data

        return {"ticker": resolved, "statement_type": statement, "data": data}

    except Exception as e:
        return {"error": f"Failed to get {statement} statement for {resolved}: {str(e)}"}


def get_price_history(ticker: str, period: str) -> dict:
    """Get historical stock price data with performance metrics.
    Use this when the user asks how a stock has performed over time,
    what the 52-week high/low is, price returns, volatility, or moving averages.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
        period: Lookback period. Must be one of:
                1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        hist = t.history(period=period)

        if hist.empty:
            return {"error": f"No price history available for {resolved} over {period}"}

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        high = float(hist["High"].max())
        low = float(hist["Low"].min())
        total_return = ((end_price - start_price) / start_price) * 100
        avg_volume = float(hist["Volume"].mean())

        sma_50 = float(hist["Close"].tail(50).mean()) if len(hist) >= 50 else None
        sma_200 = float(hist["Close"].tail(200).mean()) if len(hist) >= 200 else None

        daily_returns = hist["Close"].pct_change(fill_method=None).dropna()
        if len(daily_returns) > 1:
            volatility = float(daily_returns.std() * (252 ** 0.5) * 100)
        else:
            volatility = None

        return {
            "ticker": resolved,
            "period": period,
            "start_date": str(hist.index[0].date()),
            "end_date": str(hist.index[-1].date()),
            "start_price": round(start_price, 2),
            "current_price": round(end_price, 2),
            "period_high": round(high, 2),
            "period_low": round(low, 2),
            "total_return_pct": round(total_return, 2),
            "avg_daily_volume": int(avg_volume),
            "sma_50": round(sma_50, 2) if sma_50 else "Insufficient data",
            "sma_200": round(sma_200, 2) if sma_200 else "Insufficient data",
            "annualized_volatility_pct": round(volatility, 2) if volatility else "N/A",
        }

    except Exception as e:
        return {"error": f"Failed to get price history for {resolved}: {str(e)}"}


def get_analyst_recommendations(ticker: str) -> dict:
    """Get analyst recommendations, consensus rating, and price targets.
    Use this when the user asks what analysts think, buy/sell ratings,
    target prices, or broker recommendations.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        info = t.info
        result = {"ticker": resolved}

        result["current_price"] = round(
            float(info.get("currentPrice") or info.get("regularMarketPrice", 0)), 2
        )

        try:
            targets = t.analyst_price_targets
            if targets is not None:
                result["price_targets"] = {
                    "low": targets.get("low"),
                    "mean": targets.get("mean"),
                    "median": targets.get("median"),
                    "high": targets.get("high"),
                    "number_of_analysts": targets.get("numberOfAnalystOpinions"),
                }
            else:
                result["price_targets"] = "Not available"
        except Exception:
            result["price_targets"] = "Not available"

        try:
            recs = t.recommendations
            if recs is not None and not recs.empty:
                rec_list = []
                for _, row in recs.tail(12).iterrows():
                    rec_list.append({
                        "firm": str(row.get("Firm", row.get("firm", "Unknown"))),
                        "grade": str(row.get("To Grade", row.get("toGrade", "N/A"))),
                        "action": str(row.get("Action", row.get("action", "N/A"))),
                    })
                result["recent_recommendations"] = rec_list
            else:
                result["recent_recommendations"] = "Not available"
        except Exception:
            result["recent_recommendations"] = "Not available"

        try:
            summary = t.recommendations_summary
            if summary is not None and not summary.empty:
                result["recommendation_summary"] = summary.to_dict(orient="records")
        except Exception:
            pass

        return result

    except Exception as e:
        return {"error": f"Failed to get analyst data for {resolved}: {str(e)}"}


def get_stock_news(ticker: str) -> dict:
    """Get recent news articles about a stock.
    Use this when the user asks about recent news, developments, events,
    announcements, or what is happening with a company.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        news = t.news

        if not news:
            return {"ticker": resolved, "news": "No recent news available for this stock."}

        articles = []
        for item in news[:8]:
            published = item.get("providerPublishTime", "")
            if isinstance(published, (int, float)) and published > 0:
                from datetime import datetime
                try:
                    published = datetime.fromtimestamp(published).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    published = str(published)

            articles.append({
                "title": item.get("title", "No title"),
                "publisher": item.get("publisher", "Unknown"),
                "link": item.get("link", ""),
                "published": str(published),
            })

        return {"ticker": resolved, "news_count": len(articles), "articles": articles}

    except Exception as e:
        return {"error": f"Failed to get news for {resolved}: {str(e)}"}


def get_ownership_info(ticker: str) -> dict:
    """Get major shareholders, institutional holders, and insider transactions.
    Use this when the user asks who owns the stock, promoter holding,
    FII/DII holding, institutional investors, or insider buying/selling.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        result = {"ticker": resolved}

        try:
            major = t.major_holders
            if major is not None and not major.empty:
                breakdown = {}
                for _, row in major.iterrows():
                    breakdown[str(row.iloc[1]).strip()] = str(row.iloc[0]).strip()
                result["holder_breakdown"] = breakdown
            else:
                result["holder_breakdown"] = "Not available"
        except Exception:
            result["holder_breakdown"] = "Not available"

        try:
            inst = t.institutional_holders
            if inst is not None and not inst.empty:
                holders = []
                for _, row in inst.head(10).iterrows():
                    pct = row.get("pctHeld", row.get("pctheld", None))
                    holders.append({
                        "name": str(row.get("Holder", row.get("holder", "Unknown"))),
                        "shares": int(row.get("Shares", row.get("shares", 0))),
                        "pct_held": round(float(pct) * 100, 2) if pct and pct == pct else "N/A",
                        "value": round(float(row.get("Value", row.get("value", 0))), 2),
                    })
                result["top_institutional_holders"] = holders
            else:
                result["top_institutional_holders"] = "Not available"
        except Exception:
            result["top_institutional_holders"] = "Not available"

        try:
            insider = t.insider_transactions
            if insider is not None and not insider.empty:
                txns = []
                for _, row in insider.head(10).iterrows():
                    shares = row.get("Shares", row.get("shares", 0))
                    txns.append({
                        "insider": str(row.get("Insider", row.get("insider", "Unknown"))),
                        "relation": str(row.get("Relation", row.get("relation", ""))),
                        "transaction": str(row.get("Transaction", row.get("transaction", ""))),
                        "shares": int(shares) if shares and shares == shares else 0,
                        "date": str(row.get("Start Date", row.get("startDate", ""))),
                    })
                result["recent_insider_transactions"] = txns
            else:
                result["recent_insider_transactions"] = "Not available"
        except Exception:
            result["recent_insider_transactions"] = "Not available"

        return result

    except Exception as e:
        return {"error": f"Failed to get ownership info for {resolved}: {str(e)}"}


def get_dividend_history(ticker: str) -> dict:
    """Get the full dividend payment history and growth trend for a stock.
    Use this when the user asks about dividend consistency, payout history,
    dividend growth, whether a company has paid dividends regularly, or
    dividend yield trends.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        divs = t.dividends

        if divs is None or divs.empty:
            return {
                "ticker": resolved,
                "has_dividends": False,
                "message": "No dividend history found. This company may not pay dividends.",
            }

        total_payments = len(divs)
        years_of_data = (divs.index[-1] - divs.index[0]).days / 365.25
        latest = float(divs.iloc[-1])

        annual = divs.resample("YE").sum()
        annual_dict = {}
        for date, val in annual.tail(5).items():
            annual_dict[str(date.year)] = round(float(val), 2)

        cagr = None
        if len(annual) >= 3:
            first_val = float(annual.iloc[-min(5, len(annual))])
            last_val = float(annual.iloc[-1])
            n = min(5, len(annual)) - 1
            if first_val > 0 and n > 0:
                cagr = round(((last_val / first_val) ** (1 / n) - 1) * 100, 2)

        info = t.info
        current_yield = info.get("dividendYield")
        if current_yield and current_yield == current_yield:
            current_yield = round(float(current_yield) * 100, 2)
        else:
            current_yield = "N/A"

        return {
            "ticker": resolved,
            "has_dividends": True,
            "total_payments": total_payments,
            "years_of_data": round(years_of_data, 1),
            "latest_dividend_per_share": round(latest, 2),
            "annual_dividends_last_5y": annual_dict,
            "dividend_cagr_pct": cagr if cagr else "Insufficient data for CAGR",
            "current_dividend_yield_pct": current_yield,
        }

    except Exception as e:
        return {"error": f"Failed to get dividend history for {resolved}: {str(e)}"}


def calculate_graham_value(ticker: str) -> dict:
    """Calculate Benjamin Grahams intrinsic value for a stock using his formula:
    V = EPS x (8.5 + 2g) x 4.4 / Y

    Where EPS = trailing earnings per share, g = expected growth rate (capped at 15%),
    Y = current AAA corporate bond yield (approximated at 5%).
    Graham recommended buying only when price is at least 33% below intrinsic value.

    Use this when the user asks for Graham valuation, intrinsic value,
    whether a stock is undervalued or overvalued, or margin of safety.

    Args:
        ticker: Stock ticker symbol (e.g. RELIANCE.NS, AAPL, TCS).
    """
    resolved = _resolve_ticker(ticker)
    try:
        t = yf.Ticker(resolved)
        info = t.info
        # Graham requires 7+ years of earnings track record
        first_trade = info.get("firstTradeDateEpochUtc")
        if first_trade:
            first_date = datetime.datetime.fromtimestamp(first_trade, tz=datetime.timezone.utc)
            years_listed = (datetime.datetime.now(tz=datetime.timezone.utc) - first_date).days / 365.25
            if years_listed < 7:
                return {
                    "ticker": resolved,
                    "graham_applicable": False,
                    "years_listed": round(years_listed, 1),
                    "error": (
                        f"Graham analysis not applicable: {resolved} has only "
                        f"~{round(years_listed, 1)} years of trading history. "
                        f"Graham required a minimum of 7 years of consistent earnings "
                        f"data before trusting any valuation formula. Companies with "
                        f"shorter track records lack the earnings stability evidence "
                        f"his intrinsic value formula assumes."
                    ),
                }

        eps = info.get("trailingEps")
        if not eps or eps <= 0:
            return {
                "ticker": resolved,
                "error": f"Cannot compute Graham value: trailing EPS is {eps} (negative or unavailable). "
                         "Grahams formula only works for profitable companies.",
            }

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        growth = info.get("earningsGrowth")
        if growth and growth > 0:
            g = min(growth * 100, 15.0)
        else:
            rev_growth = info.get("revenueGrowth")
            if rev_growth and rev_growth > 0:
                g = min(rev_growth * 100, 15.0)
            else:
                g = 5.0

        Y = 5.0
        intrinsic_value = eps * (8.5 + 2 * g) * 4.4 / Y

        if current_price and current_price > 0:
            margin = ((intrinsic_value - current_price) / current_price) * 100
            if margin > 33:
                verdict = "UNDERVALUED — meets Grahams 33% margin of safety"
            elif margin > 0:
                verdict = "SLIGHTLY UNDERVALUED — but does NOT meet 33% margin of safety"
            else:
                verdict = "OVERVALUED — price exceeds Graham intrinsic value"
        else:
            margin = None
            verdict = "Cannot determine (price data unavailable)"

        return {
            "ticker": resolved,
            "current_price": round(current_price, 2) if current_price else "N/A",
            "trailing_eps": round(eps, 2),
            "growth_rate_used_pct": round(g, 2),
            "aaa_bond_yield_used_pct": Y,
            "graham_intrinsic_value": round(intrinsic_value, 2),
            "margin_of_safety_pct": round(margin, 2) if margin is not None else "N/A",
            "verdict": verdict,
            "formula_breakdown": f"V = {round(eps,2)} x (8.5 + 2x{round(g,2)}) x 4.4 / {Y} = {round(intrinsic_value,2)}",
            "note": "Growth rate capped at 15% per Grahams conservatism. AAA yield approximated at 5%. "
                    "Graham recommended buying ONLY with >33% margin of safety.",
        }

    except Exception as e:
        return {"error": f"Failed to calculate Graham value for {resolved}: {str(e)}"}


def find_investments(market: str) -> dict:
    """Find the best investment candidates from the pre-scored universe of ~4500 Indian stocks.
    Reads from universe_scored.csv which is updated monthly via universe_updater.py.

    Use this when the user asks to find, discover, or recommend stocks to invest in,
    or asks which stocks are the best buys, or wants investment ideas.

    The 4 frameworks scored are:
    1. Graham — P/E <= 15 AND P/B <= 1.5 (deep value)
    2. Greenblatt — ROE > 15% AND Earnings Yield > 5% (magic formula / capital efficiency)
    3. Dorsey — ROE > 15% AND D/E < 50% (quality + financial health; moat is qualitative)
    4. Trajectory — (Revenue Growth > 0% OR Net Income Growth > 0%) AND (Debt Growth < 0% OR D/E < 50%)

    Args:
        market: Which market to screen. Use 'india' or 'all' (both return Indian stocks).
    """
    df = universe_df
    # Strip value traps pre-flagged by universe_updater
    if "quality_pass" in df.columns:
        df = df[df["quality_pass"] != False]

    _tiers = selector.score_tiers(df)
    tier_4 = df[_tiers == 4].copy()
    tier_3 = df[_tiers == 3].copy()
    tier_2 = df[_tiers == 2].copy()


    

    # Rank-sum sorting within each tier (value + quality + momentum)
    def apply_rank_sort(tier_df):
        if tier_df.empty:
            return tier_df
        t = tier_df.copy()
        t["_pe_sort"] = t["pe"].apply(lambda x: x if pd.notna(x) else 9999)
        t["_roe_sort"] = t["roe_pct"].apply(lambda x: -x if pd.notna(x) else 9999)
        t["_rev_sort"] = t["rev_growth"].apply(lambda x: -x if pd.notna(x) else 9999)
        # Stocks near 52-week lows rank higher (more negative pct_from_high = better value)
        t["_high_sort"] = t["pct_from_high"].apply(lambda x: x if pd.notna(x) else 0)
        t = t.sort_values(["_pe_sort", "_high_sort", "_roe_sort", "_rev_sort"])
        return t.drop(columns=["_pe_sort", "_roe_sort", "_rev_sort", "_high_sort"])

    tier_4 = apply_rank_sort(tier_4)
    tier_3 = apply_rank_sort(tier_3)
    tier_2 = apply_rank_sort(tier_2)

    def to_list(tier_df, max_n=10):
        entries = []
        for _, row in tier_df.head(max_n).iterrows():
            _app = set(selector._applicable_frameworks(row))
            entries.append({
                "ticker": row["ticker"],
                "name": row.get("name", row["ticker"]) if pd.notna(row.get("name")) else row["ticker"],
                "sector": row.get("sector", "N/A") if pd.notna(row.get("sector")) else "N/A",
                "price": round(row["price"], 2) if pd.notna(row.get("price")) else "N/A",
                "pe": round(row["pe"], 2) if pd.notna(row.get("pe")) else "N/A",
                "pb": round(row["pb"], 2) if pd.notna(row.get("pb")) else "N/A",
                "roe_pct": round(row["roe_pct"], 2) if pd.notna(row.get("roe_pct")) else "N/A",
                "de_pct": round(row["de"], 2) if pd.notna(row.get("de")) else "N/A",
                "earnings_yield_pct": round(row["earnings_yield"], 2) if pd.notna(row.get("earnings_yield")) else "N/A",
                "dividend_yield_pct": round(row["dividend_yield_pct"], 2) if pd.notna(row.get("dividend_yield_pct")) else "N/A",
                "rev_growth_pct": round(row["rev_growth"], 2) if pd.notna(row.get("rev_growth")) else "N/A",
                "ni_growth_pct": round(row["ni_growth"], 2) if pd.notna(row.get("ni_growth")) else "N/A",
                "debt_growth_pct": round(row["debt_growth"], 2) if pd.notna(row.get("debt_growth")) else "N/A",
                "score": selector.score_label(row),
                # LYNCH WAS MISSING from both lists — pre-existing, not a W2
                # bug, but it meant the model never saw the fifth framework.
                # Abstained frameworks appear in NEITHER list: not passed, and
                # emphatically not failed.
                "passed": [f for f, k in _FW_COLS if k in _app
                           and pd.notna(row.get(selector.PASS_FLAG[k]))
                           and row.get(selector.PASS_FLAG[k])],
                "failed": [f for f, k in _FW_COLS if k in _app
                           and pd.notna(row.get(selector.PASS_FLAG[k]))
                           and not row.get(selector.PASS_FLAG[k])],
                "abstained": [f for f, k in _FW_COLS if k not in _app],
                "years_of_data": int(row["years_of_data"]) if pd.notna(row.get("years_of_data")) else 0,
                "pct_from_52w_high": round(row["pct_from_high"], 1) if pd.notna(row.get("pct_from_high")) else "N/A",
                "pct_from_52w_low": round(row["pct_from_low"], 1) if pd.notna(row.get("pct_from_low")) else "N/A",
                "pe_vs_historical": round(row["pe_vs_avg"], 1) if pd.notna(row.get("pe_vs_avg")) else "N/A",
                "beta": round(row["beta"], 2) if pd.notna(row.get("beta")) else "N/A",
            })
        return entries

    updated = df["updated_date"].iloc[0] if "updated_date" in df.columns else "Unknown"

    return {
        "market": "india",
        "stocks_in_universe": len(df),
        "data_as_of": updated,
        "perfect_consensus_4_of_4": {
            "count": len(tier_4),
            "top_10": to_list(tier_4),
        },
        "strong_consensus_3_of_4": {
            "count": len(tier_3),
            "top_10": to_list(tier_3),
        },
        "moderate_consensus_2_of_4": {
            "count": len(tier_2),
            "top_10": to_list(tier_2),
        },
        "note": "Pre-scored universe of ~4500 Indian stocks (NSE + BSE). Data updated monthly. After presenting results, use search_book to explain WHY each investment style delivers returns, citing Graham, Greenblatt, and Dorsey.",
    }

def get_sip_candidates(sip_amount: int, time_horizon: str, investor_type: str,
                       review_freq: str, avoid_sectors: str = "[]",
                       min_acceptable_score: int = 3,
                       philosophy: str = "growth_at_fair_price",
                       acceptable_tradeoff: str = "any",
                       framework_weights: str = "{}") -> dict:
    """Build a deterministic, IPS-compliant SIP portfolio from the scored universe.

    THIS FUNCTION NO LONGER SELECTS ANYTHING. selector.select_portfolio does,
    and it is a pure function with no Streamlit and no network — which is what
    lets backtest_runner.py call the REAL selector instead of a lookalike.

    What was deleted here, and why:

      candidates.sort(key=lambda c: c.get("diversification_rank", 999))

    That line was the terminal sort. Everything ranked above it was discarded.
    diversification_rank came from a greedy minimum-variance loop over a
    covariance matrix built from daily closes. A thinly traded stock has flat
    closes on no-trade days, hence ZERO returns, hence a downward-biased sigma;
    non-synchronous trading biases its correlations down too. So argmin(variance)
    is mechanically argmax(staleness). The loop was a staleness detector wearing
    a Markowitz costume, and it was choosing the stocks. That is why every
    portfolio was penny stocks, and why every portfolio was IDENTICAL regardless
    of the questionnaire: diversification_rank contains no user information.

    Args:
        sip_amount: Monthly SIP amount in INR.
        time_horizon: short | medium | long
        investor_type: defensive | balanced | enterprising
        review_freq: passive | moderate | active
        avoid_sectors: JSON list of sector names to exclude, e.g. '["Energy"]'
        min_acceptable_score: 4 | 3 | 2 — the hard gate from builder Q8.
        philosophy: deep_value | growth_at_fair_price | quality_compounder | contrarian
        acceptable_tradeoff: any | ok_fail_graham | ok_fail_trajectory_lynch | ok_fail_dorsey_buffett
        framework_weights: JSON dict of the five framework weights.
    """
    try:
        _avoid = json.loads(avoid_sectors) if isinstance(avoid_sectors, str) else (avoid_sectors or [])
    except Exception:
        _avoid = []
    try:
        _fw = json.loads(framework_weights) if isinstance(framework_weights, str) else (framework_weights or {})
    except Exception:
        _fw = {}

    _profile = st.session_state.get("builder_profile") or {}
    _ips = _profile.get("ips_policy") or generate_ips(_profile or {
        "investor_type": investor_type, "time_horizon": time_horizon,
        "sip_amount": sip_amount, "philosophy": philosophy,
    })

    policy = {
        "sip_amount": sip_amount,
        "avoid_sectors": _avoid,
        "min_acceptable_score": int(min_acceptable_score),
        "philosophy": philosophy,
        "acceptable_tradeoff": acceptable_tradeoff,
        "framework_weights": _fw or _ips.get("framework_weights") or {},
        # THE tilt source. Without this key _resolve_weights falls silently to
        # the legacy per-philosophy weights, and FRAMEWORK_AXIS_COMPOSITION —
        # the whole W0 relative layer's route into selection — never runs.
        # derive_demand_tilt() writes it onto the profile at build time and it
        # is persisted to Supabase; it just never got copied into the policy.
        # Same failure as the Q7/Q8/Q9 one documented below: computed, stored,
        # never handed over.
        "demand_tilt": _profile.get("demand_tilt") or {},
        "allocation_policy": _ips.get("allocation_policy", {}),
        "portfolio_sizing": _ips.get("portfolio_sizing", {}),
    }

    # ── Price history for the staleness filter and the covariance tiebreak ──
    # Only for Tier-1 survivors. investable_tickers() applies the SAME floor the
    # selector will, so the turnover threshold lives in exactly one place.
    #
    # Batched + single-threaded ON PURPOSE. One yf.download of 250 tickers fans
    # curl_cffi across many threads and peaks memory assembling one wide frame.
    # Both are fine on a dev box but are exactly the two things that differ on
    # the 1 GB Linux container where the build segfaulted: curl_cffi's threaded
    # native path (different backend than Windows) and a memory spike. Chunks of
    # 50 with threads off, concatenated on the date index: identical data, flat
    # memory, no concurrent native calls.
    _price_history = None
    try:
        _tk = selector.investable_tickers(universe_df, sip_amount, _avoid, limit=250)
        if len(_tk) >= 2:
            _end = datetime.date.today()
            _start = _end - datetime.timedelta(days=365)
            print(f"[BUILD] price_download_start: {len(_tk)} tickers (batched, threads off)",
                  file=sys.stderr, flush=True)
            _closes = []
            for _i in range(0, len(_tk), 50):
                _batch = _tk[_i:_i + 50]
                _h = yf.download(_batch, start=_start.strftime("%Y-%m-%d"),
                                 end=_end.strftime("%Y-%m-%d"), progress=False,
                                 auto_adjust=True, group_by="column", threads=False)
                if _h is None or _h.empty:
                    continue
                _cl = _h["Close"] if "Close" in _h else _h
                if isinstance(_cl, pd.Series):
                    _cl = _cl.to_frame(_batch[0])
                _closes.append(_cl)
                print(f"[BUILD]   batch {_i // 50 + 1}: {len(_batch)} tickers -> {_cl.shape}",
                      file=sys.stderr, flush=True)
            if _closes:
                _price_history = pd.concat(_closes, axis=1)
            print(f"[BUILD] price_download_done: "
                  f"{None if _price_history is None else _price_history.shape}",
                  file=sys.stderr, flush=True)
    except Exception as _e:
        # Non-blocking. Without prices the selector skips staleness and the
        # correlation tiebreak; the merit ranking is unaffected. Never silently
        # substitute a stale frame.
        print(f"Price history unavailable, selecting without covariance: {_e}",
              file=sys.stderr, flush=True)

    print("[BUILD] entering selector", file=sys.stderr, flush=True)
    result = selector.select_portfolio(universe_df, policy, _price_history)
    print(f"[BUILD] select_done: {len(result.get('holdings', []))} holdings",
          file=sys.stderr, flush=True)

    # ── Firm-distress gate ──────────────────────────────────────────────────
    # Asymmetric by design: dropping a candidate at build time is cheap and
    # reversible, so "medium" confidence suffices. Rather than patching a
    # replacement into the finished list (the old code's approach, which could
    # violate a quota it had already satisfied), exclude the distressed tickers
    # from the universe and re-run the whole deterministic selection. Same code
    # path, same guarantees, no special case.
    _dropped = {}
    try:
        _names = [h["name"] for h in result["holdings"] if h.get("name")]
        if _names:
            _distressed = _detect_firm_distress(_names, min_confidence="medium")
            if _distressed:
                _bad = {h["ticker"] for h in result["holdings"]
                        if h.get("name") in _distressed}
                _dropped = {n: d["reason"] for n, d in _distressed.items()}
                _clean = universe_df[~universe_df["ticker"].isin(_bad)]
                result = selector.select_portfolio(_clean, policy, _price_history)
                result.setdefault("warnings", []).append(
                    f"{len(_bad)} candidate(s) dropped on distress signals.")
    except Exception:
        pass
    st.session_state._distress_dropped = _dropped

    recommended_portfolio = result["holdings"]
    st.session_state._last_candidates = recommended_portfolio
    st.session_state._selection_diagnostics = result.get("diagnostics", {})
    st.session_state._selection_warnings = result.get("warnings", [])
    st.session_state._selection_rejections = result.get("rejections", [])

    # ── Web grounding: THREE separate searches (tax / inflation / sector) ──
    # Separated by half-life and purpose. Each failure is isolated so a tax
    # timeout never nulls the inflation scalar. B-mode: prose + extracted
    # scalar per field, assembled into a machine-comparable macro_snapshot.
    from datetime import date as _date
    web_grounding = {}
    _ts = st.session_state.get("_tool_status")

    def _status(msg):
        try:
            if _ts:
                _ts.update(label=msg, state="running")
        except Exception:
            pass

    # 1) TAX — structural (LTCG/STCG, holding period, exemption)
    _tax_ctx, _tax_scalar = "Web search unavailable.", {}
    try:
        _status("🌐 Fetching capital-gains tax rules...")
        _r = get_web_context(
            "India equity capital gains tax LTCG STCG rate "
            "holding period exemption limit latest"
        )
        _tax_ctx = _r.get("context", _r.get("error", "Unavailable"))
        _tax_scalar = _extract_macro_scalar(_tax_ctx, "tax")
    except Exception:
        pass
    web_grounding["tax"] = _tax_ctx

    # 2) INFLATION — RBI forward CPI PROJECTION (not spot print)
    _inf_ctx, _inf_scalar = "Web search unavailable.", {}
    try:
        _status("🌐 Fetching RBI inflation projection...")
        _r = get_web_context(
            "RBI monetary policy CPI inflation projection forward estimate "
            "next fiscal year FY2027 outlook"
        )
        _inf_ctx = _r.get("context", _r.get("error", "Unavailable"))
        _inf_scalar = _extract_macro_scalar(_inf_ctx, "inflation")
    except Exception:
        pass
    web_grounding["inflation"] = _inf_ctx

    # 3) SECTOR — outlook for the sectors we are ACTUALLY allocating into.
    # This used to read `candidates` — the ~200-name pre-selection pool that
    # the deleted greedy loop built. So it searched the outlook for sectors the
    # user might never hold. The selected holdings are the right set, and they
    # arrive ordered by rank, so dict.fromkeys keeps the strongest sectors first.
    _top_sectors = list(dict.fromkeys(
        h.get("sector", "") for h in recommended_portfolio
        if h.get("sector") and h.get("sector") != "N/A"
    ))[:5]
    _sec_ctx = ""
    if _top_sectors:
        try:
            _status("🌐 Checking sector outlook...")
            _r = get_web_context(
                f"India stock market sector outlook 2026 "
                f"{' '.join(_top_sectors)} headwinds tailwinds "
                f"recent developments"
            )
            _sec_ctx = _r.get("context", _r.get("error", "Unavailable"))
        except Exception:
            _sec_ctx = "Web search unavailable."
    web_grounding["sector_outlook"] = _sec_ctx

    # ── Assemble structured snapshot; park for register_portfolio to persist ──
    st.session_state._macro_snapshot = {
        "as_of": _date.today().isoformat(),
        "tax": {"context": _tax_ctx, **_tax_scalar},
        "inflation": {"context": _inf_ctx, **_inf_scalar},
        "sector_outlook": {"context": _sec_ctx, "sectors": _top_sectors},
    }

    return {
        "investor_profile": {
            "sip_amount_inr": sip_amount,
            "time_horizon": time_horizon,
            "investor_type": investor_type,
            "review_frequency": review_freq,
        },
        # min_stocks / max_stocks are gone. n is ENDOGENOUS:
        #   n = min(pool_size, affordable_n, ips_target)
        # No padding to a book number. Evans & Archer's 12-18 was measured on
        # RANDOMLY selected portfolios, where stock #15 has the same expected
        # return as stock #1 and diversification is free. Under a ranking, stock
        # #15 is your fifteenth-best idea and costs expected return. The floor
        # that remains (n >= 10) is a RUIN constraint, derived from the SEBI 10%
        # single-stock cap this IPS already applies — not a variance argument.
        "selection_diagnostics": result.get("diagnostics", {}),
        "selection_warnings": result.get("warnings", []),
        "near_misses": result.get("rejections", []),
        "tier1_rejections": result.get("rejects", {}),
        "recommended_portfolio": recommended_portfolio,
        "recommended_portfolio_instruction": (
            "CRITICAL: 'recommended_portfolio' is DETERMINISTIC and FINAL. "
            "It was NOT built by minimum-variance optimization — that method ranks stocks "
            "by how STALE their prices are (no-trade days produce zero returns, so sigma "
            "and correlation are both biased down), and it was removed. "
            "Each holding was ranked against its OWN SECTOR on five framework sub-scores, "
            "weighted by the investor's stated philosophy, after clearing their score gate. "
            "Cap-tier floors, sector caps and the single-stock limit were satisfied as "
            "integer quotas before any discretionary slot was filled. "
            "In PHASE 2: call register_portfolio with EXACTLY these stocks. "
            "Do NOT add, remove, reorder, or change allocation_pct. "
            "Every holding carries a '_trace'. Your ONLY job is to phrase it. "
            "The system builds the portfolio; you explain it truthfully."
        ),
        "web_grounding": web_grounding,
        "web_grounding_instruction": (
            "MANDATORY — Reilly & Brown Step 2 data above. "
            "You MUST use this in your portfolio rationale: "
            "(1) State the current inflation rate and compute "
            "real expected return (nominal minus inflation). "
            "(2) State current LTCG/STCG rates and mention "
            "holding period implications. "
            "(3) Factor sector outlook into selection — flag "
            "headwinds/tailwinds per stock. "
            "(4) Reference business cycle positioning if data "
            "indicates a clear phase."
        ),
        "selection_instruction": (
            # `candidates` is gone. There is no pre-filtered pool for you to pick from
            # — that pool, and your access to it, was how hallucinated tickers got in.
            f"'recommended_portfolio' holds {len(recommended_portfolio)} stocks, selected "
            f"from {result.get('diagnostics', {}).get('pool_size', 0)} that cleared the "
            f"investor's {result.get('diagnostics', {}).get('min_acceptable_score', '?')}+ "
            f"score gate. There is no candidate pool for you to choose from. "
            f"PHASE 1 (message starts with [BUILDER_PROFILE]): Do NOT output a portfolio table. "
            f"Review the holdings and ask the user 1-3 clarification questions. "
            f"Questions must NEVER offer options that violate the ALLOCATION POLICY. "
            f"PHASE 2 (user answers): Present the portfolio with 2-3 layman-friendly sentences "
            f"per stock, drawn from its '_trace'. If 'selection_warnings' is non-empty, state "
            f"the reason honestly rather than apologising for the stock count. "
            f"Then call register_portfolio with EXACTLY these stocks. "
            f"Use web_grounding for macro and sector context only — it never changes selection."
        ),
    }



def get_csv_financial_data(ticker: str) -> dict:
    """
    Reads the pre-scored universe database and returns the specific row for the requested ticker.
    Includes pre-computed verdict tier, formatted deep metrics, book reasoning, and pass pattern analysis.
    The verdict tier is DETERMINISTIC — the LLM must use it, never override it.
    """
    resolved = _resolve_ticker(ticker)
    try:
        company_data = universe_df[universe_df['ticker'] == resolved]
        
        if company_data.empty:
            company_data = universe_df[universe_df['name'].str.contains(ticker, case=False, na=False)]
            
        if company_data.empty:
            return {"error": f"No proprietary CSV data found for {ticker}."}
            
        row = company_data.iloc[0].fillna("N/A").to_dict()

        # ── Sprint 7: Inject deterministic verdict context ──
        pass_dict = {
            "graham_pass": bool(row.get("graham_pass")) if row.get("graham_pass") != "N/A" else False,
            "greenblatt_pass": bool(row.get("greenblatt_pass")) if row.get("greenblatt_pass") != "N/A" else False,
            "dorsey_pass": bool(row.get("dorsey_pass")) if row.get("dorsey_pass") != "N/A" else False,
            "trajectory_pass": bool(row.get("trajectory_pass")) if row.get("trajectory_pass") != "N/A" else False,
            "lynch_pass": False if (row.get("lynch_pass") in (None, "", "N/A") or pd.isna(row.get("lynch_pass"))) else bool(row.get("lynch_pass")),
        }
        score = int(row.get("score", 0)) if row.get("score") != "N/A" else 0
        quality_pass = bool(row.get("quality_pass")) if row.get("quality_pass") != "N/A" else False
        manip = int(row.get("schilit_manipulation_score", 0)) if row.get("schilit_manipulation_score") != "N/A" else 0

        _app_set = set(selector._applicable_frameworks(row))
        _n_app = len(_app_set)
        # Short names: get_pattern_key and _get_failure_principles use "dorsey",
        # FRAMEWORKS uses "dorsey_buffett". Bound HERE because both
        # get_pass_pattern_reasoning and get_pattern_meaning_for consume it below.
        _abstained = tuple("dorsey" if f == "dorsey_buffett" else f
                           for f in selector.FRAMEWORKS if f not in _app_set)
        verdict = verdict_engine.get_verdict_tier(score, quality_pass, pass_dict,
                                                  manip, n_applicable=_n_app)
        verdict_reason = verdict_engine.get_verdict_reason(verdict, score, pass_dict, manip)

        # Get user's investment philosophy if available (from active portfolio profile)
        philosophy = None
        try:
            active_pf = st.session_state.get("sb_selected_portfolio")
            if active_pf:
                _prof = active_pf.get("portfolio_profile") or {}
                philosophy = _prof.get("philosophy")
        except Exception:
            pass

        book_reasoning = verdict_engine.get_pass_pattern_reasoning(
            score, pass_dict, verdict, philosophy, abstained=_abstained)
        deep_formatted = verdict_engine.format_deep_metrics_for_llm(row)
        # Gate on the NORMALISED verdict, not the raw integer: a 2-of-4 stock is
        # a CONDITIONAL BUY and should be explained like one. get_pattern_meaning_for
        # returns None when any framework abstained — see its docstring.
        pattern_meaning = (verdict_engine.get_pattern_meaning_for(pass_dict, _abstained)
                           if verdict == "CONDITIONAL BUY" else None)

        st.session_state._last_verdict_tier = verdict
        row["_verdict_tier"] = verdict
        row["_verdict_reason"] = verdict_reason
        row["_verdict_emoji"] = verdict_engine.VERDICT_EMOJI.get(verdict, "")
        row["_deep_metrics_formatted"] = deep_formatted
        row["_book_reasoning"] = book_reasoning
        if pattern_meaning:
            row["_pass_pattern_meaning"] = pattern_meaning

        return row
    except Exception as e:
        return {"error": f"Error reading CSV data: {str(e)}"}

def get_macro_context(ticker: str) -> dict:
    """
    Returns macro context: sector, Nifty 5-day performance, India VIX (fear gauge),
    and sector-level index performance where available.
    """
    resolved = _resolve_ticker(ticker)
    try:
        stock = yf.Ticker(resolved)
        sector = stock.info.get('sector', 'Unknown Sector')
        
        result = {"ticker": resolved, "sector": sector}

        # Nifty 50 momentum
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="1mo")
            if not hist.empty and len(hist) >= 2:
                result["nifty_50_1d_pct"] = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100, 2)
                result["nifty_50_5d_pct"] = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-6 if len(hist) >= 6 else 0]) - 1) * 100, 2)
                result["nifty_50_1mo_pct"] = round(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100, 2)
                result["nifty_50_close"] = round(float(hist['Close'].iloc[-1]), 2)
        except Exception:
            result["nifty_50_5d_pct"] = "N/A"

        # India VIX (fear gauge)
        try:
            vix = yf.Ticker("^INDIAVIX")
            vix_hist = vix.history(period="5d")
            if not vix_hist.empty:
                vix_val = round(float(vix_hist['Close'].iloc[-1]), 2)
                result["india_vix"] = vix_val
                if vix_val < 13:
                    result["vix_interpretation"] = "Low fear — market is complacent, potential for sudden correction"
                elif vix_val < 20:
                    result["vix_interpretation"] = "Normal range — market pricing moderate uncertainty"
                elif vix_val < 30:
                    result["vix_interpretation"] = "Elevated fear — market expects significant moves"
                else:
                    result["vix_interpretation"] = "Panic levels — historically a contrarian buy signal per Graham"
        except Exception:
            result["india_vix"] = "N/A"

        # Sector-specific index (map common sectors to Nifty sector indices)
        SECTOR_INDICES = {
            "Technology": "^CNXIT",
            "Financial Services": "^CNXFIN",
            "Energy": "^CNXENERGY",
            "Consumer Cyclical": "^CNXCONSUMER",
            "Healthcare": "^CNXPHARMA",
            "Basic Materials": "^CNXMETAL",
            "Industrials": "^CNXINFRA",
            "Consumer Defensive": "^CNXFMCG",
            "Real Estate": "^CNXREALTY",
            "Utilities": "^CNXPSUBANK",
        }

        sector_idx = SECTOR_INDICES.get(sector)
        if sector_idx:
            try:
                sidx = yf.Ticker(sector_idx)
                sidx_hist = sidx.history(period="1mo")
                if not sidx_hist.empty and len(sidx_hist) >= 2:
                    result["sector_index"] = sector_idx
                    result["sector_1d_pct"] = round(((sidx_hist['Close'].iloc[-1] / sidx_hist['Close'].iloc[-2]) - 1) * 100, 2)
                    result["sector_5d_pct"] = round(((sidx_hist['Close'].iloc[-1] / sidx_hist['Close'].iloc[-6 if len(sidx_hist) >= 6 else 0]) - 1) * 100, 2)
                    result["sector_1mo_pct"] = round(((sidx_hist['Close'].iloc[-1] / sidx_hist['Close'].iloc[0]) - 1) * 100, 2)
            except Exception:
                pass

        return result

    except Exception as e:
        return {"error": f"Error fetching macro context: {str(e)}"}


def _sanitize_for_json(obj):
    """Replace NaN/Inf with None so Gemini gets valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    # Catch Python float, numpy.float64, numpy.float32, and any numeric type
    try:
        if math.isnan(obj) or math.isinf(obj):
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    # Convert stray numpy scalars to Python native so json.dumps never chokes
    if hasattr(obj, 'item'):
        return obj.item()
    return obj

def build_user_context():
    """Build compact client state summary for the system prompt."""
    if not st.session_state.get("sb_user_id"):
        return "Client is not logged in. They can use chat for stock analysis but cannot save portfolios or access personalized features."
    sb = get_supabase()
    try:
        ports = sb.table("portfolios").select(
            "name, sip_amount, time_horizon, investor_type, current_value, current_return_pct, is_paper"
        ).eq("user_id", st.session_state.sb_user_id).execute().data or []
    except Exception:
        ports = []
    try:
        wl_count = len(sb.table("watchlist").select("id").eq(
            "user_id", st.session_state.sb_user_id
        ).execute().data or [])
    except Exception:
        wl_count = 0

    name = st.session_state.get("_profile_name", "")
    if not ports and wl_count == 0:
        ctx = "New client — no portfolios, no watchlist."
        if name:
            ctx = f"Name: {name}. " + ctx
        return ctx

    lines = []
    if name:
        lines.append(f"Name: {name}")
    real_ports = [p for p in ports if not p.get("is_paper")]
    paper_ports = [p for p in ports if p.get("is_paper")]
    if real_ports:
        lines.append(f"{len(real_ports)} active portfolio(s):")
        for p in real_ports:
            val = f"{fmt_inr(p.get('current_value', 0))}" if p.get('current_value') else "not yet valued"
            ret = f"({p.get('current_return_pct', 0):+.1f}%)" if p.get('current_return_pct') is not None else ""
            lines.append(f"  {p['name']} — {p.get('investor_type', '')} / {p.get('time_horizon', '')} / {fmt_inr(p.get('sip_amount', 0))}/mo — {val} {ret}")
    if paper_ports:
        lines.append(f"{len(paper_ports)} paper portfolio(s) (practice mode)")
    if wl_count:
        lines.append(f"Watchlist: {wl_count} stock(s)")
    return "\n".join(lines)

def resolve_stock(query: str) -> dict:
    """Find the exact ticker for a stock the client mentions by name or partial name.
    Call this BEFORE any analysis tool when the client uses a company name instead of a full ticker (like X.NS or X.BO).
    Do NOT call this for conversational questions, platform questions, or when the ticker is already known.

    Args:
        query: The company name or partial name to search for.
    """
    matches = fuzzy_search_universe(query, universe_df)
    good = [m for m in matches if m["match_score"] > 0.4]

    if not good:
        return {"status": "no_match", "message": f"No stocks found matching '{query}'. Ask the client for the exact company name or ticker."}

    # Single match or one dominant match — resolve directly
    if len(good) == 1:
        m = good[0]
        return {"status": "resolved", "ticker": m["ticker"], "name": m["name"], "score": m["score"]}
    if good[0]["match_score"] >= 0.92 and good[1]["match_score"] < 0.80:
        m = good[0]
        return {"status": "resolved", "ticker": m["ticker"], "name": m["name"], "score": m["score"]}

    # Multiple plausible matches — show disambiguation buttons to client
    st.session_state.pending_disambiguation = {
        "original_query": st.session_state.get("_last_user_message", query),
        "matches": good[:10],
    }
    return {
        "status": "ambiguous",
        "message": f"Multiple companies match '{query}'. The client will see buttons to pick the right one — wait for their selection before proceeding.",
        "matches": [{"ticker": m["ticker"], "name": m["name"]} for m in good[:6]]
    }


def navigate_to(view: str) -> dict:
    """Open a platform feature for the client. Call this when the client wants to ACT on a feature, not when they are just asking about it.

    Args:
        view: The feature to open. One of: builder, import, portfolios, watchlist, backtest, settings
    """
    valid = {"builder", "import", "portfolios", "watchlist", "backtest", "settings"}
    if view not in valid:
        return {"error": f"Unknown view '{view}'. Valid: {', '.join(sorted(valid))}"}
    st.session_state._pending_navigate = view
    view_labels = {"builder": "Portfolio Builder", "import": "Import", "portfolios": "My Portfolios",
                   "watchlist": "Watchlist", "backtest": "Does It Work?", "settings": "Settings"}
    return {"status": f"Opening {view_labels.get(view, view)}. The client will see it momentarily."}

def get_web_context(query: str) -> dict:
    """Search the web for recent real-world context about a company, sector, or macro topic.
    Use this to ground your analysis in current events. Call for EVERY stock in a positive verdict
    (STRONG BUY, BUY, CONDITIONAL BUY) and for EVERY shortlisted candidate during portfolio construction.
    Also use for macro context like inflation outlook, RBI policy, or sector headwinds.

    Args:
        query: A specific search query (e.g. "Infosys recent news regulatory actions 2026",
               "India pharma sector headwinds 2026", "India CPI inflation RBI outlook 2026").
    """
    import requests as _req

    # ── Primary: Google Custom Search (no LLM, no Gemini quota) ──
    _gcs_key = st.secrets.get("GOOGLE_SEARCH_API_KEY")
    _gcs_cx = st.secrets.get("GOOGLE_CSE_ID")

    if _gcs_key and _gcs_cx:
        try:
            resp = _req.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": _gcs_key, "cx": _gcs_cx, "q": query, "num": 5},
                timeout=10,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    snippets = []
                    for item in items:
                        source = item.get("displayLink", "")
                        title = item.get("title", "")
                        snippet = item.get("snippet", "").replace("\n", " ")
                        snippets.append(f"• {title} ({source}): {snippet}")
                    return {"query": query, "context": "\n".join(snippets)}
                return {"query": query, "context": "No material recent developments found."}
            # 429 = daily quota exhausted → fall through to Gemini fallback
            if resp.status_code != 429:
                return {"query": query, "error": f"Search API returned {resp.status_code}"}
        except Exception:
            pass  # fall through to Gemini fallback

    # ── Fallback: Gemini google_search grounding (only search-capable models) ──
    SEARCH_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash"]
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    search_prompt = f'''Search the web and provide a concise summary of the most recent and material information for this query:

"{query}"

Focus on:
- Material events from the last 90 days (regulatory actions, management changes, fraud/investigations, major contracts, earnings surprises, M&A)
- Sector-level headwinds or tailwinds
- Macro context if relevant (RBI policy, inflation, commodity cycles)

Rules:
- Lead with the MOST material item first
- Include dates where possible
- If nothing material is found, say "No material recent developments found" — do NOT fabricate
- Keep the summary to 5-8 bullet points maximum
- Cite source names (e.g. "per Economic Times", "per SEBI filing") where possible'''

    import time as _time
    for i, model_name in enumerate(SEARCH_MODELS):
        try:
            if i > 0:
                _time.sleep(2)
            response = client.models.generate_content(
                model=model_name,
                contents=search_prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
            summary = ""
            try:
                summary = response.text or ""
            except Exception:
                try:
                    for part in (response.candidates[0].content.parts or []):
                        if hasattr(part, 'text') and part.text:
                            summary += part.text
                except (AttributeError, IndexError, TypeError):
                    pass
            if not summary.strip():
                return {"query": query, "context": "No material recent developments found."}
            return {"query": query, "context": summary.strip()}
        except Exception:
            continue

    return {"query": query, "error": "Web search unavailable — all methods exhausted."}


def _extract_macro_scalar(context_text: str, field: str) -> dict:
    """Extract ONE structured number from web-grounding prose. B-mode.

    field: 'tax' -> {ltcg_pct, stcg_pct, ltcg_holding_months, ltcg_exemption_inr}
           'inflation' -> {rbi_cpi_projection_pct, target_fy}

    Range-validated: out-of-range values are rejected as None (a parse failure
    is safer than a confidently-wrong number feeding the review-diff).
    Returns {} on any failure — null is a first-class value, never fabricate.
    """
    # STRUCTURAL WHITELIST, not a blacklist. The previous guard named three
    # exact sentinel strings, so anything else passed -- including
    # get_web_context's own error returns, e.g. "Search API returned 403",
    # which six call sites promote into a content slot via
    # _r.get("context", _r.get("error", ...)). Gemini was then asked to extract
    # an RBI CPI projection from an error message.
    #
    # Retrieved context always contains the bullet marker (CSE branch) or
    # multi-line prose (Gemini grounding branch). An error string is a single
    # short line with neither. Blacklists fail open; whitelists fail closed.
    _t = (context_text or "").strip()
    if not _t:
        return {}
    if _t in ("No material recent developments found.", "Web search unavailable.",
              "Unavailable"):
        return {}
    if "\u2022" not in _t and "\n" not in _t and len(_t) < 200:
        return {}

    if field == "tax":
        schema_prompt = (
            "From the text below, extract India equity capital-gains tax facts as JSON. "
            "Keys (use null if a value is not clearly stated): "
            "ltcg_pct (long-term capital gains rate as a percentage number, e.g. 12.5), "
            "stcg_pct (short-term rate as a percentage number, e.g. 20.0), "
            "ltcg_holding_months (months to qualify as long-term, e.g. 12), "
            "ltcg_exemption_inr (annual LTCG exemption in rupees, e.g. 125000). "
            "Output ONLY the JSON object, no prose, no markdown.\n\nTEXT:\n" + context_text
        )
    elif field == "inflation":
        schema_prompt = (
            "From the text below, extract the RBI CPI inflation PROJECTION as JSON. "
            "Keys (use null if not clearly stated): "
            "rbi_cpi_projection_pct (the projected CPI inflation as a percentage number, "
            "e.g. 4.5 — NOT a fraction like 0.045), "
            "target_fy (the fiscal year the projection targets, e.g. 'FY2027'). "
            "Prefer RBI's official forward projection over a spot/current print. "
            "Output ONLY the JSON object, no prose, no markdown.\n\nTEXT:\n" + context_text
        )
    else:
        return {}

    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        raw = None
        for model_name in FREE_MODELS:
            try:
                resp = client.models.generate_content(model=model_name, contents=schema_prompt)
                raw = (resp.text or "").strip()
                if raw:
                    break
            except Exception:
                continue
        if not raw:
            return {}
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
    except Exception:
        return {}

    # ── Range validation — reject garbage as None ──
    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out = {}
    if field == "tax":
        ltcg = _num(parsed.get("ltcg_pct"))
        stcg = _num(parsed.get("stcg_pct"))
        hold = _num(parsed.get("ltcg_holding_months"))
        exem = _num(parsed.get("ltcg_exemption_inr"))
        out["ltcg_pct"] = ltcg if (ltcg is not None and 0 <= ltcg <= 40) else None
        out["stcg_pct"] = stcg if (stcg is not None and 0 <= stcg <= 40) else None
        out["ltcg_holding_months"] = int(hold) if (hold is not None and 0 < hold <= 60) else None
        out["ltcg_exemption_inr"] = exem if (exem is not None and 0 <= exem <= 10_000_000) else None
    elif field == "inflation":
        cpi = _num(parsed.get("rbi_cpi_projection_pct"))
        # If the model returned a fraction (0.045), lift to percent
        if cpi is not None and 0 < cpi < 1:
            cpi = cpi * 100
        out["rbi_cpi_projection_pct"] = cpi if (cpi is not None and 0 <= cpi <= 15) else None
        fy = parsed.get("target_fy")
        out["target_fy"] = str(fy) if fy else None
    return out


def _append_macro_snapshot(profile: dict, new_snap: dict, n: int = 6) -> dict:
    """Append new_snap to a bounded history; keep latest as macro_snapshot."""
    if profile is None:
        profile = {}
    if not new_snap:
        return profile
    hist = profile.get("macro_history", [])
    if not isinstance(hist, list):
        hist = []
    hist.append(new_snap)
    profile["macro_history"] = hist[-n:]
    profile["macro_snapshot"] = new_snap
    return profile


def _diff_scalar(old_val, new_val):
    """Three-case diff. None is 'no reading' — NEVER coerced to 0."""
    op = old_val is not None
    np_ = new_val is not None
    if op and np_:
        try:
            delta = round(float(new_val) - float(old_val), 3)
        except (TypeError, ValueError):
            if old_val == new_val:
                return {"status": "stable", "old": old_val, "new": new_val, "delta": None}
            return {"status": "changed", "old": old_val, "new": new_val, "delta": None}
        if delta == 0:
            return {"status": "stable", "old": old_val, "new": new_val, "delta": 0}
        return {"status": "changed", "old": old_val, "new": new_val, "delta": delta}
    if np_ and not op:
        return {"status": "new", "old": None, "new": new_val, "delta": None}
    if op and not np_:
        return {"status": "unknown_now", "old": old_val, "new": None, "delta": None}
    return {"status": "no_reading", "old": None, "new": None, "delta": None}


def _diff_macro_snapshots(old: dict, new: dict) -> dict:
    """None-aware diff of two snapshots. Missing snapshot -> {}."""
    if not old or not new:
        return {}
    o_tax, n_tax = (old.get("tax") or {}), (new.get("tax") or {})
    o_inf, n_inf = (old.get("inflation") or {}), (new.get("inflation") or {})
    return {
        "as_of_old": old.get("as_of"),
        "as_of_new": new.get("as_of"),
        "inflation_projection": _diff_scalar(
            o_inf.get("rbi_cpi_projection_pct"), n_inf.get("rbi_cpi_projection_pct")),
        "inflation_target_fy": _diff_scalar(
            o_inf.get("target_fy"), n_inf.get("target_fy")),
        "ltcg_pct": _diff_scalar(o_tax.get("ltcg_pct"), n_tax.get("ltcg_pct")),
        "stcg_pct": _diff_scalar(o_tax.get("stcg_pct"), n_tax.get("stcg_pct")),
    }


def _render_macro_diff(diff: dict) -> str:
    """Signal-only delta lines. Silent when nothing moved. Empty -> ''."""
    if not diff:
        return ""
    _o, _n = diff.get("as_of_old", "?"), diff.get("as_of_new", "?")
    fields = [
        ("RBI CPI projection", "inflation_projection", "%", True),
        ("Projection target FY", "inflation_target_fy", "", False),
        ("LTCG rate", "ltcg_pct", "%", True),
        ("STCG rate", "stcg_pct", "%", True),
    ]
    signal_states = {"changed", "new", "unknown_now"}
    has_signal = any(diff[k]["status"] in signal_states for _, k, _, _ in fields)
    if not has_signal:
        return f"_No material macro shift since last review ({_o} \u2192 {_n})._"
    lines = []
    for lbl, key, unit, isbp in fields:
        d = diff[key]
        stt = d["status"]
        if stt == "changed":
            if d["delta"] is None:
                lines.append(f"- **{lbl}**: {d['old']} \u2192 {d['new']}")
            else:
                arrow = "\u25b2" if d["delta"] > 0 else "\u25bc"
                sign = "+" if d["delta"] > 0 else ""
                bp = f" ({sign}{round(d['delta']*100)}bps)" if isbp else ""
                lines.append(f"- **{lbl}**: {d['old']}{unit} \u2192 {d['new']}{unit} {arrow}{bp}")
        elif stt == "new":
            lines.append(f"- **{lbl}**: now {d['new']}{unit} (no prior reading)")
        elif stt == "unknown_now":
            lines.append(f"- **{lbl}**: was {d['old']}{unit}, could not read this cycle \u26a0\ufe0f")
    return f"**Macro shift since last review ({_o} \u2192 {_n}):**\n" + "\n".join(lines)


def _refresh_macro_snapshot(holdings: list) -> dict:
    """Standalone macro/inflation/tax refresh for review time (no builder profile).
    Sectors derived from held stocks. Returns a fresh snapshot dict."""
    from datetime import date as _date
    _tax_ctx, _tax_scalar = "Web search unavailable.", {}
    try:
        _r = get_web_context("India equity capital gains tax LTCG STCG rate "
                             "holding period exemption limit latest")
        _tax_ctx = _r.get("context", _r.get("error", "Unavailable"))
        _tax_scalar = _extract_macro_scalar(_tax_ctx, "tax")
    except Exception:
        pass
    _inf_ctx, _inf_scalar = "Web search unavailable.", {}
    try:
        _r = get_web_context("RBI monetary policy CPI inflation projection forward estimate "
                             "next fiscal year FY2027 outlook")
        _inf_ctx = _r.get("context", _r.get("error", "Unavailable"))
        _inf_scalar = _extract_macro_scalar(_inf_ctx, "inflation")
    except Exception:
        pass
    _secs = list(dict.fromkeys(
        h.get("sector", "") for h in (holdings or [])
        if h.get("sector") and h.get("sector") != "N/A"))[:5]
    _sec_ctx = ""
    if _secs:
        try:
            _r = get_web_context(f"India stock market sector outlook 2026 {' '.join(_secs)} "
                                f"headwinds tailwinds recent developments")
            _sec_ctx = _r.get("context", _r.get("error", "Unavailable"))
        except Exception:
            _sec_ctx = "Web search unavailable."
    return {
        "as_of": _date.today().isoformat(),
        "tax": {"context": _tax_ctx, **_tax_scalar},
        "inflation": {"context": _inf_ctx, **_inf_scalar},
        "sector_outlook": {"context": _sec_ctx, "sectors": _secs},
    }


def _detect_firm_distress(firm_names: list, min_confidence: str = "high") -> dict:
    """Batched, specificity-first firm-distress detection.

    ONE CSE query for the whole basket asking only about HARD distress signals
    (SEBI/RBI enforcement, auditor resignation, default, insolvency, fraud
    charge, rating downgrade to default). The LLM then judges MATERIALITY and
    CONFIDENCE per firm. Returns {firm_name: {reason, confidence}} filtered to
    firms actually in the input list AND confidence >= min_confidence.

    Asymmetric use:
      - candidate drop-replace at build  -> min_confidence="medium" (cheap to drop)
      - held-position flag at review      -> min_confidence="high"   (can cause a real sell)

    Fail-safe: any parse/search failure returns {} — NEVER fabricate distress.
    """
    firm_names = [f for f in (firm_names or []) if f]
    if not firm_names:
        return {}

    _basket = " OR ".join(f'"{n}"' for n in firm_names[:20])
    _query = (
        f"India NSE ({_basket}) SEBI action OR RBI penalty OR auditor resignation "
        f"OR fraud charge OR insolvency OR loan default OR rating downgrade default 2026"
    )
    try:
        _r = get_web_context(_query)
        _ctx = _r.get("context", "")
    except Exception:
        return {}
    if not _ctx or _ctx.strip() in (
        "No material recent developments found.", "Web search unavailable.", "Unavailable"):
        return {}

    _valid = set(firm_names)
    _judge_prompt = (
        "You are a risk officer. Below are web snippets and a list of firms we hold or are "
        "considering. Identify ONLY firms with MATERIAL, RECENT, HARD distress — specifically: "
        "regulatory ENFORCEMENT (SEBI/RBI action or penalty), auditor resignation, fraud charges, "
        "loan default, insolvency filing, or a credit-rating downgrade to default. "
        "IGNORE soft signals: analyst opinions, 'concerns', 'probes' without action, price drops, "
        "sector chatter, competitive pressure. When in doubt, DO NOT flag — a false alarm here is "
        "worse than a miss.\n\n"
        f"FIRMS: {', '.join(firm_names)}\n\n"
        f"SNIPPETS:\n{_ctx}\n\n"
        "Respond ONLY with a JSON array (no markdown). Each element: "
        '{"firm": "<exact firm name from the list>", "reason": "<one sentence, cite the hard signal>", '
        '"confidence": "high|medium|low"}. '
        "Use 'high' ONLY when the snippet explicitly states an enforcement action/default/resignation. "
        "Return [] if none qualify."
    )

    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        _raw = None
        for _m in FREE_MODELS:
            try:
                _resp = client.models.generate_content(model=_m, contents=_judge_prompt)
                _raw = (_resp.text or "").strip()
                if _raw:
                    break
            except Exception:
                continue
        if not _raw:
            return {}
        _raw = _raw.replace("```json", "").replace("```", "").strip()
        if _raw.startswith("json"):
            _raw = _raw[4:].strip()
        _parsed = json.loads(_raw)
    except Exception:
        return {}

    if not isinstance(_parsed, list):
        return {}
    _rank = {"low": 1, "medium": 2, "high": 3}
    _threshold = _rank.get(min_confidence, 3)
    _out = {}
    for _item in _parsed:
        if not isinstance(_item, dict):
            continue
        _nm = _item.get("firm") or _item.get("name")
        if not _nm or _nm not in _valid:
            continue
        _conf = str(_item.get("confidence", "low")).lower()
        if _rank.get(_conf, 0) < _threshold:
            continue
        _out[_nm] = {"reason": _item.get("reason", ""), "confidence": _conf}
    return _out


def _explain_portfolio(recommended_portfolio: list, web_grounding: dict = None,
                       diagnostics: dict = None) -> dict:
    """LLM-as-TRANSLATOR: phrase each stock's deterministic reason-trace into
    plain English + one portfolio-level paragraph. The model is handed the
    facts (score, rank, sector role, why-chosen) and may ONLY phrase them.
    It cannot invent a reason or alter selection — the trace is authoritative.

    Returns {ticker: one_line_explanation, "_portfolio": paragraph}.
    Fail-safe: on any error, returns trace-derived template strings (no LLM),
    so an explanation always exists and is always true even if the model fails.
    """
    _SLOT = {
        "cap_quota_large": "held to meet your large-cap floor",
        "cap_quota_mid": "held to meet your mid-cap floor",
        "breadth": "the first holding in this sector",
        "conviction": "a specialist pick",
        "free": "selected on merit",
    }

    def _template(s):
        """Always-true, trace-derived. Never says 'diversifier'."""
        t = s.get("_trace", {}) or {}
        sec, depth = t.get("sector", "?"), t.get("sector_depth", "?")
        passed, failed = t.get("passed") or [], t.get("failed") or []
        n_app = len(t.get("applicable") or []) or 5

        if t.get("gate_cleared") == "conviction_sleeve":
            fw = t.get("conviction_framework", "your chosen framework")
            return (f"#{t.get('conviction_rank', '?')} of {depth} in {sec} on {fw}. "
                    f"Fails {', '.join(failed) or 'nothing'} — you told us that is "
                    f"the trade-off you accept.")

        base = (f"Passed {len(passed)} of the {n_app} frameworks that apply to it; "
                f"ranked #{t.get('rank_in_sector', '?')} of {depth} in {sec}")
        if t.get("abstained"):
            base += (f". {', '.join(t['abstained']).title()} abstains on this "
                     f"business model rather than mis-scoring it")
        return f"{base} — {_SLOT.get(t.get('slot_type'), 'selected on merit')}."

    # Always-true fallback first
    out = {s["ticker"]: _template(s) for s in recommended_portfolio}
    # Passed in, not fished out of session state. This function is now a pure
    # transform of (portfolio, grounding, diagnostics) -> explanations, which
    # means it can be unit-tested and cannot silently render '?' on a page rerun
    # where the selector never ran.
    _diag = diagnostics or {}
    out["_portfolio"] = (
        f"These {len(recommended_portfolio)} holdings were chosen deterministically from "
        f"{_diag.get('pool_size', '?')} stocks that cleared your "
        f"{_diag.get('min_acceptable_score', '?')}+ score gate. Each was ranked against "
        f"its own sector on the five frameworks, weighted by the philosophy you chose. "
        f"IPS constraints — sector caps, cap-tier floors, the single-stock limit — were "
        f"satisfied before any discretionary slot was filled."
    )

    # Build the facts payload for the translator
    _facts = []
    for s in recommended_portfolio:
        t = s.get("_trace", {}) or {}
        # Pass the WHOLE trace. Fifteen holdings now carry fifteen genuinely
        # different fact-sets, which is why the model stops writing "diversifier"
        # fifteen times. It receives different inputs, so it writes different
        # sentences. No prompt engineering required.
        _facts.append({
            "ticker": s["ticker"], "name": s.get("name", ""),
            "score": s.get("score"), "pe": s.get("pe"), "roe_pct": s.get("roe_pct"),
            **t,
        })

    _sector_ctx = ""
    if web_grounding and web_grounding.get("sector_outlook"):
        _sector_ctx = f"\nSECTOR CONTEXT (for flavour only, do not contradict the facts):\n{web_grounding.get('sector_outlook')}"

    _prompt = (
        "You are a TRANSLATOR, not an analyst. Below are FACTS about why a deterministic "
        "engine selected each stock. Phrase each into ONE plain-English sentence a layman "
        "understands. You MUST NOT invent reasons, add opinions, or contradict the facts — "
        "only rephrase what is given. Then write ONE short paragraph (2-3 sentences) on how "
        "the set works together as a diversified whole.\n\n"
        f"FACTS:\n{json.dumps(_facts, indent=1)}{_sector_ctx}\n\n"
        'Respond ONLY as JSON (no markdown): {"explanations": {"<ticker>": "<one sentence>", ...}, '
        '"portfolio": "<paragraph>"}. Every ticker in FACTS must appear in explanations.'
    )

    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        _raw = None
        for _m in FREE_MODELS:
            try:
                _resp = client.models.generate_content(model=_m, contents=_prompt)
                _raw = (_resp.text or "").strip()
                if _raw:
                    break
            except Exception:
                continue
        if not _raw:
            return out
        _raw = _raw.replace("```json", "").replace("```", "").strip()
        _parsed = json.loads(_raw)
        _exp = _parsed.get("explanations", {})
        for s in recommended_portfolio:
            if s["ticker"] in _exp and _exp[s["ticker"]]:
                out[s["ticker"]] = str(_exp[s["ticker"]])
        if _parsed.get("portfolio"):
            out["_portfolio"] = str(_parsed["portfolio"])
    except Exception:
        pass  # keep the always-true templates
    return out


TOOL_STATUS_MESSAGES = {
    "resolve_stock": "🔍 Resolving company name...",
    "get_stock_data": "📊 Fetching live market data...",
    "get_csv_financial_data": "📋 Reading framework scores...",
    "get_historical_trends": "📈 Analyzing growth trends...",
    "get_financial_statements": "📑 Pulling financial statements...",
    "get_price_history": "📉 Checking price history...",
    "get_analyst_recommendations": "🎯 Checking analyst consensus...",
    "get_stock_news": "📰 Scanning recent headlines...",
    "get_ownership_info": "👥 Checking ownership structure...",
    "get_dividend_history": "💰 Reviewing dividend history...",
    "calculate_graham_value": "🧮 Computing intrinsic value...",
    "find_investments": "🔎 Screening 4,500+ stocks...",
    "show_stock_chart": "📊 Rendering price chart...",
    "get_macro_context": "🌍 Checking macro environment...",
    "get_sip_candidates": "🏗️ Building candidate pool...",
    "register_portfolio": "💾 Registering portfolio...",
    "navigate_to": "🧭 Opening feature...",
    "search_book": "📚 Consulting investment principles...",
    "get_web_context": "🌐 Searching the web for recent context...",
    "calculator": "🧮 Crunching numbers...",
}


# ──────────────────────────────────────────────
# TOOLS REGISTRY
# ──────────────────────────────────────────────
TOOLS = [
    search_book,
    get_stock_data,
    calculator,
    get_historical_trends,
    get_financial_statements,
    get_price_history,
    get_analyst_recommendations,
    get_stock_news,
    get_ownership_info,
    get_dividend_history,
    calculate_graham_value,
    find_investments,
    show_stock_chart,
    get_csv_financial_data,
    get_macro_context,
    get_sip_candidates,
    register_portfolio,
    navigate_to,
    resolve_stock,
    get_web_context,
]

tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "calculator": calculator,
    "get_historical_trends": get_historical_trends,
    "get_financial_statements": get_financial_statements,
    "get_price_history": get_price_history,
    "get_analyst_recommendations": get_analyst_recommendations,
    "get_stock_news": get_stock_news,
    "get_ownership_info": get_ownership_info,
    "get_dividend_history": get_dividend_history,
    "calculate_graham_value": calculate_graham_value,
    "find_investments": find_investments,
    "show_stock_chart": show_stock_chart,
    "get_csv_financial_data": get_csv_financial_data,
    "get_macro_context": get_macro_context,
    "get_sip_candidates": get_sip_candidates,
    "register_portfolio": register_portfolio,
    "navigate_to": navigate_to,
    "resolve_stock": resolve_stock,
    "get_web_context": get_web_context,
}


# ──────────────────────────────────────────────
# FALLBACK ROUTER
# ──────────────────────────────────────────────
def fallback_router(prompt):
    """Deterministic routing engine that triggers when the LLM is offline."""
    prompt_lower = prompt.lower()
    response_blocks = []

    potential_tickers = re.findall(r'\b[A-Z]{1,6}(?:\.NS)?\b', prompt)

    if "mahindra" in prompt_lower: potential_tickers.append("M&M.NS")
    if "apple" in prompt_lower: potential_tickers.append("AAPL")
    if "reliance" in prompt_lower or "ril" in prompt_lower: potential_tickers.append("RELIANCE.NS")

    tickers_to_check = list(set(potential_tickers))
    valid_stock_found = False

    for ticker in tickers_to_check:
        if ticker in ["I", "A", "THE", "WHAT", "WHY", "HOW", "IS", "YES", "NO"]:
            continue

        resolved = _resolve_ticker(ticker)
        data = get_stock_data(resolved)
        if "error" not in data:
            valid_stock_found = True
            table = f"### 📊 Auto-Fetched Data for {data.get('symbol', ticker)}\n"
            table += "| Metric | Value |\n| :--- | :--- |\n"
            table += f"| **Price** | {data.get('currency', '')} {data.get('current_price', 'N/A')} |\n"
            table += f"| **P/E Ratio** | {data.get('pe_ratio', 'N/A')} |\n"
            table += f"| **P/B Ratio** | {data.get('price_to_book', 'N/A')} |\n"
            table += f"| **ROE** | {round(data.get('return_on_equity', 0) * 100, 2) if data.get('return_on_equity') else 'N/A'}% |\n\n"
            response_blocks.append(table)

    book_keywords = ["graham", "greenblatt", "dorsey", "moat", "margin", "safety", "value", "formula", "rule"]
    if not valid_stock_found or any(kw in prompt_lower for kw in book_keywords):
        book_data = search_book(prompt)
        if "error" not in book_data:
            response_blocks.append("### 📚 Auto-Fetched Knowledge Base Passages\n")
            for p in book_data["passages"].split("\n\n"):
                response_blocks.append(f"> {p}\n\n")

    if not response_blocks:
        return "❌ *Fallback System:* Could not identify a valid ticker or knowledge base match from the prompt syntax."

    return "".join(response_blocks)


# ──────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────

import datetime
current_date = datetime.date.today().strftime("%B %Y")

SYSTEM_INSTRUCTION = f"""You are a highly structured Quantitative Investment Committee acting as a single agent.
CURRENT DATE: {current_date}

Your knowledge base consists of five scoring frameworks plus three qualitative reasoning books:
SCORING FRAMEWORKS (pre-computed, deterministic):
1. Benjamin Graham (Defensive Value, Margin of Safety) — defensive_score X/8, enterprising_score X/5
2. Joel Greenblatt (The Magic Formula, Capital Efficiency) — score X/10, universe-ranked
3. Pat Dorsey + Warren Buffett (Economic Moats, Management Quality) — dorsey_buffett_score X/10
4. Historical Trajectory (1-Year Momentum & Growth) — pass/fail
5. Peter Lynch (PEG, Category-Branching) — lynch_score X/10, category: slow_grower/stalwart/fast_grower/cyclical/turnaround/asset_play
QUALITY GATE: Schilit + Mulford (manipulation_score X/10 inverted, cashflow_quality X/5)
QUALITATIVE BOOKS (for reasoning, no scored columns):
- Howard Marks — second-level thinking, cycles, risk as permanent loss, contrarianism, patient opportunism
- Phil Fisher — 15 qualitative points, scuttlebutt, shake-down opportunities, when to sell (only 3 reasons)
- Seth Klarman — margin of safety as loss-avoidance, conservative valuation, catalysts, asymmetric risk-reward

You have 11 tools available. Pick the right combination for each question — you can call multiple tools in sequence.
1. search_book — Search The Intelligent Investor and other loaded books for Graham/Greenblatt/Dorsey investment philosophy. Use for conceptual or philosophical investing questions.
2. get_stock_data — Get current snapshot: price, P/E, P/B, market cap, dividend yield, 52-week range, sector. Use for quick overviews and valuation ratios.
3. calculator — Evaluate a math expression. Use for any arithmetic.
4. get_historical_trends — Get 1-year YoY trends for Revenue, Net Income, and Debt. Use for the Trajectory framework evaluation.
5. get_financial_statements — Get 4 years of income statement, balance sheet, OR cash flow data. Call with statement='income', 'balance', or 'cashflow'. You can call this multiple times with different statement types.
6. get_price_history — Get historical price performance over 1mo/3mo/6mo/1y/2y/5y. Returns total return, high/low, moving averages, and volatility.
7. get_analyst_recommendations — Get analyst buy/hold/sell ratings and consensus price targets.
8. get_stock_news — Get recent news headlines about a company.
9. get_ownership_info — Get major shareholders, institutional holders, and insider transactions.
10. get_dividend_history — Get complete dividend payment history, annual totals, growth rate, and yield.
11. calculate_graham_value — Compute Grahams intrinsic value formula (V = EPS x (8.5 + 2g) x 4.4/Y) and margin of safety.
12. find_investments — Screen ~4500 Indian stocks (NSE + BSE) from a pre-scored universe against ALL 5 frameworks. Returns three tiers: Perfect Consensus (5/5 pass), Strong Consensus (4/5 pass), and Moderate Consensus (3/5 pass), top 10 each. Use when the user asks to find, discover, or recommend stocks, or wants investment ideas. Call with market='india' or 'all'.
13. show_stock_chart — Renders a visual 13-month line chart of a stock's closing price directly in the UI. Use this whenever the user asks for a chart, graph, or visual trajectory.
14. get_csv_financial_data — Reads the pre-scored universe database for a specific ticker to get proprietary framework scores (Graham, Greenblatt, Dorsey, Trajectory pass/fail flags).
15. get_macro_context — Gets the sector and 5-day performance of the broader market (Nifty 50) to gauge macro momentum versus asset momentum.
16. get_sip_candidates — Build a SIP portfolio from a builder profile. Takes sip_amount, time_horizon, investor_type, review_freq, and avoid_sectors (a JSON string list of sector names to exclude, e.g. '["Energy"]'; pass '[]' for no exclusions). Returns pre-filtered candidates with a min/max stock count range. You decide the exact count based on candidate quality.
17. register_portfolio — After presenting your finalized SIP portfolio, call this to register it for saving. Pass portfolio_name, investor_type, sip_amount, time_horizon, review_days (integer), stocks_json (JSON string list with ticker/name/sector/allocation_pct per item), portfolio_profile (JSON string of the full builder profile), target_amount (number, 0 if no goal), and target_date (ISO date string, empty if no goal). ALWAYS call this after presenting the final portfolio table.
18. navigate_to — Open a platform feature for the client. Pass view as one of: builder, import, portfolios, watchlist, backtest, settings. Use when they want to ACT: "build me a portfolio" → navigate_to("builder"), "I have stocks on Zerodha" → navigate_to("import"), "show my portfolios" → navigate_to("portfolios"), "any picks this week" → navigate_to("watchlist"), "does this work" → navigate_to("backtest"). Do NOT navigate when they are just asking about a feature — explain first, then offer.
19. resolve_stock — Resolve a company name to its exact ticker. Call this FIRST when the client mentions a stock by name or partial name (e.g. "infosys", "hdfc bank", "capacite") instead of a full ticker (INFY.NS). Returns the resolved ticker if unambiguous, or shows the client disambiguation buttons if multiple matches exist. If the status is "ambiguous", STOP and tell the client you have shown them options to choose from — do not proceed with analysis until they pick.
20. get_web_context — Search the web for recent real-world context. Use for: company-specific due diligence (news, regulatory actions, management changes, investigations), sector headwinds/tailwinds, macro context (inflation, RBI policy, commodity cycles). MANDATORY for every stock with a positive verdict (STRONG BUY, BUY, CONDITIONAL BUY) and every shortlisted candidate during portfolio construction. Pass a specific, targeted query — not a vague ask.

SIP PORTFOLIO PROTOCOL:
Portfolio building uses the embedded Builder form (🏗️ Build Portfolio sidebar button). When you receive a message starting with [BUILDER_PROFILE], you must execute a strict 2-Phase process.

PHASE 1: DRAFT & INTERROGATE (DO NOT SHOW THE PORTFOLIO YET)
1. Call get_sip_candidates with the profile parameters.
2. Silently construct a "V1" portfolio in your mind. Do NOT output a table, do NOT list the stocks, and do NOT call register_portfolio.
3. The candidates already contain "web_grounding" with live macro, tax, and sector data. Use this to inform your V1 draft — no separate web search needed here.
4. Analyze the trade-offs in the portfolio. Review "near_misses" — stocks that ranked highly and did not make it, with the constraint that blocked each one.
5. Output a brief, layman-friendly summary of the strategy you are considering.
6. Ask the user 1 to 3 targeted questions to refine the build.
   - RULE: Speak to them as a layman. Do NOT use jargon like "Graham", "Dorsey", "moat", "beta", or "PE expansion".
   - HARD RULE: You do NOT select stocks, add stocks, remove stocks, or change the count. `recommended_portfolio` is complete and final. There is no backfilling and no "next-best from the pool" — that instruction previously licensed you to invent tickers. If the portfolio holds fewer stocks than you expected, `selection_warnings` says exactly why (the gate, the SIP, or an infeasible cap-tier quota). Report that reason honestly. A 9-stock portfolio the user understands beats a 15-stock portfolio padded with names nobody chose.
   - Example (Near Miss): "A highly profitable company ranked #1 in Financial Services, but your sector cap was already full at 3 holdings. Worth revisiting if you'd relax that."
   - Example (Fringe Candidate): "I found a highly profitable company that fits your goals perfectly, but it's in the Energy sector which you asked to avoid. Are you open to making an exception for a top-tier performer?"
   - Example (Risk Trade-off): "To hit your target, we need a bit more growth. Would you prefer adding a fast-growing but bumpier stock, or stick to steady, slow-moving giants?"
7. Stop and wait for the user's reply.

PHASE 2: FINALIZE & REGISTER (TRIGGERED ONLY AFTER USER REPLIES)
1. When the user answers your questions, finalize the stock selection.
2. Output the final portfolio table. NOW you may explain the quantitative reasoning using book philosophies (Graham/Greenblatt/Dorsey) to educate them on why these picks were made.
3. CRITICAL: Generate this textual explanation FIRST.
4. Call register_portfolio with all fields including portfolio_profile, target_amount, and target_date. CRITICAL: Use the `decision_context` parameter to summarize the user's answers to your Phase 1 questions so the system remembers their accepted trade-offs (e.g. "User accepted volatility in Industrials for higher growth"). Do not ask for permission to save, just call the tool. YOU MUST CALL THE register_portfolio TOOL — do not just write text saying "portfolio saved." The save button ONLY appears when the tool is called. If you skip the tool call, the user CANNOT save their portfolio.

PORTFOLIO EXPANSION PROTOCOL:
When you receive a message starting with [EXPANSION], a user has increased their SIP and needs more stocks added to an existing portfolio:
1. Call get_sip_candidates with their profile parameters and the NEW, larger SIP amount.
2. A larger SIP raises `affordable_n`, so the deterministic selector returns a larger portfolio under the same IPS. The additions are simply the tickers in `recommended_portfolio` that are not already held.
3. Do NOT choose which stocks to add. The correlation tiebreak, the sector caps and the cap-tier quotas were all applied by the selector, over the full portfolio, at the new size.
4. Present the additions, using each holding's `_trace` to say why the selector placed it — its rank within its sector, the frameworks it passed, and whether it filled a cap-tier quota or an empty sector.
5. Call register_portfolio with the FULL stock list (existing + new) to update the portfolio.
   Use the existing portfolio name. The system will handle the update.

If someone asks to build a portfolio WITHOUT a [BUILDER_PROFILE] prefix, direct them to click the 🏗️ Build Portfolio button in the sidebar. If they insist or provide enough info inline, you may proceed by mapping their inputs to the profile parameters.

INVESTMENT ANALYSIS FRAMEWORK (Reilly & Brown — Investment Analysis & Portfolio Management):
When analyzing any individual stock, follow the top-down approach (Ch 9):

Step 1 — MACRO ENVIRONMENT:
  Call get_web_context for macro context FIRST ("India GDP growth inflation RBI monetary policy 2026").
  Assess: What is the current business cycle phase? (recovery → expansion → peak → contraction)
  What are interest rates doing? (rising = headwind for growth stocks, falling = tailwind)
  Is the yield curve normal, flat, or inverted? (inverted = recession warning within 12-18 months)

Step 2 — SECTOR ASSESSMENT:
  Where is this industry in its life cycle? (Ch 9: pioneering → rapid growth → mature → decline)
  Porter's Five Forces: rivalry intensity, entry barriers, substitutes, supplier power, buyer power.
  Call get_web_context for sector-specific outlook.
  Sector rotation: does the current business cycle phase favor or hurt this sector?
  (End of recession → financials. Recovery → consumer durables. Expansion → capital goods. Pre-downturn → consumer staples.)

Step 3 — COMPANY ANALYSIS:
  Apply the 5 scoring frameworks + quality gate (deterministic, already computed).
  Cross-check valuation using Ch 8 methods where data permits:
  - Is the PE justified by growth rate? (PEG ratio — Lynch dimension)
  - How does ROE compare to cost of equity? (If ROE < cost of equity, the company destroys value regardless of growth)
  - Is the company a "growth company" or a "growth STOCK"? (Ch 5: a great company at a bad price is a bad investment)
  - Check owner earnings (Buffett: net income + depreciation - capex) vs reported earnings

Step 4 — BEHAVIORAL BIAS SELF-AUDIT (Ch 5):
  Before finalizing your thesis, explicitly check:
  - ANCHORING: Am I fixated on a past price, analyst target, or initial impression?
  - CONFIRMATION BIAS: Did I only seek evidence supporting my thesis? What contradicts it?
  - REPRESENTATIVENESS: Am I confusing "good company" with "good stock"?
  - HERDING: Is this a consensus favorite? If so, the edge is already priced in.
  - OVERCONFIDENCE: What is the single biggest risk to this thesis? State it clearly.
  - THE CONSENSUS TEST (Ch 5): Superior analysis requires being BOTH correct AND different from consensus.
    If your view matches what every analyst already thinks, there is no informational advantage.

SIP PORTFOLIO CONSTRUCTION — YOU DO NOT CONSTRUCT PORTFOLIOS.
The portfolio in `recommended_portfolio` is built by deterministic code. You add
no stock, remove no stock, reorder nothing. Your only job is to translate each
holding's `_trace` into plain English.

Every holding carries a `_trace` with two ORTHOGONAL facts. Use both:

  gate_cleared — HOW it passed the user's score gate
    "merit"                 passed N of 5 frameworks outright.
    "abstention_adjusted"   a framework structurally cannot evaluate it.
                            Greenblatt's formula uses ROIC and earnings yield,
                            meaningless for a levered balance sheet — he says
                            himself not to apply it to financials or utilities.
                            So it ABSTAINS. The stock is judged on 4, not 5, and
                            held to the same FRACTION. Say so plainly; it is a
                            strength of the method, not an excuse.
    "conviction_sleeve"     the user chose a wider gate, and this stock is a
                            TOP-DECILE SPECIALIST in the framework they said
                            matters most. It fails others. Name which, and name
                            the framework that earned it its place. Use
                            `conviction_rank` — the rank that CHOSE it — never
                            `rank_in_sector`, which is the composite rank that
                            buried it.

  slot_type — WHY it got a seat
    cap_quota_large / cap_quota_mid   reserved to meet an IPS cap-tier floor
    breadth                           first name in an otherwise empty sector
    conviction                        the sleeve above
    free                              pure merit order

`rank_in_sector`, `sector_depth` and `conviction_rank` count ONLY stocks that
cleared THIS user's score gate. They are not universe ranks. Say "of the 33
Basic Materials names that met your 3+ bar", never "of 33 in the market".

Never write "chosen as a diversifier" or "sector filler". Fifteen holdings have
fifteen different reasons and the trace contains all of them.

`near_misses` lists stocks that ranked highly and did not make it, with the
binding constraint. Mention one or two. Rejections build more trust than picks.

The math: adding stock i to a portfolio reduces risk when ρ(i, portfolio) < σ(i)/σ(portfolio).
The lower the correlation, the greater the diversification benefit. Same-sector stocks typically have ρ > 0.6.
Cross-sector stocks typically have ρ = 0.2-0.4. THIS is why sector diversification matters — not as a rule of thumb, but because same-sector returns are genuinely correlated.

TOOL SELECTION RULES:
- TICKER RESOLUTION: If the client uses a company name (not a full .NS/.BO ticker), call resolve_stock FIRST. If it returns "ambiguous", stop and wait — the client will pick from buttons. If it returns "resolved", proceed with the resolved ticker. If they already gave a ticker like RELIANCE.NS, skip resolve_stock.
- For a comprehensive stock analysis: call get_stock_data + get_historical_trends + get_financial_statements (income) + calculate_graham_value + search_book.
- For "is this stock a good investment" type questions: use at minimum get_stock_data + get_historical_trends + calculate_graham_value + search_book.
- For "how has X performed" questions: use get_price_history.
- For "any news about X" questions: use get_stock_news.
- For "what do analysts think" questions: use get_analyst_recommendations.
- For "does X pay dividends" or dividend history questions: use get_dividend_history.
- For "who owns X" or insider activity questions: use get_ownership_info.
- For "find me stocks" or "recommend stocks" or "where should I invest" or "best stocks" or "screen": call find_investments, THEN call search_book to explain WHY each investment tier is attractive. Follow the SCREENING OUTPUT PROTOCOL below.
- When comparing two stocks: call the relevant tools for BOTH tickers and synthesize.
- Always prefer calling a tool over guessing. If in doubt, call it.
- For "show me a chart" or "graph" questions: use show_stock_chart.

SCREENING OUTPUT PROTOCOL (use ONLY when find_investments is called):
After calling find_investments, you MUST also call search_book with queries like "margin of safety value investing" and "economic moat competitive advantage" and "magic formula return on capital" to ground your explanation in the actual books. Then present results as follows:

### Perfect Consensus (4/4 Frameworks Pass)
Show the top 3 stocks in a table with key metrics. Then explain:
- WHY this tier represents the strongest buy signal, citing specific concepts from the books (Graham margin of safety, Greenblatt capital efficiency, Dorsey moat durability)
- What kind of returns and risk profile an investor should expect (long-term compounding, downside protection)
- Use specific philosophy from the book passages you retrieved

### Strong Consensus (3/4 Frameworks Pass)
Show the top 3 stocks in a table with key metrics AND which framework they failed. Then explain:
- What the failing framework means as a specific risk (e.g., failing Graham means overvalued despite quality; failing Trajectory means growth is slowing)
- Why 3/4 is still a strong signal and what kind of investor this suits
- Ground the explanation in book concepts

### Moderate Consensus (2/4 Frameworks Pass)
Show the top 3 stocks in a table with key metrics AND which 2 frameworks they passed and which 2 they failed. Then explain:
- What combination of passes and fails this represents (e.g., passes Graham + Trajectory = cheap and growing but low quality; passes Greenblatt + Dorsey = high quality but expensive)
- Why this tier requires more caution and due diligence, but can still be attractive for investors with a specific thesis
- What additional research or conditions would strengthen conviction
- Ground the explanation in book concepts

If no stocks pass 4/4, say so clearly. If fewer than 3 pass in a tier, show however many exist.

CRITICAL RULES:
- For full analyses, you MUST call get_stock_data AND get_historical_trends.
- You MUST evaluate the thresholds silently before generating the output.
- Do NOT "think out loud" or correct yourself in the output.
- Do NOT copy the instruction text into your response.
- Each framework MUST be evaluated using ONLY its own criteria. Cross-contamination between frameworks is an error.
SYNTHESIS PROTOCOL:
- If the local CSV tool returns framework flags (e.g., graham_pass = True/False), you MUST query search_book for the theoretical definition of that framework (e.g., 'Margin of Safety').
- Cross-reference company metrics with get_macro_context to determine if the company is outperforming or being dragged by market beta.

FRAMEWORK SCORING (5 dimensions, 0-5 composite):
Framework pass/fail and spectrum scores are PRE-COMPUTED in the universe CSV. Do NOT re-derive them.
When you call get_csv_financial_data, the response includes:
- _verdict_tier: the deterministic verdict (STRONG BUY / BUY / CONDITIONAL BUY / WATCH / AVOID / SELL)
- _verdict_reason: a short explanation of why this verdict
- _deep_metrics_formatted: all spectrum scores and key metrics, structured for your analysis
- _book_reasoning: investment principles from Marks, Fisher, and Klarman matched to this stock's specific pass/fail pattern
- _pass_pattern_meaning: (for 3/5 stocks only) what the specific framework disagreement MEANS

VERDICT PROTOCOL — MANDATORY:
The verdict tier in _verdict_tier is DETERMINISTIC. You MUST use it. NEVER override it.
Your job is to EXPLAIN the verdict, not decide it. The rules are:
Scores are "N of M", where M is how many frameworks APPLY to that business, not always 5.
Greenblatt abstains on financials and utilities (his own instruction); Lynch abstains when
the business fits none of his six categories. An abstaining framework is NOT a failure —
never describe it as one, and never say a stock "failed" a test that was not applied. The
tiers below are stated in 5-framework equivalents; the engine has already normalised for M.
- STRONG BUY (all 5 of 5 + clean): Every framework agrees, quality gate passes, no red flags.
  Requires all five to apply — a stock where any framework abstains tops out at BUY.
- BUY (4-of-5 equivalent, or 5/5 with borderline manipulation): Strong consensus with one minor gap.
- CONDITIONAL BUY (3-of-5 equivalent + quality pass): Frameworks disagree. YOU must explain the thesis AND the invalidation conditions. Read _pass_pattern_meaning and _book_reasoning for guidance.
- WATCH (2-of-5 equivalent with Graham or Dorsey anchor): Interesting but insufficient evidence. State what would need to change for an upgrade.
- AVOID (2-of-5 equivalent without Graham or Dorsey): No price floor, no quality anchor. The pass pattern is fragile.
- SELL (0-1 of 5 equivalent without Graham, OR quality gate failed): No investment thesis, or accounting red flags.

HOW TO EXPLAIN EACH TIER:
For STRONG BUY and BUY:
- Lead with the verdict badge and reason
- Walk through the spectrum scores showing WHY each framework passes
- Quantify the margin of safety (Graham intrinsic value vs. current price)
- Identify risk-reward asymmetry (bounded downside, upside from what?)
- Conclude with a Committee Note on position sizing and risk management

For CONDITIONAL BUY (the key innovation — this is where you add the most value):
- Lead with the verdict badge and reason
- State the THESIS: what story does the passing frameworks tell?
- State the ANTI-THESIS: what story does the failing frameworks tell?
- Use _book_reasoning principles to weigh these against each other
- REAL-WORLD CHECK: You MUST call get_web_context for this company. Incorporate recent developments into your thesis — a CONDITIONAL BUY without real-world context is not defensible. If recent news strengthens the thesis, say so. If it weakens it, say so and adjust monitoring triggers accordingly.
- Identify a CATALYST: what event or development would validate the thesis?
- State INVALIDATION CONDITIONS: what would change this to AVOID or SELL?
- Conclude with a Committee Note on position sizing (smaller than BUY), and what to monitor

For WATCH:
- Lead with the verdict badge and reason
- Briefly explain what's interesting (the anchor framework)
- Clearly state what's missing (the failing frameworks)
- Recommend patience: "Wait for [specific condition] before considering entry"
- Do NOT recommend purchase. WATCH means the bat stays on your shoulder.

For AVOID:
- Lead with the verdict badge and reason
- Explain what the passing frameworks see (efficiency, momentum) and why it's not enough
- Explain the specific risks from lacking both price protection and quality protection

For SELL:
- Lead with the verdict badge and reason
- If quality gate failed: explain which manipulation flags triggered (use spectrum scores)
- If score 0-1: explain that no credible investment thesis exists
- Be direct. Do not soften a SELL verdict.

VERIFICATION PROTOCOL (Mandatory for STRONG BUY, BUY, CONDITIONAL BUY):
Before finalizing your analysis for any stock with a positive verdict:
Step A: Call get_stock_data for that ticker. Read the earnings_quality block.
        If anomaly_flags contains ANY "RED FLAG" → note this prominently. The deterministic verdict stands (quality gate already accounts for this), but you MUST flag it in your explanation.
Step B: If anything looks unusual in the data (P/E < 3, ROE > 50%, dramatic YoY swings), call search_book with a relevant query to check for value trap patterns.
Step C: Call get_web_context for the specific company (e.g. "Infosys recent news SEBI regulatory 2026"). If the results reveal material negative developments (investigation, fraud allegations, management exodus, regulatory action), you MUST note this prominently in your Committee Note. The deterministic verdict stands, but the client deserves to know the real-world context. For CONDITIONAL BUY, web context is what makes the thesis defensible — without it, the verdict is incomplete.
VERIFICATION DOES NOT APPLY TO: Simple data lookups, WATCH/AVOID/SELL verdicts, conversational messages.

EXECUTION PROTOCOL:
You are an intelligent, conversational, and highly analytical Quantitative Investment Committee. You are free from rigid formatting templates, but BOUND by the deterministic verdict system.

Follow these core behavioral directives:
1. The Tiered Verdict (No Waffling): Open your analysis with the verdict tier, using the exact tier name from _verdict_tier. Display it prominently. Never use YES/NO — always use the 6-tier system.
2. Fluid Integration: Weave spectrum scores (e.g., "Graham Defensive 7/8 — fails only on dividend record") naturally into your prose. Explain the WHY behind the numbers. Use _deep_metrics_formatted as your data source.
3. Book-Grounded Reasoning: For CONDITIONAL BUY especially, use the principles from _book_reasoning to frame your thesis. Name the author and concept (e.g., "Marks's cycle awareness suggests..." or "Klarman would demand a catalyst here...").
4. Dynamic Formatting: Use markdown headers, bullet points, and bold text organically. The verdict badge should be the first thing the user sees.
5. Committee Note: Conclude with actionable risk management advice grounded in the book principles. For CONDITIONAL BUY, this MUST include monitoring triggers.

FUND MANAGER ROLE:
You are not a stock screener. You are the client's fund manager. A fund manager handles four things:
1. Investment Policy — understand the client (goals, risk appetite, constraints, life stage)
2. Asset Allocation — how to distribute capital across securities
3. Security Selection — pick specific stocks (your 5 frameworks + quality gate)
4. Portfolio Monitoring — track performance, rebalance toward targets, surface problems early

Most conversations are about step 3 (stock analysis). But when a client asks about getting started, managing money, how things work, or whether they can trust you — they are asking about steps 1, 2, or 4. Meet them where they are. Do not redirect everything to stock analysis.

PLATFORM FEATURES:
You have a navigate_to tool that opens any feature directly. Use it when the client's intent is to ACT. When their intent is to UNDERSTAND, explain first, then offer to open it.

Builder (navigate_to "builder"): Guided portfolio construction — SIP amount, horizon, risk tolerance, goals, sector preferences. You then select stocks and register the portfolio. Suggest for: new clients, "help me invest", "build a portfolio", "I have X per month", "where do I start."

Import (navigate_to "import"): Bring in existing holdings via Zerodha Kite CSV export or manual ticker entry. Suggest for: "I already have stocks", "I use Zerodha", "I have a demat account", "can I add my existing portfolio."

Portfolios (navigate_to "portfolios"): Dashboard of saved portfolios — performance, holdings, target vs actual allocation, alerts, reviews, SIP deployment, PDF reports. Suggest for: "how is my portfolio", "show my investments", "when is my next review", "deploy my SIP."

Watchlist (navigate_to "watchlist"): Track stocks without buying, paper portfolios for practice, and This Weeks Picks — a curated weekly buy list personalized to the clients profile. Suggest for: "watch this stock", "any recommendations this week", "I want to practice first", "what should I buy."

Backtest (navigate_to "backtest"): Historical simulation — does Kordents scoring actually predict returns? Quarterly backtests against Nifty 50. Suggest for: "does this actually work", "prove it", "whats your track record", "why should I believe the scores."

Settings (navigate_to "settings"): Profile name, email display, Telegram alerts setup. Suggest for: "change my name", "connect Telegram", "notifications."

EXPLAINING TRUST AND METHODOLOGY:
When a client questions credibility, methodology, or trustworthiness:
— Kordent scores ~4,500 NSE/BSE stocks using 5 frameworks extracted from 7 canonical investment books (Graham, Greenblatt, Dorsey, Buffett, Lynch, Schilit, Mulford).
— A quality gate (Schilit + Mulford) catches financial manipulation deterministically before any stock reaches the recommendation stage.
— Verdicts are deterministic and tiered (STRONG BUY through SELL). You explain them — you never override them.
— Three additional books (Marks, Fisher, Klarman) provide qualitative reasoning for nuanced cases.
— Everything is transparent: the client sees every score, every spectrum value, every pass/fail flag.
— Always offer a demonstration: "Pick a stock you know well — a company you work at, buy from, or follow — and I will show you exactly how the analysis works, with full transparency. Then you decide."
— If they ask about the backtest, offer to navigate there. Evidence over persuasion.

CLIENT CONTEXT:
__USER_CONTEXT__
Use this to inform your responses. If they have no portfolios, they are new — guide gently. If they have active portfolios, they are an existing client — focus on insights and next actions. Never recite this context unprompted. Never say "based on your profile" or "I can see you have." Just know it and respond accordingly.

CONVERSATIONAL PRINCIPLES:
1. You are a fund manager sitting across the table from your client. Warm, competent, direct. Not a help desk, not a chatbot, not a disclaimer machine.
2. Intent to act → do it or navigate. Do not describe features the client can read about. Open the page.
3. Intent to understand → explain with depth and conviction. Use search_book to ground answers in the actual texts.
4. Confused or overwhelmed → simplify, offer ONE concrete next step. "Lets start here" is better than listing five options.
5. Frustrated → acknowledge, do not become defensive. Fix the problem or clearly state what you cannot do.
6. IPOs, crypto, F&O, intraday → Kordent is built for long-term value investing. Acknowledge their interest without judgment. Explain the approach. Do not lecture.
7. "Compare X vs Y" → call tools for both tickers and synthesize. This is natural and expected.
8. Never say "I cannot do that" when a platform feature handles it. Navigate instead.
9. If you do not know something about the client, ask. A fund manager who guesses instead of asking is a bad fund manager.
"""

AUDITOR_SYSTEM_PROMPT = """You are the Chief Risk Officer and Auditor for an Investment Committee.
You are a truthful, disagreeable, first-principle thinker. Your sole job is to catch the Analyst making mistakes.

You receive THREE inputs:
1. The user's original query
2. The Analyst's draft response
3. Independent Earnings Quality Data — hard numbers YOU verify against

AUDIT CHECKLIST (use the Independent data, not the Analyst's claims):
1. For every stock where the Analyst issues STRONG BUY, BUY, or CONDITIONAL BUY: check if unusual_items_pct > 20%. If so, the positive verdict must be flagged.
2. For every stock where the Analyst issues STRONG BUY, BUY, or CONDITIONAL BUY: check if cash_conversion < 0.5. If so, the positive verdict must be flagged.
3. If the Independent data contains RED FLAG entries for a stock the Analyst recommended positively, but the Analyst did not mention or address those flags, the draft is invalid.
4. For CONDITIONAL BUY verdicts: verify the Analyst stated BOTH a thesis AND invalidation conditions. A CONDITIONAL BUY without invalidation conditions is incomplete.
5. For CONDITIONAL BUY verdicts: verify the Analyst incorporated recent real-world context (news, regulatory, sector developments) into the thesis. A CONDITIONAL BUY without real-world context is incomplete.
6. If Independent Earnings Quality Data is empty (no tickers found or no flags raised), the draft is likely safe on this dimension.

CRITICAL BYPASS RULES (Auto-Approve):
- If the Analyst is simply asking the user a question (such as the portfolio builder sequence), reply EXACTLY with: [APPROVED]
- If the Analyst issued a WATCH, AVOID, or SELL verdict or is simply conversing, reply EXACTLY with: [APPROVED]

If the Analyst's draft is fundamentally sound and no Independent data contradicts it, reply EXACTLY with: [APPROVED]
If the Independent data contradicts the Analyst's verdict, reply with: [REJECT] followed by which specific tickers failed quality checks and what the Analyst must change."""

# ──────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────
def intercept_and_rewrite_query(user_query: str) -> str:
    """
    Intercepts the layman question and translates it into strict technical 
    directives for the main execution agent using a fast model.
    """
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    
    router_prompt = f"""
    You are the Pre-Processing Routing Agent for a quantitative financial system.
    Translate this layman user query into a strict, step-by-step technical directive for the Execution Agent.

    The Execution Agent has tools: get_csv_financial_data, get_macro_context, search_book, get_stock_data, get_price_history, etc.

    User Query: "{user_query}"

    Identify the ticker symbol. Tell the agent EXACTLY which tools to use and what to cross-reference based on the query intent. 
    DO NOT ANSWER THE QUESTION. ONLY OUTPUT THE DIRECTIVE.
    """
    try:
        last_good = st.session_state.get("last_working_model")
        if last_good and last_good in FREE_MODELS:
            models_to_try = [last_good] + [m for m in FREE_MODELS if m != last_good]
        else:
            models_to_try = FREE_MODELS
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=router_prompt,
                )
                return f"SYSTEM DIRECTIVE (Translated Intent): {response.text}"
            except Exception as inner_e:
                error_msg = str(inner_e).upper()
                if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                    continue
                raise inner_e
        return user_query
    except Exception:
        return user_query


def sanitize_history(history):
    """Filters out malformed messages missing a role."""
    clean = []
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("role") in ["user", "model"]:
                clean.append(msg)
        else:
            if hasattr(msg, 'role') and msg.role in ["user", "model"]:
                clean.append(msg)
    return clean


def agent_turn(user_message, status_container=None):
    st.session_state._tool_status = status_container
    def _update_status(label):
        if status_container:
            try:
                status_container.update(label=label, state="running")
            except Exception:
                pass

    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    # Inject client context into system instruction
    _user_ctx = build_user_context()
    full_instruction = SYSTEM_INSTRUCTION.replace("__USER_CONTEXT__", _user_ctx)

    raw_history = st.session_state.get("chat_history", [])
    history = sanitize_history(raw_history)

    last_error = None
    # Prioritize the model that worked last turn
    last_good = st.session_state.get("last_working_model")
    if last_good and last_good in FREE_MODELS:
        models_to_try = [last_good] + [m for m in FREE_MODELS if m != last_good]
    else:
        models_to_try = FREE_MODELS
    for model_name in models_to_try:
        try:
            _update_status("🧠 Analyst preparing thesis...")
            # --- PHASE 1: ANALYST DRAFTS THESIS ---
            analyst_chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=full_instruction,
                    tools=TOOLS,
                ),
                history=history,
            )

            analyst_response = analyst_chat.send_message(user_message)
            all_text_parts = []

            def _extract_text(resp):
                """Get text from response even when function calls coexist."""
                try:
                    for part in resp.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
                except (AttributeError, IndexError, TypeError):
                    pass
                try:
                    return resp.text or ""
                except Exception:
                    return ""

            def _safe_function_calls(resp):
                """Extract function calls safely — returns [] if parts is None."""
                try:
                    return resp.function_calls or []
                except (AttributeError, TypeError):
                    return []

            while _safe_function_calls(analyst_response):
                text_chunk = _extract_text(analyst_response)
                if text_chunk:
                    all_text_parts.append(text_chunk)
                    
                function_responses = []
                for fc in _safe_function_calls(analyst_response):
                    _update_status(TOOL_STATUS_MESSAGES.get(fc.name, f"⚙️ Running {fc.name}..."))
                    if fc.name in tool_functions:
                        # Execute the tool function, then immediately sanitize the dictionary output
                        raw_tool_output = tool_functions[fc.name](**fc.args)
                        result = _sanitize_for_json(raw_tool_output)
                    else:
                        result = {"error": f"Unknown tool: {fc.name}"}
                        
                    function_responses.append(
                        types.Part.from_function_response(name=fc.name, response=result)
                    )
                # Send the sanitized parameters back to the chat manager
                _update_status("🧠 Analyst synthesizing results...")
                analyst_response = analyst_chat.send_message(function_responses)

            final_chunk = _extract_text(analyst_response)
            if final_chunk:
                all_text_parts.append(final_chunk)
            
            clean_parts = [p.strip() for p in all_text_parts if p.strip()]
            draft_text = "\n\n".join(clean_parts).strip()

            if not draft_text:
                recovery_prompt = (
                    "You successfully executed the register_portfolio tool, but you provided zero text to the user. "
                    "You MUST reply now with a stock-by-stock explanation of why you selected each company, "
                    "grounding your reasoning in the Graham, Greenblatt, and Dorsey frameworks. Do not output any more tool calls."
                )
                recovery_response = analyst_chat.send_message(recovery_prompt)
                draft_text = _extract_text(recovery_response).strip()

            # Builder flows: skip auditor — deterministic IPS guardrails sufficient
            if st.session_state.get("builder_profile"):
                audit_result = "[APPROVED] Builder flow — IPS guardrails active."
            else:
                # Builder: skip auditor (IPS guardrails are deterministic)
                _skip_auditor = bool(st.session_state.get("builder_profile"))
                if _skip_auditor:
                    audit_result = "[APPROVED]"
                _update_status("✅ IPS validated" if _skip_auditor else "🔒 Risk officer reviewing draft...")
                # --- PHASE 2: AUDITOR REVIEWS DRAFT (with independent data) -----
                NOISE_WORDS = {"PASS", "FAIL", "YES", "NO", "ROE", "EPS", "SIP",
                               "AND", "THE", "FOR", "NOT", "USE", "ALL", "WHY",
                               "HOW", "BUY", "TOP", "LOW", "HIGH", "CAP", "NET",
                               "YOY", "INR", "USD", "FY", "PE", "PB", "DE",
                               "SMA", "CAGR", "NAV", "IPO", "ETF", "PDF", "CSV"}
                mentioned_tickers = set(re.findall(r'\b[A-Z]{2,15}(?:\.NS|\.BO)?\b', draft_text))
                mentioned_tickers -= NOISE_WORDS
    
                quality_checks = {}
                for t in mentioned_tickers:
                    qc = get_earnings_quality_metrics(t)
                    if "error" not in qc and qc.get("anomaly_flags"):
                        quality_checks[t] = {
                            "cash_conversion": qc["cash_conversion_ratio"],
                            "unusual_items_pct": qc["unusual_items_pct_of_income"],
                            "flags": qc["anomaly_flags"],
                        }
    
                auditor_input = (
                    f"User Query: {user_message}\n\n"
                    f"Analyst Draft:\n{draft_text}\n\n"
                    f"Independent Earnings Quality Data:\n{json.dumps(quality_checks, indent=2)}"
                )
    
                if not _skip_auditor:
                    auditor_response = client.models.generate_content(
                        model=model_name,
                        contents=auditor_input,
                        config=types.GenerateContentConfig(system_instruction=AUDITOR_SYSTEM_PROMPT)
                    )
    
                    audit_result = auditor_response.text.strip()

            # --- PHASE 3: RESOLUTION ---
            if audit_result.startswith("[REJECT]"):
                # Force the Analyst to read the Auditor's rejection and rewrite
                _update_status("⚖️ Revising thesis per auditor feedback...")
                correction_prompt = f"The Chief Risk Officer REJECTED your draft with the following feedback:\n\n{audit_result}\n\nRewrite your entire analysis to comply with this feedback. CRITICAL: If you are building a portfolio, you MUST call the register_portfolio tool AGAIN with your updated stock list to overwrite the rejected database entry."
                final_response = analyst_chat.send_message(correction_prompt)
                
                # --- FIX: We must process tool calls during the correction phase too! ---
                corr_text_parts = []
                while _safe_function_calls(final_response):
                    text_chunk = _extract_text(final_response)
                    if text_chunk:
                        corr_text_parts.append(text_chunk)
                        
                    function_responses = []
                    for fc in _safe_function_calls(final_response):
                        _update_status(TOOL_STATUS_MESSAGES.get(fc.name, f"⚙️ Running {fc.name}..."))
                        if fc.name in tool_functions:
                            raw_tool_output = tool_functions[fc.name](**fc.args)
                            result = _sanitize_for_json(raw_tool_output)
                        else:
                            result = {"error": f"Unknown tool: {fc.name}"}
                            
                        function_responses.append(
                            types.Part.from_function_response(name=fc.name, response=result)
                        )
                    _update_status("🧠 Analyst synthesizing results...")
                    final_response = analyst_chat.send_message(function_responses)

                final_chunk = _extract_text(final_response)
                if final_chunk:
                    corr_text_parts.append(final_chunk)
                
                clean_corr_parts = [p.strip() for p in corr_text_parts if p.strip()]
                final_text = "\n\n".join(clean_corr_parts).strip()

                st.session_state.chat_history = analyst_chat.get_history()
                
                # Append an internal note to the UI so the user sees the system working
                st.session_state.last_working_model = model_name
                return f"*(Internal Audit Triggered: Adjusted thesis based on earnings quality)*\n\n{final_text}", model_name
            else:
                # Auditor approved
                st.session_state.chat_history = analyst_chat.get_history()
                st.session_state.last_working_model = model_name
                return draft_text, model_name

        except Exception as e:
            last_error = str(e)
            error_upper = last_error.upper()
            if any(err in error_upper for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                continue
            raise e

    raise Exception(f"All models rate-limited. Last error: {last_error}")



# ══════════════════════════════════════════════
# CHAT UI
# ══════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


USER_AVATAR = "👤"
AGENT_AVATAR = "logo.svg"

def _render_verdict_badge(text=None, stored_tier=None):
    """Render a colored verdict badge. Uses stored_tier (from tool call) when available.
    Falls back to text scanning ONLY if the text contains exactly ONE ticker pattern,
    which prevents false positives on builder responses (many tickers) and
    meta-responses like 'I provide STRONG BUY verdicts'."""
    tier = stored_tier
    if not tier and text:
        # Fallback for old messages without stored tier — require EXACTLY one ticker
        # (multi-ticker responses are builder/screener output, not single-stock verdicts)
        _ticker_hits = re.findall(r'\b[A-Z]{2,15}\.(?:NS|BO)\b', text)
        if len(set(_ticker_hits)) == 1:
            _upper = text.upper()
            for t in ["STRONG BUY", "CONDITIONAL BUY", "BUY", "WATCH", "AVOID", "SELL"]:
                if t in _upper:
                    tier = t
                    break
    if not tier:
        return None
    colors = verdict_engine.VERDICT_COLORS.get(tier, {})
    bg = colors.get("bg", "#555")
    fg = colors.get("text", "#FFF")
    emoji = verdict_engine.VERDICT_EMOJI.get(tier, "")
    st.markdown(
        f'<div style="display:inline-block;background:{bg};color:{fg};'
        f'padding:6px 18px;border-radius:6px;font-weight:700;font-size:1.1em;'
        f'margin-bottom:8px;letter-spacing:0.5px;">'
        f'{emoji} {tier}</div>',
        unsafe_allow_html=True,
    )
    return tier


if st.session_state.sb_view_mode == "chat":
    chat_area = st.container()

    # ── Flush picks made before sign-in ─────────────────────────────────────
    # Deliberately NOT inside either login handler. There are two (Google OAuth
    # and email/password) and a rule duplicated in two places is a rule that
    # will diverge. Both end in st.rerun(), so this runs on the next pass with
    # sb_user_id populated.
    if st.session_state.get("_screen_pending") and st.session_state.get("sb_user_id"):
        _p = universe_df[universe_df["ticker"].isin(st.session_state._screen_pending)]
        st.session_state._screen_pending = None
        if not _p.empty:
            _add_to_watchlist(_p)
    elif st.session_state.get("sb_user_id") and st.session_state.get("_screen_open"):
        pass  # already signed in; table stays open

    # ── Screener: one button, deterministic list, no LLM in the list path ──
    if not st.session_state.messages and "pending_prompt" not in st.session_state:
        st.markdown("")

        if not st.session_state.get("_screen_open"):
            if st.button("💎  Businesses that are getting better",
                         key="screen_open", width="stretch", type="primary"):
                st.session_state._screen_open = True
                st.rerun()
            st.caption(
                "Three years of rising revenue, rising earnings, expanding margins — "
                "and growth that wasn't borrowed. Scored daily against Graham, "
                "Greenblatt, Dorsey, Buffett and Lynch. None of them is unanimous. "
                "That's the interesting part."
            )
        else:
            _screen = selector.improving_businesses(universe_df, limit=25)
            _view = pd.DataFrame({
                "Ticker": _screen["ticker"].str.replace(r"\.(NS|BO)$", "", regex=True),
                "Company": _screen["name"],
                "Sector": _screen["sector"],
                "Size": _screen["risk_tier"],
                "Frameworks": _screen["score_applicable"].astype(str) + " of "
                              + _screen["attainable"].astype(str),
                "Fails": _screen["failed"],
                "Abstains": _screen["abstained"],
                "Revenue 3y": _screen["revenue_cagr_3y"].round(1),
                "Earnings 3y": _screen["ni_cagr_3y"].round(1),
                "P/E": _screen["pe"].round(1),
            })

            st.markdown("#### Businesses that are getting better")
            st.caption(
                "Every one of these is improving on the numbers. Not one of them is "
                "cheap by every measure — 'Fails' says which test each one loses, and "
                "'Abstains' means a framework declined to judge rather than judged badly. "
                "Select any to watch. We'll email you when their scores move."
            )

            _sel = st.dataframe(
                _view, hide_index=True, width="stretch",
                on_select="rerun", selection_mode="multi-row", key="_screen_table",
            )
            _rows = _sel.selection.rows if _sel and _sel.selection else []
            _picked = _screen.iloc[_rows] if _rows else _screen.iloc[0:0]

            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                if st.session_state.sb_user_id:
                    if st.button(f"👁  Watch {len(_picked)} selected",
                                 disabled=not len(_picked), width="stretch",
                                 type="primary", key="_screen_watch"):
                        _add_to_watchlist(_picked)
                        st.rerun()
                else:
                    # The choice survives sign-in. Asking a stranger to authenticate
                    # BEFORE they have chosen anything is what kills the funnel; the
                    # work they have already done is what carries them through it.
                    st.button(f"👁  Watch {len(_picked)} selected",
                              disabled=not len(_picked), width="stretch",
                              type="primary", key="_screen_watch_anon",
                              on_click=lambda: st.session_state.update(
                                  _screen_pending=list(_picked["ticker"])))
                    if st.session_state.get("_screen_pending"):
                        # Session state does NOT survive the Google OAuth redirect —
                        # the browser leaves the app and returns as a new session.
                        # Email/password login keeps it; Google does not. Until the
                        # picks ride through redirect_to, do not promise they will.
                        st.info("Sign in from the sidebar, then pick again — it's one click.")
            with c2:
                if st.button("💬  Explain these", disabled=not len(_picked),
                             width="stretch", key="_screen_explain"):
                    st.session_state.pending_prompt = SCREENER_EXPLAIN_PROMPT.format(
                        tickers=", ".join(_picked["ticker"]))
                    st.session_state._screen_open = False
                    st.rerun()
            with c3:
                if st.button("Close", width="stretch", key="_screen_close"):
                    st.session_state._screen_open = False
                    st.rerun()

    prompt = st.chat_input("Ask about any stock, or type a question...")

    if not prompt and "pending_prompt" in st.session_state:
        prompt = st.session_state.pop("pending_prompt")
        if prompt and st.session_state.get("pending_disambiguation"):
            st.session_state.pending_disambiguation = None

    with chat_area:
        if not st.session_state.messages:
            st.markdown("")
            st.info("Type a company name or question below to get started, or use the screeners above.")

        for msg in st.session_state.messages:
            avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):
                if msg["role"] == "assistant":
                    _render_verdict_badge(text=msg["content"], stored_tier=msg.get("verdict_tier"))
                st.markdown(msg["content"])
                if msg.get("model"):
                    st.caption(f"⚡ {msg['model']}")

        if st.session_state.get("pending_portfolio"):
            portfolio = st.session_state.pending_portfolio

            if st.session_state.sb_user_id is None:
                st.info("💡 Log in to save this portfolio to your account.")
            else:
                st.markdown("### 📋 Your SIP Portfolio")
                preview_data = []
                for s in portfolio["stocks"]:
                    preview_data.append({
                        "Stock": s.get("name", s["ticker"]),
                        "Ticker": s["ticker"],
                        "Sector": s.get("sector", "—"),
                        "Allocation": f"{s.get('allocation_pct', 0)}%",
                        "Monthly": f"{fmt_inr(portfolio['sip_amount'] * s.get('allocation_pct', 0) / 100)}",
                    })
                st.dataframe(pd.DataFrame(preview_data), hide_index=True, width="stretch")
                _paper_tag = " · 👁 Paper Portfolio" if portfolio.get("is_paper") else ""
                _goal_tag = f" · Goal: {fmt_inr(portfolio['target_amount'])}" if portfolio.get("target_amount") else ""
                st.caption(f"Total SIP: {fmt_inr(portfolio['sip_amount'])}/month · {portfolio.get('investor_type', '')} · {portfolio.get('time_horizon', '')} horizon{_goal_tag}{_paper_tag}")
                _custom_name = st.text_input(
                    "Portfolio name",
                    value=portfolio["name"],
                    placeholder=portfolio["name"],
                    key="portfolio_name_input",
                )
                if _custom_name and _custom_name.strip():
                    portfolio["name"] = _custom_name.strip()
                if st.button("💾 Save Portfolio", width="stretch"):
                    _r = _commit_portfolio(portfolio)
                    if not _r.get("ok"):
                        st.error(f"Save failed: {_r.get('error')}")
                    else:
                        st.success(f"Portfolio saved! Invested {fmt_inr(_r['invested'])} of {fmt_inr(portfolio['sip_amount'])}.")
                        if _r["unallocated"] > 0:
                            st.info(f"{fmt_inr(_r['unallocated'])} unallocated (not enough for another share of any holding).")
                        if _r.get("stale_priced"):
                            st.warning(f"⚠️ Live price unavailable for {', '.join(_r['stale_priced'])} — used last known close. Verify on Kite before paying.")
                        breakdown_data = []
                        for s in _r["allocated"]:
                            breakdown_data.append({
                                "Stock": s["name"] or s["ticker"], "Price": f"{fmt_inr(s['price'], 2)}",
                                "Shares": s["shares"], "Invested": f"{fmt_inr(s['actual_amount'])}",
                                "Target": f"{fmt_inr(portfolio['sip_amount'] * s['allocation_pct'] / 100)}",
                            })
                        st.dataframe(pd.DataFrame(breakdown_data), hide_index=True, width="stretch")
                        if portfolio.get("is_paper"):
                            st.session_state._paper_just_saved = True
                            st.session_state.sb_view_mode = "watchlist"
                        st.rerun()

        # ── Disambiguation UI (shown when awaiting user's pick) ──
        if not prompt and st.session_state.get("pending_disambiguation"):
            pd_data = st.session_state.pending_disambiguation
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(pd_data["original_query"])
            st.info("🔍 Multiple matches found. Which company did you mean?")
            _matches = pd_data["matches"]
            _ncols = min(len(_matches), 4)
            for _row_start in range(0, len(_matches), _ncols):
                _row_items = _matches[_row_start:_row_start + _ncols]
                _btn_cols = st.columns(len(_row_items))
                for _j, _m in enumerate(_row_items):
                    _i = _row_start + _j
                    with _btn_cols[_j]:
                        _lbl = f"{_m['name']} ({_m['ticker'].replace('.NS','').replace('.BO','')})"
                        if st.button(_lbl, key=f"disambig_{_i}", use_container_width=True):
                            _resolved = f"{pd_data['original_query']} (company: {_m['name']}, ticker: {_m['ticker']})"
                            st.session_state.pending_prompt = _resolved
                            st.session_state.pending_disambiguation = None
                            st.rerun()

        if prompt:
            st.session_state._last_user_message = prompt
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=USER_AVATAR):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=AGENT_AVATAR):
                response_placeholder = st.empty()
                answer = None
                model_used = None
                _status_slot = st.empty()
                with _status_slot.status("🧠 Preparing analysis...", expanded=False) as status:
                    try:
                        answer, model_used = agent_turn(prompt, status_container=status)
                        status.update(label="✅ Analysis complete", state="complete")
                    except Exception as e:
                        status.update(label="⚠️ Error encountered", state="error")
                        error_msg = str(e)
                        error_upper = error_msg.upper()
                        if any(err in error_upper for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "ALL MODELS RATE-LIMITED"]):
                            # ── Model fallback chain: same pattern as get_web_context ──
                            _current_model = st.session_state.get("last_working_model",
                                st.session_state.get("selected_model", ""))
                            _cur_idx = -1
                            for _i, _m in enumerate(FREE_MODELS):
                                if _m == _current_model:
                                    _cur_idx = _i
                                    break
                            _remaining = [m for m in FREE_MODELS[_cur_idx + 1:] if m != _current_model]

                            import time
                            _retry_ok = False
                            for _fallback in _remaining:
                                st.session_state["last_working_model"] = _fallback
                                st.warning(f"Rate limited — switching to `{_fallback}` ...")
                                time.sleep(3)
                                try:
                                    answer, model_used = agent_turn(prompt, status_container=status)
                                    status.update(label="✅ Analysis complete", state="complete")
                                    _retry_ok = True
                                    break
                                except Exception:
                                    continue

                            if not _retry_ok:
                                st.warning("All models busy. Click below to try again.")
                                st.session_state.pop("last_working_model", None)
                                st.session_state.pending_retry = prompt
                                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                                    st.session_state.messages.pop()
                        else:
                            st.error(f"Error: {error_msg[:150]}")
                            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                                st.session_state.messages.pop()
                            st.session_state.pending_retry = prompt
                if answer:
                    _status_slot.empty()
                    _vt = st.session_state.pop("_last_verdict_tier", None)
                    if _vt and not st.session_state.get("builder_profile"):
                        _render_verdict_badge(stored_tier=_vt)
                    response_placeholder.markdown(answer)
                    st.caption(f"⚡ {model_used}")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "model": model_used, "verdict_tier": _vt})
                    if st.session_state.get("pending_portfolio"):
                        st.info("⬆️ Scroll up to review and save your portfolio.")
                        st.rerun()

                    # Sprint 11: Safety net — detect when LLM presented a portfolio table
                    # but DIDN'T call register_portfolio (common with weaker models)
                    _bp = st.session_state.get("builder_profile")
                    if _bp and not st.session_state.get("pending_portfolio"):
                        # Check if the response contains ticker patterns (X.NS) suggesting a portfolio was presented
                        import re
                        _ticker_matches = re.findall(r'[A-Z]{2,20}\.NS', answer or "")
                        if len(_ticker_matches) >= 5:
                            st.warning("⚠️ Portfolio was presented but not saved to the system. Retrying registration...")
                            # Build stocks list from the tickers found in the response
                            _found_tickers = list(dict.fromkeys(_ticker_matches))  # deduplicate, preserve order
                            _auto_stocks = []
                            _equal_pct = round(100 / len(_found_tickers), 1)
                            for _ft in _found_tickers:
                                _row = universe_df[universe_df["ticker"] == _ft]
                                _auto_stocks.append({
                                    "ticker": _ft,
                                    "name": str(_row.iloc[0].get("name", _ft)) if not _row.empty else _ft,
                                    "sector": str(_row.iloc[0].get("sector", "Unknown")) if not _row.empty else "Unknown",
                                    "allocation_pct": _equal_pct,
                                })
                            _auto_result = register_portfolio(
                                portfolio_name=f"{_bp.get('investor_type', 'Growth').title()} SIP {datetime.date.today().strftime('%b %Y')}",
                                investor_type=_bp.get("investor_type", "balanced"),
                                sip_amount=_bp.get("sip_amount", 5000),
                                time_horizon=_bp.get("time_horizon", "medium"),
                                review_days=_bp.get("review_days", 90),
                                stocks_json=json.dumps(_auto_stocks),
                                portfolio_profile=json.dumps(_bp),
                                target_amount=_bp.get("target_amount", 0) or 0,
                                target_date=_bp.get("target_date", "") or "",
                                decision_context="Auto-extracted from LLM response (model failed to call register_portfolio tool)."
                            )
                            if st.session_state.get("pending_portfolio"):
                                st.info("✅ Portfolio captured. Scroll up to review and save.")
                                st.rerun()
                            elif isinstance(_auto_result, dict) and _auto_result.get("error"):
                                st.warning(f"Auto-save blocked: {_auto_result.get('error')} — {', '.join(_auto_result.get('violations', []))}")

                    if st.session_state.get("_pending_navigate"):
                        st.session_state.sb_view_mode = st.session_state._pending_navigate
                        st.session_state.pop("_pending_navigate", None)
                        st.rerun()

                    if st.session_state.get("pending_disambiguation"):
                        st.rerun()

                    # ── Score History Chart (single-stock analysis only — skip during builder) ──
                    if not st.session_state.get("builder_profile"):
                        _sh_chat_tickers = set(re.findall(r'\b[A-Z][A-Z0-9&]+\.(?:NS|BO)\b', answer))
                        _sh_disambig = re.search(r'ticker:\s*([A-Z][A-Z0-9&]+\.(?:NS|BO))', prompt)
                        _sh_chat_target = None
                        if _sh_disambig:
                            _sh_chat_target = _sh_disambig.group(1)
                        elif len(_sh_chat_tickers) == 1:
                            _sh_chat_target = list(_sh_chat_tickers)[0]

                    # ── Chat → Watchlist bridge: store positive-verdict tickers for buttons ──
                    _positive_verdicts = {"STRONG BUY", "BUY", "CONDITIONAL BUY"}
                    _answer_upper = answer.upper()
                    _has_positive = any(v in _answer_upper for v in _positive_verdicts)
                    if st.session_state.sb_user_id and _has_positive:
                        _resp_tickers = set(re.findall(r'\b[A-Z][A-Z0-9&]+\.(?:NS|BO)\b', answer))
                        _disambig = re.search(r'ticker:\s*([A-Z][A-Z0-9&]+\.(?:NS|BO))', prompt)
                        if _disambig:
                            _resp_tickers.add(_disambig.group(1))

                        _positive_tickers = []
                        for _t in _resp_tickers:
                            _t_up = _t.upper()
                            for _v in _positive_verdicts:
                                if re.search(
                                    rf'(?:{re.escape(_t_up)}.{{0,400}}{re.escape(_v)})|(?:{re.escape(_v)}.{{0,400}}{re.escape(_t_up)})',
                                    _answer_upper
                                ):
                                    _positive_tickers.append(_t)
                                    break
                        if not _positive_tickers and len(_resp_tickers) == 1:
                            _positive_tickers = list(_resp_tickers)

                        if _positive_tickers:
                            st.session_state.pending_watch_tickers = _positive_tickers

        if st.session_state.get("pending_retry"):
            if st.button("🔄 Retry last query", width="stretch"):
                st.session_state.pending_prompt = st.session_state.pop("pending_retry")
                st.rerun()

        # ── Watchlist bridge buttons (persistent across reruns) ──
        if st.session_state.get("pending_watch_tickers") and st.session_state.sb_user_id:
            _pw_sb = get_supabase()
            _pw_tickers = st.session_state.pending_watch_tickers

            try:
                _pw_ports = _pw_sb.table("portfolios").select("id").eq(
                    "user_id", st.session_state.sb_user_id
                ).execute().data or []
                _pw_port_ids = [p["id"] for p in _pw_ports]
                if _pw_port_ids:
                    _pw_held = {h["ticker"] for h in (_pw_sb.table("holdings").select(
                        "ticker"
                    ).in_("portfolio_id", _pw_port_ids).execute().data or [])}
                else:
                    _pw_held = set()
            except Exception:
                _pw_held = set()

            try:
                _pw_watched = {w["ticker"] for w in (_pw_sb.table("watchlist").select(
                    "ticker"
                ).eq("user_id", st.session_state.sb_user_id).execute().data or [])}
            except Exception:
                _pw_watched = set()

            _any_actionable = False
            for _yt in _pw_tickers:
                _bare = _yt.replace(".NS", "").replace(".BO", "")
                if _yt in _pw_held:
                    st.caption(f"✅ {_bare} — already in your portfolio")
                    if KITE_ENABLED:
                        st.link_button(f"🛒 Buy {_bare} on Kite", kite_buy_url(_yt), use_container_width=True)
                elif _yt in _pw_watched:
                    st.caption(f"👁 {_bare} — already on your watchlist")
                    if KITE_ENABLED:
                        st.link_button(f"🛒 Buy {_bare} on Kite", kite_buy_url(_yt), use_container_width=True)
                else:
                    _any_actionable = True
                    if KITE_ENABLED:
                        _yt_c1, _yt_c2 = st.columns(2)
                        with _yt_c1:
                            _yt_watch = st.button(f"👁 Watch {_bare}", key=f"watch_{_yt}", use_container_width=True)
                        with _yt_c2:
                            st.link_button(f"🛒 Buy on Kite", kite_buy_url(_yt), use_container_width=True)
                    else:
                        _yt_watch = st.button(f"👁 Watch {_bare}", key=f"watch_{_yt}", use_container_width=True)
                    if _yt_watch:
                        _wl_row = universe_df[universe_df["ticker"] == _yt]
                        _wl_data = {
                            "user_id": st.session_state.sb_user_id,
                            "ticker": _yt,
                            "name": str(_wl_row["name"].iloc[0]) if not _wl_row.empty else _bare,
                            "score_when_added": int(_wl_row["score"].iloc[0]) if not _wl_row.empty and pd.notna(_wl_row["score"].iloc[0]) else None,
                            "quality_when_added": bool(_wl_row["quality_pass"].iloc[0]) if not _wl_row.empty and "quality_pass" in _wl_row.columns and pd.notna(_wl_row["quality_pass"].iloc[0]) else None,
                            # See the bulk-add site: entry thesis, captured now
                            # because it is unreconstructable later. None when
                            # the ticker is not in today's universe — that is an
                            # honest absence, and it labels as unknown_inputs.
                            "entry_trace": (selector.build_watch_trace(_wl_row.iloc[0])
                                            if not _wl_row.empty else None),
                        }
                        try:
                            _pw_sb.table("watchlist").insert(_wl_data).execute()
                            st.session_state.pending_watch_tickers = [t for t in _pw_tickers if t != _yt]
                            if not st.session_state.pending_watch_tickers:
                                del st.session_state["pending_watch_tickers"]
                            st.rerun()
                        except Exception as _we:
                            st.error(f"Failed: {_we}")

            if not _any_actionable:
                st.session_state.pop("pending_watch_tickers", None)
elif st.session_state.sb_view_mode == "watchlist":
    st.markdown("### 👁 My Watchlist")

    if st.session_state.sb_user_id is None:
        st.warning("Please log in to view your watchlist.")
    else:
        _wl_tab_recs, _wl_tab_stocks, _wl_tab_paper = st.tabs(["📋 This Week's Picks", "📊 Stocks", "📁 Paper Portfolios"])

        with _wl_tab_recs:
            _rec_sb = get_supabase()
            try:
                _rec_resp = _rec_sb.table("weekly_recommendations").select("*").eq(
                    "user_id", st.session_state.sb_user_id
                ).gt("expires_at", datetime.datetime.utcnow().isoformat()).eq("acted_on", False).execute()
                _rec_items = _rec_resp.data or []
            except Exception:
                _rec_items = []

            if not _rec_items:
                st.info("No recommendations right now. Check back Monday morning — Kordent curates a personalized buy list every week.")
            else:
                for _rec in _rec_items:
                    _rec_horizon = _rec.get("time_horizon", "medium").title()
                    _rec_type = _rec.get("investor_type", "balanced")
                    _rec_budget = float(_rec.get("budget_inr") or 0)
                    _rec_stocks = _rec.get("stocks", [])
                    if isinstance(_rec_stocks, str):
                        import json as _rjson
                        try:
                            _rec_stocks = _rjson.loads(_rec_stocks)
                        except Exception:
                            _rec_stocks = []

                    if not _rec_stocks:
                        continue

                    # Expiry countdown
                    try:
                        _exp = datetime.datetime.fromisoformat(_rec["expires_at"].replace("Z", "+00:00"))
                        _now = datetime.datetime.now(datetime.timezone.utc)
                        _hours_left = max(0, int((_exp - _now).total_seconds() / 3600))
                    except Exception:
                        _hours_left = None

                    with st.container(border=True):
                        _exp_tag = f" · ⏳ {_hours_left}h left" if _hours_left is not None else ""
                        st.markdown(f"**📋 {_rec_horizon} Horizon Picks**{_exp_tag}")
                        st.caption(f"Based on your {_rec_type} profile · Budget: {fmt_inr(_rec_budget)}")

                        _rec_rows = []
                        _rec_total = 0
                        for _rs in _rec_stocks:
                            _rec_rows.append({
                                "Stock": _rs["name"],
                                "Ticker": _rs["ticker"],
                                "Score": _score_label(_rs["ticker"], _rs.get("score")),
                                "Verdict": _rs["verdict"],
                                "Price": f"{fmt_inr(_rs['price'])}",
                                "Shares": _rs["shares"],
                                "Amount": f"{fmt_inr(_rs['amount'])}",
                            })
                            _rec_total += _rs["amount"]
                        st.dataframe(pd.DataFrame(_rec_rows), hide_index=True, use_container_width=True)
                        st.caption(f"Total: {fmt_inr(_rec_total)} of {fmt_inr(_rec_budget)} budget")

                        _rc1, _rc2 = st.columns(2)
                        with _rc1:
                            if KITE_ENABLED:
                                _rec_kite = [{"ticker": s["ticker"], "quantity": s["shares"]} for s in _rec_stocks if s["shares"] > 0]
                                if _rec_kite:
                                    st.link_button("🛒 Buy All on Kite", kite_basket_url(_rec_kite), use_container_width=True)
                        with _rc2:
                            if st.button("⏭ Skip This Week", key=f"skip_rec_{_rec['id']}", use_container_width=True):
                                try:
                                    _rec_sb.table("weekly_recommendations").update(
                                        {"acted_on": True}
                                    ).eq("id", _rec["id"]).execute()
                                    st.rerun()
                                except Exception as _re:
                                    st.error(f"Failed: {_re}")

        with _wl_tab_stocks:
            _w_sb = get_supabase()
            try:
                _w_resp = _w_sb.table("watchlist").select("*").eq(
                    "user_id", st.session_state.sb_user_id
                ).order("added_date", desc=True).execute()
                _w_items = _w_resp.data or []
            except Exception as _w_err:
                st.error(f"Failed to load watchlist: {_w_err}")
                _w_items = []
    
            # Fetch today's watchlist alerts (auto-expire after 24h)
            _wl_alerts_by_ticker = {}
            if _w_items:
                try:
                    _wl_alert_resp = _w_sb.table("portfolio_alerts").select("*").eq(
                        "user_id", st.session_state.sb_user_id
                    ).eq("alert_date", datetime.date.today().isoformat()).in_(
                        "alert_type", ["watchlist_score_up", "watchlist_score_down",
                                       "watchlist_quality_flip", "watchlist_near_low"]
                    ).execute()
                    for _wa in (_wl_alert_resp.data or []):
                        _wl_alerts_by_ticker.setdefault(_wa["ticker"], []).append(_wa)
                except Exception:
                    pass
    
            if not _w_items:
                st.info("Your watchlist is empty. Analyze a stock in chat — if it gets a YES verdict, you'll see a Watch button.")
            else:
                for _w in _w_items:
                    _w_ticker = _w["ticker"]
                    _w_name = _w.get("name") or _w_ticker
                    _w_bare = _w_ticker.replace(".NS", "").replace(".BO", "")
                    _w_added_score = _w.get("score_when_added")
                    _w_added_quality = _w.get("quality_when_added")
                    _w_note = _w.get("note") or ""
                    _w_added_date = _w.get("added_date", "")
                    _w_days = 0
                    if _w_added_date:
                        try:
                            _w_days = (datetime.date.today() - datetime.date.fromisoformat(str(_w_added_date))).days
                        except Exception:
                            pass
    
                    # Current data from universe_df
                    _w_cur_score = "?"
                    _w_cur_quality = None
                    _w_sector = "—"
                    _w_pe = "—"
                    _w_pb = "—"
                    try:
                        _w_row = universe_df[universe_df["ticker"] == _w_ticker]
                        if not _w_row.empty:
                            if pd.notna(_w_row["score"].iloc[0]):
                                _w_cur_score = int(_w_row["score"].iloc[0])
                            if "quality_pass" in _w_row.columns and pd.notna(_w_row["quality_pass"].iloc[0]):
                                _w_cur_quality = bool(_w_row["quality_pass"].iloc[0])
                            _w_sector = str(_w_row["sector"].iloc[0]) if pd.notna(_w_row.get("sector", pd.Series([None])).iloc[0]) else "—"
                            _w_pe = round(float(_w_row["pe"].iloc[0]), 1) if pd.notna(_w_row.get("pe", pd.Series([None])).iloc[0]) else "—"
                            _w_pb = round(float(_w_row["pb"].iloc[0]), 2) if pd.notna(_w_row.get("pb", pd.Series([None])).iloc[0]) else "—"
                    except NameError:
                        pass
    
                    with st.container(border=True):
                        _wh1, _wh2 = st.columns([4, 1])
                        with _wh1:
                            # Score delta
                            _w_delta_str = ""
                            if _w_added_score is not None and _w_cur_score != "?":
                                _w_diff = _w_cur_score - _w_added_score
                                if _w_diff > 0:
                                    _w_delta_str = f"  ↑{_w_diff} since added"
                                elif _w_diff < 0:
                                    _w_delta_str = f"  ↓{abs(_w_diff)} since added"
                            st.markdown(f"**{_w_name}** ({_w_bare})")
                            # Denominator from the live row. The DELTA stays a
                            # raw-integer difference: score_when_added is stored
                            # raw by design, so both sides share a scale.
                            _w_lbl = (selector.score_label(_w_row.iloc[0], score=_w_cur_score)
                                      if not _w_row.empty and _w_cur_score != "?" else f"{_w_cur_score}")
                            st.caption(f"Score: {_w_lbl}{_w_delta_str} · {_w_sector} · PE {_w_pe} · PB {_w_pb} · Watching {_w_days}d")
    
                            # Quality flip warning
                            if _w_added_quality is not None and _w_cur_quality is not None and _w_added_quality != _w_cur_quality:
                                if _w_cur_quality:
                                    st.success("Quality flipped to PASS since you added this.")
                                else:
                                    st.warning("Quality flipped to FAIL since you added this.")
    
                        with _wh2:
                            if KITE_ENABLED:
                                st.link_button("🛒 Buy", kite_buy_url(_w_ticker), use_container_width=True)
                            if st.button("✕ Remove", key=f"wl_rm_{_w['id']}", use_container_width=True):
                                try:
                                    _w_sb.table("watchlist").delete().eq("id", _w["id"]).execute()
                                    st.rerun()
                                except Exception as _e:
                                    st.error(f"Failed: {_e}")
                        # Daily alerts (auto-expire after 24h)
                        _card_alerts = _wl_alerts_by_ticker.get(_w_ticker, [])
                        for _ca in _card_alerts:
                            # Severity now carries the good/bad reading this block
                            # used to re-derive. quality_flip parsed its own detail
                            # JSON to work out direction — duplicating a judgment
                            # portfolio_tracker had already made when it wrote the
                            # row. One source, read here.
                            _ca_headline = _ca.get("headline", "")
                            _ca_sev = _ca.get("severity") or "info"
                            _say_wl = {"danger": st.error,
                                       "warning": st.warning}.get(_ca_sev, st.success)
                            _say_wl(f"{_ca_headline}")
    
                        
                        # Editable note
                        _w_new_note = st.text_input(
                            "Note", value=_w_note, key=f"wl_note_{_w['id']}",
                            placeholder="e.g. Waiting for PE to drop below 12",
                            label_visibility="collapsed"
                        )
                        if _w_new_note != _w_note:
                            try:
                                _w_sb.table("watchlist").update({"note": _w_new_note}).eq("id", _w["id"]).execute()
                            except Exception:
                                pass

                        with st.expander("📊 Score Trend"):
                            render_score_history_chart(_w_sb, _w_ticker, stock_name=_w_name,
                                chart_key=f"sh_wl_{_w['id']}")


        with _wl_tab_paper:
            if st.session_state.get("_paper_just_saved"):
                st.success("Paper portfolio saved! Track its performance here.")
                st.session_state.pop("_paper_just_saved", None)

            _pp_sb = get_supabase()
            try:
                _pp_resp = _pp_sb.table("portfolios").select("*").eq(
                    "user_id", st.session_state.sb_user_id
                ).eq("is_paper", True).order("created_at", desc=True).execute()
                _pp_ports = _pp_resp.data or []
            except Exception as _pp_err:
                st.error(f"Failed to load paper portfolios: {_pp_err}")
                _pp_ports = []

            if not _pp_ports:
                st.info("No paper portfolios yet. Use 🏗️ Build Portfolio and check 'Watch only' to create one.")
            else:
                for _pp in _pp_ports:
                    with st.container(border=True):
                        st.markdown(f"**👁 {_pp['name']}**")

                        try:
                            _pp_h_resp = _pp_sb.table("holdings").select("*").eq(
                                "portfolio_id", _pp["id"]
                            ).execute()
                            _pp_holdings = _pp_h_resp.data or []
                        except Exception:
                            _pp_holdings = []

                        if _pp_holdings:
                            _pp_enriched = enrich_holdings_live(_pp_holdings, cache_key=f"paper_{_pp['id']}")
                            _pp_invested = sum(h.get("shares", 0) * h.get("price_at_entry", 0) for h in _pp_enriched)
                            _pp_current = sum(h.get("current_value", 0) for h in _pp_enriched)
                            _pp_ret = ((_pp_current - _pp_invested) / _pp_invested * 100) if _pp_invested > 0 else 0

                            _pm1, _pm2, _pm3 = st.columns(3)
                            with _pm1:
                                st.metric("Invested", f"{fmt_inr(_pp_invested)}")
                            with _pm2:
                                st.metric("Current Value", f"{fmt_inr(_pp_current)}")
                            with _pm3:
                                st.metric("Return", f"{_pp_ret:+.1f}%")

                            _pp_rows = []
                            for _h in _pp_enriched:
                                _h_entry = _h.get("price_at_entry", 0)
                                _h_now = _h.get("current_price", 0)
                                _h_sh = _h.get("shares", 0)
                                _h_pnl = (_h_now - _h_entry) * _h_sh
                                _h_ret = ((_h_now - _h_entry) / _h_entry * 100) if _h_entry > 0 else 0
                                _pp_rows.append({
                                    "Stock": _h.get("name") or _h.get("ticker", ""),
                                    "Shares": _h_sh,
                                    "Entry": f"{fmt_inr(_h_entry, 2)}",
                                    "Now": f"{fmt_inr(_h_now, 2)}",
                                    "P&L": f"{fmt_inr(_h_pnl)}",
                                    "Return": f"{_h_ret:+.1f}%",
                                })
                            st.dataframe(pd.DataFrame(_pp_rows), hide_index=True, use_container_width=True)

                            _pp_profile = _pp.get("portfolio_profile") or {}
                            if isinstance(_pp_profile, str):
                                try:
                                    _pp_profile = json.loads(_pp_profile)
                                except Exception:
                                    _pp_profile = {}
                            _pp_cap_parts = [_pp.get("created_at", "")[:10]]
                            if _pp.get("investor_type"):
                                _pp_cap_parts.append(_pp["investor_type"])
                            if _pp.get("time_horizon"):
                                _pp_cap_parts.append(f"{_pp['time_horizon']} horizon")
                            if _pp.get("target_amount"):
                                _pp_cap_parts.append(f"Goal: {fmt_inr(_pp['target_amount'])}")
                            st.caption(" · ".join(_pp_cap_parts))
                        else:
                            st.caption("No holdings recorded.")

                        # ── Action buttons ──
                        if KITE_ENABLED and _pp_holdings:
                            _paper_kite = [{"ticker": h.get("ticker", ""), "quantity": h.get("shares", 1)}
                                           for h in _pp_holdings if h.get("ticker")]
                            if _paper_kite:
                                st.link_button("🛒 Buy All on Kite", kite_basket_url(_paper_kite), use_container_width=True)
                        _pp_c1, _pp_c2 = st.columns(2)
                        with _pp_c1:
                            if st.button("🚀 Make This Real", key=f"make_real_{_pp['id']}", use_container_width=True):
                                st.session_state[f"confirm_real_{_pp['id']}"] = True
                        with _pp_c2:
                            if st.button("🗑️ Delete", key=f"del_paper_{_pp['id']}", use_container_width=True, type="secondary"):
                                st.session_state[f"confirm_del_paper_{_pp['id']}"] = True

                        # ── Make This Real confirmation ──
                        if st.session_state.get(f"confirm_real_{_pp['id']}"):
                            st.warning("This converts to a real portfolio at **current market prices** (not original paper prices). XIRR and tracking restart from today.")
                            if KITE_ENABLED and _pp_holdings:
                                _conv_kite = [{"ticker": h.get("ticker", ""), "quantity": h.get("shares", 1)} for h in _pp_holdings if h.get("ticker")]
                                if _conv_kite:
                                    st.link_button("🛒 Step 1: Buy on Kite first", kite_basket_url(_conv_kite), use_container_width=True)
                            _rc1, _rc2 = st.columns(2)
                            with _rc1:
                                _confirm_label = "Step 2: Confirm conversion" if KITE_ENABLED else "Yes, make it real"
                                if st.button(f"✅ {_confirm_label}", key=f"real_yes_{_pp['id']}", use_container_width=True):
                                    try:
                                        _conv_today = datetime.date.today().isoformat()
                                        # Reset holdings to current market prices
                                        for _rh in _pp_holdings:
                                            try:
                                                _rh_price = yf.Ticker(_rh.get("ticker", "")).fast_info.last_price or _rh.get("price_at_entry", 0)
                                            except Exception:
                                                _rh_price = _rh.get("price_at_entry", 0)
                                            _pp_sb.table("holdings").update({
                                                "price_at_entry": round(_rh_price, 2),
                                                "sip_amount_inr": round(_rh.get("shares", 0) * _rh_price, 2),
                                                "entry_date": _conv_today,
                                            }).eq("id", _rh["id"]).execute()
                                        # Delete old paper transactions — XIRR restarts from today
                                        try:
                                            _pp_sb.table("sip_transactions").delete().eq("portfolio_id", _pp["id"]).execute()
                                        except Exception:
                                            pass
                                        # Create fresh buy transactions at current prices
                                        _nifty_c = None
                                        for _rh in _pp_holdings:
                                            _rh_shares = float(_rh.get("shares") or 0)
                                            _rh_price_new = float(_rh.get("price_at_entry") or 0)
                                            # Use the just-updated price if available
                                            try:
                                                _rh_price_new = yf.Ticker(_rh.get("ticker", "")).fast_info.last_price or _rh_price_new
                                            except Exception:
                                                pass
                                            _rh_amt = round(_rh_shares * _rh_price_new, 2)
                                            if _rh_amt > 0:
                                                _nifty_c = record_transaction(
                                                    _pp_sb, _pp["id"], st.session_state.sb_user_id,
                                                    _rh.get("ticker", ""), _rh_shares, _rh_price_new, _rh_amt,
                                                    "buy", _nifty_c
                                                )
                                        # Adjust goal date: paper period doesn't count
                                        _conv_update = {
                                            "is_paper": False,
                                            "paper_converted_at": _conv_today,
                                        }
                                        _old_goal = _pp.get("target_date")
                                        _pp_created = str(_pp.get("created_at", ""))[:10]
                                        if _old_goal and _pp_created:
                                            try:
                                                _goal_dt = datetime.date.fromisoformat(str(_old_goal)[:10])
                                                _created_dt = datetime.date.fromisoformat(_pp_created)
                                                _conv_dt = datetime.date.fromisoformat(_conv_today)
                                                _paper_days = (_conv_dt - _created_dt).days
                                                _new_goal = (_goal_dt + datetime.timedelta(days=_paper_days)).isoformat()
                                                _conv_update["target_date"] = _new_goal
                                            except (ValueError, TypeError):
                                                pass
                                        _pp_sb.table("portfolios").update(_conv_update).eq("id", _pp["id"]).execute()
                                        # Clear portfolio history (clean start)
                                        try:
                                            _pp_sb.table("portfolio_history").delete().eq("portfolio_id", _pp["id"]).execute()
                                        except Exception:
                                            pass
                                        st.session_state.pop(f"confirm_real_{_pp['id']}", None)
                                        _conv_msg = "Portfolio is now real! Fresh transactions recorded — XIRR starts from today. Find it in My Portfolios."
                                        if _conv_update.get("target_date") and _old_goal:
                                            _conv_msg += f"\n\n📅 Goal date adjusted from {str(_old_goal)[:10]} to {_conv_update['target_date']} — the portfolio spent {_paper_days} days in paper mode."
                                        st.success(_conv_msg)
                                        st.rerun()
                                    except Exception as _re:
                                        st.error(f"Failed: {_re}")
                            with _rc2:
                                if st.button("Cancel", key=f"real_no_{_pp['id']}", use_container_width=True):
                                    st.session_state.pop(f"confirm_real_{_pp['id']}", None)
                                    st.rerun()

                        # ── Delete confirmation ──
                        if st.session_state.get(f"confirm_del_paper_{_pp['id']}"):
                            st.warning("Delete this paper portfolio? This cannot be undone.")
                            _dc1, _dc2 = st.columns(2)
                            with _dc1:
                                if st.button("Yes, delete", key=f"del_yes_{_pp['id']}", use_container_width=True):
                                    try:
                                        _pp_sb.table("sip_transactions").delete().eq("portfolio_id", _pp["id"]).execute()
                                        _pp_sb.table("portfolio_alerts").delete().eq("portfolio_id", _pp["id"]).execute()
                                        _pp_sb.table("portfolio_history").delete().eq("portfolio_id", _pp["id"]).execute()
                                        _pp_sb.table("holdings").delete().eq("portfolio_id", _pp["id"]).execute()
                                        _pp_sb.table("portfolios").delete().eq("id", _pp["id"]).execute()
                                        st.session_state.pop(f"confirm_del_paper_{_pp['id']}", None)
                                        st.rerun()
                                    except Exception as _de:
                                        st.error(f"Failed: {_de}")
                            with _dc2:
                                if st.button("Cancel", key=f"del_no_{_pp['id']}", use_container_width=True):
                                    st.session_state.pop(f"confirm_del_paper_{_pp['id']}", None)
                                    st.rerun()

elif st.session_state.sb_view_mode == "import":
    st.markdown("### 📥 Import Your Existing Portfolio")
    st.caption("Onboard your current holdings to analyze them using the Kordent framework.")
    
    if st.session_state.sb_user_id is None:
        st.warning("Please log in via the sidebar to save and analyze an existing portfolio.")
    else:
        # 1. Meta Information Gathering
        with st.container(border=True):
            st.markdown("#### ⚙️ Portfolio Metadata & Goals")
            p_name = st.text_input("Portfolio Name", placeholder="e.g., My Main Brokerage")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                total_invested = st.number_input("Total Amount Invested Till Date (INR)", min_value=0, value=100000, step=5000)
                sip_amt = st.number_input("Current Monthly SIP Amount (INR)", min_value=0, value=10000, step=1000)
            with col_m2:
                inv_type = st.selectbox("Investment Goal Profile", ["defensive", "balanced", "enterprising"], index=1)
                horizon = st.selectbox("Time Horizon", ["short", "medium", "long"], index=1)
        
        # Initialize holding pool if not present
        if "import_holding_pool" not in st.session_state:
            st.session_state.import_holding_pool = []

        # Create section tabs below metadata
        tab_kite, tab_manual = st.tabs(["📤 Import from Kite", "📊 Add Holdings Manually"])

        # 2a. Kite CSV Import (one-click)
        with tab_kite:
            with st.container(border=True):
                st.markdown("#### 📤 Import from Kite")
                st.caption("Go to Kite Console → Holdings → Download CSV, then upload it here.")
                _kite_csv = st.file_uploader("Upload Kite Holdings CSV", type=["csv"], key="kite_csv_upload")
                if _kite_csv is not None:
                    try:
                        _kdf = pd.read_csv(_kite_csv)
                        # Normalize column names (strip whitespace, lowercase for matching)
                        _kdf.columns = [c.strip() for c in _kdf.columns]
                        _instr_col = next((c for c in _kdf.columns if c.lower() == "instrument"), None)
                        _qty_col = next((c for c in _kdf.columns if c.lower().startswith("qty")), None)
                        _avg_col = next((c for c in _kdf.columns if "avg" in c.lower() and "cost" in c.lower()), None)
                        if not _instr_col or not _qty_col or not _avg_col:
                            st.error(f"Could not find required columns. Found: {list(_kdf.columns)}. Expected: Instrument, Qty., Avg. cost")
                        else:
                            _matched = []
                            _unmatched = []
                            _universe_tickers = set(universe_df["ticker"].tolist()) if universe_df is not None else set()
                            for _, _kr in _kdf.iterrows():
                                _sym = str(_kr[_instr_col]).strip().upper()
                                _qty = int(float(_kr[_qty_col])) if pd.notna(_kr[_qty_col]) else 0
                                _avg = float(_kr[_avg_col]) if pd.notna(_kr[_avg_col]) else 0.0
                                if _qty <= 0 or _avg <= 0:
                                    continue
                                # Try NSE first, then BSE
                                _resolved = None
                                for _suffix in [".NS", ".BO"]:
                                    if f"{_sym}{_suffix}" in _universe_tickers:
                                        _resolved = f"{_sym}{_suffix}"
                                        break
                                if _resolved:
                                    _urow = universe_df[universe_df["ticker"] == _resolved].iloc[0]
                                    _matched.append({
                                        "ticker": _resolved,
                                        "name": _urow.get("name", _sym),
                                        "shares": _qty,
                                        "price": _avg,
                                    })
                                else:
                                    _unmatched.append(_sym)
                            if _matched:
                                st.success(f"Matched {len(_matched)} of {len(_matched) + len(_unmatched)} instruments to Kordent universe.")
                                _preview_df = pd.DataFrame(_matched)
                                st.dataframe(_preview_df[["name", "ticker", "shares", "price"]], hide_index=True, use_container_width=True)
                                if _unmatched:
                                    st.warning(f"Could not match: {', '.join(_unmatched)}. These will be skipped.")
                                if st.button("✅ Use these holdings", use_container_width=True, key="kite_csv_accept"):
                                    st.session_state.import_holding_pool = _matched
                                    st.rerun()
                            else:
                                st.error("No instruments matched the Kordent universe. Check the CSV format.")
                                if _unmatched:
                                    st.caption(f"Unmatched: {', '.join(_unmatched)}")
                    except Exception as _csv_err:
                        st.error(f"Failed to parse CSV: {_csv_err}")

        # 2b. Manual Asset Adder (Searchable Dropdown)
        with tab_manual:
            with st.container(border=True):
                st.markdown("#### 📊 Add Holdings Manually")
                
                # Create a list of "Company Name (TICKER)" from your existing universe_df
                stock_options = [
                    f"{row.get('name', row['ticker'])} ({row['ticker']})" 
                    for _, row in universe_df.iterrows()
                ]
                
                selected_stock = st.selectbox("🔍 Search & Select Company (Type to filter)", stock_options)
                
                # Preview CSV price as default, but let user override with actual buy price
                _ticker_preview = selected_stock.split("(")[-1].replace(")", "").strip()
                _row_preview = universe_df[universe_df["ticker"] == _ticker_preview]
                _default_price = float(_row_preview["price"].iloc[0]) if len(_row_preview) and pd.notna(_row_preview["price"].iloc[0]) else 0.0
                
                col_sh, col_px = st.columns(2)
                with col_sh:
                    shares_to_add = st.number_input("Shares Owned", min_value=1, value=10, step=1)
                with col_px:
                    price_paid = st.number_input("Avg Buy Price (₹)", min_value=0.01, value=_default_price, format="%.2f")
                
                if st.button("➕ Add to List", width="stretch"):
                    ticker_resolved = _ticker_preview
                    company_name = selected_stock.split(" (")[0]
                    
                    st.session_state.import_holding_pool.append({
                        "ticker": ticker_resolved,
                        "name": company_name,
                        "shares": shares_to_add,
                        "price": price_paid
                    })
                    st.success(f"Added {ticker_resolved} to your staging list.")
                    st.rerun()

        # 3. Present Staging List Table
        if st.session_state.import_holding_pool:
            st.markdown("#### Staging Review Table")
            staging_df = pd.DataFrame(st.session_state.import_holding_pool)
            st.dataframe(staging_df[["name", "ticker", "shares", "price"]], hide_index=True, width="stretch")
            
            if st.button("🗑️ Clear List"):
                st.session_state.import_holding_pool = []
                st.rerun()
                
            # 4. Save and Trigger Instant Review
            if st.button("💾 Save & Run Instant Analysis", width="stretch"):
                if not p_name:
                    st.error("Please provide a name for this portfolio.")
                else:
                    try:
                        sb = get_supabase()
                        review_days = 90 if horizon == "medium" else (180 if horizon == "long" else 30)
                        next_review = (datetime.date.today() + datetime.timedelta(days=review_days)).isoformat()
                        next_sip = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
                        
                        _port_data = {
                            "user_id": st.session_state.sb_user_id,
                            "name": p_name,
                            "investor_type": inv_type,
                            "sip_amount": sip_amt,
                            "time_horizon": horizon,
                            "review_freq": str(review_days),
                            "next_review_date": next_review,
                            "next_sip_date": next_sip,
                            "is_paper": False
                        }
                        
                        port_resp = sb.table("portfolios").insert(_port_data).execute()
                        portfolio_id = port_resp.data[0]["id"]
                        
                        for s in st.session_state.import_holding_pool:
                            row = universe_df[universe_df["ticker"] == s["ticker"]]
                            pe = float(row["pe"].iloc[0]) if len(row) and pd.notna(row["pe"].iloc[0]) else None
                            roe = float(row["roe_y0"].iloc[0]) if len(row) and "roe_y0" in row.columns and pd.notna(row["roe_y0"].iloc[0]) else None
                            score = int(row["score"].iloc[0]) if len(row) and pd.notna(row["score"].iloc[0]) else None
                            sect = str(row["sector"].iloc[0]) if len(row) and "sector" in row.columns and pd.notna(row["sector"].iloc[0]) else "Unknown"
                            
                            _imp_invested = round(s["shares"] * s["price"], 2)
                            sb.table("holdings").insert({
                                "portfolio_id": portfolio_id, 
                                "ticker": s["ticker"], 
                                "name": s["name"],
                                "sector": sect, 
                                "allocation_pct": 0, 
                                "shares": s["shares"],
                                "sip_amount_inr": _imp_invested, 
                                "price_at_entry": s["price"],
                                "pe_at_entry": pe, 
                                "roe_at_entry": roe, 
                                "score_at_entry": score
                            }).execute()
                            record_transaction(sb, portfolio_id, st.session_state.sb_user_id, s["ticker"], s["shares"], s["price"], _imp_invested, "buy")
                        
                        st.session_state.import_holding_pool = []
                        st.session_state[f"auto_trigger_review_{portfolio_id}"] = True
                        st.session_state.sb_view_mode = "portfolios"
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to onboard portfolio: {e}")

elif st.session_state.sb_view_mode == "builder":
    st.markdown("### 🏗️ Build Your Portfolio")
    st.caption("Answer a few simple questions — no financial jargon, we promise.")

    def _sip_future_value(monthly, annual_pct, years):
        n = years * 12
        r = annual_pct / 12 / 100
        if r <= 0:
            return monthly * n
        return monthly * ((1 + r)**n - 1) / r * (1 + r)

    if st.session_state.sb_user_id is None:
        st.warning("Please log in via the sidebar to build a portfolio.")
    else:
        # ── SIP Calculator (interactive, outside form) ──
        st.markdown("#### 💰 SIP Calculator")
        st.caption("See how your money grows with compounding. Adjust to explore.")

        _calc_sip = st.number_input(
            "Monthly investment (₹)",
            min_value=500, max_value=10000000, value=5000, step=500,
            help="Start small — you can always increase later.",
        )
        _calc_slider_cols = st.columns(2)
        with _calc_slider_cols[0]:
            _calc_years = st.slider(
                "⏳ Time period (years)", min_value=1, max_value=30, value=15,
            )
        with _calc_slider_cols[1]:
            _calc_return = st.slider(
                "📈 Expected return (% p.a.)",
                min_value=6.0, max_value=20.0, value=13.0, step=0.5,
                help="8%: Conservative · 12%: Equity avg · 16%+: Aggressive",
            )

        _total_invested = _calc_sip * 12 * _calc_years
        _future_value = _sip_future_value(_calc_sip, _calc_return, _calc_years)
        _est_returns = _future_value - _total_invested

        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("Invested", f"{fmt_inr(_total_invested)}")
        _m2.metric("Est. Returns", f"{fmt_inr(_est_returns)}")
        _m3.metric("Total Value", f"{fmt_inr(_future_value)}")

        # Stacked bar chart — the compounding hockey stick
        _yr_range = list(range(1, _calc_years + 1))
        _inv_arr = [_calc_sip * 12 * y for y in _yr_range]
        _fv_arr = [_sip_future_value(_calc_sip, _calc_return, y) for y in _yr_range]
        _ret_arr = [fv - inv for fv, inv in zip(_fv_arr, _inv_arr)]

        _chart_fig = go.Figure()
        _chart_fig.add_trace(go.Bar(
            name="Invested", x=[f"Yr {y}" for y in _yr_range],
            y=_inv_arr, marker_color="#9CA3AF",
        ))
        _chart_fig.add_trace(go.Bar(
            name="Returns", x=[f"Yr {y}" for y in _yr_range],
            y=_ret_arr, marker_color="#10B981",
        ))
        _chart_fig.update_layout(
            barmode="stack", height=280,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            xaxis_title="", yaxis_title="",
            yaxis_tickprefix="₹",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_chart_fig, use_container_width=True, key="sip_calc_chart")

        # Optional goal mode
        _calc_has_goal = st.checkbox("🎯 I have a specific savings goal", value=False)
        _calc_goal_amt = 0
        if _calc_has_goal:
            _calc_goal_amt = st.number_input(
                "Target amount (₹)", min_value=10000, value=max(10000, int(_future_value)), step=100000,
            )
            if _calc_goal_amt > _total_invested:
                _lo, _hi = 0.0, 50.0
                for _ in range(100):
                    _mid = (_lo + _hi) / 2
                    if _sip_future_value(_calc_sip, _mid, _calc_years) < _calc_goal_amt:
                        _lo = _mid
                    else:
                        _hi = _mid
                _req_return = round((_lo + _hi) / 2, 1)
                if _req_return > 20:
                    st.warning(f"⚠️ Requires ~{_req_return}% annual returns — consider increasing your SIP or extending the time period.")
                elif abs(_req_return - _calc_return) < 1:
                    st.success("✅ Your current settings already reach this goal!")
                else:
                    st.info(f"📊 To reach {fmt_inr(_calc_goal_amt)} in {_calc_years} years, you'd need ~{_req_return}% annual returns.")
            else:
                st.success("✅ Your SIP already covers this goal even without any returns!")

        st.divider()
        st.markdown("#### 📋 Your Preferences")

        # Age (outside form — interactive, like SIP calculator)
        _calc_age = st.number_input(
            "🎂 Your age",
            min_value=18, max_value=80, value=30, step=1,
            help="Helps us tailor risk and time horizon to your life stage.",
        )

        with st.form("portfolio_builder_form"):
            # Q4 — Risk tolerance
            _b_risk_resp = st.radio(
                "📉 If your portfolio dropped 20% in a month, would you:",
                options=[
                    "Buy more — it's on sale!",
                    "Hold and wait for recovery",
                    "Sell some to sleep better",
                ],
                index=1,
            )

            st.divider()

            # Q5 — Sector exclusions
            _b_avoid = st.multiselect(
                "🚫 Any industries you want to avoid?",
                options=[
                    "Energy", "Basic Materials", "Utilities",
                    "Real Estate", "Financial Services", "Industrials",
                ],
                default=[],
                help="Select sectors to stay away from. Leave empty for no preference.",
            )

            st.divider()

            # Q6 — Income need only. The volatility half moved out; Q4 owns that job.
            _b_pref_resp = st.radio(
                "🎚️ Do you need this portfolio to pay you along the way?",
                options=[
                    "Pay me regularly — dividends matter",
                    "Purely growth — reinvest everything",
                ],
                index=1,
                help="This is about income need, not risk — your comfort with drops is covered above.",
            )

            st.divider()

            # Q7 — Investing identity. Each option maps to exactly ONE axis.
            # Dropped "diamonds in the rough": it asked the user to lower a SAFETY floor
            # (turnaround risk) while dressed as a style preference. Floors are not concessions.
            _b_philosophy = st.radio(
                "🧠 Which investing style sounds most like you?",
                options=[
                    "Businesses growing fast — I'll pay a fair price for that",
                    "Bargains — sound businesses trading below what they're worth",
                    "The most durable businesses — I'll accept a full price and slower growth",
                    "Balanced — no strong preference",
                ],
                index=3,
                help="This sets which quality we lean toward when two stocks are otherwise close.",
            )

            st.divider()

            # Q8 — Selectivity (Sprint 6)
            _b_selectivity = st.radio(
                "🎯 How selective should we be?",
                options=[
                    "Only the best — 4+ frameworks must agree",
                    "Quality picks — 3+ frameworks agree",
                    "Open to opportunities — 2+ with a compelling reason",
                ],
                index=1,
                help="Lower selectivity means more opportunities but each needs stronger individual merit.",
            )

            st.divider()

            # Q9 — Acceptable trade-off (Sprint 6)
            _b_tradeoff = st.radio(
                "⚖️ If a stock fails one test, which trade-off are you most comfortable with?",
                options=[
                    "Not the cheapest, but growing fast with strong earnings",
                    "Not the fastest grower, but deeply undervalued right now",
                    "Not the widest moat, but priced so low the risk-reward is worth it",
                    "No preference — I trust the system to decide",
                ],
                index=3,
                help="This tells us which 'imperfect' stocks might actually be perfect for you.",
            )

            st.divider()

            # Q10 — Paper portfolio toggle
            _b_is_paper = st.checkbox(
                "👁 Watch only — don't invest yet (paper portfolio)",
                value=False,
                help="Track how this portfolio would perform without putting real money in.",
            )

            _b_submitted = st.form_submit_button("Build My Portfolio →", use_container_width=True)

        if _b_submitted:
            # ── Map responses to profile dict ──
            _risk_map = {
                "Buy more — it's on sale!": "aggressive",
                "Hold and wait for recovery": "moderate",
                "Sell some to sleep better": "conservative",
            }
            _pref_map = {
                "Pay me regularly — dividends matter": "income",
                "Purely growth — reinvest everything": "growth",
            }
            _b_risk = _risk_map.get(_b_risk_resp, "moderate")
            _b_pref = _pref_map.get(_b_pref_resp, "growth")
            if _calc_years <= 3:
                _b_time = "short"
            elif _calc_years <= 7:
                _b_time = "medium"
            else:
                _b_time = "long"

            # investor_type from risk × preference
            if _b_risk == "conservative" or _b_pref == "income":
                _b_inv_type = "defensive"
            elif _b_risk == "aggressive" and _b_pref == "growth":
                _b_inv_type = "enterprising"
            else:
                _b_inv_type = "balanced"

            # review cadence from investor_type
            _rev_map = {"defensive": ("passive", 180), "balanced": ("moderate", 90), "enterprising": ("active", 60)}
            _b_rev_freq, _b_rev_days = _rev_map.get(_b_inv_type, ("moderate", 90))

            # Goal from SIP calculator
            _b_target_amt = _calc_goal_amt if _calc_has_goal and _calc_goal_amt > 0 else None
            _b_target_dt = datetime.date.today() + datetime.timedelta(days=int(_calc_years * 365.25))

            # Sprint 6: Map philosophy, selectivity, trade-off
            _philosophy_map = {
                "Businesses growing fast — I'll pay a fair price for that": "growth_at_fair_price",
                "Bargains — sound businesses trading below what they're worth": "deep_value",
                "The most durable businesses — I'll accept a full price and slower growth": "quality_compounder",
                "Balanced — no strong preference": "balanced",
            }
            _selectivity_map = {
                "Only the best — 4+ frameworks must agree": 4,
                "Quality picks — 3+ frameworks agree": 3,
                "Open to opportunities — 2+ with a compelling reason": 2,
            }
            _tradeoff_map = {
                "Not the cheapest, but growing fast with strong earnings": "ok_fail_graham",
                "Not the fastest grower, but deeply undervalued right now": "ok_fail_trajectory_lynch",
                "Not the widest moat, but priced so low the risk-reward is worth it": "ok_fail_dorsey_buffett",
                "No preference — I trust the system to decide": "any",
            }

            _b_philosophy_val = _philosophy_map.get(_b_philosophy, "balanced")
            _b_min_score = _selectivity_map.get(_b_selectivity, 3)
            _b_tradeoff_val = _tradeoff_map.get(_b_tradeoff, "any")

            # Sprint 6: Framework weights based on philosophy
            _framework_weights = {
                "deep_value":           {"graham": 35, "greenblatt": 25, "dorsey_buffett": 15, "trajectory": 10, "lynch": 15},
                "growth_at_fair_price": {"graham": 10, "greenblatt": 15, "dorsey_buffett": 20, "trajectory": 25, "lynch": 30},
                "quality_compounder":   {"graham": 15, "greenblatt": 15, "dorsey_buffett": 35, "trajectory": 15, "lynch": 20},
                "contrarian":           {"graham": 25, "greenblatt": 30, "dorsey_buffett": 10, "trajectory": 20, "lynch": 15},
                "balanced":             {"graham": 20, "greenblatt": 20, "dorsey_buffett": 20, "trajectory": 20, "lynch": 20},
            }

            _b_profile = {
                "sip_amount": _calc_sip,
                "target_amount": _b_target_amt if _b_target_amt else None,
                "target_date": _b_target_dt.isoformat() if _b_target_amt else None,
                "expected_return_pct": _calc_return,
                "risk": _b_risk,
                "avoid_sectors": _b_avoid,
                "preference": _b_pref,
                "investor_type": _b_inv_type,
                "time_horizon": _b_time,
                "review_freq": _b_rev_freq,
                "review_days": _b_rev_days,
                "is_paper": _b_is_paper,
                # Sprint 6 additions
                "philosophy": _b_philosophy_val,
                "min_acceptable_score": _b_min_score,
                "acceptable_tradeoff": _b_tradeoff_val,
                "framework_weights": _framework_weights.get(_b_philosophy_val, {}),
            }
            # Demand tilt from Q4/Q6/Q7/Q9. Stored now, consumed by selection later
            # (see Consumption Map) — a lean on the relative layer only.
            _b_profile["demand_tilt"] = derive_demand_tilt(_b_profile)
            _b_demand_warn = detect_demand_contradiction(_b_profile)
            if _b_demand_warn:
                _b_profile["demand_contradiction"] = _b_demand_warn
                st.info(_b_demand_warn)
            # Sprint 11: Generate IPS from profile (book is the standard)
            _b_ips = generate_ips(_b_profile, age=_calc_age)
            _b_profile["ips_policy"] = _b_ips
            _b_profile["age"] = _calc_age
            st.session_state.builder_profile = _b_profile

            # ── DETERMINISTIC BUILD — no LLM in construction ──
            # The system builds the portfolio from universe_scored + IPS.
            # No chat, no clarification questions, no LLM stock-picking.
            import json as _json
            # Q7, Q8 and Q9 were computed, stored in _b_profile, handed to
            # generate_ips — and then never reached the selector. Five arguments
            # went in; philosophy, min_acceptable_score, acceptable_tradeoff and
            # framework_weights were dropped on the floor. That is why every
            # answer produced the same fifteen stocks.
            with st.spinner("🏗️ Building your portfolio — scoring the universe and selecting your stocks. This takes a moment…"):
                _cand_result = get_sip_candidates(
                    sip_amount=_calc_sip,
                    time_horizon=_b_time,
                    investor_type=_b_inv_type,
                    review_freq=_b_rev_freq,
                    avoid_sectors=_json.dumps(_b_avoid or []),
                    min_acceptable_score=_b_min_score,
                    philosophy=_b_philosophy_val,
                    acceptable_tradeoff=_b_tradeoff_val,
                    framework_weights=_json.dumps(_framework_weights.get(_b_philosophy_val, {})),
                )
            st.session_state._built_portfolio = _cand_result.get("recommended_portfolio", [])
            st.session_state._built_web_grounding = _cand_result.get("web_grounding", {})
            st.session_state._built_diagnostics = _cand_result.get("selection_diagnostics", {})
            st.session_state._built_warnings = _cand_result.get("selection_warnings", [])
            st.session_state._built_near_misses = _cand_result.get("near_misses", [])
            st.session_state._built_is_paper = _b_is_paper
            st.session_state.sb_view_mode = "build_result"
            st.rerun()

elif st.session_state.sb_view_mode == "build_result":
    st.markdown("### \U0001F4CA Your Portfolio")
    _built = st.session_state.get("_built_portfolio", [])
    _bwg = st.session_state.get("_built_web_grounding", {})
    _bdiag = st.session_state.get("_built_diagnostics", {})
    _bprof = st.session_state.get("builder_profile", {}) or {}

    if not _built:
        st.error("The deterministic builder returned no stocks. This usually means the "
                 "universe filters were too strict for your constraints. Try a lower minimum "
                 "score or a broader risk profile.")
        if st.button("\u2190 Back to Builder"):
            st.session_state.builder_profile = None
            st.session_state.sb_view_mode = "builder"
            st.rerun()
    else:
        # Distress drops (if any) surfaced honestly
        _dropped = st.session_state.get("_distress_dropped", {})
        if _dropped:
            _dl = "; ".join(f"{n} ({r})" for n, r in _dropped.items())
            # No longer "and replaced". Distressed tickers are excluded from the
            # universe and the whole selection is re-run deterministically —
            # the old patch-in-a-replacement could violate a quota already met.
            st.warning(f"\u26A0\uFE0F Removed for distress signals: {_dl}")

        # Honest under-diversification, surfaced rather than padded away.
        # A 9-stock portfolio the user understands beats a 15-stock portfolio
        # padded with names nobody chose.
        for _w in st.session_state.get("_built_warnings", []):
            st.warning(f"\u26A0\uFE0F {_w}")

        # Translator: phrase the deterministic reason-traces (LLM = translator only)
        if "_built_explanations" not in st.session_state:
            with st.spinner("Preparing explanations..."):
                st.session_state._built_explanations = _explain_portfolio(_built, _bwg, _bdiag)
        _expl = st.session_state._built_explanations

        st.markdown(f"_{_expl.get('_portfolio', '')}_")

        import pandas as _pd
        _rows = []
        for _s in _built:
            _rows.append({
                "Stock": _s.get("name", _s["ticker"]),
                "Ticker": _s["ticker"],
                "Sector": _s.get("sector", ""),
                "Score": _s.get("score", ""),
                "Alloc %": _s.get("allocation_pct", ""),
                "Why": _expl.get(_s["ticker"], ""),
            })
        # Render as Markdown, NOT st.dataframe. st.dataframe serializes every
        # column through pyarrow, and pyarrow 25.0.0 on the Cloud runtime
        # segfaults (native, uncatchable) inside pandas_compat.convert_column
        # on this table — confirmed by the faulthandler dump. A segfault can't
        # be caught, so the only reliable fix is to not call the crashing path.
        # Markdown touches no pyarrow. Cells are scrubbed of invalid unicode
        # (lone surrogates from LLM 'Why' text) and pipe/newline chars.
        def _md_cell(_v):
            _t = "" if _v is None else str(_v)
            _t = _t.encode("utf-8", "replace").decode("utf-8")
            return _t.replace("|", "\\|").replace("\n", " ").strip()
        _cols = ["Stock", "Ticker", "Sector", "Score", "Alloc %", "Why"]
        _md = "| " + " | ".join(_cols) + " |\n"
        _md += "| " + " | ".join(["---"] * len(_cols)) + " |\n"
        for _row in _rows:
            _md += "| " + " | ".join(_md_cell(_row[_c]) for _c in _cols) + " |\n"
        st.markdown(_md)

        # ── Shortfall disclosure at DECISION time (before saving) ──
        # If the deterministic selector could not reach the IPS target even from
        # the widened 200-deep pool, that is genuine scarcity — show WHY now, so
        # the user decides with the tradeoff visible, not swallowed post-hoc.
        _ips_target = ((_bprof.get("ips_policy", {}) or {}).get("portfolio_sizing", {})
                       or {}).get("actual", 0)
        if _ips_target and len(_built) < _ips_target:
            st.warning(
                f"⚠️ **{len(_built)} stocks vs your IPS target of {_ips_target}.** "
                f"Your universe cannot supply {_ips_target} sector-legal, "
                f"quality-passing names under these constraints — {len(_built)} is the "
                f"diversified maximum. Reaching {_ips_target} would require breaching a "
                f"sector cap or admitting illiquid/low-quality stocks, both of which "
                f"*raise* unsystematic risk. Per Reilly & Brown, the constraint-limited "
                f"count is the book-compliant answer. You may still save it below."
            )

        _pname = st.text_input("Portfolio name",
            value=f"{_bprof.get('philosophy', 'Quality')} SIP - {datetime.date.today().strftime('%B %Y')}")

        _c1, _c2 = st.columns(2)
        with _c1:
            if st.button("\U0001F4BE Save Portfolio", width="stretch", type="primary"):
                # register_portfolio validates IPS + auto-fixes + stages pending_portfolio
                _reg = register_portfolio(
                    portfolio_name=_pname,
                    investor_type=_bprof.get("investor_type", "balanced"),
                    sip_amount=_bprof.get("sip_amount", 5000),
                    time_horizon=_bprof.get("time_horizon", "long"),
                    review_days=_bprof.get("review_days", 180),
                    stocks_json=json.dumps([
                        {"ticker": s["ticker"], "name": s.get("name", ""),
                         "sector": s.get("sector", ""), "allocation_pct": s.get("allocation_pct", 0)}
                        for s in _built
                    ]),
                    portfolio_profile=json.dumps(_bprof),
                )
                if _reg.get("error"):
                    st.error(f"Save blocked: {_reg['error']}")
                else:
                    # register_portfolio staged pending_portfolio; commit it
                    _pending = st.session_state.get("pending_portfolio")
                    if not _pending:
                        st.error("Internal error: portfolio was not staged for save.")
                    else:
                        _pending["is_paper"] = st.session_state.get("_built_is_paper", False)
                        _res = _commit_portfolio(_pending)
                        if not _res.get("ok"):
                            st.error(f"Save failed: {_res.get('error')}")
                        else:
                            st.success(f"Portfolio saved! Invested {fmt_inr(_res['invested'])} of {fmt_inr(_bprof.get('sip_amount', 5000))}.")
                            if _res.get("stale_priced"):
                                st.warning(f"⚠️ Live price unavailable for {', '.join(_res['stale_priced'])} — used last known close. Verify on Kite before paying.")
                            st.session_state.builder_profile = None
                            # Every key set alongside _built_portfolio must be cleared
                            # with it. A stale _built_diagnostics would render last
                            # build's pool size against this build's holdings.
                            for _k in ("_built_portfolio", "_built_web_grounding", "_built_is_paper",
                                       "_distress_dropped", "_built_explanations",
                                       "_built_diagnostics", "_built_warnings", "_built_near_misses"):
                                st.session_state.pop(_k, None)
                            st.session_state.sb_view_mode = "portfolios"
                            st.rerun()
        with _c2:
            if st.button("\u2190 Rebuild", width="stretch"):
                st.session_state.builder_profile = None
                for _k in ("_built_explanations", "_built_diagnostics",
                           "_built_warnings", "_built_near_misses"):
                    st.session_state.pop(_k, None)
                st.session_state.sb_view_mode = "builder"
                st.rerun()

elif st.session_state.sb_view_mode == "portfolios":
    st.markdown("### \U0001F4C1 My Portfolios")
    sb = get_supabase()
    try:
        port_resp = sb.table("portfolios").select("*").eq(
            "user_id", st.session_state.sb_user_id
        ).order("created_at", desc=True).execute()
        portfolios = [p for p in (port_resp.data or []) if not p.get("is_paper")]
    except Exception as e:
        st.error(f"Failed to load portfolios: {e}")
        portfolios = []

    if not portfolios:
        st.info("No saved portfolios yet. Click 🏗️ Build Portfolio in the sidebar to get started!")
    else:
        for port in portfolios:
            with st.container(border=True):
                _pv = port.get("current_value")
                _pr = port.get("current_return_pct")
                _header = f"**{port['name']}**"
                if _pv:
                    _header += f" · {fmt_inr(_pv)}"
                if _pr is not None:
                    _header += f" ({_pr:+.1f}%)"
                _div = port.get("diversification_score")
                if _div is not None:
                    _div_emoji = "🟢" if _div >= 70 else "🟡" if _div >= 40 else "🔴"
                    _header += f" · {_div_emoji} {_div}/100"
                st.markdown(_header)
                # ── Light refresh (#2): re-fetch live prices, recompute value &
                # return only. Risk metrics (Sharpe/beta/drawdown) are left as the
                # daily tracker wrote them — they don't move intraday and need a
                # year of history to recompute. cache_key=None forces a fresh pull.
                if st.button("↻ Refresh", key=f"refresh_port_{port['id']}",
                             help="Re-fetch live prices and recompute value & return"):
                    try:
                        _rh = (sb.table("holdings")
                               .select("ticker, shares, price_at_entry, sip_amount_inr")
                               .eq("portfolio_id", port["id"]).execute().data) or []
                        if _rh:
                            _enr = enrich_holdings_live(_rh, cache_key=None)
                            # Was recomputing (value - surviving cost basis) and
                            # writing it over the tracker's correct figure, which
                            # made every refresh after a sale re-introduce the bug.
                            _e = portfolio_money(sb, port["id"], _enr, port.get("benchmark_ticker"))
                            sb.table("portfolios").update({
                                "current_value": _e["total_assets"],
                                "current_return_pct": _e["return_pct"],
                                "cash_balance": _e["cash_balance"],
                                "realized_pnl": _e["realized_pnl"],
                                "withdrawn": _e["withdrawn"],
                            }).eq("id", port["id"]).execute()
                            _rp_txt = f"{_e['return_pct']:+.1f}%" if _e["return_pct"] is not None else "n/a"
                            st.toast(f"Refreshed: {fmt_inr(_e['total_assets'])} ({_rp_txt})")
                            st.rerun()
                        else:
                            st.toast("No holdings to refresh.")
                    except Exception as _e:
                        st.error(f"Refresh failed: {_e}")

                # ── Withdraw cash (Sprint 15) ──
                # Sale proceeds sit as cash until they are redeployed or taken out.
                # Without this the ledger never learns the money left: the next buy
                # gets funded from cash that no longer exists, external capital is
                # understated, and every return after that point is overstated.
                _cash_now = float(port.get("cash_balance") or 0)
                if _cash_now > 0:
                    with st.expander(f"💸 Withdraw cash ({fmt_inr(_cash_now)} uninvested)"):
                        st.caption(
                            "Record money you moved out of your broker to your own bank. "
                            "This does not reduce your invested capital — that already "
                            "happened. It records value coming back to you, which is what "
                            "keeps your return honest."
                        )
                        _wd_amt = st.number_input(
                            "Amount withdrawn (Rs.)", min_value=0.0, max_value=float(_cash_now),
                            value=float(_cash_now), format="%.2f", key=f"wd_amt_{port['id']}",
                        )
                        if st.button("Record withdrawal", key=f"wd_go_{port['id']}", width="stretch"):
                            if _wd_amt <= 0:
                                st.error("Enter an amount above 0.")
                            else:
                                try:
                                    record_withdrawal(sb, port, st.session_state.sb_user_id, _wd_amt)
                                    _e2 = portfolio_money(
                                        sb, port["id"],
                                        enrich_holdings_live(
                                            sb.table("holdings").select("*").eq(
                                                "portfolio_id", port["id"]).execute().data or [],
                                            cache_key=str(port["id"])),
                                        port.get("benchmark_ticker"))
                                    sb.table("portfolios").update({
                                        "current_value": _e2["total_assets"],
                                        "current_return_pct": _e2["return_pct"],
                                        "cash_balance": _e2["cash_balance"],
                                        "realized_pnl": _e2["realized_pnl"],
                                        "withdrawn": _e2["withdrawn"],
                                    }).eq("id", port["id"]).execute()
                                    st.success(f"Recorded withdrawal of {fmt_inr(_wd_amt)}.")
                                    st.rerun()
                                except Exception as _we:
                                    st.error(f"Withdrawal not recorded: {_we}")
                _px = port.get("xirr_pct")
                if _px is not None:
                    _nx = port.get("nifty_xirr_pct")
                    _xirr_parts = [f"XIRR: {_px:+.1f}%"]
                    if _nx is not None:
                        _xirr_parts.append(f"Alpha: {round(_px - _nx, 1):+.1f}%")
                    st.caption(" · ".join(_xirr_parts))
                if port.get("paper_converted_at"):
                    st.caption(f"📋 Converted from paper on {str(port['paper_converted_at'])[:10]}")

                _beta = port.get("portfolio_beta")
                _sharpe = port.get("sharpe_ratio")
                _alpha = port.get("jensen_alpha")
                if _beta is not None:
                    _header += f" · β {_beta:.2f}"
                if _sharpe is not None:
                    _header += f" · Sharpe {_sharpe:.2f}"
                if _alpha is not None:
                    _alpha_sign = "+" if _alpha >= 0 else ""
                    _header += f" · α {_alpha_sign}{_alpha*100:.1f}%"

                # ── Alert Banner ──
                try:
                    alerts_resp = sb.table("portfolio_alerts").select("*").eq(
                        "portfolio_id", port["id"]
                    ).eq("is_read", False).order("created_at", desc=True).execute()
                    port_alerts = alerts_resp.data

                    # Broadcast alerts (new_entry) now handled by weekly recommendations tab
                except Exception:
                    port_alerts = []

                if port_alerts:
                    for alert in port_alerts:
                        a_type = alert["alert_type"]
                        if a_type in ("opportunity", "new_entry"):
                            continue
                        a_id = alert["id"]
                        detail = alert.get("detail") or {}
                        if isinstance(detail, str):
                            import json as _json
                            try:
                                detail = _json.loads(detail)
                            except Exception:
                                detail = {}

                        # Type picks WHICH card, severity picks HOW LOUD. These
                        # were one value until now, which is why a portfolio-level
                        # warning (ticker '_portfolio') fell into the holding card
                        # below, looked up a holding named '_portfolio', found
                        # none, and told the user it had already been sold.
                        #
                        # `or "danger"` covers rows written before the severity
                        # column existed. Those age out within 7 days via the
                        # cleanup in portfolio_tracker, so this fallback is
                        # temporary — but it must not be a crash in the meantime.
                        _sev = alert.get("severity") or "danger"
                        _say = {"danger": st.error, "warning": st.warning,
                                "info": st.info}.get(_sev, st.error)

                        if a_type in ("score_drop", "quality_fail", "price_crash"):
                            _say(f"🛡️ **{alert['headline']}**")
                            
                            # Replaced expander with a permanently open, bordered container
                            with st.container(border=True):
                                st.markdown("##### Defend Position")

                                # W1 drift reason. Only score_drop writes these
                                # keys, so quality_fail and price_crash fall
                                # through untouched. This is not decoration: the
                                # label already set the severity above, and an
                                # amber card with no stated reason is worse than
                                # the red one it replaced.
                                _dr = detail.get("drift_reason")
                                if _dr:
                                    _nf = detail.get("drift_newly_failing") or []
                                    _pf = detail.get("drift_per_framework") or {}
                                    if _nf:
                                        st.markdown(_drift_flip_line(
                                            "No longer passes", _nf, _pf))
                                    _note = _DRIFT_ALERT_NOTE.get(_dr)
                                    if _note:
                                        st.caption(_note)
                                
                                ticker = alert.get("ticker", "")
                                # Fetch holding for this portfolio directly from Supabase
                                try:
                                    h_resp = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).eq("ticker", ticker).execute()
                                    h_match = h_resp.data[0] if h_resp.data else None
                                except Exception:
                                    h_match = None

                                if h_match:
                                    max_shares = h_match.get("shares", 0)
                                    sell_qty = st.number_input(
                                        f"Shares to sell (of {max_shares})",
                                        min_value=0, max_value=max_shares, value=max_shares,
                                        key=f"defend_qty_{a_id}"
                                    )
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        _sell_px = st.number_input(
                                            "Sell price (Rs.)", min_value=0.0,
                                            value=float(live_price(ticker, h_match.get("price_at_entry", 0))),
                                            format="%.2f", key=f"defend_px_{a_id}",
                                            help="Defaults to last traded price. Override with your actual Kite fill.",
                                        )
                                        if st.button("🛡️ Confirm Sell", key=f"defend_confirm_{a_id}", width="stretch"):
                                            if sell_qty > 0 and _sell_px > 0:
                                                # Ledger first. If this insert fails the holding is
                                                # left untouched and the user can retry; the reverse
                                                # order loses the proceeds forever.
                                                try:
                                                    record_transaction(
                                                        sb, port["id"], st.session_state.sb_user_id,
                                                        ticker, sell_qty, _sell_px,
                                                        round(sell_qty * _sell_px, 2), "sell",
                                                        benchmark_ticker=port.get("benchmark_ticker"),
                                                        raise_on_error=True,
                                                    )
                                                except Exception as _e:
                                                    st.error(f"Sale not recorded, holding unchanged: {_e}")
                                                    st.stop()
                                                new_shares = max_shares - sell_qty
                                                if new_shares <= 0:
                                                    sb.table("holdings").delete().eq("id", h_match["id"]).execute()
                                                else:
                                                    new_invested = new_shares * h_match.get("price_at_entry", 0)
                                                    sb.table("holdings").update({
                                                        "shares": new_shares,
                                                        "sip_amount_inr": round(new_invested, 2)
                                                    }).eq("id", h_match["id"]).execute()
                                                sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                                st.success(f"Sold {sell_qty} shares of {ticker}.")
                                                st.rerun()
                                            elif sell_qty > 0:
                                                st.error("Enter a sell price above 0.")
                                    with c2:
                                        if st.button("Dismiss", key=f"defend_dismiss_{a_id}", width="stretch"):
                                            sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                            st.rerun()
                                else:
                                    st.caption("Holding not found — may have been sold already.")
                                    if st.button("Dismiss", key=f"defend_dismiss_nf_{a_id}"):
                                        sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                        st.rerun()

                        elif a_type == "goal_drift":
                            st.success(f"⚡ **{alert['headline']}**")

                            with st.container(border=True):
                                ticker = alert.get("ticker", "")
                                opp_name = detail.get("name", ticker)
                                live_price = float(detail.get("price", 0)) if detail.get("price") else 0.0
                                act_now = detail.get("act_now", False)
                                budget_left = float(port.get("sip_budget_remaining") or 0)
                                # Cash on hand BEFORE this buy, so the opportunity
                                # cap can be charged the external portion only.
                                _cash_before = float(port.get("cash_balance") or 0)

                                suggested_qty = int(budget_left // live_price) if live_price > 0 and budget_left > 0 else 0

                                if act_now and suggested_qty > 0:
                                    st.markdown(f"Budget left this month: **{fmt_inr(budget_left)}** · Price: **{fmt_inr(live_price, 2)}** · Suggested: **{suggested_qty} shares** (~{fmt_inr(suggested_qty * live_price)})")
                                elif live_price > budget_left and budget_left > 0:
                                    st.info(f"One share costs {fmt_inr(live_price)} but only {fmt_inr(budget_left)} left in this month's opportunity budget. Consider this at your next review.")
                                elif budget_left <= 0:
                                    st.info("This month's opportunity budget is used up. Noted for your weekly summary.")
                                else:
                                    st.markdown(f"Price: **{fmt_inr(live_price, 2)}**")

                                if KITE_ENABLED and live_price > 0:
                                    _kite_qty = suggested_qty if suggested_qty > 0 else 1
                                    st.link_button("🛒 Buy on Kite", kite_buy_url(ticker, quantity=_kite_qty), use_container_width=True)
                                    st.caption("After buying, confirm what you did:")

                                col_sh, col_px = st.columns(2)
                                with col_sh:
                                    buy_qty = st.number_input(
                                        "Shares to buy",
                                        min_value=0, value=suggested_qty, key=f"buy_qty_{a_id}"
                                    )
                                with col_px:
                                    buy_price = st.number_input(
                                        "Price per share (₹)",
                                        min_value=0.0, value=live_price,
                                        format="%.2f", key=f"buy_price_{a_id}"
                                    )

                                c1, c2 = st.columns(2)
                                with c1:
                                    if st.button("✅ Bought", key=f"buy_confirm_{a_id}", use_container_width=True):
                                        if buy_qty > 0 and buy_price > 0:
                                            try:
                                                invested = round(buy_qty * buy_price, 2)
                                                sb.table("holdings").insert({
                                                    "portfolio_id": port["id"],
                                                    "ticker": ticker,
                                                    "name": opp_name,
                                                    "sector": detail.get("sector", ""),
                                                    "allocation_pct": 0,
                                                    "shares": buy_qty,
                                                    "sip_amount_inr": invested,
                                                    "price_at_entry": round(buy_price, 2),
                                                    "score_at_entry": detail.get("score"),
                                                }).execute()
                                                record_transaction(sb, port["id"], st.session_state.sb_user_id, ticker, buy_qty, buy_price, invested, "buy",
                                                                   benchmark_ticker=port.get("benchmark_ticker"))
                                                # The opportunity cap limits NEW capital, not gross
                                                # spend. Buying with idle sale proceeds deploys nothing
                                                # new, so it must not burn the month's allowance.
                                                _external_spent = max(0.0, invested - _cash_before)
                                                new_budget = max(0, budget_left - _external_spent)
                                                sb.table("portfolios").update({
                                                    "sip_budget_remaining": round(new_budget, 2)
                                                }).eq("id", port["id"]).execute()
                                                sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                                st.success(f"Tracked {buy_qty} shares of {opp_name}. Budget remaining: {fmt_inr(new_budget)}")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Failed: {e}")
                                        else:
                                            st.warning("Enter shares and price.")
                                with c2:
                                    if st.button("⏭ Didn't buy", key=f"buy_skip_{a_id}", use_container_width=True):
                                        sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                        st.rerun()

                        
                        elif a_type == "goal_drift":
                            # Severity now varies: danger below half the needed
                            # CAGR, warning above it. Previously the severe case
                            # was routed to a different BRANCH entirely by writing
                            # alert_type='danger', so this block only ever ran for
                            # the mild case.
                            _say(f"🎯 **{alert['headline']}**")
                            with st.container(border=True):
                                actual = detail.get("actual_cagr_pct", 0)
                                needed = detail.get("needed_cagr_pct", 0)
                                months = detail.get("months_remaining", 0)
                                # Was: "Consider increasing your SIP or reviewing
                                # your picks — but don't chase risk." That is a
                                # recommendation about the user's money, which
                                # this system does not make. What replaces it is
                                # arithmetic, not counsel: a shortfall has
                                # exactly three variables, and naming them is
                                # strictly more useful than nominating one.
                                st.markdown(
                                    f"Your portfolio is growing at **{actual:.1f}%** "
                                    f"against the **{needed:.1f}%** needed to reach "
                                    f"your goal in {months} months.")
                                st.caption(
                                    "A gap like this closes in one of three ways: "
                                    "contributing more, allowing more time, or earning "
                                    "a higher return. The first two are yours to set. "
                                    "The third is not.")
                                if st.button("✗ Dismiss", key=f"goaldrift_dismiss_{a_id}", use_container_width=True):
                                    sb.table("portfolio_alerts").update({"is_read": True}).eq("id", a_id).execute()
                                    st.rerun()

                        elif a_type in ("sector_concentration", "low_diversification"):
                            # Portfolio-level: ticker is '_portfolio', so there is
                            # no holding to defend. Before the type/severity split
                            # these fell into the holding card above, looked up a
                            # holding named '_portfolio', found none, and rendered
                            # "Holding not found — may have been sold already."
                            _say(f"📊 **{alert['headline']}**")
                            with st.container(border=True):
                                if a_type == "sector_concentration":
                                    st.markdown(
                                        f"{detail.get('sector', 'One sector')} is "
                                        f"{detail.get('weight_pct', 0)}% of your holdings. "
                                        "Concentration raises the cost of being wrong about "
                                        "a single industry.")
                                else:
                                    st.markdown(
                                        f"Diversification score {detail.get('score', 0)}/100. "
                                        "A low score means your holdings tend to move together, "
                                        "so they cushion each other less than the count suggests.")
                                if st.button("✗ Dismiss", key=f"portrisk_dismiss_{a_id}",
                                             use_container_width=True):
                                    sb.table("portfolio_alerts").update(
                                        {"is_read": True}).eq("id", a_id).execute()
                                    st.rerun()

                        elif a_type == "review_due":
                            # Written every run and, until now, matched by no
                            # branch here — visible only in the PDF export. It is
                            # the one alert whose entire purpose is to be seen.
                            _say(f"⏰ **{alert['headline']}**")
                            with st.container(border=True):
                                _od = detail.get("days_overdue", 0)
                                st.markdown(
                                    f"Your scheduled review is {_od} day"
                                    f"{'s' if _od != 1 else ''} overdue. Reviewing is how "
                                    "drift gets caught early — it does not commit you to "
                                    "trading anything.")
                                if st.button("✗ Dismiss", key=f"reviewdue_dismiss_{a_id}",
                                             use_container_width=True):
                                    sb.table("portfolio_alerts").update(
                                        {"is_read": True}).eq("id", a_id).execute()
                                    st.rerun()

                
                # Check if this specific portfolio is in "SIP edit mode"
                is_editing_sip = st.session_state.get(f"edit_sip_{port['id']}", False)

                if is_editing_sip:
                    # Edit Mode UI
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        new_sip = st.number_input(
                            "New SIP Amount (₹)", 
                            value=int(port.get('sip_amount', 0)), 
                            step=1000, 
                            key=f"new_sip_input_{port['id']}", 
                            label_visibility="collapsed"
                        )
                    with col2:
                        if st.button("💾 Save", key=f"save_sip_{port['id']}", width="stretch"):
                            old_sip = int(port.get('sip_amount', 0))
                            try:
                                sb.table("portfolios").update({"sip_amount": new_sip}).eq("id", port["id"]).execute()
                                st.session_state[f"edit_sip_{port['id']}"] = False

                                # Sprint 11: SIP expansion check (Reilly & Brown Ch 6)
                                _profile = port.get("portfolio_profile") or {}
                                _ips = _profile.get("ips_policy") or {}
                                _old_affordable = max(3, old_sip // 500)
                                _new_affordable = max(3, new_sip // 500)
                                _current_count = len([h for h in sb.table("holdings").select("id").eq("portfolio_id", port["id"]).execute().data])
                                BOOK_MIN = 12

                                if new_sip > old_sip and _new_affordable > _current_count and _current_count < BOOK_MIN:
                                    # SIP increased AND can now support more stocks AND portfolio is under-diversified
                                    _can_add = min(_new_affordable, BOOK_MIN) - _current_count
                                    st.success(f"SIP updated to {fmt_inr(new_sip)}/mo!")
                                    st.info(
                                        f"📈 **Portfolio expansion recommended.** Your portfolio has {_current_count} stocks. "
                                        f"With {fmt_inr(new_sip)}/mo, you can support up to {min(_new_affordable, BOOK_MIN)} stocks "
                                        f"(book minimum: {BOOK_MIN} for adequate diversification). "
                                        f"Adding {_can_add} diversifying stocks would reduce your unsystematic risk."
                                    )
                                    # Set flag for expansion flow
                                    st.session_state[f"expand_portfolio_{port['id']}"] = {
                                        "current_count": _current_count,
                                        "target_count": min(_new_affordable, BOOK_MIN),
                                        "can_add": _can_add,
                                        "new_sip": new_sip,
                                    }
                                elif new_sip > old_sip:
                                    st.success(f"SIP updated to {fmt_inr(new_sip)}/mo! Extra capital will be distributed proportionally.")
                                else:
                                    st.success("Updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    with col3:
                        if st.button("❌", key=f"cancel_sip_{port['id']}", width="stretch"):
                            st.session_state[f"edit_sip_{port['id']}"] = False
                            st.rerun()
                    
                    # Show the rest of the caption without the SIP amount while editing
                    st.caption(
                        f"Created: {port['created_at'][:10]} · "
                        f"{port.get('investor_type', '—')} · "
                        f"{port.get('time_horizon', '—')} horizon · "
                        f"Review: every {port.get('review_freq', '90')} days · "
                        f"Next: {port.get('next_review_date', '—')}"
                    )
                else:
                    # Normal Mode UI with Edit Button
                    col_cap, col_btn = st.columns([11, 1])
                    with col_cap:
                        st.caption(
                            f"Created: {port['created_at'][:10]} · "
                            f"{port.get('investor_type', '—')} · "
                            f"**{fmt_inr(port.get('sip_amount', 0))}/mo** · "
                            f"{port.get('time_horizon', '—')} horizon · "
                            f"Review: every {port.get('review_freq', '90')} days · "
                            f"Next: {port.get('next_review_date', '—')}"
                        )
                    with col_btn:
                        if st.button("✏️", key=f"trigger_edit_sip_{port['id']}", help="Edit SIP Amount"):
                            st.session_state[f"edit_sip_{port['id']}"] = True
                            st.rerun()

                # Sprint 11: Portfolio expansion flow
                _expand = st.session_state.get(f"expand_portfolio_{port['id']}")
                if _expand:
                    with st.container(border=True):
                        st.markdown(f"### 🌱 Portfolio Expansion")
                        st.write(
                            f"Your portfolio has **{_expand['current_count']}** stocks. "
                            f"The book recommends at least **12** for meaningful diversification. "
                            f"We can add **{_expand['can_add']}** stocks from under-represented sectors "
                            f"to reduce unsystematic risk."
                        )
                        col_go, col_skip = st.columns(2)
                        with col_go:
                            if st.button("🔍 Review Expansion Candidates", key=f"expand_go_{port['id']}", use_container_width=True):
                                # Trigger expansion via chat — send a builder-like prompt
                                _held_tickers = [h["ticker"] for h in sb.table("holdings").select("ticker, sector").eq("portfolio_id", port["id"]).execute().data]
                                _held_sectors = [h.get("sector", "") for h in sb.table("holdings").select("ticker, sector").eq("portfolio_id", port["id"]).execute().data]
                                _sector_counts = {}
                                for _s in _held_sectors:
                                    _sector_counts[_s] = _sector_counts.get(_s, 0) + 1

                                _expand_prompt = (
                                    f"[EXPANSION] My portfolio '{port['name']}' currently has {_expand['current_count']} stocks "
                                    f"across these sectors: {dict(_sector_counts)}. "
                                    f"I just increased my SIP to {fmt_inr(_expand['new_sip'])}/mo. "
                                    f"I need {_expand['can_add']} MORE stocks to reach the book minimum of 12. "
                                    f"The portfolio philosophy is {port.get('investor_type', 'balanced')}. "
                                    f"CRITICAL: New stocks MUST be from sectors NOT already heavily represented. "
                                    f"Existing tickers (do NOT duplicate): {_held_tickers}. "
                                    f"Use get_sip_candidates to find candidates, then filter for LOW correlation "
                                    f"with existing holdings. Present the expansion candidates with diversification rationale."
                                )
                                st.session_state.pending_prompt = _expand_prompt
                                st.session_state.sb_view_mode = "chat"
                                st.session_state.pop(f"expand_portfolio_{port['id']}", None)
                                st.rerun()
                        with col_skip:
                            if st.button("Skip — distribute proportionally", key=f"expand_skip_{port['id']}", use_container_width=True):
                                st.session_state.pop(f"expand_portfolio_{port['id']}", None)
                                st.rerun()

                try:
                    hold_resp = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute()
                    holdings = hold_resp.data
                except Exception:
                    holdings = []

                if holdings:
                    display_holdings = enrich_holdings_live(holdings, cache_key=str(port["id"]))
                    hold_df = pd.DataFrame(display_holdings)
                    display_cols = {
                        "name": "Stock", "ticker": "Ticker", "sector": "Sector", "shares": "Shares",
                        "price_at_entry": "Entry ₹", "current_price": "CMP ₹",
                        "sip_amount_inr": "Invested", "current_value": "Value",
                        "allocation_pct": "Target %", "actual_allocation_pct": "Actual %", "score_at_entry": "Score",
                    }
                    available = {k: v for k, v in display_cols.items() if k in hold_df.columns}
                    st.dataframe(hold_df[list(available.keys())].rename(columns=available), hide_index=True, width="stretch")
                else:
                    st.caption("No holdings found.")

                # TODO: Score Trend per holding — Streamlit selectbox + plotly rerender bug.
                # Revisit when migrating to kordent.in (React frontend).
                # Helper render_score_history_chart() is ready; issue is selectbox not triggering chart refresh.

                # ── Stacked Absolute Chart + XIRR ──
                try:
                    hist_resp = sb.table("portfolio_history").select(
                        "date, total_value, cash_balance, withdrawn, cumulative_invested, nifty_shadow_value"
                    ).eq("portfolio_id", port["id"]).order("date").execute()
                    hist_data = hist_resp.data

                    if hist_data and len(hist_data) >= 2:
                        hist_df = pd.DataFrame(hist_data)
                        hist_df["date"] = pd.to_datetime(hist_df["date"])
                        _bench = selector.describe_benchmark(port)

                        has_invested = "cumulative_invested" in hist_df.columns and hist_df["cumulative_invested"].notna().sum() >= 2
                        has_shadow = "nifty_shadow_value" in hist_df.columns and hist_df["nifty_shadow_value"].notna().sum() >= 2

                        # total_value is holdings only. Uninvested sale proceeds
                        # are real assets; without adding them the line drops on
                        # every sale date and never recovers — a fictitious loss.
                        # Rows written before cash_balance existed are NULL, and
                        # no sale had been recorded then, so 0 is exactly right.
                        for _c in ("cash_balance", "withdrawn"):
                            if _c in hist_df.columns:
                                hist_df[_c] = hist_df[_c].fillna(0)
                            else:
                                hist_df[_c] = 0
                        # Withdrawn money must stay ON the line. Without it the
                        # series drops by the withdrawal on the day it happens
                        # and never recovers - a cliff that is not a loss, sitting
                        # under a flat invested line that correctly did not move.
                        hist_df["total_assets"] = (hist_df["total_value"]
                                                   + hist_df["cash_balance"]
                                                   + hist_df["withdrawn"])

                        fig = go.Figure()

                        # 1. Principal Baseline (shaded area)
                        if has_invested:
                            fig.add_trace(go.Scatter(
                                x=hist_df["date"], y=hist_df["cumulative_invested"],
                                fill="tozeroy", fillcolor="rgba(29, 78, 216, 0.08)",
                                line=dict(color="rgba(29, 78, 216, 0.25)", width=1),
                                name="Invested",
                                hovertemplate="₹%{y:,.0f}<extra>Invested</extra>",
                            ))

                        # 2. Shadow Benchmark (dashed)
                        if has_shadow:
                            fig.add_trace(go.Scatter(
                                x=hist_df["date"], y=hist_df["nifty_shadow_value"] + hist_df["withdrawn"],
                                line=dict(color="#9CA3AF", width=1.5, dash="dash"),
                                name=f"{_bench['label']} shadow",
                                hovertemplate="₹%{y:,.0f}<extra>" + _bench['label'] + " shadow</extra>",
                            ))

                        # 3. Reality Line (bold) — holdings + uninvested cash
                        fig.add_trace(go.Scatter(
                            x=hist_df["date"], y=hist_df["total_assets"],
                            line=dict(color="#1D4ED8", width=2.5),
                            name="Portfolio",
                            hovertemplate="₹%{y:,.0f}<extra>Portfolio</extra>",
                        ))

                        fig.update_layout(
                            margin=dict(l=0, r=0, t=10, b=0),
                            height=320,
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            yaxis=dict(tickprefix="₹", tickformat=",", gridcolor="rgba(0,0,0,0.05)"),
                            xaxis=dict(showgrid=False),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )

                        st.plotly_chart(fig, use_container_width=True, key=f"stacked_{port['id']}")
                        st.caption(
                            f"Benchmark: {_bench['label']} — {_bench['reason']}. Locked at "
                            f"registration; the shadow is what these same SIPs would be worth "
                            f"in {_bench['label']} ({_bench['ticker']})."
                        )

                        # ── Metrics below chart ──
                        # Live, not the last history row: the ledger and live
                        # prices are both current, whereas the history row is as
                        # of the last tracker run. One basis for every number here.
                        _econ = portfolio_money(sb, port["id"], display_holdings,
                                                port.get("benchmark_ticker"))
                        last_shadow = float(hist_df["nifty_shadow_value"].iloc[-1]) if has_shadow else None
                        days_tracked = (hist_df["date"].iloc[-1] - hist_df["date"].iloc[0]).days

                        port_xirr, nifty_xirr = compute_portfolio_xirr(_econ, last_shadow)

                        last_invested = _econ["external_capital"]
                        last_val = _econ["total_assets"]

                        if last_invested and last_invested > 0:
                            profit = _econ["total_pnl"]
                            simple_ret = _econ["return_pct"] if _econ["return_pct"] is not None else 0.0

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Capital Invested", f"{fmt_inr(last_invested)}",
                                      help="What you actually paid in from outside. A sale converts "
                                           "shares to cash inside the portfolio; it does not give you "
                                           "capital back, so this never falls when you sell.")
                            m2.metric("Total Assets" if _econ["cash_balance"] > 0 else "Current Value",
                                      f"{fmt_inr(last_val)}",
                                      help="Market value of holdings plus any uninvested cash from sales.")
                            m3.metric("P&L", f"{fmt_inr(profit)}", delta=f"{simple_ret:+.1f}%")
 
                            _costs_paid = _econ.get("total_costs_paid") or 0
                            if (_econ["cash_balance"] > 0 or _econ["realized_pnl"] != 0
                                    or _econ["withdrawn"] > 0 or _costs_paid > 0):
                                _parts = [f"Holdings {fmt_inr(_econ['market_value'])}",
                                          f"Cash {fmt_inr(_econ['cash_balance'])}"]
                                if _econ["withdrawn"] > 0:
                                    _parts.append(f"Withdrawn {fmt_inr(_econ['withdrawn'])}")
                                st.caption(
                                    " · ".join(_parts) +
                                    f" — realized {fmt_inr(_econ['realized_pnl'])}, "
                                    f"unrealized {fmt_inr(_econ['unrealized_pnl'])}, "
                                    f"costs {fmt_inr(_costs_paid)}."
                                )
                                if _econ["unreconciled_withdrawal"] > 0:
                                    st.warning(
                                        f"Withdrawals exceed recorded cash by "
                                        f"{fmt_inr(_econ['unreconciled_withdrawal'])}. A sale or "
                                        f"contribution is missing from the ledger — returns "
                                        f"shown here are understated until it is added."
                                    )
 
                            # ── Charges: gross to net, and the cost of leaving ──
                            # Three things a user cannot get anywhere else, in one
                            # place: what the market gave them, what the broker and
                            # the government took, and what it would cost to walk
                            # away today. The headline P&L above stays NET, because
                            # net is what they actually have; gross belongs here,
                            # beside the charge that explains the difference.
                            #
                            # Behind an expander whose LABEL carries the numbers, so
                            # the figures are visible without a click and the
                            # arithmetic is one click away. The exit estimate is the
                            # decision-relevant half and the ledger cannot know it —
                            # it is computed fresh on every render from the holdings
                            # and prices on screen.
                            _n_scrips = len({h.get("ticker") for h in (display_holdings or [])
                                             if h.get("ticker")})
                            _mv_now = _econ["market_value"]
                            _dp_total = _n_scrips * costs.RATES["dp_per_scrip_sell"]
                            _exit_cost = (_dp_total + _mv_now * costs.sell_rate()) if _n_scrips else 0.0
                            _gross_pnl = _econ.get("gross_pnl")
                            _gross_ret = _econ.get("gross_return_pct")
 
                            if _costs_paid > 0 or _exit_cost > 0:
                                _lbl = f"Charges — {fmt_inr(_costs_paid, 2)} paid so far"
                                if _exit_cost > 0 and _mv_now > 0:
                                    _lbl += (f", about {fmt_inr(_exit_cost, 2)} to exit everything "
                                             f"today ({_exit_cost / _mv_now * 100:.2f}%)")
                                with st.expander(_lbl):
                                    _rows = []
                                    if _gross_pnl is not None:
                                        _rows.append(("Gain/loss before charges", fmt_inr(_gross_pnl),
                                                      f"{_gross_ret:+.2f}%" if _gross_ret is not None else "—"))
                                    _rows.append(("Buying charges paid",
                                                  "\u2212" + fmt_inr(_econ.get("buy_costs_paid") or 0, 2), ""))
                                    _rows.append(("Selling charges paid",
                                                  "\u2212" + fmt_inr(_econ.get("sell_costs_paid") or 0, 2), ""))
                                    _rows.append(("**Gain/loss after charges**", f"**{fmt_inr(profit)}**",
                                                  f"**{simple_ret:+.2f}%**"))
                                    st.table(pd.DataFrame(
                                        _rows, columns=["", "Amount", "On capital paid in"]
                                    ).set_index(""))
 
                                    if _econ.get("cost_rows_missing"):
                                        # A total that understates must say so.
                                        st.caption(
                                            f"{_econ['cost_rows_missing']} older transaction(s) "
                                            f"were recorded before charges were tracked, so the "
                                            f"charges above are understated.")
 
                                    if _n_scrips and _mv_now > 0:
                                        st.caption(
                                            f"**Not yet paid:** selling all {_n_scrips} holdings "
                                            f"today would cost about {fmt_inr(_exit_cost, 2)} — "
                                            f"{_exit_cost / _mv_now * 100:.2f}% of what they are "
                                            f"worth. {fmt_inr(_dp_total, 2)} of that is the flat "
                                            f"depository charge of "
                                            f"{fmt_inr(costs.RATES['dp_per_scrip_sell'], 2)} per "
                                            f"stock, which does not shrink with position size — "
                                            f"it is the reason small positions are expensive to "
                                            f"leave.")
                                    st.caption(
                                        f"Charges are modelled on Zerodha equity delivery rates "
                                        f"as verified on {costs.RATES_VERIFIED}, not read from a "
                                        f"contract note. Your broker's actual bill may differ. "
                                        f"Market impact and slippage are not included.")
 
                            m4, m5 = st.columns(2)
                            if port_xirr is not None:
                                m4.metric("XIRR", f"{port_xirr:+.1f}%")
                            else:
                                if days_tracked < 90:
                                    m4.metric("Return", f"{simple_ret:+.1f}%", help="XIRR becomes meaningful after 3+ months")
                                else:
                                    m4.metric("Return", f"{simple_ret:+.1f}%")

                            if port_xirr is not None and nifty_xirr is not None:
                                alpha = round(port_xirr - nifty_xirr, 1)
                                m5.metric(f"Alpha vs {_bench['label']}", f"{alpha:+.1f}%", delta=f"{_bench['label']} XIRR {nifty_xirr:+.1f}%")
                            elif has_shadow and last_shadow and last_invested > 0:
                                # Mirror the portfolio side exactly:
                                #   portfolio (total_assets + withdrawn - ext) / ext
                                #   benchmark (shadow_value  + withdrawn - ext) / ext
                                # NOT + cash_balance. That term was correct only
                                # while the shadow shrank on every sale; once the
                                # shadow tracked external flows instead, adding
                                # cash counted the same proceeds on both sides and
                                # inflated the benchmark by the full sale amount.
                                nifty_simple = (((last_shadow + _econ["withdrawn"]) - last_invested) / last_invested) * 100
                                alpha_simple = simple_ret - nifty_simple
                                m5.metric(f"vs {_bench['label']}", f"{alpha_simple:+.1f}%", delta=f"{_bench['label']} {nifty_simple:+.1f}%")
                            else:
                                m5.metric(f"vs {_bench['label']}", "—")

                            # ── Sprint 13 §3: risk metrics — three a retail SIP
                            # investor can act on up front, the other eight behind
                            # a methodology expander, and Jensen's alpha in plain
                            # English (what a layman means by "is this better than
                            # the market — or did I just get lucky with small caps?").
                            _jensen = port.get("jensen_alpha")
                            _mkt = port.get("market_return")
                            _dd = port.get("max_drawdown")
                            _dd_prov = port.get("max_drawdown_provisional")
                            _sharpe_v = port.get("sharpe_ratio")

                            if _sharpe_v is not None or _dd is not None:
                                r1, r2 = st.columns(2)
                                if _sharpe_v is not None:
                                    r1.metric("Sharpe (risk-adjusted)", f"{_sharpe_v:.2f}",
                                              help="Return earned per unit of total risk. Above 1 is "
                                                   "good; negative means the volatility wasn't paid for.")
                                    _sl, _shi = port.get("sharpe_low"), port.get("sharpe_high")
                                    if _sl is not None and _shi is not None:
                                        r1.caption(f"range {_sl:.2f}–{_shi:.2f} (widens on short history)")
                                if _dd is not None:
                                    if _dd_prov:
                                        r2.metric("Worst fall so far", f"{_dd*100:.1f}%",
                                                  help="Short history (under ~6 months) — not yet a "
                                                       "reliable risk estimate; the real worst fall is "
                                                       "likely deeper than this.")
                                        r2.caption("⚠️ short history — provisional")
                                    else:
                                        r2.metric("Max drawdown", f"{_dd*100:.1f}%",
                                                  help="Deepest peak-to-trough fall over the period.")

                            if _jensen is not None:
                                _jp = _jensen * 100
                                _dir = "ahead of" if _jp >= 0 else "behind"
                                _tail = ("Genuine selection edge, if it holds up."
                                         if _jp >= 0 else "The picks have lagged so far.")
                                st.caption(
                                    f"📈 About **{_jp:+.1f} points** {_dir} what your risk level alone "
                                    f"would predict — the part that came from *which stocks were picked*, "
                                    f"after stripping out how the market moved and how much risk you took. "
                                    f"{_tail}"
                                )

                            _adv = [
                                ("Portfolio beta (β)", port.get("portfolio_beta"), "{:.2f}",
                                 "Sensitivity to the market. 1.0 moves with it; below 1 is calmer."),
                                ("Sortino ratio", port.get("sortino_ratio"), "{:.2f}",
                                 "Like Sharpe, but penalises only downside volatility."),
                                ("Treynor ratio", port.get("treynor_ratio"), "{:.4f}",
                                 "Excess return per unit of market risk. Not directly actionable for a SIP."),
                                ("Information ratio", port.get("information_ratio"), "{:.2f}",
                                 "Consistency of out/under-performance vs your assigned benchmark ETF."),
                                ("CAPM expected return", port.get("capm_expected_return"), "{:.1%}",
                                 "What the model says you 'should' earn for your beta."),
                                ("Semi-deviation", port.get("semi_deviation"), "{:.1%}",
                                 "Volatility of below-average returns only."),
                                ("Annualised return", port.get("annual_return"), "{:.1%}",
                                 "Simulated on these holdings' PRIOR year — not this portfolio's own track record yet."),
                                ("Annualised volatility", port.get("annual_std"), "{:.1%}",
                                 "Standard deviation of returns, annualised."),
                            ]
                            if any(v is not None for _, v, _f, _n in _adv):
                                with st.expander("Methodology & advanced metrics"):
                                    _ranges = {"Sortino ratio": ("sortino_low", "sortino_high"),
                                               "Treynor ratio": ("treynor_low", "treynor_high")}
                                    for _label, _val, _fmt, _note in _adv:
                                        if _val is not None:
                                            _line = f"**{_label}: {_fmt.format(_val)}** — {_note}"
                                            _rk = _ranges.get(_label)
                                            if _rk:
                                                _lo, _hi = port.get(_rk[0]), port.get(_rk[1])
                                                if _lo is not None and _hi is not None:
                                                    _line += f" (range {_fmt.format(_lo)}–{_fmt.format(_hi)})"
                                            st.markdown(_line)
                                    _rfr = port.get("rfr_used")
                                    _hd = port.get("metrics_history_days")
                                    _foot = []
                                    if _rfr is not None:
                                        _foot.append(f"risk-free rate used: {_rfr*100:.1f}%")
                                    if _hd is not None:
                                        _foot.append(f"history: {_hd} trading days")
                                    if _foot:
                                        st.caption(" · ".join(_foot))
                                    st.caption("Ratios are shown as single points here; the honest "
                                               "ranges that widen on short history are stored per metric.")
                        else:
                            st.caption(f"Portfolio: {fmt_inr(last_val)} · {days_tracked} days tracked")

                    elif hist_data and len(hist_data) == 1:
                        st.caption("📈 Growth chart available after 2+ days of tracking.")
                except Exception:
                    pass  # Fail silently if history table doesn't exist yet

                # ── Goal Tracker ──
                _goal_amt = port.get("target_amount")
                _goal_date = port.get("target_date")
                if _goal_amt and _goal_date:
                    try:
                        _goal_amt = float(_goal_amt)
                        _sip = float(port.get("sip_amount") or 0)
                        _cur_val = float(port.get("current_value") or 0)

                        # Compute actual CAGR from portfolio_history (need 6+ months)
                        _goal_cagr = None
                        try:
                            _gh = sb.table("portfolio_history").select(
                                "date, total_value"
                            ).eq("portfolio_id", port["id"]).order("date").execute().data or []
                            if len(_gh) >= 120:  # ~6 months of weekday entries
                                _g_first = float(_gh[0]["total_value"])
                                _g_first_d = datetime.date.fromisoformat(_gh[0]["date"])
                                _g_days = max(1, (datetime.date.today() - _g_first_d).days)
                                if _g_first > 0 and _cur_val > 0:
                                    _goal_cagr = (_cur_val / _g_first) ** (365 / _g_days) - 1
                        except Exception:
                            pass

                        _proj = compute_goal_projection(_cur_val, _sip, _goal_amt, _goal_date, _goal_cagr)

                        if _proj and _proj.get("months_remaining", 0) > 0:
                            with st.container(border=True):
                                st.markdown("**🎯 Goal Tracker**")

                                # Status cards
                                _status = _proj["status"]
                                _gap = _proj["gap"]
                                _months = _proj["months_remaining"]
                                _years = _months / 12

                                if _status == "ahead":
                                    st.success(f"You're ahead of target by {fmt_inr(abs(_gap))}")
                                elif _status == "on_track":
                                    st.success(f"On track — within 5% of your {fmt_inr(_goal_amt)} goal")
                                else:
                                    st.warning(f"Behind target by {fmt_inr(abs(_gap))}")

                                gc1, gc2, gc3 = st.columns(3)
                                gc1.metric("Goal", f"{fmt_inr(_goal_amt)}")
                                gc2.metric("Projected", f"{fmt_inr(_proj['projected_value'])}")
                                gc3.metric("Time Left", f"{_years:.1f} yrs" if _years >= 1 else f"{_months} mo")

                                if _proj.get("using_default"):
                                    st.caption("Projected at 12% historical Nifty CAGR. Your actual trajectory will appear after 6 months of data.")
                                elif _proj.get("actual_cagr") is not None:
                                    _ac = _proj["actual_cagr"] * 100
                                    _nc = _proj["needed_cagr"] * 100 if _proj.get("needed_cagr") is not None else None
                                    _cagr_note = f"Your trailing CAGR: {_ac:.1f}%"
                                    if _nc is not None:
                                        _cagr_note += f" · Needed: {_nc:.1f}%"
                                    st.caption(_cagr_note)

                                # Projection chart
                                _cp = _proj.get("current_points", [])
                                _np = _proj.get("needed_points", [])
                                if _cp:
                                    _goal_fig = go.Figure()

                                    # Target line (horizontal)
                                    _goal_fig.add_trace(go.Scatter(
                                        x=[_cp[0]["date"], _cp[-1]["date"]],
                                        y=[_goal_amt, _goal_amt],
                                        line=dict(color="#10B981", width=1.5, dash="dot"),
                                        name="Goal",
                                        hovertemplate="₹%{y:,.0f}<extra>Goal</extra>",
                                    ))

                                    # Needed trajectory
                                    if _np:
                                        _goal_fig.add_trace(go.Scatter(
                                            x=[p["date"] for p in _np],
                                            y=[p["value"] for p in _np],
                                            line=dict(color="#F59E0B", width=1.5, dash="dash"),
                                            name="Needed",
                                            hovertemplate="₹%{y:,.0f}<extra>Needed</extra>",
                                        ))

                                    # Current trajectory
                                    _goal_fig.add_trace(go.Scatter(
                                        x=[p["date"] for p in _cp],
                                        y=[p["value"] for p in _cp],
                                        line=dict(color="#1D4ED8", width=2),
                                        name="Projected",
                                        hovertemplate="₹%{y:,.0f}<extra>Projected</extra>",
                                    ))

                                    _goal_fig.update_layout(
                                        margin=dict(l=0, r=0, t=10, b=0),
                                        height=250,
                                        hovermode="x unified",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        yaxis=dict(tickprefix="₹", tickformat=",", gridcolor="rgba(0,0,0,0.05)"),
                                        xaxis=dict(showgrid=False),
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                    )
                                    st.plotly_chart(_goal_fig, use_container_width=True, key=f"goal_{port['id']}")

                                # SIP adjustment suggestion
                                if _status == "behind" and _proj.get("sip_increase") and _proj["sip_increase"] > 0:
                                    st.caption(f"💡 Increasing your SIP by {fmt_inr(_proj['sip_increase'])}/mo (to {fmt_inr(_sip + _proj['sip_increase'])}) could close the gap at your current growth rate.")

                        elif _proj and _proj.get("status") == "achieved":
                            with st.container(border=True):
                                st.markdown("**🎯 Goal Tracker**")
                                st.success(f"Goal reached! Your portfolio ({fmt_inr(_cur_val)}) exceeds your target of {fmt_inr(_goal_amt)}.")
                        elif _proj and _proj.get("status") == "missed":
                            with st.container(border=True):
                                st.markdown("**🎯 Goal Tracker**")
                                st.warning(f"Goal deadline passed. Current: {fmt_inr(_cur_val)} vs Target: {fmt_inr(_goal_amt)}. Gap: {fmt_inr(abs(_proj['gap']))}.")
                    except Exception:
                        pass

                # ── PDF Export (two-step: generate then download) ──
                report_key = f"report_ready_{port['id']}"

                if st.session_state.get(report_key):
                    _share_col, _dl_col = st.columns(2)
                    with _share_col:
                        _share_key = f"share_url_{port['id']}"
                        if st.session_state.get(_share_key):
                            st.code(st.session_state[_share_key], language=None)
                            st.caption("Link expires in 24 hours")
                        else:
                            if st.button("🔗 Share Report", key=f"share_btn_{port['id']}",
                                         use_container_width=True):
                                _redact = st.session_state.get(f"redact_{port['id']}", True)
                                with st.spinner("Generating share link..."):
                                    if _redact:
                                        _share_h = st.session_state.get(f"_pdf_holdings_{port['id']}", [])
                                        _share_pdf = generate_portfolio_pdf(
                                            port,
                                            _share_h,
                                            redact_holdings=True,
                                            chart_buf=None, narrative=None,
                                            xirr_data=None, score_data=None,
                                            econ=portfolio_money(sb, port["id"], _share_h,
                                                                 port.get("benchmark_ticker")),
                                        )
                                    else:
                                        _share_pdf = st.session_state[report_key]
                                    result = upload_shared_report(sb, st.session_state.sb_user_id, _share_pdf)
                                    if result.get("url"):
                                        st.session_state[_share_key] = result["url"]
                                        st.rerun()
                                    else:
                                        st.error(result.get("error", "Upload failed"))
                            st.checkbox("Hide stock names in shared report",
                                        value=True, key=f"redact_{port['id']}")
                    with _dl_col:
                        st.download_button(
                            label="⬇️ Download Report",
                            data=st.session_state[report_key],
                            file_name=f"Kordent_{re.sub(r'[^a-zA-Z0-9]', '_', port.get('name', 'portfolio'))}_{datetime.date.today().isoformat()}.pdf",
                            mime="application/pdf",
                            key=f"pdf_download_{port['id']}",
                            use_container_width=True,
                        )
                    if st.button("✕ Clear", key=f"pdf_clear_{port['id']}"):
                        del st.session_state[report_key]
                        st.session_state.pop(f"share_url_{port['id']}", None)
                        st.rerun()
                else:
                    if st.button("📄 Generate Alpha Report", key=f"pdf_gen_{port['id']}", width="stretch"):
                        with st.spinner("Building Alpha Report — analyzing holdings, computing XIRR, generating narrative..."):
                            try:
                                hold_for_pdf = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute().data or []
                                hold_for_pdf = enrich_holdings_live(hold_for_pdf, cache_key=f"pdf_{port['id']}")
                                hist_for_pdf = sb.table("portfolio_history").select(
                                    "date, total_value, nifty_value, cumulative_invested, nifty_shadow_value"
                                ).eq("portfolio_id", port["id"]).order("date").execute().data or []
                                alerts_for_pdf = sb.table("portfolio_alerts").select("*").eq("portfolio_id", port["id"]).eq("is_read", False).execute().data or []
 
                                # Chart (Plotly stacked → PNG via kaleido, matplotlib fallback)
                                chart_buf = generate_portfolio_chart(hist_for_pdf)
 
                                # Economics + XIRR (same basis, one computation)
                                _pdf_econ = portfolio_money(sb, port["id"], hold_for_pdf,
                                                            port.get("benchmark_ticker"))
                                _pdf_nifty_sh = hist_for_pdf[-1].get("nifty_shadow_value") if hist_for_pdf else None
                                xirr_data = compute_portfolio_xirr(_pdf_econ, _pdf_nifty_sh)
 
                                # Goal projection + chart
                                goal_data = None
                                goal_chart_buf = None
                                if port.get("target_amount") and port.get("target_date"):
                                    _pdf_actual_cagr = None
                                    if hist_for_pdf and len(hist_for_pdf) >= 2:
                                        _first_d = datetime.date.fromisoformat(str(hist_for_pdf[0]["date"]))
                                        _last_d = datetime.date.fromisoformat(str(hist_for_pdf[-1]["date"]))
                                        _days_active = (_last_d - _first_d).days
                                        if _days_active >= 180:
                                            _first_v = hist_for_pdf[0]["total_value"]
                                            _last_v = hist_for_pdf[-1]["total_value"]
                                            if _first_v and _first_v > 0:
                                                _pdf_actual_cagr = (_last_v / _first_v) ** (365.0 / _days_active) - 1
 
                                    goal_data = compute_goal_projection(
                                        _pdf_cur_val, port.get("sip_amount", 0),
                                        port["target_amount"], port["target_date"],
                                        actual_cagr=_pdf_actual_cagr
                                    )
 
                                    # Render goal chart as PNG
                                    if goal_data and goal_data.get("current_points"):
                                        _cp = goal_data["current_points"]
                                        _np = goal_data.get("needed_points", [])
                                        _ga = port["target_amount"]
                                        _gf = go.Figure()

                                        # Sprint 12: probability FAN behind the deterministic line.
                                        # Replaces the illusion of one certain path with a p10-p90
                                        # band from a block-bootstrap of real Nifty months. The
                                        # confidence caveat rides in the title, inseparable.
                                        _fan_title = None
                                        try:
                                            import stats as _kstats
                                            _months = goal_data.get("months_remaining") or len(_cp)
                                            _dist = _kstats.project_goal_distribution(
                                                _pdf_cur_val, port.get("sip_amount", 0),
                                                port["target_amount"], _months)
                                            if _dist and _dist.get("fan"):
                                                _fan = _dist["fan"]
                                                _step = _dist.get("fan_step_months", 1)
                                                _base = _cp[0]["date"]
                                                _fdates = [_add_months(_base, (i + 1) * _step)
                                                           for i in range(len(_fan["p50"]))]
                                                # upper (p90) then lower (p10) with fill between
                                                _gf.add_trace(go.Scatter(
                                                    x=_fdates, y=_fan["p90"], mode="lines",
                                                    line=dict(width=0), showlegend=False,
                                                    hoverinfo="skip"))
                                                _gf.add_trace(go.Scatter(
                                                    x=_fdates, y=_fan["p10"], mode="lines",
                                                    line=dict(width=0), fill="tonexty",
                                                    fillcolor="rgba(29,78,216,0.12)",
                                                    name="10th-90th percentile",
                                                    hoverinfo="skip"))
                                                _ph = _dist.get("prob_hit_target")
                                                _conf = _dist.get("confidence_note", "")
                                                _pht = (f"{_ph*100:.0f}% chance of reaching "
                                                        f"Rs.{port['target_amount']:,.0f}"
                                                        if _ph is not None else "")
                                                _fan_title = ("If your portfolio behaves like the "
                                                              f"{_dist.get('benchmark','index')}: "
                                                              f"{_pht}. {_conf}")
                                        except Exception as _fe:
                                            print(f"Goal fan skipped (non-blocking): {type(_fe).__name__}: {_fe}")

                                        _gf.add_trace(go.Scatter(
                                            x=[_cp[0]["date"], _cp[-1]["date"]], y=[_ga, _ga],
                                            line=dict(color="#10B981", width=1.5, dash="dot"), name="Goal"))
                                        if _np:
                                            _gf.add_trace(go.Scatter(
                                                x=[p["date"] for p in _np], y=[p["value"] for p in _np],
                                                line=dict(color="#F59E0B", width=1.5, dash="dash"), name="Needed"))
                                        _gf.add_trace(go.Scatter(
                                            x=[p["date"] for p in _cp], y=[p["value"] for p in _cp],
                                            line=dict(color="#1D4ED8", width=2), name="Projected"))
                                        _gf.update_layout(
                                            margin=dict(l=10, r=10, t=(46 if _fan_title else 30), b=10), height=280, width=800,
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                            yaxis=dict(tickprefix="Rs.", tickformat=","),
                                            title=(dict(text=_fan_title, font=dict(size=9), x=0.0, xanchor="left") if _fan_title else None),
                                            plot_bgcolor="white", paper_bgcolor="white")
                                        goal_chart_buf = _plotly_to_png(_gf, width=800, height=280)
 
                                # Score data from universe
                                score_data = {}
                                try:
                                    for h in hold_for_pdf:
                                        _t = h.get("ticker", "")
                                        _r = universe_df[universe_df["ticker"] == _t]
                                        if not _r.empty:
                                            score_data[_t] = {
                                                "score": int(_r["score"].iloc[0]) if pd.notna(_r["score"].iloc[0]) else 0,
                                                "graham_pass": bool(_r["graham_pass"].iloc[0]) if pd.notna(_r.get("graham_pass", pd.Series([None])).iloc[0]) else False,
                                                "greenblatt_pass": bool(_r["greenblatt_pass"].iloc[0]) if pd.notna(_r.get("greenblatt_pass", pd.Series([None])).iloc[0]) else False,
                                                "dorsey_pass": bool(_r["dorsey_pass"].iloc[0]) if pd.notna(_r.get("dorsey_pass", pd.Series([None])).iloc[0]) else False,
                                                "trajectory_pass": bool(_r["trajectory_pass"].iloc[0]) if pd.notna(_r.get("trajectory_pass", pd.Series([None])).iloc[0]) else False,
                                                "quality_pass": bool(_r["quality_pass"].iloc[0]) if "quality_pass" in _r.columns and pd.notna(_r["quality_pass"].iloc[0]) else False,
                                            }
                                except NameError:
                                    pass
 
                                # Sector momentum
                                _pdf_sectors = tuple(set(h.get("sector", "") for h in hold_for_pdf if h.get("sector")))
                                sector_data = get_sector_momentum(_pdf_sectors) if _pdf_sectors else {}
 
                                # User name
                                _pdf_name = None
                                try:
                                    _name_resp = sb.table("profiles").select("full_name").eq("id", st.session_state.sb_user_id).limit(1).execute()
                                    if _name_resp.data and _name_resp.data[0].get("full_name"):
                                        _pdf_name = _name_resp.data[0]["full_name"]
                                except Exception:
                                    pass
 
                                # LLM narrative
                                narrative = generate_portfolio_narrative(port, hold_for_pdf, collection, score_data=score_data)
 
                                # Build PDF
                                pdf_bytes = generate_portfolio_pdf(
                                    port, hold_for_pdf, hist_for_pdf, alerts_for_pdf,
                                    chart_buf=chart_buf, narrative=narrative,
                                    xirr_data=xirr_data, goal_data=goal_data,
                                    goal_chart_buf=goal_chart_buf,
                                    sector_data=sector_data, score_data=score_data,
                                    user_name=_pdf_name, econ=_pdf_econ,
                                )
                                st.session_state[f"_pdf_holdings_{port['id']}"] = hold_for_pdf
                                st.session_state[report_key] = pdf_bytes
                                st.rerun()
                            except Exception as e:
                                st.error(f"Report generation failed: {e}")

                # ── Standalone Health Check (when not in review) ──
                _review_imminent = False
                if port.get("next_review_date"):
                    try:
                        _rd = datetime.date.fromisoformat(str(port["next_review_date"]))
                        _review_imminent = (_rd - datetime.date.today()).days <= 7
                    except (ValueError, TypeError):
                        pass

                if not st.session_state.get(f"review_data_{port['id']}") and not _review_imminent:
                    hc_key = f"health_check_{port['id']}"
                    if st.session_state.get(hc_key):
                        hc = st.session_state[hc_key]
                        with st.container(border=True):
                            st.markdown("**Health Check Results**")
                            d_score = hc["diversification_score"]
                            d_color = "🟢" if d_score >= 70 else "🟡" if d_score >= 40 else "🔴"
                            st.metric("Diversification Score", f"{d_color} {d_score}/100")
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Avg Beta", hc["avg_beta"] or "N/A")
                            with m2:
                                pe_val = hc["avg_pe_vs_historical"]
                                pe_label = f"{pe_val:+.1f}%" if pe_val is not None else "N/A"
                                st.metric("PE vs History", pe_label)
                            with m3:
                                high_val = hc["avg_pct_from_52w_high"]
                                high_label = f"{high_val:.1f}%" if high_val is not None else "N/A"
                                st.metric("From 52w High", high_label)
                            sector_dist = hc["sector_distribution"]
                            if sector_dist:
                                sector_df = pd.DataFrame([
                                    {"Sector": s, "Stocks": c, "Weight": f"{c/sum(sector_dist.values())*100:.0f}%"}
                                    for s, c in sorted(sector_dist.items(), key=lambda x: -x[1])
                                ])
                                st.dataframe(sector_df, hide_index=True, width="stretch")
                            for w in hc.get("warnings", []):
                                st.warning(w)
                            if hc.get("narrative"):
                                st.markdown("---")
                                st.markdown(hc["narrative"])

                        # ── Actionable recommendations ──
                        hc_actions = hc.get("actions", [])
                        if hc_actions:
                            st.markdown("---")
                            st.markdown("**Execute Recommendations**")
                            for ai, act in enumerate(hc_actions):
                                act_type = act.get("type", "")
                                act_ticker = act.get("ticker", "")
                                act_reason = act.get("reason", "")

                                if act_type == "reduce":
                                    target_pct = act.get("target_alloc_pct", 0)
                                    if st.button(
                                        f"📉 Reduce {act_ticker} to {target_pct}% — {act_reason}",
                                        key=f"hc_reduce_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            sb.table("holdings").update(
                                                {"allocation_pct": target_pct}
                                            ).eq("portfolio_id", port["id"]).eq("ticker", act_ticker).execute()
                                            st.success(f"Updated {act_ticker} allocation to {target_pct}%.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "sell":
                                    sell_shares = act.get("shares", 0)
                                    label = f"🔴 Sell all {act_ticker}" if sell_shares == 0 else f"🔴 Sell {sell_shares} shares of {act_ticker}"
                                    if st.button(
                                        f"{label} — {act_reason}",
                                        key=f"hc_sell_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            # Both branches now resolve the holding FIRST: the old
                                            # "sell all" path deleted the row without ever knowing
                                            # how many shares left the portfolio, so the sale could
                                            # not be written to the ledger even in principle.
                                            h_resp = sb.table("holdings").select("*").eq(
                                                "portfolio_id", port["id"]
                                            ).eq("ticker", act_ticker).execute()
                                            h = (h_resp.data or [None])[0]
                                            if not h:
                                                st.warning(f"{act_ticker} is no longer held.")
                                                st.stop()
                                            _held = int(h.get("shares") or 0)
                                            _qty = _held if sell_shares == 0 else min(int(sell_shares), _held)
                                            _px = live_price(act_ticker, h.get("price_at_entry", 0))
                                            if _qty <= 0 or _px <= 0:
                                                st.error("No live price for this ticker — sale not recorded. "
                                                         "Use Review, where you can enter the fill price.")
                                                st.stop()
                                            record_transaction(
                                                sb, port["id"], st.session_state.sb_user_id,
                                                act_ticker, _qty, _px, round(_qty * _px, 2), "sell",
                                                benchmark_ticker=port.get("benchmark_ticker"),
                                                raise_on_error=True,
                                            )
                                            new_shares = _held - _qty
                                            if new_shares <= 0:
                                                sb.table("holdings").delete().eq("id", h["id"]).execute()
                                                st.success(f"Sold all {_qty} shares of {act_ticker} at {fmt_inr(_px)}.")
                                            else:
                                                new_invested = new_shares * h.get("price_at_entry", 0)
                                                sb.table("holdings").update({
                                                    "shares": new_shares,
                                                    "sip_amount_inr": round(new_invested, 2)
                                                }).eq("id", h["id"]).execute()
                                                st.success(f"Sold {_qty} shares of {act_ticker} at {fmt_inr(_px)}.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "add":
                                    btn_key = f"hc_action_{port['id']}_{ai}"
                                    action_msg_key = f"hc_action_msg_{port['id']}"
                                    act_name = act.get("name", act_ticker)
                                    act_sector = act.get("sector", "")
                                    act_score = act.get("score", 0)
                                    suggested_pct = act.get("suggested_alloc_pct", 10)
                                    add_state_key = f"hc_add_form_{port['id']}_{ai}"

                                    if st.button(
                                        f"Add {act_name} ({act_ticker}) — {act_reason}",
                                        key=btn_key, width="stretch"
                                    ):
                                        st.session_state[add_state_key] = True

                                    if st.session_state.get(add_state_key):
                                        with st.container(border=True):
                                            # Fetch live price
                                            try:
                                                _live = yf.Ticker(act_ticker).fast_info
                                                _live_price = round(float(_live.last_price), 2)
                                            except Exception:
                                                _live_price = 100.0

                                            # Suggest shares from SIP and allocation
                                            _sip = port.get("sip_amount", 10000)
                                            _budget = _sip * suggested_pct / 100
                                            _suggested_qty = max(1, int(_budget / _live_price)) if _live_price > 0 else 1

                                            st.caption(
                                                f"Sector: {act_sector} · Score: {_score_label(act_ticker, act_score)} · "
                                                f"PE: {act.get('pe', 'N/A')} · Price: {fmt_inr(_live_price, 2)} · "
                                                f"Budget ({suggested_pct}% of {fmt_inr(_sip)}): {fmt_inr(_budget)}"
                                            )
                                            ac1, ac2 = st.columns(2)
                                            with ac1:
                                                add_qty = st.number_input(
                                                    "Shares to buy", min_value=1, value=_suggested_qty,
                                                    key=f"hc_add_qty_{port['id']}_{ai}"
                                                )
                                            with ac2:
                                                add_price = st.number_input(
                                                    "Price per share (₹)", min_value=0.01,
                                                    value=_live_price,
                                                    format="%.2f", key=f"hc_add_price_{port['id']}_{ai}"
                                                )
                                            bc1, bc2 = st.columns(2)
                                            with bc1:
                                                if st.button("Confirm Add", key=f"hc_add_confirm_{port['id']}_{ai}", width="stretch"):
                                                    try:
                                                        invested = round(add_qty * add_price, 2)
                                                        sb.table("holdings").insert({
                                                            "portfolio_id": port["id"],
                                                            "ticker": act_ticker,
                                                            "name": act_name,
                                                            "sector": act_sector,
                                                            "allocation_pct": suggested_pct,
                                                            "shares": add_qty,
                                                            "sip_amount_inr": invested,
                                                            "price_at_entry": round(add_price, 2),
                                                            "score_at_entry": act_score,
                                                        }).execute()
                                                        record_transaction(sb, port["id"], st.session_state.sb_user_id, act_ticker, add_qty, add_price, invested, "buy")

                                                        # Normalize all allocations to sum to 100%
                                                        all_h = sb.table("holdings").select("id, allocation_pct").eq(
                                                            "portfolio_id", port["id"]
                                                        ).execute().data or []
                                                        if all_h:
                                                            raw_total = sum(h["allocation_pct"] for h in all_h)
                                                            non_zero = [h for h in all_h if h["allocation_pct"] > 0]
                                                            if len(non_zero) < len(all_h) / 2:
                                                                # Most are zero — reset to equal allocation
                                                                equal_pct = round(100 / len(all_h), 1)
                                                                for h in all_h:
                                                                    sb.table("holdings").update(
                                                                        {"allocation_pct": equal_pct}
                                                                    ).eq("id", h["id"]).execute()
                                                            elif raw_total > 0:
                                                                for h in all_h:
                                                                    normalized = round(h["allocation_pct"] / raw_total * 100, 1)
                                                                    sb.table("holdings").update(
                                                                        {"allocation_pct": normalized}
                                                                    ).eq("id", h["id"]).execute()

                                                        st.session_state[action_msg_key] = f"Added {act_name}. All allocations normalized to 100%."
                                                        del st.session_state[add_state_key]
                                                        st.rerun()
                                                    except Exception as e:
                                                        st.error(f"Failed: {e}")
                                            with bc2:
                                                if st.button("Cancel", key=f"hc_add_cancel_{port['id']}_{ai}", width="stretch"):
                                                    del st.session_state[add_state_key]
                                                    st.rerun()

                                elif act_type == "investigate":
                                    btn_key = f"hc_action_{port['id']}_{ai}"
                                    inv_key = f"hc_inv_result_{port['id']}_{ai}"
                                    if st.button(
                                        f"Investigate {act_ticker} — {act_reason}",
                                        key=btn_key, width="stretch"
                                    ):
                                        with st.spinner(f"Investigating {act_ticker}..."):
                                            try:
                                                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                                                stock_data = get_stock_data(act_ticker)
                                                book_data = search_book(f"{act_reason} investment risk")
                                                inv_prompt = (
                                                    f"You are Kordent's analyst investigating a specific concern about {act_ticker}.\n\n"
                                                    f"CONCERN: {act_reason}\n\n"
                                                    f"STOCK DATA:\n{json.dumps(stock_data, indent=2, default=str)}\n\n"
                                                    f"BOOK CONTEXT:\n{book_data.get('passages', '')[:800]}\n\n"
                                                    f"Write a focused 150-word investigation: what does the data show about this concern? "
                                                    f"Is the concern valid? What should the investor do? Cite book principles."
                                                )
                                                last_good = st.session_state.get("last_working_model")
                                                models = [last_good] + [m for m in FREE_MODELS if m != last_good] if last_good else FREE_MODELS
                                                for model in models:
                                                    try:
                                                        resp = client.models.generate_content(model=model, contents=inv_prompt)
                                                        st.session_state[inv_key] = resp.text
                                                        st.session_state.last_working_model = model
                                                        break
                                                    except Exception as e:
                                                        error_msg = str(e).upper()
                                                        if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                                                            continue
                                                        break
                                            except Exception as e:
                                                st.session_state[inv_key] = f"Investigation failed: {e}"
                                        st.rerun()

                                    if st.session_state.get(inv_key):
                                        with st.container(border=True):
                                            st.markdown(st.session_state[inv_key])
                                            if st.button("Dismiss", key=f"inv_dismiss_{port['id']}_{ai}"):
                                                del st.session_state[inv_key]
                                                st.rerun()
                        if st.button("✕ Close", key=f"hc_close_{port['id']}"):
                            del st.session_state[hc_key]
                            st.rerun()
                    else:
                        if st.button("🩺 Health Check", key=f"hc_btn_{port['id']}", width="stretch"):
                            with st.spinner("Diagnosing portfolio against book principles..."):
                                try:
                                    hc_holdings = sb.table("holdings").select("*").eq("portfolio_id", port["id"]).execute().data or []
                                    hc_holdings = enrich_holdings_live(hc_holdings, cache_key=str(port["id"]))
                                    hc_result = generate_health_check(port, hc_holdings, universe_df, collection)
                                    if hc_result:
                                        st.session_state[hc_key] = hc_result
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Health check failed: {e}")

                
                # --- NEW COLLISION PROTOCOL & SIP MODULE ---
                today = datetime.date.today()
                _auto_key = f"auto_trigger_review_{port['id']}"
                _auto_run = st.session_state.pop(_auto_key, False)
                
                # 1. Calculate Review Clock
                review_date = None
                rev_due_days = 0
                if port.get("next_review_date"):
                    try:
                        review_date = datetime.date.fromisoformat(str(port["next_review_date"]))
                        rev_due_days = (review_date - today).days
                    except (ValueError, TypeError):
                        pass

                # 2. Calculate SIP Clock
                sip_date_str = port.get("next_sip_date")
                sip_due_days = 0
                if sip_date_str:
                    try:
                        sip_due_days = (datetime.date.fromisoformat(str(sip_date_str)) - today).days
                    except (ValueError, TypeError):
                        pass

                _review_clicked = False
                
                if holdings:
                    if _auto_run or rev_due_days <= 0:
                        # STATE 1: PRIORITY OVERRIDE (Review is Due)
                        if not _auto_run:
                            st.warning(f"📅 Review overdue by {abs(rev_due_days)} days! You must evaluate fundamentals before deploying more capital.")
                        _review_clicked = st.button("🔄 Review Portfolio", key=f"review_{port['id']}", width="stretch")
                    
                    else:
                        # STATE 2 & 3: FLEXIBLE SIP DEPLOYMENT (Due or Ad-hoc)
                        if sip_due_days <= 0:
                            st.success(f"💰 Monthly SIP of {fmt_inr(port.get('sip_amount', 0))} is due!")
                            _deploy_btn = st.button("💵 Deploy SIP", key=f"deploy_sip_{port['id']}", width="stretch")
                        else:
                            st.caption(f"📅 Next Review due in {rev_due_days} days ({review_date.isoformat() if review_date else '—'})")
                            st.caption(f"💰 Next SIP due in {sip_due_days} days ({sip_date_str})")
                            _deploy_btn = st.button("💵 Deploy Capital Early (Mid-Cycle)", key=f"deploy_sip_{port['id']}", width="stretch", type="secondary")
                        
                        if _deploy_btn:
                            st.session_state[f"active_sip_{port['id']}"] = True
                        
                        if st.session_state.get(f"active_sip_{port['id']}"):
                            with st.container(border=True):
                                st.markdown("**Flexible Capital Deployment**")
                                
                                # --- REALITY INPUT: Change amount or Skip ---
                                exec_c1, exec_c2 = st.columns([2, 1])
                                with exec_c1:
                                    exec_amount = st.number_input(
                                        "Amount to deploy today (₹)",
                                        min_value=0, value=int(port.get('sip_amount', 0)), step=1000,
                                        key=f"exec_amt_{port['id']}"
                                    )
                                with exec_c2:
                                    if st.button("⏭️ Skip This Cycle", key=f"skip_sip_{port['id']}", width="stretch"):
                                        # Reset timer, clear state, no database holdings updated
                                        new_sip_date = (today + datetime.timedelta(days=30)).isoformat()
                                        sb.table("portfolios").update({"next_sip_date": new_sip_date}).eq("id", port["id"]).execute()
                                        del st.session_state[f"active_sip_{port['id']}"]
                                        st.rerun()

                                if exec_amount > 0:
                                    # Hand raw holdings with TRUE target weights to
                                    # allocate_shares — it owns all breadth/gap logic.
                                    # No pre-chewing: no deficit weighting, no filtering
                                    # of on-target names (breadth rule must see them all).
                                    sip_stocks = []
                                    for h in display_holdings:
                                        if h.get("allocation_pct", 0) <= 0:
                                            continue
                                        sip_stocks.append({
                                            "ticker": h["ticker"],
                                            "name": h.get("name", h["ticker"]),
                                            "allocation_pct": h.get("allocation_pct", 0),
                                            "price": h.get("current_price", h.get("price_at_entry", 1)),
                                            "id": h["id"],
                                            "old_shares": h["shares"],
                                            "old_entry": h.get("price_at_entry", 1),
                                        })

                                    if sip_stocks:
                                        _existing = {h["ticker"]: h["shares"] for h in display_holdings}
                                        allocated, _ = allocate_shares(sip_stocks, exec_amount, existing_shares=_existing)
                                        
                                        # ── TRACK LIVE SPENDING ──
                                        live_spent = 0
                                        
                                        for a in allocated:
                                            c1, c2, c3 = st.columns([2,1,1])
                                            with c1: st.markdown(f"**{a['name']}**")
                                            with c2: 
                                                live_q = st.number_input("Shares", value=a["shares"], key=f"sip_q_{port['id']}_{a['ticker']}")
                                            with c3: 
                                                live_p = st.number_input("Price (₹)", value=float(a["price"]), key=f"sip_p_{port['id']}_{a['ticker']}")
                                            
                                            # Add the live input values to our running total
                                            live_spent += (live_q * live_p)
                                        
                                        # Calculate real-time unallocated cash
                                        live_unallocated = exec_amount - live_spent
                                        
                                        # --- FRACTIONAL SAVER: Smart Cash Drag Mitigation ---
                                        if live_unallocated > 0:
                                            affordable = [s for s in sip_stocks if 0 < s["price"] <= live_unallocated]
                                            if affordable:
                                                best_opt = sorted(affordable, key=lambda x: x["price"], reverse=True)[0]
                                                extra_shares = int(live_unallocated // best_opt["price"])
                                                st.info(f"💡 **{fmt_inr(live_unallocated)} unallocated.** You can't hit exact target percentages, but you could buy **{extra_shares} more share(s) of {best_opt['name']}** ({fmt_inr(best_opt['price'], 2)}) to put that cash to work. Just increase the shares above.")
                                            else:
                                                st.caption(f"ℹ️ {fmt_inr(live_unallocated)} unallocated (not enough to buy any of your holdings). Leave it in your bank.")
                                        elif live_unallocated < 0:
                                            st.warning(f"⚠️ You have exceeded your deployment amount by {fmt_inr(abs(live_unallocated))}.")
                                        else:
                                            st.success("✅ Perfect allocation! Zero unallocated cash.")
                                        if KITE_ENABLED:
                                            _deploy_kite = []
                                            for a in allocated:
                                                _q = st.session_state.get(f"sip_q_{port['id']}_{a['ticker']}", a["shares"])
                                                if _q > 0:
                                                    _deploy_kite.append({"ticker": a["ticker"], "quantity": _q})
                                            if _deploy_kite:
                                                st.link_button("🛒 Buy on Kite", kite_basket_url(_deploy_kite), use_container_width=True)
                                                st.caption("After buying on Kite, confirm below:")
                                        bc1, bc2 = st.columns(2)
                                        with bc1:
                                            if st.button("✅ Confirm Purchase", key=f"conf_sip_{port['id']}", width="stretch"):
                                                try:
                                                    for a in allocated:
                                                        buy_q = st.session_state[f"sip_q_{port['id']}_{a['ticker']}"]
                                                        buy_p = st.session_state[f"sip_p_{port['id']}_{a['ticker']}"]
                                                        if buy_q > 0:
                                                            new_total_shares = a["old_shares"] + buy_q
                                                            old_value = a["old_shares"] * a["old_entry"]
                                                            new_value = buy_q * buy_p
                                                            new_avg_price = (old_value + new_value) / new_total_shares
                                                            sb.table("holdings").update({
                                                                "shares": new_total_shares,
                                                                "price_at_entry": round(new_avg_price, 2),
                                                                "sip_amount_inr": round(new_total_shares * new_avg_price, 2)
                                                            }).eq("id", a["id"]).execute()
                                                            record_transaction(sb, port["id"], st.session_state.sb_user_id, a["ticker"], buy_q, buy_p, round(new_value, 2), "buy")
                                                    
                                                    # Bump the SIP timer by 30 days based on *today's* real-world action
                                                    new_sip_date = (today + datetime.timedelta(days=30)).isoformat()
                                                    sb.table("portfolios").update({"next_sip_date": new_sip_date}).eq("id", port["id"]).execute()
                                                    
                                                    del st.session_state[f"active_sip_{port['id']}"]
                                                    st.success("Capital Deployed Successfully!")
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Failed: {e}")
                                        with bc2:
                                            if st.button("Cancel", key=f"canc_sip_{port['id']}", width="stretch"):
                                                del st.session_state[f"active_sip_{port['id']}"]
                                                st.rerun()
                                    else:
                                        st.warning("No active holdings found to allocate to.")

                        if _review_clicked or _auto_run:
                            with st.spinner("Analyzing holdings with market context and book philosophy..."):
                                # ── Macro refresh + diff (review-diff capability) ──
                                try:
                                    _prof = port.get("portfolio_profile") or {}
                                    _prev_snap = _prof.get("macro_snapshot")
                                    _new_snap = _refresh_macro_snapshot(holdings)
                                    _macro_diff = _diff_macro_snapshots(_prev_snap, _new_snap)
                                    _prof = _append_macro_snapshot(_prof, _new_snap, n=6)
                                    sb.table("portfolios").update(
                                        {"portfolio_profile": _prof}).eq("id", port["id"]).execute()
                                    st.session_state[f"_macro_diff_{port['id']}"] = _macro_diff
                                    # Bridge the diff into the recommendation engine (in-memory only)
                                    _prof_for_rec = dict(_prof)
                                    _prof_for_rec["_pending_macro_diff"] = _macro_diff
                                    port["portfolio_profile"] = _prof_for_rec
                                except Exception:
                                    st.session_state[f"_macro_diff_{port['id']}"] = {}

                                enriched = build_review_context(holdings, port)
                                # demand_tilt lives on the PROFILE, not inside
                                # ips_policy, so the policy handed to the drift
                                # re-selection lacked it — review re-ranked every
                                # holding under different weights than the build
                                # used. That manufactures drift: a stock flagged
                                # "outranked" because the QUESTION changed, not
                                # the company. Build and review must score by the
                                # same rule or the diff is meaningless.
                                # Falsy ips_policy stays falsy: compute_thesis_drift
                                # treats that as "drift undefined", and a tilt
                                # alone is not a mandate.
                                _pp = port.get("portfolio_profile") or {}
                                _ips_pol = _pp.get("ips_policy")
                                if _ips_pol and _pp.get("demand_tilt"):
                                    _ips_pol = {**_ips_pol,
                                                "demand_tilt": _pp["demand_tilt"]}
                                _drift = selector.compute_thesis_drift(
                                    holdings,
                                    _ips_pol,
                                    universe_df,
                                )
                                llm_recs = generate_review_recommendations(
                                    enriched, port.get("investor_type", "balanced"),
                                    port.get("time_horizon", "medium"),
                                    port
                                )

                                # Merge LLM recommendations with enriched data
                                total_entry = 0
                                total_current = 0
                                review_rows = []

                                # ── Firm-distress gate (held positions): batched, ONE query ──
                                # min_confidence="high" — this can prompt a real, tax-incurring
                                # sell, so the bar is near-certain. Flags for HUMAN confirmation,
                                # never auto-sells. Distinct from the deterministic red-flag override.
                                _held_distress = {}
                                try:
                                    _held_names = [h.get("name", "") for h in enriched if h.get("name")]
                                    _held_distress = _detect_firm_distress(_held_names, min_confidence="high")
                                except Exception:
                                    _held_distress = {}

                                for h in enriched:
                                    total_entry += h["entry_price"] * h["shares"]
                                    total_current += h["now_price"] * h["shares"]

                                    # Find LLM recommendation for this ticker
                                    llm_rec = None
                                    if llm_recs:
                                        llm_rec = next((r for r in llm_recs if r.get("ticker") == h["ticker"]), None)

                                    if llm_rec:
                                        raw_action = llm_rec.get("action", "HOLD").upper()
                                        reasoning = llm_rec.get("reasoning", "")
                                        confidence = llm_rec.get("confidence", "medium")
                                        sell_qty = llm_rec.get("sell_qty", 0)

                                        if "SELL ALL" in raw_action:
                                            action = f"🔴 SELL ALL ({h['shares']})"
                                            sell_qty = h["shares"]
                                        elif "SELL HALF" in raw_action:
                                            sell_qty = max(1, h["shares"] // 2)
                                            action = f"🟠 SELL {sell_qty} of {h['shares']}"
                                        elif "BUY" in raw_action:
                                            action = "🟢 BUY MORE"
                                            sell_qty = 0
                                        else:
                                            action = "🟢 HOLD"
                                            sell_qty = 0
                                    else:
                                        # Mechanical fallback
                                        sc = h["score_change"]
                                        if h["has_red_flags"]:
                                            action = f"🔴 SELL ALL ({h['shares']})"
                                            reasoning = "Earnings quality red flags detected. Graham warns against value traps."
                                            confidence = "high"
                                            sell_qty = h["shares"]
                                        elif sc <= -2:
                                            sell_qty = max(1, h["shares"] // 2)
                                            action = f"🟠 SELL {sell_qty} of {h['shares']}"
                                            reasoning = "Score dropped sharply. Review fundamentals."
                                            confidence = "medium"
                                        elif sc <= -1:
                                            action = "🟡 HOLD (watch)"
                                            reasoning = "Slight deterioration. Monitor next review."
                                            confidence = "medium"
                                            sell_qty = 0
                                        elif sc == 0:
                                            action = "🟢 HOLD"
                                            reasoning = "Fundamentals stable."
                                            confidence = "high"
                                            sell_qty = 0
                                        else:
                                            action = "🟢 BUY MORE"
                                            reasoning = "Score improved."
                                            confidence = "medium"
                                            sell_qty = 0

                                    
                                    # ── DETERMINISTIC RED FLAG OVERRIDE ──
                                    # The LLM is not trusted on quality failures.
                                    # If has_red_flags is True, force SELL ALL regardless.
                                    if h["has_red_flags"] and "SELL ALL" not in action:
                                        action = f"🔴 SELL ALL ({h['shares']})"
                                        sell_qty = h["shares"]
                                        reasoning = (
                                            f"OVERRIDE: Earnings quality RED FLAGS detected — "
                                            f"{', '.join(h['quality_flags'])}. "
                                            f"Graham warns against value traps where reported earnings "
                                            f"are inflated by non-recurring items. Forced SELL."
                                        )
                                        confidence = "high"

                                    # ── DETERMINISTIC BELOW-THRESHOLD OVERRIDE ──
                                    # Score 0 = no thesis. Score 1 without Graham = below buy threshold.
                                    # Both warrant exit. Score 1 WITH Graham = deep value exception, hold.
                                    #
                                    # GAP CLOSED 2026-07: the score-1 branch now
                                    # asks selector.forced_exit_applies, which
                                    # holds every stock to the same FRACTION
                                    # (<= 1/5 of applicable frameworks). At n < 5
                                    # the ceiling is 0, so only a total thesis
                                    # failure forces an exit. The score-0 branch
                                    # is unchanged and needs no guard — "every
                                    # applicable framework fails" is
                                    # denominator-invariant.
                                    if h["now_score"] == 0 and "SELL" not in action:
                                        action = f"🔴 SELL ALL ({h['shares']})"
                                        sell_qty = h["shares"]
                                        reasoning = (
                                            f"Score is {_score_label(h['ticker'], 0)} — every applicable "
                                            f"framework fails. "
                                            f"No investment thesis exists. Redeploy capital."
                                        )
                                        confidence = "high"
                                    elif h["now_score"] == 1 and selector.forced_exit_applies(universe_df, h["ticker"], 1) and "SELL" not in action:
                                        # Check if the lone pass is Graham (deep value exception)
                                        urow_check = universe_df[universe_df["ticker"] == h["ticker"]]
                                        graham_alive = (
                                            len(urow_check) > 0 
                                            and "graham_pass" in urow_check.columns 
                                            and urow_check["graham_pass"].iloc[0] == True
                                        )
                                        if not graham_alive:
                                            action = f"🔴 SELL ALL ({h['shares']})"
                                            sell_qty = h["shares"]
                                            reasoning = (
                                                f"Score dropped to {_score_label(h['ticker'], 1)} without "
                                                f"Graham pass — below the buy threshold and no deep value "
                                                f"exception applies. Thesis has eroded."
                                            )
                                            confidence = "high"

                                    # ── FIRM-DISTRESS FLAG (human-confirmation, not auto-sell) ──
                                    # A high-confidence hard distress signal (SEBI/RBI action,
                                    # default, auditor exit). This is a CONVICTION-EXIT prompt for
                                    # the human — it does NOT force a sell like the red-flag override.
                                    _h_name = h.get("name", "")
                                    if _h_name in _held_distress and "SELL ALL" not in action:
                                        _dr = _held_distress[_h_name]["reason"]
                                        action = f"⚠️ REVIEW: possible distress ({h['shares']})"
                                        reasoning = (
                                            f"DISTRESS SIGNAL: {_dr} "
                                            f"This is a potential thesis-breaking event flagged for YOUR "
                                            f"decision — not an automatic sell. If the thesis is broken, "
                                            f"exit; if it's noise, hold. Verify before acting."
                                        )
                                        confidence = "review"

                                    
                                    mkt_note = ""
                                    if h["market_relative"] is not None:
                                        if h["market_relative"] > 5:
                                            mkt_note = f"Outperformed Nifty by {h['market_relative']:+.1f}%"
                                        elif h["market_relative"] < -5:
                                            mkt_note = f"Underperformed Nifty by {h['market_relative']:+.1f}%"
                                        else:
                                            mkt_note = f"In line with market ({h['market_relative']:+.1f}% vs Nifty)"

                                    review_rows.append({
                                        "Stock": h["name"], "Shares": h["shares"],
                                        "Entry": f"{fmt_inr(h['entry_price'], 2)}", "Now": f"{fmt_inr(h['now_price'], 2)}",
                                        "P&L": f"{fmt_inr(h['pnl'])}", "Return": f"{h['stock_return']:+.1f}%",
                                        "Score": f"{h['entry_score']}→{h['now_score']}", "Trend": h.get("score_trend", "—"), "Action": action,
                                        "_reasoning": reasoning, "_confidence": confidence,
                                        "_market_note": mkt_note, "_book_passage": h["book_passage"],
                                        "_sell_qty": sell_qty, "_holding_id": h["holding_id"],
                                        "_ticker": h["ticker"], "_sector": h["sector"],
                                        "_entry_price": h["entry_price"], "_now_price": h["now_price"],
                                        "_thesis_drift": _drift.get(h["ticker"]),
                                    })

                                # Auto-run health check during review
                                hc_result = generate_health_check(port, enrich_holdings_live(holdings, cache_key=str(port["id"])), universe_df, collection)

                                st.session_state[f"review_data_{port['id']}"] = {
                                    "rows": review_rows, "total_entry": total_entry,
                                    "total_current": total_current, "holdings": holdings,
                                    "enriched": enriched, "health_check": hc_result,
                                }

                                try:
                                    next_days = int(port.get("review_freq", 90))
                                except (ValueError, TypeError):
                                    next_days = 90
                                new_review = (today + datetime.timedelta(days=next_days)).isoformat()
                                try:
                                    sb.table("portfolios").update({"next_review_date": new_review}).eq("id", port["id"]).execute()
                                except Exception:
                                    pass

                review_state = st.session_state.get(f"review_data_{port['id']}")
                if review_state:
                    review_rows = review_state["rows"]
                    total_entry = review_state["total_entry"]
                    total_current = review_state["total_current"]
                    rev_holdings = review_state["holdings"]

                    port_pnl = total_current - total_entry
                    port_ret = (port_pnl / total_entry * 100) if total_entry > 0 else 0

                    # Position-level, NOT portfolio-level: total_entry is the cost
                    # basis of the open positions under review. It excludes closed
                    # positions and any uninvested cash, so it is deliberately
                    # labelled differently from the portfolio numbers above.
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Cost basis (open)", f"{fmt_inr(total_entry)}")
                    m2.metric("Value now", f"{fmt_inr(total_current)}")
                    m3.metric("Unrealized P&L", f"{fmt_inr(port_pnl)}", delta=f"{port_ret:+.1f}%")

                    # Portfolio-level Nifty alpha
                    _enriched_data = review_state.get("enriched", [])
                    _nifty_ret = _enriched_data[0].get("nifty_return") if _enriched_data else None
                    if _nifty_ret is not None:
                        _alpha = round(port_ret - _nifty_ret, 1)
                        m4.metric("vs Nifty", f"{_alpha:+.1f}%", delta=f"Nifty {_nifty_ret:+.1f}%")
                    else:
                        m4.metric("vs Nifty", "—")

                    # ── Simplified review surface ──
                    # The user sees a crisp SELL / HOLD verdict per holding with a
                    # one-line takeaway. ALL the committee analysis — the exact
                    # SELL ALL/HALF/BUY MORE nuance, full reasoning, confidence,
                    # thesis drift, quality data, book passage — is preserved behind
                    # the "Why?" expander. Nothing is dropped; only the surface is
                    # simplified. SELL verdicts get a one-tap Kite sell.
                    enriched_data = review_state.get("enriched", [])
                    for r in review_rows:
                        _act = r["Action"]
                        _is_sell = "SELL" in _act
                        _verdict = "SELL" if _is_sell else "HOLD"
                        _vicon = "🔴" if _is_sell else "🟢"
                        _one = (r.get("_reasoning") or "").strip()
                        if _one:
                            _one = _one.split(". ")[0].rstrip(".") + "."

                        with st.container(border=True):
                            _c1, _c2 = st.columns([3, 1])
                            with _c1:
                                st.markdown(f"### {_vicon} {_verdict} — {r['Stock']}")
                                if _one:
                                    st.markdown(_one)
                                st.caption(f"Return {r.get('Return','—')} · Score {r.get('Score','—')} · {r.get('Shares','—')} shares")
                            with _c2:
                                if _is_sell:
                                    _qty = int(r.get("_sell_qty") or r.get("Shares") or 0)
                                    if _qty > 0:
                                        try:
                                            st.link_button(f"Sell {_qty} on Kite",
                                                           kite_sell_url(r["_ticker"], _qty),
                                                           use_container_width=True)
                                        except Exception:
                                            pass

                            with st.expander("Why? — full analysis"):
                                st.markdown(f"**Committee verdict:** {_act}  ·  confidence: {r.get('_confidence','—')}")
                                if r.get("_reasoning"):
                                    st.markdown(r["_reasoning"])
                                if r.get("_market_note"):
                                    st.caption(r["_market_note"])
                                _dfmt = _format_thesis_drift(r.get("_thesis_drift"))
                                if _dfmt:
                                    _badge, _dmd = _dfmt
                                    _icon = {"broken": "🔴", "weak": "🟠", "strong": "🟢",
                                             "intact": "🟢", "neutral": "⚪"}.get(_badge, "⚪")
                                    st.markdown(f"{_icon} **Thesis drift** — {_dmd}")
                                matching = next((h for h in enriched_data if h.get("ticker") == r["_ticker"]), None)
                                if matching:
                                    st.caption(
                                        f"Flags: {matching.get('quality_flags','N/A')} · "
                                        f"Cash conversion: {matching.get('cash_conversion','N/A')} · "
                                        f"Red flags: {matching.get('has_red_flags', False)} · "
                                        f"ROE trend: {matching.get('roe_trend', [])}")
                                if r.get("_book_passage"):
                                    st.caption(f"📖 {r['_book_passage']}")

                    # ── Macro shift since last review (review-diff) ──
                    _mdiff = st.session_state.get(f"_macro_diff_{port['id']}")
                    if _mdiff:
                        _mtext = _render_macro_diff(_mdiff)
                        if _mtext:
                            with st.container(border=True):
                                st.markdown(_mtext)

                    # ── Health Check (inline during review) ──
                    hc = review_state.get("health_check")
                    if hc:
                        with st.container(border=True):
                            st.markdown("**Health Check Results**")
                            d_score = hc["diversification_score"]
                            d_color = "🟢" if d_score >= 70 else "🟡" if d_score >= 40 else "🔴"
                            st.metric("Diversification Score", f"{d_color} {d_score}/100")

                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("Avg Beta", hc["avg_beta"] or "N/A")
                            with m2:
                                pe_val = hc["avg_pe_vs_historical"]
                                pe_label = f"{pe_val:+.1f}%" if pe_val is not None else "N/A"
                                st.metric("PE vs History", pe_label)
                            with m3:
                                high_val = hc["avg_pct_from_52w_high"]
                                high_label = f"{high_val:.1f}%" if high_val is not None else "N/A"
                                st.metric("From 52w High", high_label)

                            sector_dist = hc["sector_distribution"]
                            if sector_dist:
                                sector_df = pd.DataFrame([
                                    {"Sector": s, "Stocks": c, "Weight": f"{c/sum(sector_dist.values())*100:.0f}%"}
                                    for s, c in sorted(sector_dist.items(), key=lambda x: -x[1])
                                ])
                                st.dataframe(sector_df, hide_index=True, width="stretch")

                            for w in hc.get("warnings", []):
                                st.warning(w)

                            if hc.get("narrative"):
                                st.markdown("---")
                                st.markdown(hc["narrative"])

                        # ── Actionable recommendations ──
                        hc_actions = hc.get("actions", [])
                        if hc_actions:
                            st.markdown("---")
                            st.markdown("**Execute Recommendations**")
                            for ai, act in enumerate(hc_actions):
                                act_type = act.get("type", "")
                                act_ticker = act.get("ticker", "")
                                act_reason = act.get("reason", "")

                                if act_type == "reduce":
                                    target_pct = act.get("target_alloc_pct", 0)
                                    if st.button(
                                        f"📉 Reduce {act_ticker} to {target_pct}% — {act_reason}",
                                        key=f"hc_reduce_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            sb.table("holdings").update(
                                                {"allocation_pct": target_pct}
                                            ).eq("portfolio_id", port["id"]).eq("ticker", act_ticker).execute()
                                            st.success(f"Updated {act_ticker} allocation to {target_pct}%.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "sell":
                                    sell_shares = act.get("shares", 0)
                                    label = f"🔴 Sell all {act_ticker}" if sell_shares == 0 else f"🔴 Sell {sell_shares} shares of {act_ticker}"
                                    if st.button(
                                        f"{label} — {act_reason}",
                                        key=f"hc_sell_{port['id']}_{ai}",
                                        width="stretch"
                                    ):
                                        try:
                                            # Both branches now resolve the holding FIRST: the old
                                            # "sell all" path deleted the row without ever knowing
                                            # how many shares left the portfolio, so the sale could
                                            # not be written to the ledger even in principle.
                                            h_resp = sb.table("holdings").select("*").eq(
                                                "portfolio_id", port["id"]
                                            ).eq("ticker", act_ticker).execute()
                                            h = (h_resp.data or [None])[0]
                                            if not h:
                                                st.warning(f"{act_ticker} is no longer held.")
                                                st.stop()
                                            _held = int(h.get("shares") or 0)
                                            _qty = _held if sell_shares == 0 else min(int(sell_shares), _held)
                                            _px = live_price(act_ticker, h.get("price_at_entry", 0))
                                            if _qty <= 0 or _px <= 0:
                                                st.error("No live price for this ticker — sale not recorded. "
                                                         "Use Review, where you can enter the fill price.")
                                                st.stop()
                                            record_transaction(
                                                sb, port["id"], st.session_state.sb_user_id,
                                                act_ticker, _qty, _px, round(_qty * _px, 2), "sell",
                                                benchmark_ticker=port.get("benchmark_ticker"),
                                                raise_on_error=True,
                                            )
                                            new_shares = _held - _qty
                                            if new_shares <= 0:
                                                sb.table("holdings").delete().eq("id", h["id"]).execute()
                                                st.success(f"Sold all {_qty} shares of {act_ticker} at {fmt_inr(_px)}.")
                                            else:
                                                new_invested = new_shares * h.get("price_at_entry", 0)
                                                sb.table("holdings").update({
                                                    "shares": new_shares,
                                                    "sip_amount_inr": round(new_invested, 2)
                                                }).eq("id", h["id"]).execute()
                                                st.success(f"Sold {_qty} shares of {act_ticker} at {fmt_inr(_px)}.")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed: {e}")

                                elif act_type == "investigate":
                                    btn_key = f"hc_action_{port['id']}_{ai}"
                                    inv_key = f"hc_inv_result_{port['id']}_{ai}"
                                    if st.button(
                                        f"Investigate {act_ticker} — {act_reason}",
                                        key=btn_key, width="stretch"
                                    ):
                                        with st.spinner(f"Investigating {act_ticker}..."):
                                            try:
                                                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                                                stock_data = get_stock_data(act_ticker)
                                                book_data = search_book(f"{act_reason} investment risk")
                                                inv_prompt = (
                                                    f"You are Kordent's analyst investigating a specific concern about {act_ticker}.\n\n"
                                                    f"CONCERN: {act_reason}\n\n"
                                                    f"STOCK DATA:\n{json.dumps(stock_data, indent=2, default=str)}\n\n"
                                                    f"BOOK CONTEXT:\n{book_data.get('passages', '')[:800]}\n\n"
                                                    f"Write a focused 150-word investigation: what does the data show about this concern? "
                                                    f"Is the concern valid? What should the investor do? Cite book principles."
                                                )
                                                last_good = st.session_state.get("last_working_model")
                                                models = [last_good] + [m for m in FREE_MODELS if m != last_good] if last_good else FREE_MODELS
                                                for model in models:
                                                    try:
                                                        resp = client.models.generate_content(model=model, contents=inv_prompt)
                                                        st.session_state[inv_key] = resp.text
                                                        st.session_state.last_working_model = model
                                                        break
                                                    except Exception as e:
                                                        error_msg = str(e).upper()
                                                        if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "404", "NOT_FOUND"]):
                                                            continue
                                                        break
                                            except Exception as e:
                                                st.session_state[inv_key] = f"Investigation failed: {e}"
                                        st.rerun()

                                    if st.session_state.get(inv_key):
                                        with st.container(border=True):
                                            st.markdown(st.session_state[inv_key])
                                            if st.button("Dismiss", key=f"inv_dismiss_{port['id']}_{ai}"):
                                                del st.session_state[inv_key]
                                                st.rerun()

                    # ── Update form — ALL holdings ──
                    sell_stocks = [(i, r) for i, r in enumerate(review_rows) if "SELL" in r["Action"]]

                    st.markdown("---")

                    # Review is now strictly analytical. SIP is handled mechanically elsewhere.
                    # We only deploy freed capital from sells during a review.
                    cycle_amount = 0

                    sip_stocks = []
                    for r in review_rows:
                        if "SELL" not in r["Action"]:
                            h_match = next((h for h in rev_holdings if h.get("id") == r["_holding_id"]), {})
                            alloc = h_match.get("allocation_pct", 0)
                            non_sell_count = len([x for x in review_rows if "SELL" not in x["Action"]])
                            if alloc == 0 and non_sell_count > 0:
                                alloc = 100 / non_sell_count
                            sip_stocks.append({
                                "ticker": r["_ticker"], "name": r["Stock"],
                                "allocation_pct": alloc, "price": r["_now_price"],
                            })
                    sip_alloc = {}
                    unallocated_sip = cycle_amount
                    if sip_stocks and cycle_amount > 0:
                        _existing = {h.get("ticker", ""): h.get("shares", 0) for h in rev_holdings}
                        allocated, unallocated_sip = allocate_shares(sip_stocks, cycle_amount, existing_shares=_existing)
                        for a in allocated:
                            sip_alloc[a["ticker"]] = a["shares"]
                        st.caption(f"💰 This cycle ({review_days} days): {fmt_inr(cycle_amount)} to invest — suggested shares pre-filled below")
                        if unallocated_sip > 0:
                            st.caption(f"{fmt_inr(unallocated_sip)} unallocatable (not enough for another share)")
                    else:
                        st.caption("Update what you actually did at your broker since last review:")

                    # ── Kite basket for buy-side stocks ──
                    if KITE_ENABLED and sip_stocks:
                        _kite_buys = [{"ticker": s["ticker"], "quantity": sip_alloc.get(s["ticker"], 1)}
                                      for s in sip_stocks if sip_alloc.get(s["ticker"], 0) > 0]
                        if not _kite_buys:
                            _kite_buys = [{"ticker": s["ticker"], "quantity": 1} for s in sip_stocks]
                        if _kite_buys:
                            st.link_button("🛒 Buy All on Kite", kite_basket_url(_kite_buys), use_container_width=True)
                            st.caption("After executing on your broker, confirm below:")

                    for i, r in enumerate(review_rows):
                        h_id = r["_holding_id"]
                        if "SELL" in r["Action"]:
                            st.number_input(
                                f"🔴 {r['Stock']} — shares sold (of {r['Shares']})",
                                min_value=0, max_value=r["Shares"], value=r["_sell_qty"],
                                key=f"sold_{port['id']}_{h_id}"
                            )
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                suggested = sip_alloc.get(r["_ticker"], 0)
                                st.number_input(
                                    f"{'🟢' if 'BUY' in r['Action'] else '📥'} {r['Stock']} — shares bought",
                                    min_value=0, value=suggested, key=f"add_qty_{port['id']}_{h_id}"
                                )
                            with c2:
                                st.number_input(
                                    f"{r['Stock']} — price paid (₹)",
                                    min_value=0.0, value=float(r["_now_price"]), format="%.2f", key=f"add_price_{port['id']}_{h_id}"
                                )
                            st.number_input(
                                f"🔻 {r['Stock']} — shares sold (of {r['Shares']})",
                                min_value=0, max_value=r["Shares"], value=0,
                                key=f"manual_sold_{port['id']}_{h_id}"
                            )

                    # ── Replacement candidates if sells exist ──
                    candidates = []
                    if sell_stocks:
                        freed = 0
                        for idx, r in sell_stocks:
                            sell_qty = st.session_state.get(f"sold_{port['id']}_{r['_holding_id']}", 0)
                            price = r["_now_price"]
                            freed += sell_qty * price
                        remaining_sectors = []
                        for i, r in enumerate(review_rows):
                            is_sell = any(si == i for si, _ in sell_stocks)
                            if not is_sell:
                                remaining_sectors.append(r.get("_sector", ""))
                            else:
                                sold_qty = st.session_state.get(f"sold_{port['id']}_{r['_holding_id']}", 0)
                                if r["Shares"] - sold_qty > 0:
                                    remaining_sectors.append(r.get("_sector", ""))
                        all_tickers = [r["_ticker"] for r in review_rows]
                        candidates = find_replacement_candidates(
                            port.get("investor_type", "balanced"), port.get("time_horizon", "medium"),
                            all_tickers, remaining_sectors
                        )
                        if candidates:
                            st.markdown("---")
                            total_repl_budget = freed + unallocated_sip
                            st.markdown(f"**Replacement candidates** ({fmt_inr(freed)} freed + {fmt_inr(unallocated_sip)} SIP = **{fmt_inr(total_repl_budget)}** to deploy)")
                            cand_df = pd.DataFrame(candidates)
                            cand_display = cand_df[["name", "ticker", "sector", "price", "score", "pe", "roe_pct"]].rename(columns={
                                "name": "Stock", "ticker": "Ticker", "sector": "Sector",
                                "price": "Price", "score": "Score", "pe": "P/E", "roe_pct": "ROE %"
                            })
                            st.dataframe(cand_display, hide_index=True, width="stretch")
                            per_slot = total_repl_budget / len(candidates) if candidates else 0
                            if KITE_ENABLED:
                                _repl_kite = [{"ticker": c["ticker"],
                                               "quantity": max(1, int(per_slot // c["price"])) if c["price"] > 0 else 1}
                                              for c in candidates]
                                st.link_button("🛒 Buy Replacements on Kite", kite_basket_url(_repl_kite), use_container_width=True)
                            st.caption("Shares pre-filled from total budget. Set to 0 to skip a stock.")
                            for c in candidates:
                                suggested = max(1, int(per_slot // c["price"])) if c["price"] > 0 else 0
                                col_name, col_qty, col_px = st.columns([2, 1, 1])
                                with col_name:
                                    st.markdown(f"**{c['name'].strip()}** ({c['ticker']})")
                                with col_qty:
                                    st.number_input("Shares", min_value=0, value=suggested, key=f"repl_qty_{port['id']}_{c['ticker']}")
                                with col_px:
                                    st.number_input("Price (₹)", min_value=0.0, value=float(c["price"]), format="%.2f", key=f"repl_px_{port['id']}_{c['ticker']}")
                            # ── Budget tracker ──
                            spent = 0
                            for c in candidates:
                                rq = st.session_state.get(f"repl_qty_{port['id']}_{c['ticker']}", 0)
                                rp = st.session_state.get(f"repl_px_{port['id']}_{c['ticker']}", 0.0)
                                if rq > 0:
                                    spent += rq * rp
                            remaining = total_repl_budget - spent
                            if remaining >= 0:
                                st.caption(f"💰 Budget: {fmt_inr(total_repl_budget)} — Allocated: {fmt_inr(spent)} = {fmt_inr(remaining)} remaining")
                            else:
                                st.warning(f"Over-allocated by {fmt_inr(abs(remaining))}. Budget: {fmt_inr(total_repl_budget)}, Allocated: {fmt_inr(spent)}")

                    # ── Single update button ──
                    if st.button("✅ Portfolio Updated", key=f"apply_{port['id']}", width="stretch"):
                        def _apply_sale(_r, _h_id, _qty):
                            """Ledger first, then holdings. Returns False and leaves the
                            holding untouched if the sale cannot be recorded."""
                            _px = _r.get("_now_price") or _r.get("_entry_price") or 0
                            if _qty <= 0 or _px <= 0:
                                st.error(f"No usable price for {_r.get('_ticker')} — sale not recorded.")
                                return False
                            try:
                                record_transaction(
                                    sb, port["id"], st.session_state.sb_user_id,
                                    _r["_ticker"], _qty, _px, round(_qty * _px, 2), "sell",
                                    benchmark_ticker=port.get("benchmark_ticker"),
                                    raise_on_error=True,
                                )
                            except Exception as _e:
                                st.error(f"Sale of {_r.get('_ticker')} not recorded, holding unchanged: {_e}")
                                return False
                            _new = _r["Shares"] - _qty
                            if _new <= 0:
                                sb.table("holdings").delete().eq("id", _h_id).execute()
                            else:
                                sb.table("holdings").update({
                                    "shares": _new,
                                    "sip_amount_inr": round(_new * _r["_entry_price"], 2),
                                }).eq("id", _h_id).execute()
                            return True

                        for i, r in enumerate(review_rows):
                            h_id = r["_holding_id"]
                            if "SELL" in r["Action"]:
                                sold = st.session_state.get(f"sold_{port['id']}_{h_id}", 0)
                                if sold > 0:
                                    _apply_sale(r, h_id, sold)
                            else:
                                manual_sold = st.session_state.get(f"manual_sold_{port['id']}_{h_id}", 0)
                                if manual_sold > 0:
                                    _apply_sale(r, h_id, manual_sold)
                                else:
                                    new_qty = st.session_state.get(f"add_qty_{port['id']}_{h_id}", 0)
                                    buy_price = st.session_state.get(f"add_price_{port['id']}_{h_id}", 0.0)
                                    if new_qty > 0 and buy_price > 0:
                                        old_shares = r["Shares"]
                                        old_price = r["_entry_price"]
                                        total_shares = old_shares + new_qty
                                        avg_price = ((old_shares * old_price) + (new_qty * buy_price)) / total_shares
                                        sb.table("holdings").update({
                                            "shares": total_shares,
                                            "price_at_entry": round(avg_price, 2),
                                            "sip_amount_inr": round(total_shares * avg_price, 2),
                                        }).eq("id", h_id).execute()
                                        record_transaction(sb, port["id"], st.session_state.sb_user_id, r["_ticker"], new_qty, buy_price, round(new_qty * buy_price, 2), "buy")
                        if sell_stocks and candidates:
                            for c in candidates:
                                qty = st.session_state.get(f"repl_qty_{port['id']}_{c['ticker']}", 0)
                                px = st.session_state.get(f"repl_px_{port['id']}_{c['ticker']}", 0.0)
                                if qty > 0 and px > 0:
                                    urow = universe_df[universe_df["ticker"] == c["ticker"]]
                                    sc_val = int(urow["score"].iloc[0]) if len(urow) and pd.notna(urow["score"].iloc[0]) else None
                                    pe_val = float(urow["pe"].iloc[0]) if len(urow) and pd.notna(urow["pe"].iloc[0]) else None
                                    roe_val = float(urow["roe_y0"].iloc[0]) if len(urow) and "roe_y0" in urow.columns and pd.notna(urow["roe_y0"].iloc[0]) else None
                                    _repl_invested = round(qty * px, 2)
                                    sb.table("holdings").insert({
                                        "portfolio_id": port["id"], "ticker": c["ticker"], "name": c["name"],
                                        "sector": c["sector"], "allocation_pct": 0, "shares": qty,
                                        "sip_amount_inr": _repl_invested, "price_at_entry": round(px, 2),
                                        "pe_at_entry": pe_val, "roe_at_entry": roe_val, "score_at_entry": sc_val,
                                    }).execute()
                                    record_transaction(sb, port["id"], st.session_state.sb_user_id, c["ticker"], qty, px, _repl_invested, "buy")
                        st.session_state.pop(f"review_data_{port['id']}", None)
                        st.success("Portfolio updated.")
                        st.rerun()

                    if st.button("✕ Close Review", key=f"close_review_{port['id']}"):
                        st.session_state.pop(f"review_data_{port['id']}", None)
                        st.rerun()


                col_r, col_d = st.columns([3, 1])
                with col_r:
                    new_name = st.text_input("Rename", value=port["name"], key=f"rename_{port['id']}", label_visibility="collapsed")
                    if new_name != port["name"]:
                        if st.button("Save Name", key=f"save_name_{port['id']}"):
                            try:
                                sb.table("portfolios").update({"name": new_name}).eq("id", port["id"]).execute()
                                st.success("Renamed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Rename failed: {e}")
                with col_d:
                    if st.button("🗑️ Delete", key=f"delete_{port['id']}", type="secondary"):
                        st.session_state[f"confirm_delete_{port['id']}"] = True

                if st.session_state.get(f"confirm_delete_{port['id']}"):
                    st.warning("Are you sure? This cannot be undone.")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes, delete", key=f"confirm_yes_{port['id']}"):
                            try:
                                sb.table("sip_transactions").delete().eq("portfolio_id", port["id"]).execute()
                                sb.table("portfolio_alerts").delete().eq("portfolio_id", port["id"]).execute()
                                sb.table("portfolio_history").delete().eq("portfolio_id", port["id"]).execute()
                                sb.table("holdings").delete().eq("portfolio_id", port["id"]).execute()
                                sb.table("portfolios").delete().eq("id", port["id"]).execute()
                                st.session_state.pop(f"confirm_delete_{port['id']}", None)
                                st.success("Deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                    with c2:
                        if st.button("Cancel", key=f"confirm_no_{port['id']}"):
                            st.session_state.pop(f"confirm_delete_{port['id']}", None)
                            st.rerun()

# ──────────────────────────────────────────────
# SETTINGS VIEW (Sprint 9)
# ──────────────────────────────────────────────
elif st.session_state.sb_view_mode == "settings":
    st.markdown("### ⚙️ Settings")

    if st.session_state.sb_user_id is None:
        st.warning("Please log in to access settings.")
    else:
        _set_sb = get_supabase()

        # ── Profile ──
        with st.container(border=True):
            st.markdown("**Profile**")
            st.caption(f"Email: {st.session_state.sb_user_email}")

            _set_name = st.text_input(
                "Full Name",
                value=st.session_state.get("_profile_name", ""),
                key="settings_name_input"
            )
            if st.button("Save Name", key="settings_save_name"):
                if _set_name and _set_name.strip():
                    try:
                        _set_sb.table("profiles").upsert({
                            "id": st.session_state.sb_user_id,
                            "full_name": _set_name.strip()
                        }, on_conflict="id").execute()
                        st.session_state["_profile_name"] = _set_name.strip()
                        st.success("Name updated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
                else:
                    st.warning("Enter a name.")

        # ── Telegram ──
        with st.container(border=True):
            st.markdown("**Telegram Alerts**")

            if not st.session_state.get("_tg_checked"):
                try:
                    _tg_prof = _set_sb.table("profiles").select("telegram_chat_id").eq(
                        "id", st.session_state.sb_user_id).limit(1).execute()
                    st.session_state["_tg_connected"] = bool(
                        _tg_prof.data and _tg_prof.data[0].get("telegram_chat_id"))
                except Exception:
                    st.session_state["_tg_connected"] = False
                st.session_state["_tg_checked"] = True

            if st.session_state.get("_tg_connected"):
                st.caption("✅ Connected — you receive daily updates and alerts on Telegram.")
                if st.button("Disconnect Telegram", key="tg_disconnect_settings"):
                    try:
                        _set_sb.table("profiles").update({"telegram_chat_id": None}).eq(
                            "id", st.session_state.sb_user_id).execute()
                        st.session_state["_tg_connected"] = False
                        st.session_state.pop("_tg_link_code", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
            else:
                st.caption("Get daily portfolio updates, danger alerts, and SIP reminders on Telegram.")
                if st.session_state.get("_tg_link_code"):
                    st.markdown("Send this to **@KordentAIBot** on Telegram:")
                    st.code(f"/start {st.session_state['_tg_link_code']}", language=None)
                    st.caption("You'll get a confirmation within the hour. Refresh this page to check.")
                    if st.button("🔄 Check connection", key="tg_recheck_settings", use_container_width=True):
                        st.session_state.pop("_tg_checked", None)
                        st.session_state.pop("_tg_link_code", None)
                        st.rerun()
                else:
                    if st.button("Connect Telegram", key="tg_connect_settings", use_container_width=True):
                        import random
                        code = str(random.randint(100000, 999999))
                        try:
                            _set_sb.table("profiles").update({"telegram_link_code": code}).eq(
                                "id", st.session_state.sb_user_id).execute()
                            st.session_state["_tg_link_code"] = code
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")

# ──────────────────────────────────────────────
# DOES IT WORK? VIEW (Sprint 6)
# ──────────────────────────────────────────────
elif st.session_state.sb_view_mode == "backtest":
    # ──────────────────────────────────────────────
    # DOES IT WORK? (Sprint 12 — honest clock, not a fabricated backtest)
    # ──────────────────────────────────────────────
    # The old version rendered a strategy-vs-Nifty CAGR chart from a backtest
    # that CANNOT exist yet: the point-in-time archive is days old, and a
    # durable-business thesis cannot be judged on weeks of forward returns.
    # This page tells that truth structurally: it opens with the count (starting
    # near zero) and the DATE the numbers start to mean something, then shows the
    # things that ARE measurable today — descriptive facts about the scorer,
    # each labelled as description, never as a performance claim.
    import json as _json
    import datetime as _dtm

    st.markdown("### 📊 Does It Work?")

    CLOCK_START = _dtm.date(2026, 7, 10)   # dividend + trajectory_pass fixes landed

    # ── Film: raw snapshots archived (collection cadence, ~1/weekday) ──
    clean_snaps = 0
    last_snap = None
    try:
        with open("archive_manifest.json") as _mf:
            _recs = _json.load(_mf)
        _clean = [r for r in _recs
                  if r.get("schema_version", 0) >= 1
                  and str(r.get("date", "")) >= CLOCK_START.isoformat()]
        clean_snaps = len(_clean)
        if _clean:
            last_snap = _clean[-1].get("date")
    except Exception:
        clean_snaps = 0

    # ── Grades: the runner is the single authority on the reading ──
    _bt = {}
    try:
        with open("backtest_summary.json") as _bf:
            _bt = _json.load(_bf)
    except Exception:
        _bt = {}
    _status = _bt.get("status", "NO_ARCHIVE")
    _matured = _bt.get("matured_cohorts", 0)
    _quarters = _bt.get("independent_quarters", 0)
    _ladder = _bt.get("ladder", {}) or {}
    _spread = _bt.get("spread_high_minus_low", {}) or {}
    _first_reading = _bt.get("first_reading_date")

    st.markdown(
        "**We can't hand you a fabricated backtest — so here is exactly where the "
        "real evidence stands today.**  \n"
        "The test is simple: photograph every stock's score, wait one holding "
        "horizon, then check whether higher-scored buckets earned more. A good or "
        "bad quarter lifts every bucket together, so it cancels out of the "
        "high-vs-low spread. The reading below is real but noisy at first, and "
        "sharpens as the archive spans more independent quarters.")

    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        st.metric("Snapshots archived", f"{clean_snaps}",
                  help="Point-in-time archives of the whole scored universe — the "
                       "raw film. Grows by one every weekday. Collection cadence, "
                       "not evidence: overlapping days aren't independent.")
    with _c2:
        st.metric("Matured cohorts", f"{_matured}",
                  delta=(f"≈{_quarters} independent quarter"
                         f"{'s' if _quarters != 1 else ''}") if _matured else None,
                  delta_color="off",
                  help="Cohorts whose full holding horizon has elapsed, so they can "
                       "be graded. The spread's band uses the independent-quarter "
                       "count, never the raw cohort count.")
    with _c3:
        if _status == "OK":
            st.metric("Reading", "live",
                      help="A cohort has matured. The ladder below is the current, "
                           "still-noisy reading; it tightens over time.")
        else:
            st.metric("First reading", _first_reading or "—",
                      help="Projected date the first cohort matures = earliest "
                           "clean snapshot + one holding horizon.")

    if _status == "OK" and _ladder:
        st.markdown("**Forward return by entry score** — buy-and-hold, one horizon, "
                    "averaged across matured cohorts:")
        _order = [b for b in ["5", "4", "3", "2", "1", "0"] if b in _ladder]
        _cols = st.columns(len(_order))
        for _col, _b in zip(_cols, _order):
            _v = _ladder[_b]
            _tag = " ·sampled" if _v.get("sampled") else ""
            _col.metric(f"Score {_b}{_tag}", f"{_v['mean_fwd_return'] * 100:+.1f}%",
                        help=f"{_v.get('cohorts', 0)} cohorts")
        _pt = _spread.get("point")
        if _pt is not None:
            _ci = _spread.get("ci95")
            _band = (f"95% CI {_ci[0] * 100:+.1f}% … {_ci[1] * 100:+.1f}%"
                     if _ci else "band forms once a 2nd cohort matures")
            st.metric("High-score minus low-score spread", f"{_pt * 100:+.1f}%",
                      delta=_band, delta_color="off",
                      help="mean(scores 4,5) − mean(scores 0,1) forward return — the "
                           "one number for 'better score = better return', with the "
                           "overlap-adjusted band.")
        _mono = _bt.get("ladder_monotonic")
        if _mono is not None:
            st.caption(("✅ Ladder is monotonic (5 ≥ 4 ≥ … ≥ 0) on current means."
                        if _mono else
                        "⚠️ Ladder isn't cleanly monotonic yet — expected while the "
                        "cohort count is low.")
                       + "  Survivorship note: delisted names are dropped, biasing "
                         "surviving buckets upward.")
    else:
        st.info("No cohort has matured yet, so there is no return reading to show. "
                "The archive is filling in real time — every screener online claims "
                "backtested alpha; we're showing you the film accumulate instead.")

    st.caption(f"Clock started {CLOCK_START.strftime('%d %b %Y')}"
               + (f" · last snapshot {last_snap}" if last_snap else "")
               + (f" · horizon {_bt.get('horizon_trading_days', '?')} trading days"
                  if _bt else ""))

    st.divider()

    # ── What we CAN say today: descriptive stats about the scorer ────────
    st.markdown("#### What we can measure today")
    st.caption("These describe the **scorer** — how it sorts the market right "
               "now. None of them is a claim about returns; that needs the clock "
               "above to fill. A base rate is how often a test fires, never "
               "whether firing predicts profit.")

    try:
        import stats as _kstats
        _udf = globals().get("universe_df")
        if _udf is not None and len(_udf):
            _us = _kstats.compute_universe_stats(_udf)

            # base rates
            _br = _us.get("base_rates", {})
            if _br:
                st.markdown("**How selective each framework is**")
                _spread = _us.get("base_rate_spread", {})
                _rows = "".join(
                    f"| {v['label']} | {v['pass_rate']*100:.1f}% | {v['pass_count']:,} |\n"
                    for v in _br.values())
                st.markdown(
                    "| Framework | Passes | Count |\n|---|---|---|\n" + _rows)
                if _spread:
                    st.caption(
                        f"The rarest test ({_br.get(_spread.get('rarest',''),{}).get('label','')}) "
                        f"fires {_spread.get('ratio','?')}× less often than the "
                        f"commonest. Rarity is not merit — it's shown so you can "
                        f"see how much each filter actually removes.")

            # correlation cluster — the real finding
            _cl = _us.get("least_orthogonal_pair", {})
            _cluster = _us.get("correlated_cluster", [])
            if _cl and _cl.get("labels"):
                _a, _b = _cl["labels"]
                st.markdown("**Are the five frameworks independent?**")
                st.markdown(
                    f"Mostly — but not entirely. **{_a}** and **{_b}** tend to "
                    f"pass the same stocks (φ = {_cl.get('phi')}). "
                    + (f"They're part of a cluster of {len(_cluster)} correlated "
                       f"pairs. " if len(_cluster) > 1 else "")
                    + "That matters: a stock clearing three *correlated* tests is "
                      "a weaker signal than one clearing three *independent* ones. "
                      "We'd rather you know that than hide it.")

            # score pyramid — why 2/5 and 3/5 matter
            _sd = _us.get("score_distribution", {})
            if _sd:
                st.markdown("**How the whole market scores (0–5)**")
                import plotly.graph_objects as _go
                _fig = _go.Figure(_go.Bar(
                    x=[str(k) for k in _sd.keys()],
                    y=list(_sd.values()),
                    marker_color="#1D4ED8"))
                _fig.update_layout(
                    height=240, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title="Score", yaxis_title="Stocks",
                    plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(_fig, use_container_width=True, key="diw_score_dist")
                _top = _sd.get(4, 0) + _sd.get(5, 0)
                _mid = _sd.get(2, 0) + _sd.get(3, 0)
                st.caption(
                    f"Only {_top:,} stocks score 4 or 5 — any screener can find "
                    f"those. The {_mid:,} stocks at 2 and 3 are where matching the "
                    f"right business to the right investor actually earns its keep.")

            # trajectory cliff
            _tc = _us.get("trajectory_cliff", {})
            if _tc and _tc.get("at_boundary"):
                st.markdown("**A known rough edge**")
                st.markdown(
                    f"The Trajectory gate is a hard cut at {_tc['gate']}. "
                    f"**{_tc['at_boundary']:,} stocks sit at exactly "
                    f"{_tc['boundary_score']}** — one point short — versus "
                    f"{_tc['passing']:,} that clear it. That's a cliff, not a "
                    f"slope, and it's on our list to revisit. We show our sharp "
                    f"edges, not just our clean ones.")
        else:
            st.warning("Universe not loaded — descriptive stats unavailable.")
    except Exception as _e:
        st.warning(f"Descriptive stats unavailable: {type(_e).__name__}: {_e}")

    st.divider()

    # ── What we CAN'T say yet, and why ───────────────────────────────────
    with st.expander("Why isn't there a backtest here yet?"):
        st.markdown("""
A backtest needs two things we don't yet have honestly:

**Point-in-time data.** To ask "what would this have picked in 2023, and how did
it do?", we need the scores *as they were then* — not today's numbers applied to
the past, which is look-ahead, not a test. That archive only started
accumulating cleanly on **10 July 2026**, the day two scoring bugs were fixed.
It grows one snapshot per weekday. The counter above is that archive filling up.

**A horizon that matches the thesis.** Kordent asks you to hold durable
businesses for years. A one-month forward return says nothing about that — it
measures one-month momentum, a different and arguably worse question. The first
*hint* about the ranking arrives at roughly six monthly cohorts; a defensible
answer takes about a year.

**On free data, the past can't be bought back.** Reconstructing history from
free financial feeds fails on restatement (today's numbers aren't what was
filed then) and the four-year data window. So we wait, with a clean clock,
rather than ship a number that looks precise and means nothing.

When the clock is ready, this page will show the one test that matters: does the
ranking beat picking blindly from the same stocks that already cleared the
filters? Both sides drawn from the same universe, so survivorship bias cancels.
Until then, zero — honestly.
""")
