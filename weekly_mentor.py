"""
WEEKLY MENTOR EMAIL
==================
Runs every Monday at 3:30 UTC (9:00 AM IST) via GitHub Actions.
Reads alerts written by the daily engine, validates, deduplicates,
synthesizes via Gemini into one mentor-style email per user.

Zero alerts are generated here. This script only consumes.
Adds a macro overlay (RBI policy, yield curve, sector rotation) per Reilly & Brown.

Usage:  python weekly_mentor.py
Cron:   '30 3 * * 1'
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, datetime, timedelta
import json
import verdict_engine
import selector          # score_label + _applicable_frameworks (no Streamlit dep)
from collections import defaultdict

import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types
from supabase import create_client, Client
import requests as _requests


def _html_esc(text):
    """Escape HTML special chars for Telegram."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
 
def send_telegram(chat_id, text, bot_token):
    """Send a message via Telegram Bot API (HTML parse mode). Non-blocking."""
    try:
        _requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  Telegram send failed: {e}")
 
APP_URL = "https://shivam1kedia.streamlit.app"
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]

def get_macro_context(gemini_key):
    """Fetch current macro environment for portfolio management overlay.
    Reilly & Brown Ch 9: leading indicators, yield curve, sector rotation."""
    client = genai.Client(api_key=gemini_key)
    queries = [
        "India RBI repo rate monetary policy stance inflation outlook July 2026",
        "India 10 year government bond yield curve shape recession signal 2026",
        "India stock market sector rotation which sectors outperforming underperforming 2026",
    ]
    results = []
    for q in queries:
        for model in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"Provide a concise 3-4 bullet summary of: {q}",
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                    ),
                )
                if response.text:
                    results.append(response.text.strip())
                    break
            except Exception:
                continue
    return "\n\n".join(results) if results else ""


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def fetch_nifty_weekly():
    """Get Nifty 50 weekly return percentage."""
    try:
        hist = yf.Ticker("^NSEI").history(period="7d")
        if len(hist) >= 2:
            return round((float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100, 2)
    except Exception as e:
        print(f"Nifty fetch failed: {e}")
    return None


def validate_alerts(alerts, universe_df):
    """
    Re-check each alert against current universe data.
    Drop opportunities/new_entries whose score or quality flipped.
    Portfolio-level alerts (ticker starts with _) pass through.
    """
    if universe_df is None:
        return alerts

    valid = []
    for a in alerts:
        ticker = a.get("ticker", "")
        a_type = a.get("alert_type", "")

        # Portfolio-level alerts — no ticker to validate
        if ticker.startswith("_") or not ticker:
            valid.append(a)
            continue

        row = universe_df[universe_df["ticker"] == ticker]
        if row.empty:
            continue  # vanished from universe — skip silently

        cur_score = int(row["score"].iloc[0]) if pd.notna(row["score"].iloc[0]) else 0
        cur_quality = bool(row["quality_pass"].iloc[0]) if "quality_pass" in row.columns and pd.notna(row["quality_pass"].iloc[0]) else True

        # Opportunities and new entries must still pass quality + score gate
        if a_type in ("opportunity", "new_entry"):
            if cur_score < 3 or not cur_quality:
                continue

        valid.append(a)

    dropped = len(alerts) - len(valid)
    if dropped:
        print(f"  Validation dropped {dropped} stale alerts.")
    return valid

def build_weekly_recommendations(user_id, user_ports, universe_df, supabase):
    """
    Build personalized weekly stock recommendations per time_horizon.
    Writes to weekly_recommendations table. Returns list of rec dicts for email.
    """
    if universe_df is None or not user_ports:
        return []

    # Collect ALL held tickers across all user's portfolios
    held_tickers = set()
    for p in user_ports:
        try:
            h_resp = supabase.table("holdings").select("ticker").eq("portfolio_id", p["id"]).execute()
            for h in (h_resp.data or []):
                held_tickers.add(h["ticker"])
        except Exception:
            pass

    # Group portfolios by time_horizon
    horizon_groups = defaultdict(list)
    for p in user_ports:
        if p.get("is_paper"):
            continue
        horizon = p.get("time_horizon", "medium")
        horizon_groups[horizon].append(p)

    recommendations = []
    expires_at = (datetime.utcnow() + timedelta(hours=24)).isoformat()

    for horizon, ports in horizon_groups.items():
        # Dominant profile = largest SIP of this horizon
        dominant = max(ports, key=lambda p: float(p.get("sip_amount") or 0))
        investor_type = dominant.get("investor_type", "balanced")
        max_sip = max(float(p.get("sip_amount") or 0) for p in ports)
        budget = round(max_sip * 0.3)

        if budget <= 0:
            continue

        # Filter universe
        df = universe_df.copy()
        df = df[df["score"] >= 3]
        if "quality_pass" in df.columns:
            df = df[df["quality_pass"] != False]
        df = df[~df["ticker"].isin(held_tickers)]
        df = df[pd.notna(df["pe"]) & (df["pe"] > 0)]
        df = df[pd.notna(df["price"]) & (df["price"] > 0)]

        # Profile-specific framework filtering
        if investor_type == "defensive":
            if len(df[df["graham_pass"] == True]) >= 3:
                df = df[df["graham_pass"] == True]
        elif investor_type == "enterprising":
            if len(df[df["trajectory_pass"] == True]) >= 3:
                df = df[df["trajectory_pass"] == True]
        else:
            quality = df[(df["greenblatt_pass"] == True) | (df["dorsey_pass"] == True)]
            if len(quality) >= 3:
                df = quality

        # Horizon filtering
        if horizon == "short":
            large = df[pd.notna(df.get("market_cap")) & (df["market_cap"] > 1e10)]
            if len(large) >= 3:
                df = large

        # Affordability filter
        _max_n = max(1, min(10, int(budget // 750)))
        _min_n = max(2, _max_n // 3)
        _max_price = budget / _min_n
        affordable = df[df["price"] <= _max_price]
        if len(affordable) >= 3:
            df = affordable

        # Rank: score desc, PE asc
        df = df.sort_values(["score", "pe"], ascending=[False, True])

        # Pick top N
        n_stocks = max(2, min(5, int(budget // 750)))
        df = df.head(n_stocks)

        if df.empty:
            continue

        # Build stock list with share counts
        stocks = []
        per_stock_budget = budget / len(df)
        total_amount = 0

        for _, row in df.iterrows():
            price = round(float(row["price"]), 2)
            shares = int(per_stock_budget / price) if price > 0 else 0
            if shares <= 0:
                continue
            amount = round(shares * price, 2)
            total_amount += amount

            score = int(row["score"])
            pass_dict = {
                "graham_pass": bool(row.get("graham_pass")) if pd.notna(row.get("graham_pass")) else False,
                "greenblatt_pass": bool(row.get("greenblatt_pass")) if pd.notna(row.get("greenblatt_pass")) else False,
                "dorsey_pass": bool(row.get("dorsey_pass")) if pd.notna(row.get("dorsey_pass")) else False,
                "trajectory_pass": bool(row.get("trajectory_pass")) if pd.notna(row.get("trajectory_pass")) else False,
                "lynch_pass": bool(row.get("lynch_pass")) if pd.notna(row.get("lynch_pass")) else False,
            }
            # n_applicable, not the default 5. Greenblatt abstains on
            # financials/utilities and Lynch on a business his six categories
            # cannot place; judging a stock against tests it never took
            # understates the verdict — and this one goes out by email unreviewed.
            _n_app = len(selector._applicable_frameworks(row))
            v = verdict_engine.get_verdict_tier(score, True, pass_dict,
                                                n_applicable=_n_app)

            stocks.append({
                "ticker": row["ticker"],
                "name": str(row.get("name", row["ticker"])),
                "sector": str(row.get("sector", "N/A")),
                "score": score,
                # "3 of 4", not "3/5" — computed HERE because this is where the
                # universe row is in scope; both render sites below just read it.
                "score_label": selector.score_label(row, score),
                "verdict": v,
                "price": price,
                "shares": shares,
                "amount": amount,
            })

        if not stocks:
            continue

        try:
            supabase.table("weekly_recommendations").upsert({
                "user_id": user_id,
                "time_horizon": horizon,
                "investor_type": investor_type,
                "expires_at": expires_at,
                "budget_inr": budget,
                "stocks": json.dumps(stocks),
                "acted_on": False,
            }, on_conflict="user_id,time_horizon").execute()
        except Exception as e:
            print(f"  Weekly rec upsert failed for {horizon}: {e}")
            continue

        recommendations.append({
            "horizon": horizon,
            "investor_type": investor_type,
            "budget": budget,
            "stocks": stocks,
            "total_amount": total_amount,
        })

    return recommendations


def deduplicate_alerts(alerts):
    """
    Same stock + same alert type across multiple portfolios → keep one.
    Attach list of portfolio_ids it appeared in for context.
    """
    groups = defaultdict(list)
    for a in alerts:
        key = f"{a.get('ticker', '')}:{a.get('alert_type', '')}"
        groups[key].append(a)

    deduped = []
    for key, group in groups.items():
        rep = group[0]  # keep first occurrence
        if len(group) > 1:
            rep["_appeared_in_count"] = len(group)
        deduped.append(rep)

    return deduped


def get_portfolio_summaries(ports, supabase, nifty_weekly_pct):
    """Build a summary dict for each portfolio."""
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    summaries = []

    for p in ports:
        pid = p["id"]
        current_val = float(p.get("current_value") or 0)
        ret_pct = float(p.get("current_return_pct") or 0)
        sip = float(p.get("sip_amount") or 0)

        # Detect first-week portfolios
        created_str = str(p.get("created_at", ""))[:10]
        is_first_week = created_str >= cutoff

        # Weekly return from portfolio_history (adjusted for inflows)
        weekly_ret = None
        if not is_first_week:
            try:
                hist = supabase.table("portfolio_history").select("total_value").eq(
                    "portfolio_id", pid
                ).gte("date", cutoff).order("date").limit(1).execute()
                if hist.data:
                    week_start_val = float(hist.data[0]["total_value"])
                    if week_start_val > 0:
                        # Subtract inflows so SIP deposits don't appear as gains
                        inflows = 0
                        try:
                            txn_resp = supabase.table("sip_transactions").select(
                                "amount_inr"
                            ).eq("portfolio_id", pid).eq(
                                "transaction_type", "buy"
                            ).gte("transaction_date", cutoff).execute()
                            for t in (txn_resp.data or []):
                                inflows += float(t.get("amount_inr") or 0)
                        except Exception:
                            pass
                        weekly_ret = round(
                            (current_val - week_start_val - inflows) / week_start_val * 100, 2
                        )
            except Exception:
                pass

        summaries.append({
            "name": p.get("name", "Unnamed"),
            "id": pid,
            "value": current_val,
            "return_pct": ret_pct,
            "weekly_return_pct": weekly_ret,
            "nifty_weekly_pct": nifty_weekly_pct,
            "sip": sip,
            "is_first_week": is_first_week,
            "goal_amount": p.get("target_amount"),
            "goal_date": p.get("target_date"),
            "link": f"{APP_URL}/?portfolio={pid}",
            "is_paper": p.get("is_paper", False),
            "xirr_pct": float(p["xirr_pct"]) if p.get("xirr_pct") is not None else None,
            "nifty_xirr_pct": float(p["nifty_xirr_pct"]) if p.get("nifty_xirr_pct") is not None else None,
        })

    return summaries


def build_gemini_prompt(name, summaries, alerts, recommendations=None, macro_context=""):
    """
    Build the LLM prompt for the weekly email.
    Book passages come from the alerts themselves — no book loading needed.
    """
    # Portfolio summary block
    port_block = ""
    for s in summaries:
        if s.get("is_first_week"):
            weekly = "First week — no comparison yet"
        elif s['weekly_return_pct'] is not None:
            weekly = f"{s['weekly_return_pct']:+.2f}%"
        else:
            weekly = "N/A"
        nifty = f"{s['nifty_weekly_pct']:+.2f}%" if s['nifty_weekly_pct'] is not None else "N/A"
        goal_line = ""
        if s["goal_amount"]:
            goal_line = f"\n  Goal: ₹{float(s['goal_amount']):,.0f} by {s['goal_date']}"

        _paper_tag = " 👁 (Paper — not yet invested)" if s.get("is_paper") else ""
        _xirr_line = ""
        if s.get("xirr_pct") is not None:
            _xirr_line = f"\n  XIRR: {s['xirr_pct']:+.1f}%"
            if s.get("nifty_xirr_pct") is not None:
                _alpha = round(s["xirr_pct"] - s["nifty_xirr_pct"], 1)
                _xirr_line += f" | Nifty XIRR: {s['nifty_xirr_pct']:+.1f}% | Alpha: {_alpha:+.1f}%"
        port_block += f"""
Portfolio: {s['name']}{_paper_tag}
  Value: ₹{s['value']:,.0f} | Overall return: {s['return_pct']:+.1f}%{_xirr_line}
  This week: {weekly} (Nifty: {nifty}){goal_line}
  View: {s['link']}
"""

    # Alerts block with book passages
    danger_block = ""
    opp_block = ""
    info_block = ""

    for a in alerts:
        detail = a.get("detail") or {}
        if isinstance(detail, str):
            import json
            try:
                detail = json.loads(detail)
            except Exception:
                detail = {}

        passages = detail.get("book_passages", [])
        passage_text = ""
        if passages:
            passage_text = " | Book context: " + "; ".join(
                f"[{p['author']}] {p['text'][:200]}" for p in passages[:2]
            )

        appeared = ""
        if a.get("_appeared_in_count", 0) > 1:
            appeared = f" (flagged in {a['_appeared_in_count']} portfolios)"

        # W1 label as a CONSTRAINT, not decoration. portfolio_tracker already
        # persists drift_reason via classify_score_change; it was simply never
        # read here, so the model was asked to explain a score move with nothing
        # deterministic to explain it from — and it inferred a cause.
        # The classifier decides; the model only phrases.
        _reason = detail.get("drift_reason")
        _cause = ""
        if _reason:
            _cause = f" [CAUSE: {_reason}"
            _fw = detail.get("drift_frameworks")
            if _fw:
                _fw_txt = ", ".join(_fw) if isinstance(_fw, list) else str(_fw)
                _cause += f"; frameworks: {_fw_txt}"
            _cause += "]"

        line = f"- {a['headline']}{_cause}{appeared}{passage_text}\n"

        # Severity-first, same as build_plain_fallback. alert_type no longer
        # carries "danger"/"overvalued" — severity does, and this block sorts
        # what the CIO email leads with. On the OLD strings every score_drop and
        # quality_fail fell into ALSO NOTING and the LLM was told to treat a real
        # break as routine — on every weekly email, not just the fallback.
        a_type = a.get("alert_type", "")
        if a_type in ("opportunity", "new_entry"):
            opp_block += line
        elif a.get("severity") == "danger":
            danger_block += line
        else:
            info_block += line

    alerts_block = ""
    if danger_block:
        alerts_block += f"NEEDS ATTENTION:\n{danger_block}\n"
    if opp_block:
        alerts_block += f"OPPORTUNITIES:\n{opp_block}\n"
    if info_block:
        alerts_block += f"ALSO NOTING:\n{info_block}\n"

    if not alerts_block.strip():
        alerts_block = "No significant alerts this week. Quiet weeks are good weeks.\n"

    recs_block = ""
    if recommendations:
        for rec in recommendations:
            recs_block += f"\n{rec['horizon'].upper()} HORIZON ({rec['investor_type']} profile, ₹{rec['budget']:,} budget):\n"
            for s in rec["stocks"]:
                recs_block += f"  {s['name']} ({s['ticker']}) — {s.get('score_label') or s['score']} {s['verdict']} — {s['shares']} shares @ ₹{s['price']:,.0f} = ₹{s['amount']:,.0f}\n"
            recs_block += f"  Total deployed: ₹{rec['total_amount']:,.0f}\n"
    if not recs_block:
        recs_block = "No new recommendations this week.\n"

    prompt = f"""You are Kordent's Chief Investment Officer writing a weekly email to {name}.

TODAY: {date.today().strftime('%A, %B %d, %Y')}

PORTFOLIO DATA:
{port_block}

MACRO ENVIRONMENT (Reilly & Brown Ch 9 — use this to contextualize portfolio advice):
{macro_context if macro_context else "Macro data unavailable this week."}

THIS WEEK'S ALERTS:
{alerts_block}
THIS WEEK'S PICKS:
{recs_block}

Write a warm, personal weekly email. Rules:

1. Address {name} by name. First line should be a one-sentence summary of their week — not a greeting.
2. Give each portfolio its own paragraph with a clear header. Mention value and weekly change vs Nifty. Be honest — if they underperformed, say so plainly. For first-week portfolios, welcome them and set expectations.
3. For each alert, explain WHY it matters using the book context provided. Reference Graham, Greenblatt, or Dorsey naturally — "Graham would say..." not "According to Benjamin Graham's The Intelligent Investor..."
3a. Where an alert carries a [CAUSE: ...] tag, that cause is DETERMINISTIC — computed by comparing the stock's frameworks against the record from when it was added. You MUST use it and MUST NOT contradict it or substitute a cause of your own. Translate it into plain English:
   - fundamental — the business itself weakened. This is the serious one.
   - valuation — the PRICE moved; the business did not change.
   - relative_rank — the business held, but peers improved past it.
   - mixed — both the business and the price moved. Say so; do not pick one.
   - unclear / unknown_inputs — we cannot attribute the change. SAY THAT PLAINLY ("the score moved but we can't say why yet"). Do not guess a reason to fill the gap.
   If an alert has NO [CAUSE:] tag, do not speculate about why a score moved — describe what changed and stop.
3b. Scores read "N of M", where M is how many frameworks APPLY to that business — not always 5. Greenblatt does not apply to banks or utilities (his own instruction), and Lynch does not apply to a business his six categories cannot place. A framework that abstains was NEVER TESTED. Never call it failed, missed, or a weakness, and never say a stock "only got 3 out of 5" when it scored 3 of 4.
4. If there are weekly picks, present them as a curated buy list — mention each stock briefly with why it fits their profile. These expire in 24 hours so convey gentle urgency.
5. If a goal is set, give a one-line status: on track, behind, or ahead.
6. If a portfolio is marked "Paper", note it's a watchlist portfolio — tracking performance without real money. Be encouraging about what they're learning from the simulation.
7. MACRO AWARENESS (Reilly & Brown Ch 9): If macro data is available, open with a 1-2 sentence market environment read. Mention which sectors the current business cycle phase favors. If the yield curve or RBI policy signals caution, say so — a fund manager who ignores macro is irresponsible. But keep it brief and actionable, not academic.
8. End with a patience reminder — one sentence, not preachy. Vary it each week.
9. Include the portfolio link(s) so they can take action.
10. Sign off as "Kordent"

TONE: You're a wise friend who happens to be great with money. Not a robot, not a salesperson, not a professor. Use simple language. If you must use a financial term, define it in parentheses.
LENGTH: Under 400 words. Shorter is better.
FORMAT: Plain text. Use emojis sparingly — 2-3 max in the whole email, only where they add warmth.
DO NOT: Use bullet points. Use "Dear". Use "I hope this finds you well". Use "Best regards". Use corporate jargon."""

    return prompt


def call_gemini(prompt, gemini_key):
    """Call Gemini with model fallback. Returns narrative string or None."""
    client = genai.Client(api_key=gemini_key)
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            if any(k in str(e) for k in ("429", "RESOURCE_EXHAUSTED", "404", "NOT_FOUND", "503", "UNAVAILABLE")):
                print(f"  {model} unavailable or rate-limited, trying next...")
                continue
            print(f"  {model} error: {e}")
            return None
    return None


def build_plain_fallback(name, summaries, alerts, recommendations=None):
    """Decent plain-text email when Gemini is unavailable."""
    lines = [f"{name}, here's your week at a glance.\n"]

    for s in summaries:
        if s.get("is_first_week"):
            weekly = "first week"
        elif s['weekly_return_pct'] is not None:
            weekly = f"{s['weekly_return_pct']:+.2f}%"
        else:
            weekly = "no data"
        nifty = f"Nifty {s['nifty_weekly_pct']:+.2f}%" if s['nifty_weekly_pct'] is not None else ""
        lines.append(f"📊 {s['name']}: ₹{s['value']:,.0f} ({weekly} this week, {nifty})")
        if s.get("xirr_pct") is not None:
            _xirr_fb = f"   XIRR: {s['xirr_pct']:+.1f}%"
            if s.get("nifty_xirr_pct") is not None:
                _xirr_fb += f" | Alpha: {round(s['xirr_pct'] - s['nifty_xirr_pct'], 1):+.1f}%"
            lines.append(_xirr_fb)
        lines.append(f"   {s['link']}")

        lines.append("")

    # Severity-first. Before the alert_type/severity split, "danger" WAS an
    # alert_type; now it is a value in the severity column and alert_type holds
    # the real kind (score_drop, quality_fail, ...). 'overvalued' was deleted
    # entirely. Grouping on the old strings silently classified ZERO alerts as
    # serious and dropped every real danger into "also noting". goal_drift now
    # varies danger/warning, so it lands here only when ITS severity is danger.
    _EXCLUDED = ("opportunity", "new_entry")
    dangers = [a for a in alerts
               if a.get("severity") == "danger"
               and a.get("alert_type") not in _EXCLUDED]
    others = [a for a in alerts
              if a.get("severity") != "danger"
              and a.get("alert_type") not in _EXCLUDED]

    if dangers:
        lines.append("⚠️ Needs your attention:")
        for a in dangers:
            lines.append(f"  {a['headline']}")
        lines.append("")

    if recommendations:
        lines.append("📋 This Week's Picks (24h only):")
        for rec in recommendations:
            lines.append(f"  {rec['horizon'].title()} horizon ({rec['investor_type']}, ₹{rec['budget']:,} budget):")
            for s in rec["stocks"]:
                lines.append(f"    {s['name']} — {s.get('score_label') or s['score']} {s['verdict']} — {s['shares']}× ₹{s['price']:,.0f}")
            lines.append(f"    Total: ₹{rec['total_amount']:,.0f}")
        lines.append("")

    if others:
        lines.append("📋 Also noting:")
        for a in others:
            lines.append(f"  {a['headline']}")
        lines.append("")

    lines.append("Wealth is built in the waiting. See you next week.")
    lines.append("— Kordent")

    return "\n".join(lines)


def build_subject(name, summaries, alerts):
    """Subject line with emotional context — a mentor who feels what you feel."""
    if summaries:
        weighted_ret = 0
        total_weight = 0
        for s in summaries:
            if s.get("weekly_return_pct") is not None and s["value"] > 0:
                weighted_ret += s["weekly_return_pct"] * s["value"]
                total_weight += s["value"]

        if total_weight > 0:
            r = weighted_ret / total_weight
            pct = f"({r:+.1f}%)"

            if r >= 3:
                return f"🟢 {name}, weeks like this are why you stay patient {pct}"
            elif r >= 1:
                return f"🟢 {name}, steady progress — your portfolio grew {pct}"
            elif r >= 0:
                return f"📊 {name}, quiet week {pct} — that's how compounding feels"
            elif r >= -2:
                return f"📊 {name}, a small dip {pct} — nothing to worry about"
            elif r >= -5:
                return f"🟡 {name}, bumpy week {pct} — let's look at what matters"
            else:
                return f"🔴 {name}, tough week {pct} — here's the bigger picture"

    danger_count = sum(1 for a in alerts
                       if a.get("severity") == "danger"
                       and a.get("alert_type") not in ("opportunity", "new_entry"))
    opp_count = 0  # Opportunities now handled by weekly recommendations

    if danger_count and opp_count:
        return f"📊 {name}, a few things to look at this week"
    elif danger_count:
        return f"📊 {name}, something needs your attention"
    elif opp_count:
        return f"📊 {name}, spotted something interesting for you"

    return f"📊 {name}, your week in a minute"


def send_email(recipient, subject, body, smtp_user, smtp_pass):
    """Send a single plain-text email."""
    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = recipient
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        print(f"  ✓ Sent to {recipient}: {subject}")
        return True
    except Exception as e:
        print(f"  ✗ Failed for {recipient}: {e}")
        return False


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_weekly_mentor():
    print(f"Kordent Weekly Mentor — {date.today().strftime('%A, %B %d, %Y')}")
    print("=" * 50)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    smtp_user = os.environ.get("ALERT_EMAIL")
    smtp_pass = os.environ.get("ALERT_EMAIL_PASSWORD")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")
    if not smtp_user or not smtp_pass:
        raise ValueError("Missing ALERT_EMAIL or ALERT_EMAIL_PASSWORD")

    supabase: Client = create_client(url, key)

    # ── Load universe for validation ──
    universe_df = None
    if os.path.exists("universe_scored.csv"):
        universe_df = pd.read_csv("universe_scored.csv")
        print(f"Universe loaded: {len(universe_df)} stocks")
    else:
        print("Warning: universe_scored.csv not found. Skipping validation.")

    # ── Nifty weekly return ──
    nifty_weekly = fetch_nifty_weekly()
    if nifty_weekly is not None:
        print(f"Nifty 50 weekly: {nifty_weekly:+.2f}%")

    # ── Fetch all users ──
    profiles = supabase.table("profiles").select("id, email, full_name").execute().data or []
    if not profiles:
        print("No users. Exiting.")
        return

    user_map = {p["id"]: p for p in profiles}

    # ── Fetch all portfolios ──
    all_portfolios = supabase.table("portfolios").select("*").execute().data or []
    user_portfolios = defaultdict(list)
    for p in all_portfolios:
        user_portfolios[p["user_id"]].append(p)

    # ── Fetch alerts from past 7 days ──
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    all_alerts = supabase.table("portfolio_alerts").select("*").gte(
        "alert_date", cutoff
    ).execute().data or []

    # Group alerts by user (broadcast alerts go to everyone)
    user_alerts = defaultdict(list)
    for a in all_alerts:
        uid = a.get("user_id")
        if uid:
            user_alerts[uid].append(a)
        else:
            for p in profiles:
                user_alerts[p["id"]].append(a)

    print(f"\nProcessing {len(profiles)} users, {len(all_alerts)} raw alerts...\n")

    sent = 0
    skipped = 0

    for uid, profile in user_map.items():
        email = profile.get("email")
        if not email:
            skipped += 1
            continue

        name = profile.get("full_name") or email.split("@")[0].title()
        ports = user_portfolios.get(uid, [])
        raw = user_alerts.get(uid, [])

        # Skip users with no portfolios AND no alerts
        if not ports and not raw:
            skipped += 1
            continue

        print(f"[{name}] {len(ports)} portfolios, {len(raw)} raw alerts")

        # ── Pipeline: validate → deduplicate ──
        validated = validate_alerts(raw, universe_df)
        deduped = deduplicate_alerts(validated)

        # Opportunities now handled by weekly recommendations — exclude from email alerts
        final_alerts = [a for a in deduped if a.get("alert_type") not in ("opportunity", "new_entry")]

        # Build weekly recommendations (writes to weekly_recommendations table)
        user_recs = build_weekly_recommendations(uid, ports, universe_df, supabase)

        print(f"  After pipeline: {len(final_alerts)} alerts, {len(user_recs)} recommendation sets")

        # ── Portfolio summaries ──
        summaries = get_portfolio_summaries(ports, supabase, nifty_weekly)

        # ── Generate email ──
        subject = build_subject(name, summaries, final_alerts)

        body = None
        if gemini_key:
            # Sprint 11: Macro overlay for portfolio management (Ch 9)
            macro_env = ""
            try:
                macro_env = get_macro_context(gemini_key)
                if macro_env:
                    print(f"  Macro context fetched ({len(macro_env)} chars)")
            except Exception as e:
                print(f"  Macro fetch failed (non-blocking): {e}")

            prompt = build_gemini_prompt(name, summaries, final_alerts, user_recs, macro_context=macro_env)
            body = call_gemini(prompt, gemini_key)
            if body:
                print(f"  LLM narrative generated ({len(body)} chars)")
            else:
                print(f"  LLM failed — using fallback")

        if not body:
            body = build_plain_fallback(name, summaries, final_alerts, user_recs)

        # ── Send ──
        if send_email(email, subject, body, smtp_user, smtp_pass):
            sent += 1
 
        # ── Telegram mirror ──
        _tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if _tg_token:
            _tg_chat = profile.get("telegram_chat_id")
            if _tg_chat:
                _tg_body = f"<b>{_html_esc(subject)}</b>\n\n{_html_esc(body)}\n\n<a href='{APP_URL}'>Open Kordent</a>"
                # Telegram max is 4096 chars
                if len(_tg_body) > 4096:
                    _tg_body = _tg_body[:4090] + "..."
                send_telegram(_tg_chat, _tg_body, _tg_token)
                print(f"  Telegram sent to {name}")

    print(f"\n{'=' * 50}")
    print(f"Done. Sent: {sent} | Skipped: {skipped}")


if __name__ == "__main__":
    run_weekly_mentor()
