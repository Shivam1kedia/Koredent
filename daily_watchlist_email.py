"""
DAILY WATCHLIST EMAIL
=====================
Runs every weekday ~2 hours after the daily engine (13:30 UTC / 7:00 PM IST).
Sends a daily mentor-style email to each user who has stocks on their watchlist.
Covers ONLY individual watched stocks — NOT watchlist portfolios (those go
in the weekly mentor email).

Purpose: Remind users daily about their watched stocks, teach investing
principles, and nudge toward buying when conditions are right.

Zero alerts are generated here. This script only consumes alerts written
by portfolio_tracker.py and current data from universe_scored.csv.

Usage:  python daily_watchlist_email.py
Cron:   '0 12 * * 1-5'  (weekdays, 12:00 UTC = 5:30 PM IST)
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from google import genai
from supabase import create_client, Client
import watchlist_reasons  # weekly rotating "reasons to buy" engine

# score_label — the true denominator, from the one place that knows it.
# selector imports only math/numpy/pandas (no Streamlit), so it is safe in a
# headless Actions job. Degrades to a bare number rather than failing the send.
try:
    import selector
except Exception:
    selector = None


APP_URL = "https://shivam1kedia.streamlit.app"
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.1-pro-preview",
]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def enrich_watchlist(watchlist_items, universe_df, price_cache):
    """
    Cross-reference each watchlist item with universe_scored.csv
    to get current score, quality, PE, PB, sector, 52-week range.
    """
    enriched = []
    for wl in watchlist_items:
        ticker = wl["ticker"]
        name = wl.get("name") or ticker
        added_score = wl.get("score_when_added")
        added_quality = wl.get("quality_when_added")
        added_date = wl.get("added_date", "")

        days_watched = 0
        if added_date:
            try:
                days_watched = (date.today() - date.fromisoformat(str(added_date))).days
            except Exception:
                pass

        item = {
            "ticker": ticker,
            "name": name,
            "note": wl.get("note") or "",
            "days_watched": days_watched,
            "added_score": added_score,
            "added_quality": added_quality,
            "current_score": None,
            "score_label": "?",
            "current_quality": None,
            "sector": "Unknown",
            "pe": None,
            "pb": None,
            "current_price": None,
            "week52_low": None,
            "week52_high": None,
            "pct_from_low": None,
            "dividend_yield_pct": None,
            "roe_pct": None,
        }

        if universe_df is not None:
            row = universe_df[universe_df["ticker"] == ticker]
            if not row.empty:
                r = row.iloc[0]
                item["current_score"] = int(r["score"]) if pd.notna(r.get("score")) else None
                # "3 of 4", not "3/5". Computed HERE because this is the only
                # place the universe row is in scope; the three render sites
                # below just read it. A framework that abstains was never
                # applied, so counting it in the denominator states the stock
                # failed a test it never took.
                item["score_label"] = (
                    selector.score_label(r, item["current_score"])
                    if selector and item["current_score"] is not None
                    else (str(item["current_score"])
                          if item["current_score"] is not None else "?"))
                item["current_quality"] = bool(r["quality_pass"]) if pd.notna(r.get("quality_pass")) else None
                item["sector"] = str(r["sector"]) if pd.notna(r.get("sector")) else "Unknown"
                item["pe"] = round(float(r["pe"]), 1) if pd.notna(r.get("pe")) else None
                item["pb"] = round(float(r["pb"]), 2) if pd.notna(r.get("pb")) else None
                item["week52_low"] = round(float(r["week52_low"]), 2) if pd.notna(r.get("week52_low")) else None
                item["week52_high"] = round(float(r["week52_high"]), 2) if pd.notna(r.get("week52_high")) else None
                item["pct_from_low"] = round(float(r["pct_from_low"]), 1) if pd.notna(r.get("pct_from_low")) else None
                item["dividend_yield_pct"] = round(float(r["dividend_yield_pct"]), 2) if pd.notna(r.get("dividend_yield_pct")) else None
                item["roe_pct"] = round(float(r["roe_pct"]), 1) if pd.notna(r.get("roe_pct")) else None

        # Current price from cache (populated by portfolio_tracker earlier)
        if ticker in price_cache:
            item["current_price"] = round(price_cache[ticker], 2)
        else:
            try:
                p = yf.Ticker(ticker).fast_info.last_price
                if p:
                    item["current_price"] = round(p, 2)
                    price_cache[ticker] = p
            except Exception:
                pass

        # ── Weekly rotating "reasons to buy" (watchlist_reasons engine) ──
        # Anchor rotation to when the score improved (or when added). Week 0 leads
        # with the improvement; each later week surfaces the next 2 strongest
        # top-quartile facts. When a stock runs out of genuine facts, it rests.
        item["_reasons"] = []
        item["_show_improvement"] = False
        item["_improved_days"] = None
        try:
            _improved_on = wl.get("score_improved_on")
            _anchor = _improved_on or added_date
            _wk = 0
            if _anchor:
                try:
                    _wk = max(0, (date.today() - date.fromisoformat(str(_anchor))).days // 7)
                except Exception:
                    _wk = 0
            _R = watchlist_reasons.rank_reasons(ticker, universe_df) if universe_df is not None else []
            item["_reasons"] = [r["text"] for r in _R[_wk * 2:_wk * 2 + 2]]
            if _improved_on and _wk == 0:
                try:
                    item["_improved_days"] = (date.today() - date.fromisoformat(str(_improved_on))).days
                    item["_show_improvement"] = True
                except Exception:
                    pass
        except Exception as _re:
            print(f"  reasons skipped for {ticker}: {_re}")

        enriched.append(item)

    return enriched


def build_mentor_prompt(name, stocks, alerts_for_user):
    """
    Build the Gemini prompt for a daily watchlist email.
    Tone: honest mentor teaching a layman about their specific stocks.
    """
    today_str = date.today().strftime("%A, %B %d, %Y")

    # Stock summary block
    stock_block = ""
    for s in stocks:
        score_line = s.get("score_label") or "?"
        score_delta = ""
        if s['added_score'] is not None and s['current_score'] is not None:
            diff = s['current_score'] - s['added_score']
            if diff > 0:
                score_delta = f" (↑{diff} since you started watching)"
            elif diff < 0:
                score_delta = f" (↓{abs(diff)} since you started watching)"

        quality_line = ""
        if s['current_quality'] is not None:
            quality_line = f"Quality: {'PASS' if s['current_quality'] else 'FAIL'}"
            if s['added_quality'] is not None and s['added_quality'] != s['current_quality']:
                quality_line += " ⚠ CHANGED since added"

        price_line = f"Price: ₹{s['current_price']:,.2f}" if s['current_price'] else "Price: unavailable"
        low_line = ""
        if s['pct_from_low'] is not None:
            low_line = f" | {s['pct_from_low']:.1f}% above 52-week low (₹{s['week52_low']})"

        pe_line = f"PE: {s['pe']}" if s['pe'] else "PE: N/A"
        roe_line = f"ROE: {s['roe_pct']}%" if s['roe_pct'] else ""
        note_line = f"Your note: \"{s['note']}\"" if s['note'] else ""

        # This week's rotating angle: improvement headline + top-quartile facts
        reasons_line = ""
        if s.get("_show_improvement") and s.get("current_score") is not None:
            _dd = s.get("_improved_days")
            _when = "today" if _dd == 0 else (f"{_dd} days ago" if _dd else "recently")
            reasons_line = f"THIS WEEK'S ANGLE — score improved to {s.get('score_label') or s['current_score']} {_when}."
        if s.get("_reasons"):
            _facts = "; ".join(s["_reasons"])
            reasons_line += (f" Also notable: {_facts}." if reasons_line
                             else f"THIS WEEK'S ANGLE — {_facts}.")

        stock_block += f"""
{s['name']} ({s['ticker'].replace('.NS','').replace('.BO','')})
  Sector: {s['sector']} | Watching for {s['days_watched']} days
  Score: {score_line}{score_delta} | {quality_line}
  {price_line}{low_line}
  {pe_line} | {roe_line}
  {note_line}
  {reasons_line}
"""

    # Today's alerts for this user's watchlist
    alerts_block = ""
    if alerts_for_user:
        for a in alerts_for_user:
            detail = a.get("detail") or {}
            if isinstance(detail, str):
                try:
                    detail = json.loads(detail)
                except Exception:
                    detail = {}

            passages = detail.get("book_passages", [])
            passage_text = ""
            if passages:
                passage_text = " [Wisdom: " + "; ".join(
                    f"{p['author']} — {p['text'][:150]}" for p in passages[:1]
                ) + "]"

            # W1 label as a CONSTRAINT, not decoration. portfolio_tracker already
            # persists drift_reason via classify_score_change; it was simply never
            # read here, so rule 4 asked the model to judge "red flag or noise"
            # with nothing deterministic to judge from — and it inferred a cause.
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

            alerts_block += f"- {a['headline']}{_cause}{passage_text}\n"
    else:
        alerts_block = "No significant changes today."

    prompt = f"""You are a wise investing mentor writing a brief daily email to {name} about the stocks they're watching.

TODAY: {today_str}

THEIR WATCHLIST:
{stock_block}

TODAY'S ALERTS:
{alerts_block}

Write a warm, honest daily email. Rules:

1. Address {name} by name. Open with one sentence about today's picture — not a greeting.
2. Go through each stock briefly. For stocks with alerts today, explain what happened and what it means in simple terms. For stocks with no alerts, a one-liner ("still steady", "no change — patience pays") is enough.
3. If a stock's score improved or is near its 52-week low, gently remind them this could be a buying opportunity — but don't push. Say something like "this is the kind of setup Graham looked for" or "Dorsey would call this buying quality at a discount."
3a. If a stock has a "THIS WEEK'S ANGLE" line, LEAD that stock's mention with it — it is a real, verified fact about where this stock ranks in the whole market (e.g. top 10% on earnings yield). State it plainly and explain the term simply. These angles ROTATE weekly so the stock stays fresh; do NOT invent facts beyond the angle given, and do NOT hype — if the angle reads implausibly strong (e.g. a >80% discount), soften to "unusually cheap" rather than quoting the raw number. If a stock has no angle line, keep it to a brief one-liner; it is resting this cycle.
4. If a stock's quality flipped to FAIL or score dropped, be honest. Explain what it means and whether it's a red flag or just noise. A mentor doesn't sugarcoat.
4a. Where an alert carries a [CAUSE: ...] tag, that cause is DETERMINISTIC — computed by comparing the stock's frameworks against the record from when it was added. You MUST use it and MUST NOT contradict or replace it with a cause of your own. Translate it into plain English:
   - fundamental — the business itself weakened. This is the serious one.
   - valuation — the PRICE moved; the business did not change. Often means it got more expensive, or cheaper.
   - relative_rank — the business held, but peers improved past it.
   - mixed — both the business and the price moved. Say so; do not pick one.
   - unclear / unknown_inputs — we cannot attribute the change. SAY THAT PLAINLY ("the score moved but we can't say why yet"). Do not guess a reason to fill the gap.
   If an alert has NO [CAUSE:] tag, do not speculate about why the score moved — describe what changed and stop.
4b. Scores read "N of M", where M is how many frameworks APPLY to that business — not always 5. Greenblatt does not apply to banks or utilities (his own instruction), and Lynch does not apply to a business his six categories cannot place. A framework that abstains was NEVER TESTED. Never describe it as failed, missed, or a weakness, and never say a stock "only got 3 out of 5" when it scored 3 of 4.
5. If the user left a personal note on a stock, reference it naturally. E.g., "You said you're waiting for PE below 12 — it's at 14 today, getting closer."
6. End with ONE short educational nugget — a concept from Graham, Greenblatt, or Dorsey explained simply (e.g., "Here's what 'margin of safety' actually means in practice..."). Rotate topics daily. Don't repeat what you've said before.
7. Sign off with a brief patience reminder. Vary it daily.
8. Sign off as "Kordent"

TONE: You're the knowledgeable older friend they trust with money questions. You explain things simply without being condescending. You're honest about bad news. You NEVER use phrases like "Dear", "I hope this email finds you", "Best regards", or corporate speak.
LENGTH: Under 300 words. Brevity is respect for their time.
FORMAT: Plain text only. No markdown, no bullet points, no headers. 1-2 emojis max for warmth, not decoration.
IMPORTANT: Explain financial terms in parentheses the first time — assume they've never taken a finance class."""

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


def build_plain_fallback(name, stocks, alerts):
    """Clean data-only email when Gemini fails."""
    lines = [f"{name}, here's your daily watchlist update.\n"]

    for s in stocks:
        score_str = s.get("score_label") or "?"
        price_str = f"₹{s['current_price']:,.2f}" if s['current_price'] else "price unavailable"

        delta = ""
        if s['added_score'] is not None and s['current_score'] is not None:
            diff = s['current_score'] - s['added_score']
            if diff > 0:
                delta = f" (↑{diff})"
            elif diff < 0:
                delta = f" (↓{abs(diff)})"

        lines.append(f"📊 {s['name']}: {price_str} | Score: {score_str}{delta}")

        if s['pct_from_low'] is not None and s['pct_from_low'] <= 10:
            lines.append(f"   Near 52-week low ({s['pct_from_low']:.1f}% above)")

        if s.get('note'):
            lines.append(f"   Your note: {s['note']}")

        lines.append("")

    # Alerts
    wl_alerts = [a for a in alerts if a.get("alert_type", "").startswith("watchlist_")]
    if wl_alerts:
        lines.append("Today's changes:")
        for a in wl_alerts:
            lines.append(f"  {a['headline']}")
        lines.append("")

    lines.append(f"Review your watchlist: {APP_URL}")
    lines.append("\nWealth is built in the waiting. — Kordent")

    return "\n".join(lines)


def build_subject(name, stocks, alerts):
    """Short, informative subject line."""
    wl_alerts = [a for a in alerts if a.get("alert_type", "").startswith("watchlist_")]

    score_ups = [a for a in wl_alerts if a["alert_type"] == "watchlist_score_up"]
    near_lows = [a for a in wl_alerts if a["alert_type"] == "watchlist_near_low"]
    score_downs = [a for a in wl_alerts if a["alert_type"] == "watchlist_score_down"]
    quality_flips = [a for a in wl_alerts if a["alert_type"] == "watchlist_quality_flip"]

    if score_ups:
        stock_name = score_ups[0].get("ticker", "").replace(".NS", "").replace(".BO", "")
        return f"📈 {name}, {stock_name} just improved"

    if near_lows:
        stock_name = near_lows[0].get("ticker", "").replace(".NS", "").replace(".BO", "")
        return f"👁 {name}, {stock_name} near its 52-week low"

    if score_downs or quality_flips:
        return f"📊 {name}, heads up on your watchlist"

    return f"👁 {name}, your daily watchlist check-in"


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

def run_daily_watchlist_email():
    print(f"Kordent Daily Watchlist Email — {date.today().strftime('%A, %B %d, %Y')}")
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

    # ── Load universe for enrichment ──
    universe_df = None
    if os.path.exists("universe_scored.csv"):
        universe_df = pd.read_csv("universe_scored.csv")
        print(f"Universe loaded: {len(universe_df)} stocks")
    else:
        print("Warning: universe_scored.csv not found.")

    # ── Fetch all watchlist items ──
    all_watchlist = supabase.table("watchlist").select("*").execute().data or []
    if not all_watchlist:
        print("No watchlist items across any user. Exiting.")
        return

    # Group by user
    from collections import defaultdict
    user_watchlist = defaultdict(list)
    for wl in all_watchlist:
        user_watchlist[wl["user_id"]].append(wl)

    print(f"Found {len(all_watchlist)} watchlist items across {len(user_watchlist)} users.")

    # ── Fetch profiles for names/emails ──
    user_ids = list(user_watchlist.keys())
    profiles = supabase.table("profiles").select("id, email, full_name").in_(
        "id", user_ids
    ).execute().data or []
    profile_map = {p["id"]: p for p in profiles}

    # ── Fetch today's watchlist alerts ──
    today_str = date.today().isoformat()
    today_alerts = supabase.table("portfolio_alerts").select("*").eq(
        "alert_date", today_str
    ).in_(
        "alert_type", ["watchlist_score_up", "watchlist_score_down",
                        "watchlist_quality_flip", "watchlist_near_low"]
    ).execute().data or []

    user_alerts = defaultdict(list)
    for a in today_alerts:
        uid = a.get("user_id")
        if uid:
            user_alerts[uid].append(a)

    print(f"Today's watchlist alerts: {len(today_alerts)}")

    # ── Price cache (shared across users to avoid duplicate yf calls) ──
    price_cache = {}

    # ── Process each user ──
    sent = 0
    skipped = 0

    for uid, wl_items in user_watchlist.items():
        profile = profile_map.get(uid)
        if not profile or not profile.get("email"):
            skipped += 1
            continue

        email = profile["email"]
        name = profile.get("full_name") or email.split("@")[0].title()

        print(f"\n[{name}] {len(wl_items)} watched stocks")

        # Enrich watchlist with current data
        enriched = enrich_watchlist(wl_items, universe_df, price_cache)
        alerts = user_alerts.get(uid, [])

        # Build subject
        subject = build_subject(name, enriched, alerts)

        # Generate email body
        body = None
        if gemini_key:
            prompt = build_mentor_prompt(name, enriched, alerts)
            body = call_gemini(prompt, gemini_key)
            if body:
                print(f"  LLM narrative generated ({len(body)} chars)")
            else:
                print(f"  LLM failed — using fallback")

        if not body:
            body = build_plain_fallback(name, enriched, alerts)

        # Send
        if send_email(email, subject, body, smtp_user, smtp_pass):
            sent += 1

    print(f"\n{'=' * 50}")
    print(f"Done. Sent: {sent} | Skipped: {skipped}")


if __name__ == "__main__":
    run_daily_watchlist_email()
