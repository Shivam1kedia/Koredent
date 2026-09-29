"""
Kordent Telegram Bot — Interactive Commands
=============================================
Handles /start (linking), /portfolio, /score, /watchlist, /help.
Runs as a scheduled job (GitHub Actions cron every 5 min).
Uses raw Telegram Bot API via requests — no python-telegram-bot dependency.

Environment variables:
    TELEGRAM_BOT_TOKEN
    SUPABASE_URL
    SUPABASE_KEY
"""

import os
import sys
import requests
import pandas as pd
from supabase import create_client

# score_label + PASS_FLAG: the denominator and the framework->column mapping live
# in ONE place. selector imports only math/numpy/pandas — no Streamlit — so it is
# safe in a headless Actions job. Degrades to a bare number if unavailable rather
# than taking the bot down.
try:
    import selector
except Exception:
    selector = None

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
APP_URL = "https://kordent.streamlit.app"


def _html_esc(text):
    """Escape HTML special chars for Telegram."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_message(chat_id, text):
    """Send a Telegram message (HTML parse mode). Max 4096 chars."""
    if len(text) > 4096:
        text = text[:4090] + "..."
    try:
        requests.post(f"{BASE_URL}/sendMessage", json={
            "chat_id": chat_id, "text": text, "parse_mode": "HTML"
        }, timeout=10)
    except Exception as e:
        print(f"  Send failed to {chat_id}: {e}")


def get_updates(offset=None):
    """Fetch pending updates from Telegram."""
    params = {"timeout": 0, "allowed_updates": ["message"]}
    if offset:
        params["offset"] = offset
    try:
        resp = requests.get(f"{BASE_URL}/getUpdates", params=params, timeout=15)
        return resp.json().get("result", [])
    except Exception as e:
        print(f"getUpdates failed: {e}")
        return []


def get_user_by_chat_id(supabase, chat_id):
    """Look up user_id from telegram_chat_id. Returns profile dict or None."""
    try:
        resp = supabase.table("profiles").select("id, full_name").eq(
            "telegram_chat_id", str(chat_id)).limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def handle_start(supabase, chat_id, text):
    """Handle /start and /start CODE for account linking."""
    parts = text.strip().split()
    if len(parts) >= 2:
        code = parts[1].strip()
        try:
            resp = supabase.table("profiles").select("id").eq(
                "telegram_link_code", code).limit(1).execute()
            if resp.data:
                user_id = resp.data[0]["id"]
                supabase.table("profiles").update({
                    "telegram_chat_id": str(chat_id),
                    "telegram_link_code": None,
                }).eq("id", user_id).execute()
                send_message(chat_id,
                    "✅ <b>Connected!</b>\n\n"
                    "You'll now receive:\n"
                    "• Daily portfolio updates\n"
                    "• Danger alerts (instant)\n"
                    "• SIP reminders on the 1st\n"
                    "• Weekly mentor emails\n\n"
                    f"<a href='{APP_URL}'>Open Kordent</a>")
            else:
                send_message(chat_id,
                    "❌ Invalid or expired code.\n\n"
                    f"Generate a new one from <a href='{APP_URL}'>Kordent</a> → Telegram Alerts.")
        except Exception as e:
            send_message(chat_id, f"Something went wrong. Try again.\n({_html_esc(str(e)[:100])})")
    else:
        send_message(chat_id,
            "👋 <b>Welcome to Kordent!</b>\n\n"
            "To link your account:\n"
            f"1. Open <a href='{APP_URL}'>Kordent</a>\n"
            "2. Click 📱 Telegram Alerts in the sidebar\n"
            "3. Click Connect Telegram\n"
            "4. Send the code here: /start CODE\n\n"
            "Already linked? Try /portfolio or /help\n\n"
            "<i>Commands are processed every hour. Alerts and daily updates are delivered instantly.</i>")


def handle_portfolio(supabase, chat_id):
    """Show user's portfolio summaries."""
    profile = get_user_by_chat_id(supabase, chat_id)
    if not profile:
        send_message(chat_id,
            "Account not linked. Use /start CODE to connect.\n"
            f"Get your code from <a href='{APP_URL}'>Kordent</a>.")
        return

    try:
        ports = supabase.table("portfolios").select(
            "name, current_value, current_return_pct, sip_amount, is_paper"
        ).eq("user_id", profile["id"]).execute().data or []
    except Exception as e:
        send_message(chat_id, f"Failed to fetch portfolios: {_html_esc(str(e)[:100])}")
        return

    real = [p for p in ports if not p.get("is_paper")]
    paper = [p for p in ports if p.get("is_paper")]

    if not real and not paper:
        send_message(chat_id, f"No portfolios yet. <a href='{APP_URL}'>Build one on Kordent</a>.")
        return

    lines = [f"<b>📊 Your Portfolios</b>\n"]

    for p in real:
        name = _html_esc(p.get("name", "Portfolio"))
        val = p.get("current_value", 0)
        ret = p.get("current_return_pct", 0)
        sip = p.get("sip_amount", 0)
        sign = "+" if ret >= 0 else ""
        lines.append(
            f"<b>{name}</b>\n"
            f"Rs. {val:,.0f} ({sign}{ret:.1f}%)\n"
            f"SIP: Rs. {sip:,.0f}/mo\n"
        )

    if paper:
        lines.append("<i>Paper Portfolios:</i>")
        for p in paper:
            name = _html_esc(p.get("name", "Paper"))
            val = p.get("current_value", 0)
            ret = p.get("current_return_pct", 0)
            sign = "+" if ret >= 0 else ""
            lines.append(f"  {name}: Rs. {val:,.0f} ({sign}{ret:.1f}%)")

    lines.append(f"\n<a href='{APP_URL}'>Open Kordent</a>")
    send_message(chat_id, "\n".join(lines))


def handle_score(supabase, chat_id, text, universe_df):
    """Look up a stock's score from universe CSV."""
    parts = text.strip().split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        send_message(chat_id,
            "Usage: /score TICKER\n\n"
            "Examples:\n"
            "/score HDFCBANK\n"
            "/score TCS.NS\n"
            "/score RELIANCE")
        return

    query = parts[1].strip().upper()
    # Normalize: strip .NS/.BO suffix for matching, then try both
    bare = query.replace(".NS", "").replace(".BO", "")

    if universe_df is None or universe_df.empty:
        send_message(chat_id, "Universe data not available. Try again later.")
        return

    # Try exact ticker match first, then partial name match
    row = universe_df[universe_df["ticker"].str.upper() == f"{bare}.NS"]
    if row.empty:
        row = universe_df[universe_df["ticker"].str.upper() == f"{bare}.BO"]
    if row.empty:
        row = universe_df[universe_df["ticker"].str.upper() == query]
    if row.empty and "name" in universe_df.columns:
        row = universe_df[universe_df["name"].str.upper().str.contains(bare, na=False)].head(1)

    if row.empty:
        send_message(chat_id, f"No match for <b>{_html_esc(query)}</b> in the universe.")
        return

    r = row.iloc[0]
    ticker = r.get("ticker", "?")
    name = r.get("name", ticker)
    score = int(r["score"]) if pd.notna(r.get("score")) else "?"
    sector = r.get("sector", "—")
    pe = f"{r['pe']:.1f}" if pd.notna(r.get("pe")) else "—"

    # LYNCH WAS MISSING from this list entirely — the bot has never shown the
    # fifth framework. An ABSTAINING framework renders as "—", not ❌: Greenblatt
    # does not apply to financials/utilities and Lynch does not apply to a
    # business his six categories cannot place. A ❌ says the stock was tested
    # and failed, which for those rows is false.
    _app = set(selector._applicable_frameworks(r)) if selector else None
    fw_lines = []
    for label, fw, key in [("Graham", "graham", "graham_pass"),
                           ("Greenblatt", "greenblatt", "greenblatt_pass"),
                           ("Dorsey", "dorsey_buffett", "dorsey_pass"),
                           ("Trajectory", "trajectory", "trajectory_pass"),
                           ("Lynch", "lynch", "lynch_pass")]:
        if _app is not None and fw not in _app:
            fw_lines.append(f"  {label}: — (does not apply)")
        elif key in r and pd.notna(r[key]):
            fw_lines.append(f"  {label}: {'✅' if r[key] else '❌'}")

    quality = ""
    if "quality_pass" in r.index and pd.notna(r.get("quality_pass")):
        quality = "✅ Quality pass" if r["quality_pass"] else "❌ Quality fail"

    msg = (
        f"<b>{_html_esc(name)}</b> ({_html_esc(ticker)})\n\n"
        f"Score: <b>{selector.score_label(r, score) if selector else score}</b>\n"
        f"Sector: {_html_esc(sector)}\n"
        f"PE: {_html_esc(pe)}\n\n"
        f"<b>Frameworks:</b>\n" + "\n".join(fw_lines)
    )
    if quality:
        msg += f"\n\n{quality}"

    msg += f"\n\n<a href='{APP_URL}'>Full analysis on Kordent</a>"
    send_message(chat_id, msg)


def handle_watchlist(supabase, chat_id, universe_df):
    """Show user's watchlist with current scores."""
    profile = get_user_by_chat_id(supabase, chat_id)
    if not profile:
        send_message(chat_id,
            "Account not linked. Use /start CODE to connect.")
        return

    try:
        wl = supabase.table("watchlist").select("ticker, note").eq(
            "user_id", profile["id"]).execute().data or []
    except Exception as e:
        send_message(chat_id, f"Failed to fetch watchlist: {_html_esc(str(e)[:100])}")
        return

    if not wl:
        send_message(chat_id, f"Watchlist is empty. <a href='{APP_URL}'>Add stocks on Kordent</a>.")
        return

    lines = [f"<b>👁 Your Watchlist</b> ({len(wl)} stocks)\n"]

    for w in wl:
        ticker = w.get("ticker", "?")
        bare = ticker.replace(".NS", "").replace(".BO", "")
        score = "?"
        if universe_df is not None and not universe_df.empty:
            row = universe_df[universe_df["ticker"] == ticker]
            if not row.empty:
                _sc = row["score"].iloc[0]
                if pd.notna(_sc):
                    score = (selector.score_label(row.iloc[0], int(_sc))
                             if selector else f"{int(_sc)}")
        note_text = f" — {_html_esc(w['note'][:40])}" if w.get("note") else ""
        lines.append(f"<b>{_html_esc(bare)}</b>  {score}{note_text}")

    lines.append(f"\n<a href='{APP_URL}'>Open Kordent</a>")
    send_message(chat_id, "\n".join(lines))


def handle_help(chat_id):
    """Show available commands."""
    send_message(chat_id,
        "<b>Kordent Bot Commands</b>\n\n"
        "/portfolio — Your portfolio values and returns\n"
        "/score TICKER — Score lookup (e.g. /score HDFCBANK)\n"
        "/watchlist — Your watched stocks with scores\n"
        "/help — This message\n\n"
        "<i>Commands are processed every hour. Daily updates, danger alerts, and SIP reminders are delivered instantly.</i>\n\n"
        f"<a href='{APP_URL}'>Open Kordent</a>")


def main():
    if not BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN not set. Exiting.")
        sys.exit(0)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("Missing SUPABASE_URL or SUPABASE_KEY. Exiting.")
        sys.exit(1)

    supabase = create_client(url, key)

    # Load universe CSV
    universe_df = None
    if os.path.exists("universe_scored.csv"):
        universe_df = pd.read_csv("universe_scored.csv")
        print(f"Loaded universe: {len(universe_df)} stocks")
    else:
        print("Warning: universe_scored.csv not found.")

    # Fetch and process updates
    updates = get_updates()
    if not updates:
        print("No pending updates.")
        return

    print(f"Processing {len(updates)} updates...")

    for update in updates:
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if not chat_id or not text:
            continue

        cmd = text.split()[0].lower()

        if cmd == "/start":
            handle_start(supabase, chat_id, text)
        elif cmd == "/portfolio":
            handle_portfolio(supabase, chat_id)
        elif cmd == "/score":
            handle_score(supabase, chat_id, text, universe_df)
        elif cmd == "/watchlist":
            handle_watchlist(supabase, chat_id, universe_df)
        elif cmd == "/help":
            handle_help(chat_id)
        else:
            send_message(chat_id,
                f"Unknown command. Try /help for available commands.")

    # Confirm all processed updates
    last_id = updates[-1]["update_id"]
    get_updates(offset=last_id + 1)
    print(f"Confirmed {len(updates)} updates (through ID {last_id}).")


if __name__ == "__main__":
    main()
