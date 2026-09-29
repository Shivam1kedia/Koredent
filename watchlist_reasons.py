"""
watchlist_reasons.py — universe-relative "reasons to buy" for the watchlist email.

For a given stock, ranks its OWN facts by how much of a positive outlier it is
across the whole scored universe, and returns the strongest not-yet-shown ones.
A fact only qualifies if the stock sits in the favourable top 25% of the market
on it (or clears a book threshold) — so we surface genuine distinctions, never
"above average" filler. Rotates: each email-week shows the next strongest facts
the user hasn't seen yet; when a stock runs out of top-quartile facts, it rests.

No new data: every fact is a column already in universe_scored.csv.
"""
import pandas as pd

# column -> (direction, kind, template)
#   direction: 'high' favourable = bigger better; 'low' = smaller better
#   kind: 'pct' value is already a percent; 'frac' multiply by 100; 'raw' as-is
# Curated to real, self-justifying investment facts across the five lenses.
REASON_MAP = {
    # ── Valuation / cheapness ──
    "pe":                          ("low",  "raw",  "a P/E of just {v:.1f}"),
    "pb":                          ("low",  "raw",  "a price-to-book of {v:.2f}"),
    "graham_margin_of_safety_pct": ("high", "pct",  "trading ~{v:.0f}% below Graham fair value"),
    "greenblatt_earnings_yield":   ("high", "pct",  "an earnings yield of {v:.1f}%"),
    "lynch_peg":                   ("low",  "raw",  "a PEG ratio of {v:.2f}"),
    "graham_ncav_ratio":           ("high", "raw",  "net current assets {v:.1f}x its market value"),
    # REMOVED 2026-07-28 — buffett_margin_of_safety_pct derives from
    # deep_metrics.py:496-503, a Gordon growth model oe_ps*(1+g)/(r-g) with
    # r = INDIA_10Y_BOND_RATE (7%) and g = min(ni_cagr, 15%). Two failure modes,
    # and REASON_MAP's percentile ranking turns the second into a win:
    #   g >= r  -> intrinsic value None -> excluded. Most healthy growers.
    #   g -> r- -> denominator -> 0 -> MoS asymptotes to ~100% -> guaranteed
    #              top quartile -> surfaced as a leading reason to buy,
    #              DISPLACING genuine facts.
    # The metric therefore favours companies growing just BELOW the government
    # bond rate and hides everything growing faster — the inverse of what an
    # owner-earnings margin of safety should select for. The column is still
    # computed and stored; only the user-facing surface is withdrawn, pending a
    # replacement intrinsic-value model. Do not re-add without that model.
    # ── Quality / profitability ──
    "roe_pct":                     ("high", "pct",  "a return on equity of {v:.0f}%"),
    "profit_margin":               ("high", "frac", "a net profit margin of {v:.0f}%"),
    "schilit_cfo_ni_ratio":        ("high", "raw",  "operating cash flow {v:.1f}x its reported profit"),
    "dorsey_cash_return":          ("high", "pct",  "a cash return on capital of {v:.1f}%"),
    "mulford_cash_margin":         ("high", "pct",  "a cash margin of {v:.0f}%"),
    # ── Growth ──
    "revenue_cagr_3y":             ("high", "pct",  "revenue compounding at {v:.0f}% a year"),
    "ni_cagr_3y":                  ("high", "pct",  "earnings compounding at {v:.0f}% a year"),
    "graham_eps_growth_pct_4y":    ("high", "pct",  "EPS up {v:.0f}% over four years"),
    # ── Balance-sheet strength ──
    "de":                          ("low",  "raw",  "debt-to-equity of only {v:.1f}"),
    "current_ratio":               ("high", "raw",  "a current ratio of {v:.1f}"),
    "dorsey_interest_coverage":    ("high", "raw",  "interest covered {v:.0f}x by earnings"),
    "dorsey_quick_ratio":          ("high", "raw",  "a quick ratio of {v:.1f}"),
    # ── Income ──
    "dividend_yield_pct":          ("high", "pct",  "a {v:.1f}% dividend yield"),
}

TOP_QUARTILE = 25  # a fact qualifies only in the favourable top 25% of the universe


def _val(v, kind):
    if v is None or pd.isna(v):
        return None
    return v * 100 if kind == "frac" else v


def rank_reasons(ticker, universe_df):
    """All top-quartile facts for `ticker`, strongest first.
    Returns list of dicts: {column, text, top_pct, strength}."""
    row = universe_df[universe_df["ticker"] == ticker]
    if row.empty:
        return []
    row = row.iloc[0]
    out = []
    for col, (direction, kind, tmpl) in REASON_MAP.items():
        if col not in universe_df.columns:
            continue
        raw = row.get(col)
        if raw is None or pd.isna(raw):
            continue
        series = universe_df[col].dropna()
        if len(series) < 50:                       # too thin to rank meaningfully
            continue
        pctile = float((series < raw).mean() * 100)   # % of universe below this value
        if direction == "high":
            if pctile < 100 - TOP_QUARTILE:            # not in top 25%
                continue
            strength, top_pct = pctile, round(100 - pctile)
        else:                                          # low is better
            if pctile > TOP_QUARTILE:
                continue
            strength, top_pct = 100 - pctile, round(pctile)
        top_pct = max(1, top_pct)
        disp = _val(raw, kind)
        text = tmpl.format(v=disp) + f" — top {top_pct}% of the market"
        out.append({"column": col, "text": text,
                    "top_pct": top_pct, "strength": strength})
    out.sort(key=lambda r: r["strength"], reverse=True)
    return out


def pick_reasons(ticker, universe_df, shown_columns=None, n=2):
    """Next `n` strongest facts not already shown. Empty => the stock has run
    out of genuine top-quartile facts and should rest this cycle."""
    shown = set(shown_columns or [])
    return [r for r in rank_reasons(ticker, universe_df) if r["column"] not in shown][:n]


if __name__ == "__main__":
    df = pd.read_csv("/mnt/user-data/uploads/universe_scored.csv")
    # test on a few high-score stocks
    hi = df[df["score"] >= 4]["ticker"].head(4).tolist()
    for tk in hi:
        print(f"\n=== {tk} (score {df[df.ticker==tk].iloc[0]['score']}) ===")
        reasons = rank_reasons(tk, df)
        for r in reasons[:8]:
            print(f"   [{r['strength']:.0f}] {r['text']}")
        print(f"   -- week1 pick:", [x["column"] for x in pick_reasons(tk, df, [], 2)])
        wk1 = [x["column"] for x in pick_reasons(tk, df, [], 2)]
        print(f"   -- week2 pick:", [x["column"] for x in pick_reasons(tk, df, wk1, 2)])
