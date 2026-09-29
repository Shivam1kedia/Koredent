"""
costs.py — single source of truth for transaction costs. (Sprint 16)

PURE. No I/O, no network, no pandas, no imports at all. Callers pass a rupee
value in and get rupees back, exactly as economics.py takes ledger rows rather
than fetching them. That purity is what lets backtest_runner, economics and the
UI all price the same trade the same way.

THE MODEL
Zerodha equity DELIVERY, resident retail. Every rate below carries its source
and the date it was verified against https://zerodha.com/charges/.

    buy   = STT + transaction + SEBI + stamp duty + GST(txn + SEBI)
    sell  = STT + transaction + SEBI +              GST(txn + SEBI) + DP charge

Brokerage is zero on delivery, so the GST base is the exchange and regulator
fees only. Stamp duty is buy-side only. The DP charge is sell-side only and is
FLAT — Rs 15.34 per scrip regardless of quantity, GST already inside it.

WHY THE FLAT CHARGE IS THE WHOLE STORY
Proportional costs are ~0.22% round trip at every size. The flat Rs 15.34 is
what makes a small position expensive:

    position     round trip     of which DP
      Rs   333       4.83%          95%
      Rs   500       3.29%          93%
      Rs 1,000       1.76%          87%
      Rs 5,000       0.53%          58%
      Rs 50,000      0.25%          12%

Break-even sizes: Rs 1,973 for a 1.0% round trip, Rs 2,908 for 0.75%,
Rs 5,528 for 0.50%. A Rs 5,000 SIP across the 10-name ruin floor buys Rs 500
positions that cost 3.29% to round-trip, which is the number every exit rule in
this product has to be designed against.

Selling costs roughly forty times what buying costs at Rs 333 (Rs 15.69 vs
Rs 0.40). Any design that reduces SELLING is worth far more than any design
that optimises buying.

WHAT THIS MODEL DOES NOT INCLUDE, AND WHY
  - Slippage and market impact. Not observable from anything Kordent stores,
    and a made-up number is worse than an acknowledged omission. Excluded from
    v1 deliberately; revisit only if attribution (Sprint 17) turns up an
    unexplained negative alpha of the right magnitude.
  - Any broker other than Zerodha. Kordent never sees a contract note — it
    routes a basket to Kite Publisher and the user places the order. Every
    figure here is MODELLED, never observed. A user on a percentage-brokerage
    broker pays more than this says.
  - Same-day aggregation of the DP charge. The DP charge is levied per scrip
    PER DAY, not per sell order, so two sells of the same scrip on one day
    incur it once. This module prices one transaction in isolation and will
    therefore overstate that case. Selling one scrip once is the ordinary case.
  - Securities lending, auction penalties, call-and-trade, physical settlement.

ROUNDING. Components are carried at full precision and only the returned total
is rounded to paise, which is how the broker's own worked examples come out.
"""

# ── Rates ────────────────────────────────────────────────────────────────
# ONE dict. Every rate in one place so a change is one edit, and every entry
# names where it came from. Verified against https://zerodha.com/charges/ on
# the date below; these DO change (STT last moved in a Union Budget), so the
# date is part of the data, not a comment.
RATES_VERIFIED = "2026-09-18"

RATES = {
    # Zero on equity delivery. Kept explicit because it is the GST base.
    "brokerage_pct": 0.0,                                  # zerodha.com/charges
    # Securities Transaction Tax, both sides.
    "stt_pct": 0.001,                                      # 0.1% buy & sell
    # Exchange transaction charge. BSE is 22% dearer than NSE, which is why
    # this module reads the ticker suffix instead of assuming NSE.
    "txn_pct": {"NSE": 0.0000307, "BSE": 0.0000375},       # 0.00307% / 0.00375%
    # SEBI turnover fee, Rs 10 per crore = 1e-6 of traded value.
    "sebi_pct": 0.000001,                                  # Rs 10/crore
    # Stamp duty, buy side only.
    "stamp_pct_buy": 0.00015,                              # 0.015% or Rs 1500/cr
    # GST on (brokerage + transaction + SEBI). NOT on STT or stamp duty.
    "gst_pct": 0.18,
    # Depository charge, sell side only, FLAT per scrip.
    # Rs 3.5 CDSL + Rs 9.5 Zerodha + Rs 2.34 GST. GST is already inside it,
    # so it must never be fed through the GST line above.
    "dp_per_scrip_sell": 15.34,
}

DEFAULT_EXCHANGE = "NSE"


def exchange_for(ticker):
    """NSE or BSE from a yfinance ticker suffix. Unknown suffix -> NSE.

    The universe carries both .NS and .BO names and the transaction charge
    differs, so the exchange is derived from the ticker rather than assumed.
    """
    t = str(ticker or "").strip().upper()
    if t.endswith(".BO"):
        return "BSE"
    return DEFAULT_EXCHANGE


def _rates_for(exchange):
    ex = str(exchange or DEFAULT_EXCHANGE).upper()
    if ex not in RATES["txn_pct"]:
        ex = DEFAULT_EXCHANGE
    return ex


def buy_rate(exchange=DEFAULT_EXCHANGE):
    """Proportional buy cost as a fraction of traded value. No flat component."""
    ex = _rates_for(exchange)
    txn = RATES["txn_pct"][ex]
    gst_base = RATES["brokerage_pct"] + txn + RATES["sebi_pct"]
    return (RATES["stt_pct"] + txn + RATES["sebi_pct"]
            + RATES["stamp_pct_buy"] + RATES["gst_pct"] * gst_base)


def sell_rate(exchange=DEFAULT_EXCHANGE):
    """Proportional sell cost as a fraction of traded value. Excludes the flat
    DP charge, which is added by sell_cost and is the dominant term at retail
    sizes — do not use this as 'the sell cost'."""
    ex = _rates_for(exchange)
    txn = RATES["txn_pct"][ex]
    gst_base = RATES["brokerage_pct"] + txn + RATES["sebi_pct"]
    return (RATES["stt_pct"] + txn + RATES["sebi_pct"]
            + RATES["gst_pct"] * gst_base)


def buy_cost(value, exchange=DEFAULT_EXCHANGE):
    """Rupees paid on top of a buy of `value` rupees."""
    v = _positive(value)
    if v is None:
        return 0.0
    return round(v * buy_rate(exchange), 2)


def sell_cost(value, exchange=DEFAULT_EXCHANGE):
    """Rupees deducted from the proceeds of a sale of `value` rupees,
    INCLUDING the flat per-scrip DP charge."""
    v = _positive(value)
    if v is None:
        return 0.0
    return round(v * sell_rate(exchange) + RATES["dp_per_scrip_sell"], 2)


def round_trip_cost(value, exchange=DEFAULT_EXCHANGE):
    """Total rupees to buy and later sell a position of `value` rupees.

    Priced at a CONSTANT value on both legs. A position that doubles pays more
    on the way out; net_return() prices that case properly and is what the
    backtest uses. This function answers 'what does a round trip cost at this
    size', which is the question the exit rules ask.
    """
    v = _positive(value)
    if v is None:
        return 0.0
    # Summed from the RATES, not from two already-rounded returns. Rounding the
    # legs first and adding them is off by up to a paisa, which is nothing here
    # but is exactly the habit that puts a rupee into a portfolio total.
    return round(v * (buy_rate(exchange) + sell_rate(exchange))
                 + RATES["dp_per_scrip_sell"], 2)


def round_trip_pct(value, exchange=DEFAULT_EXCHANGE):
    """Round-trip cost as a FRACTION of position value (0.0329 == 3.29%)."""
    v = _positive(value)
    if v is None:
        return None
    return round_trip_cost(v, exchange) / v


def min_position_for_cost_pct(target_pct, exchange=DEFAULT_EXCHANGE):
    """Smallest position whose round trip costs no more than `target_pct`.

    Round trip is  V*k + DP  where k is the summed proportional rate, so
    (V*k + DP)/V <= target  solves to  V >= DP / (target - k).

    Returns None when target_pct is at or below k: no position size is small
    enough to make the proportional floor go away, and returning a huge number
    there would read as an answer rather than as 'impossible'.
    """
    try:
        target = float(target_pct)
    except (TypeError, ValueError):
        return None
    k = buy_rate(exchange) + sell_rate(exchange)
    if target <= k:
        return None
    return round(RATES["dp_per_scrip_sell"] / (target - k), 2)


def net_return(entry_value, gross_return, exchange=DEFAULT_EXCHANGE):
    """Return after costs on a position entered at `entry_value` that moved
    `gross_return` (0.10 == +10%) before being sold.

    Prices the exit on the EXIT value, not the entry value, because STT and the
    exchange fees are charged on what the sale is actually worth. The DP charge
    is flat and does not care.

        net = (exit - sell_cost(exit) - entry - buy_cost(entry)) / entry

    Returns None if entry_value is not a positive number, so a missing price
    cannot silently become a zero-cost trade.
    """
    v = _positive(entry_value)
    if v is None:
        return None
    try:
        g = float(gross_return)
    except (TypeError, ValueError):
        return None
    exit_value = v * (1.0 + g)
    if exit_value <= 0:
        # Total loss. Nothing to sell, so no sell-side cost is incurred; the
        # buy cost is still sunk.
        return (0.0 - v - buy_cost(v, exchange)) / v
    net_proceeds = exit_value - sell_cost(exit_value, exchange)
    total_outlay = v + buy_cost(v, exchange)
    return (net_proceeds - total_outlay) / v


def _positive(value):
    """float(value) if it is a positive number, else None. One place, so a bad
    input cannot become 0.0 in one function and raise in another."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def cost_table(sizes=(333, 500, 1000, 2000, 5000, 25000, 50000),
               exchange=DEFAULT_EXCHANGE):
    """The table in the docstring, computed. For preflight and for the UI —
    nothing in this product should ever hardcode these numbers again."""
    return [{
        "position": s,
        "buy_cost": buy_cost(s, exchange),
        "sell_cost": sell_cost(s, exchange),
        "round_trip": round_trip_cost(s, exchange),
        "round_trip_pct": round(round_trip_pct(s, exchange) * 100, 2),
    } for s in sizes]


if __name__ == "__main__":
    print(f"Zerodha equity delivery, rates verified {RATES_VERIFIED}\n")
    print(f"{'position':>10} {'buy':>8} {'sell':>8} {'round trip':>11} {'%':>7}")
    for r in cost_table():
        print(f"{r['position']:>10,} {r['buy_cost']:>8.2f} {r['sell_cost']:>8.2f} "
              f"{r['round_trip']:>11.2f} {r['round_trip_pct']:>6.2f}%")
    print()
    for t in (0.010, 0.0075, 0.005):
        print(f"  break-even for a {t*100:.2f}% round trip: "
              f"Rs {min_position_for_cost_pct(t):,.0f}")
