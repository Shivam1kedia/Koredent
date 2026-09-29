"""
economics.py — single source of truth for portfolio-level money.

MODEL (a), decided Sprint 14, extended with withdrawals in Sprint 15.
Sale proceeds stay INSIDE the portfolio as cash. "Invested" means external
capital the user actually paid in from outside; a buy funded by earlier sale
proceeds contributes nothing to it. Return is measured on external capital,
never on gross turnover, and never on the surviving cost basis.

    buy_outflow      = buy_amount + cost      (cost per row, stored, from costs.py)
    sell_inflow      = sell_amount - cost
    external_capital = sum over buys of max(0, buy_outflow - cash_before)
    withdrawn        = sum of withdrawals (money that left for the user's bank)
    total_costs      = sum of cost over every buy and sell
    cash             = external_capital + sells - buys - withdrawn - total_costs
    market_value     = sum(shares * live_price)          [caller supplies]
    total_assets     = market_value + cash
    total_pnl        = total_assets + withdrawn - external_capital
    return_pct       = total_pnl / external_capital * 100

WHY A WITHDRAWAL DOES NOT REDUCE THE DENOMINATOR. Put in 10,000, it doubles,
withdraw 10,000, 10,000 of stock left. Netting withdrawals off external capital
gives a denominator of zero and a return of infinity on the most ordinary case
there is. A withdrawal is value RETURNED, so it belongs in the numerator; the
capital you committed is a historical fact that does not un-happen.

WHY NOT "invested = surviving cost basis": reducing the denominator on sale
deletes the cost basis and the outcome of the sold position together, so
realising a loss RAISES reported return. A return series that improves when you
lose money is not a return series.

realized_pnl uses weighted-average cost per ticker, which is exactly what
holdings.price_at_entry already represents (every top-up site re-averages it).

COSTS (Sprint 16). A transaction cost is money that actually left, so it
reduces cash and — when a buy cannot be funded from cash alone — it raises
external capital. It is NOT a display-time adjustment. cost_inr is a separate
stored column per row; amount_inr keeps meaning gross traded value, so the
existing archive keeps its meaning and total costs paid becomes a reportable
number rather than something reconstructed later from rates that have moved.

Every cost figure in this system is MODELLED, never observed: Kordent routes a
basket to Kite Publisher and never sees a contract note. costs.py states the
model and its assumptions. A row with cost_inr NULL is counted in
cost_rows_missing and treated as zero — the flag is what keeps "no cost was
charged" and "this row predates cost tracking" distinguishable, which is the
same reason score_history.applicable exists.

IDENTITY, WITH COSTS. total_pnl reduces algebraically to
market_value + sells - buys - total_costs, so the decomposition is THREE terms:

    total_pnl == realized_pnl + unrealized_pnl - total_costs_paid

with realized_pnl and unrealized_pnl both GROSS of costs. Proof, writing
cb_sold and cb_surv for the cost basis of sold and surviving shares:

    realized   = sells - cb_sold                    [by definition, gross]
    unrealized = market_value - cb_surv             [by definition, gross]
    buys       = cb_sold + cb_surv                  [lots are gross]
    total_pnl  = market_value + sells - buys - total_costs
               = market_value + (realized + cb_sold) - (cb_sold + cb_surv) - costs
               = (market_value - cb_surv) + realized - costs
               = unrealized + realized - total_costs_paid          QED

WHY THREE TERMS AND NOT TWO. Keeping the old two-term identity would force
costs into whichever term absorbs the remainder — unrealized_pnl — and
unrealized_pnl would stop meaning "market value minus surviving cost basis".
That breaks its correspondence to holdings.price_at_entry, which is the thing
that makes it checkable against the holdings table at all. Capitalising buy
costs into the lot basis instead would break the same correspondence from the
other side. A cost is its own kind of money leaving and gets its own line.

FEE ENTRY, decided Sprint 16, no code yet. If Kordent is ever monetised, a fee
is a THIRD transaction type ("fee"), not a cost and not a withdrawal:

  - Like a cost, it reduces cash and never reduces external_capital, so paying
    a fee lowers return rather than flattering it.
  - Unlike a cost, it is not attached to a trade, so it carries no ticker and
    accumulates into total_fees_paid — its own line, for the same reason costs
    got one. total_pnl becomes realized + unrealized - costs - fees.
  - A performance fee needs a high-water mark, which is STORED STATE on the
    portfolio row (hwm_value, hwm_date), never a formula recomputed from
    history. Recomputed, it silently resets whenever history is trimmed or a
    ledger row is corrected, and it resets in the direction that charges the
    user twice for the same gain. The fee job reads the watermark, charges
    rate * max(0, total_assets - hwm), writes the fee row, then sets
    hwm := total_assets in the same transaction.
  - Fees are charged on total_assets, not on market_value: uninvested cash
    inside the portfolio is money under management.

This is written down now because the Sprint 15 accounting correction happened
precisely because the money model was not thought through before it had users.

BENCHMARK SHADOW. shadow_units tracks what EXTERNAL flows would have bought in
the benchmark ETF, priced at each row's own nifty_price. A sell adds nothing
(no outside money moved); only the externally-funded portion of a buy adds
units, and a withdrawal removes them. Summing the stored nifty_units column
instead — +amt for buys, -amt for everything else — makes a sale look like a
withdrawal, and double-counts once a real withdrawal follows it.

The shadow is GROSS of the benchmark's own transaction costs. Buying NIFTYBEES
is an equity delivery trade and carries the same STT, stamp duty and DP charge
as anything else, so the shadow flatters the benchmark by roughly its buy-side
cost plus one DP charge on exit. Left uncorrected on purpose: the error runs
AGAINST the portfolio, and an unflattering assumption needs no defending. It is
also small at the shadow's scale, where the flat DP charge is spread over the
whole contribution rather than over one small position.

This module does NO I/O. Callers pass transaction rows in. Rows are dicts with
keys: transaction_date, created_at, id, ticker, shares, price, amount_inr,
transaction_type, nifty_price.
"""

import datetime

BUY = "buy"
WITHDRAWAL = "withdrawal"


def _sort_key(t):
    """Chronological replay order. transaction_date is the economic date;
    created_at breaks same-day ties by real insertion order, which is what
    makes 'sell then rebuy on the same day' fund itself from cash rather than
    drawing fresh external capital. id is the final deterministic tiebreak."""
    d = t.get("transaction_date") or "0001-01-01"
    c = t.get("created_at") or ""
    return (str(d)[:10], str(c), str(t.get("id") or ""))


def replay_ledger(txns):
    """Replay a portfolio's sip_transactions in chronological order.

    Returns external_capital, withdrawn, cash, realized_pnl, total_buys,
    total_sells, total_costs, external_flows [(date, signed_amount)] for XIRR,
    shadow_units, and three integrity flags: shadow_incomplete (a contribution
    had no benchmark price, so the shadow understates), unreconciled_withdrawal
    (a withdrawal exceeded known cash, so a sale or contribution is missing
    upstream), and cost_rows_missing (rows with no cost_inr, so total_costs
    understates — a row that predates cost tracking, not a free trade).
    """
    cash = 0.0
    external = 0.0
    withdrawn = 0.0
    realized = 0.0
    total_buys = 0.0
    total_sells = 0.0
    total_costs = 0.0
    buy_costs = 0.0
    sell_costs = 0.0
    cost_rows_missing = 0
    cost_debt = 0.0   # costs incurred but not yet paid for out of external draw
    shadow_units = 0.0
    shadow_incomplete = False
    unreconciled_withdrawal = 0.0
    external_flows = []
    lots = {}  # ticker -> [shares, avg_cost]

    for t in sorted(txns or [], key=_sort_key):
        ttype = str(t.get("transaction_type") or BUY).lower()
        try:
            amt = float(t.get("amount_inr") or 0.0)
            sh = float(t.get("shares") or 0.0)
            px = float(t.get("price") or 0.0)
            bpx = float(t.get("nifty_price") or 0.0)
        except (TypeError, ValueError):
            continue
        if amt <= 0:
            continue
        tk = t.get("ticker") or ""
        d = str(t.get("transaction_date") or "")[:10]

        # NULL is not zero. A row written before cost tracking existed has an
        # UNKNOWN cost; treating that as "no cost was charged" is the same
        # False-carries-two-meanings bug that score_history.applicable exists
        # to prevent. Replay with zero so the arithmetic still works, and count
        # the row so the caller can say total_costs_paid understates.
        raw_cost = t.get("cost_inr")
        if raw_cost is None:
            cost = 0.0
            if ttype != WITHDRAWAL:
                cost_rows_missing += 1
        else:
            try:
                cost = max(0.0, float(raw_cost))
            except (TypeError, ValueError):
                cost = 0.0
                cost_rows_missing += 1

        if ttype == BUY:
            total_buys += amt
            total_costs += cost
            buy_costs += cost
            cost_debt += cost
            # The cost is part of the outflow, so a buy that cash cannot cover
            # draws external capital for the cost too. It genuinely did.
            outflow = amt + cost
            shortfall = outflow - cash
            if shortfall > 0:
                external += shortfall
                cash += shortfall
                external_flows.append((d, round(shortfall, 2)))
                # THE SHADOW BUYS SECURITIES, NOT FRICTION. External money that
                # went to STT, stamp duty and the DP charge bought nothing, here
                # or in the ETF, so it must not become benchmark units. Without
                # this, a pure rotation — which draws a little external capital
                # purely to cover its own costs — would ADD shadow units, and
                # the benchmark would grow in proportion to how much the user
                # churned. The portfolio already pays for the churn in cash;
                # paying for it a second time by handing the benchmark free
                # units is the same double-count the nifty_units column had.
                paid = min(shortfall, cost_debt)
                cost_debt -= paid
                investable = shortfall - paid
                if investable > 0:
                    if bpx > 0:
                        shadow_units += investable / bpx
                    else:
                        shadow_incomplete = True
            cash -= outflow
            lot = lots.setdefault(tk, [0.0, 0.0])
            new_sh = lot[0] + sh
            if new_sh > 0:
                lot[1] = ((lot[0] * lot[1]) + amt) / new_sh
            lot[0] = new_sh

        elif ttype == WITHDRAWAL:
            take = min(amt, cash)
            if amt - take > 0.005:
                # Money left that the ledger cannot account for. Clamping keeps
                # cash >= 0; the flag makes preflight fail rather than hiding it.
                unreconciled_withdrawal += amt - take
            if take > 0:
                cash -= take
                withdrawn += take
                external_flows.append((d, -round(take, 2)))
                if bpx > 0:
                    shadow_units -= take / bpx
                else:
                    shadow_incomplete = True

        else:  # sell
            total_sells += amt
            total_costs += cost
            sell_costs += cost
            cost_debt += cost
            # Proceeds arrive NET. At retail position sizes this is the whole
            # story: the flat DP charge is ~95% of the cost of exiting a Rs 333
            # position. realized_pnl below stays GROSS on purpose — costs are
            # their own line in the decomposition, see the module docstring.
            cash += (amt - cost)
            # No shadow change: no outside money moved.
            lot = lots.get(tk)
            if lot and lot[0] > 0:
                sold = min(sh, lot[0])
                realized += sold * (px - lot[1])
                lot[0] -= sold
            # A sell with no recorded lot (holding predates the ledger)
            # contributes cash but no realized P&L. The cash is real either way.

    return {
        "external_capital": external,
        "withdrawn": withdrawn,
        "cash": max(0.0, cash),
        "realized_pnl": realized,
        "total_buys": total_buys,
        "total_sells": total_sells,
        "total_costs": total_costs,
        # Split because the two behave completely differently. Buying is ~0.12%
        # of value at any size; selling carries the flat DP charge and is the
        # side worth designing against. A single blended total hides that.
        "buy_costs": buy_costs,
        "sell_costs": sell_costs,
        "cost_rows_missing": cost_rows_missing,
        "external_flows": external_flows,
        "shadow_units": max(0.0, shadow_units),
        "shadow_incomplete": shadow_incomplete,
        "unreconciled_withdrawal": unreconciled_withdrawal,
    }


def portfolio_economics(txns, market_value, benchmark_price=None):
    """The one function every display site calls. market_value is the live
    market value of surviving holdings (sum of shares * current price).
    benchmark_price is the benchmark ETF's current close, if known.

    return_pct is None — not 0.0 — when there is no external capital to
    measure against. A wrong number is worse than no number.

    Every figure returned is NET of the transaction costs stored on the ledger
    rows. total_costs_paid is the total of those costs and cost_rows_missing is
    how many rows carried no cost figure — if it is non-zero, total_costs_paid
    UNDERSTATES and any display of it must say so.
    """
    led = replay_ledger(txns)
    ext = led["external_capital"]
    cash = led["cash"]
    wd = led["withdrawn"]
    costs = led["total_costs"]
    mv = float(market_value or 0.0)
    total_assets = mv + cash
    total_pnl = total_assets + wd - ext
    realized = led["realized_pnl"]
    # unrealized is DERIVED so the three-way identity holds by construction:
    #   total_pnl == realized + unrealized - costs
    # and it still equals market_value - cost_basis_of_surviving_shares, which
    # is what makes it checkable against holdings.price_at_entry. See the proof
    # in the module docstring.
    unrealized = total_pnl - realized + costs

    shadow_value = None
    try:
        bp = float(benchmark_price or 0.0)
        if bp > 0 and led["shadow_units"] > 0:
            shadow_value = round(led["shadow_units"] * bp, 2)
    except (TypeError, ValueError):
        shadow_value = None

    return {
        "external_capital": round(ext, 2),
        "withdrawn": round(wd, 2),
        "cash_balance": round(cash, 2),
        "market_value": round(mv, 2),
        "total_assets": round(total_assets, 2),
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_costs_paid": round(costs, 2),
        "buy_costs_paid": round(led["buy_costs"], 2),
        "sell_costs_paid": round(led["sell_costs"], 2),
        "cost_rows_missing": led["cost_rows_missing"],
        # P&L BEFORE charges. Exact, not a counterfactual: it is
        # market_value + sells - buys, the same number the system reported
        # before costs existed. gross_pnl - total_costs_paid == total_pnl.
        "gross_pnl": round(total_pnl + costs, 2),
        # Return before charges, on the SAME denominator as return_pct — the
        # capital actually paid in. It answers "of the money I put in, how much
        # did the market hand me before charges took their cut", which is a real
        # question with a real denominator. It is NOT "what I would have earned
        # in a world with no charges": in that world less capital would have
        # been needed, so the denominator would differ too. Reporting the
        # honest question rather than the flattering counterfactual.
        "gross_return_pct": (round((total_pnl + costs) / ext * 100, 2)
                             if ext > 0 else None),
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(total_pnl / ext * 100, 2) if ext > 0 else None,
        "external_flows": led["external_flows"],
        "shadow_units": led["shadow_units"],
        "shadow_value": shadow_value,
        "shadow_incomplete": led["shadow_incomplete"],
        "unreconciled_withdrawal": round(led["unreconciled_withdrawal"], 2),
        "has_ledger": bool(led["total_buys"] > 0),
    }


def xirr_flows(econ, as_of=None):
    """Convert portfolio_economics output into (dates, amounts) for pyxirr.

    Under model (a) the only true external flows are contributions (money in,
    negative to the investor) and withdrawals (money out, positive), with
    total_assets as the terminal value. Buys and sells are internal transfers
    between cash and securities and must NOT appear, or a same-day
    sell-and-rebuy shows up as a spurious round trip.
    """
    dates, amounts = [], []
    for d, amt in econ.get("external_flows", []):
        try:
            dates.append(datetime.date.fromisoformat(d))
        except (ValueError, TypeError):
            continue
        amounts.append(-float(amt))
    if not dates:
        return None, None
    dates.append(as_of or datetime.date.today())
    amounts.append(float(econ.get("total_assets") or 0.0))
    return dates, amounts
