"""
ARCHETYPE ENGINE (W2)
=====================
Assigns Lynch's six archetypes DETERMINISTICALLY from stored columns. Replaces
the priority-ordered if/elif ladder in deep_metrics.compute_classification,
which was empirically shown to be noise (see below).

Two consumers, one classifier:
    assign_archetype()
        ├─> lynch_category            (= archetype_primary, 1:1 for the six)
        └─> W2 sign-flip conditioning (drift-label interpretation)

Building a second classifier alongside the broken one would let them drift.
lynch_score's branching is UNCHANGED — it simply stops receiving a coin flip.

WHY THE OLD LADDER DIED (diagnostics, 2026-07, 4,477 fresh rows)
----------------------------------------------------------------
  graham_eps_cv > 0.5 split at the MEDIAN of the distribution it was splitting.
  pct(eps_cv > 0.5) by sector ran 0.321-0.558 — a coin flip in every sector.
  The ordering was INVERTED: Technology 0.481 > Basic Materials 0.369.
  Cyclical share by sector spanned only 24.1%-34.9% across the whole economy;
  Consumer Defensive (32.1%) outranked Basic Materials (26.9%).
  Mechanism: abs(std/mean) over 4 EPS points measures earnings NOISE, which is
  largest where the denominator is smallest. 115 rows had eps_cv > 10, which is
  arithmetically only possible for a near-breakeven firm.
  Turnaround held top precedence and fired on any rounding-error loss -> 660 hits.

  Revenue amplitude max(rev)/min(rev) over y0..y3 was tested as a replacement and
  FAILED THE SAME WAY: Solar 4.02 (rank 1), Steel 1.38 (46), Aluminum 1.27 (60),
  Oil & Gas Refining 1.26 (63). Over 4 points in a monotone series, peak/trough
  IS y0/y3 — i.e. the growth rate. It measures growth, not cyclicality.

THE ESTIMABILITY PRINCIPLE (why cyclical is declared, the rest measured)
-----------------------------------------------------------------------
  Four annual points measure LEVEL and TREND. They cannot measure CYCLES.

  fast_grower / stalwart / slow_grower / turnaround are TREND archetypes —
  4 points estimate those adequately, so they stay evidence-based.
  cyclical is a CYCLE archetype — 4 points cannot contain one peak and one
  trough, so it comes from business STRUCTURE, not measurement.

  This is not hand-labelling stocks. "Steel is a cyclical industry" is a taxonomy
  statement about a business model, sourced from Lynch, applied deterministically
  to whatever tickers carry that industry string. Structurally identical to
  universe_updater._UNEVALUABLE_INDUSTRIES, and auditable in one place.

Sources: Lynch (One Up on Wall Street) for the six categories, the growth
thresholds, and the cyclical business types. Dorsey's price-taker test concurs
on the commodity half of the list.
"""

# ─── Helpers (duplicated from deep_metrics to avoid a circular import) ───
# deep_metrics imports THIS module, so this module must not import it back.
# Semantics kept identical, including fail-closed None -> 0.0.

import math


def _sf(val, default=None):
    """Safely convert to float. Returns default for NaN/None/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _ramp(x, lo, hi):
    """Linear ramp, higher-is-better: 0 at/below lo, 1 at/above hi. None -> 0.0."""
    x = _sf(x)
    if x is None:
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _ramp_down(x, ideal, limit):
    """Linear ramp, lower-is-better: 1 at/below ideal, 0 at/above limit. None -> 0.0."""
    x = _sf(x)
    if x is None:
        return 0.0
    if limit == ideal:
        return 1.0 if x <= ideal else 0.0
    return max(0.0, min(1.0, (limit - x) / (limit - ideal)))


# ═══════════════════════════════════════════════════════════════════════
# DECLARED CYCLICAL INDUSTRIES
# ═══════════════════════════════════════════════════════════════════════
# Rule: cyclical iff the business is a PRICE-TAKER ON COMMODITY OUTPUT, or its
# demand is CAPEX / BIG-TICKET DISCRETIONARY. Both are Lynch's structural
# markers. Borderline tiebreaker: include iff the industry has had an observable
# multi-year revenue/earnings DOWNTURN in Indian markets driven by a cycle rather
# than a one-off policy or pandemic shock. A one-off shock is not a cycle — it
# does not recur, and normalizing earnings over it does nothing useful.
#
# Strings verified 2026-07 against the yfinance industry vocabulary actually
# present in universe_scored.csv (125 industries over 2,300 classified rows).
# An industry NOT on this list is NOT cyclical — fail-safe direction, we never
# assume cyclical without a declared reason. Blank industry -> not cyclical.
#
# ── KNOWN WEAKEST BOUNDARY ────────────────────────────────────────────
# "Specialty Chemicals" (118 rows) is OUT; "Chemicals" (41) is IN. This single
# split moves more names than any other decision here, and it rests entirely on
# yfinance's taxonomy being meaningful — which is UNVERIFIED. Indian "Specialty
# Chemicals" genuinely contains sticky formulation businesses, but almost
# certainly also contains commodity intermediates that are pure price-takers.
# The cost is asymmetric in BOTH directions:
#   false negative (commodity chemical -> not cyclical): judged on trailing PE at
#     peak margins -> value trap. The exact failure W2 exists to prevent.
#   false positive (specialty compounder -> cyclical): judged on 3-4y average
#     earnings -> systematically understated -> never bought. A real cost, not
#     free conservatism.
# No clean resolution with 4 years of data. A firm-level margin-stability
# override inside chemicals was considered and rejected as scope creep.
# Revisit as a one-line edit here if evidence appears.
#
# Two smaller heterogeneity flags, same character, lower stakes:
#   "Textile Manufacturing" (105) mixes cyclical spinning mills with steadier
#     garment exporters.
#   "Auto Parts" (94) increasingly contains export/EV-content names on secular
#     trends rather than the domestic auto cycle.
#
# ── EMPIRICAL STATUS ──────────────────────────────────────────────────
# This list stands on the BOOKS ALONE. Revenue-amplitude corroboration was
# attempted and failed (see module docstring). Coverage is 888/2,300 = 38.6% of
# classified rows, against a 42.8% Basic Materials + Industrials + Energy +
# Real Estate base — coherent, but a coin flip and a real signal can share a
# base rate. Treat as a sourced assertion, not a finding.
# ═══════════════════════════════════════════════════════════════════════

CYCLICAL_INDUSTRIES = frozenset({
    # ── Metals & mining: price-takers on globally quoted output ──
    "Steel",
    "Aluminum",
    "Copper",
    "Other Industrial Metals & Mining",
    "Other Precious Metals & Mining",
    "Thermal Coal",
    "Coking Coal",

    # ── Energy: price-takers on crude/gas, plus their capex chain ──
    "Oil & Gas Refining & Marketing",
    "Oil & Gas Equipment & Services",
    "Oil & Gas E&P",
    "Oil & Gas Drilling",
    "Oil & Gas Integrated",

    # ── Bulk process industries: commodity output, cost-curve competition ──
    "Chemicals",                      # NB: "Specialty Chemicals" deliberately OUT
    "Paper & Paper Products",
    "Lumber & Wood Production",
    "Building Materials",             # cement — classic Indian capex cycle

    # ── Autos: big-ticket discretionary, and their supply chain ──
    "Auto Manufacturers",
    "Auto Parts",
    "Auto & Truck Dealerships",

    # ── Capital goods & construction: demand IS the capex cycle ──
    "Engineering & Construction",
    "Farm & Heavy Construction Machinery",
    "Specialty Industrial Machinery",
    "Metal Fabrication",

    # ── Transport: capacity cycles, freight/fare price-taking ──
    "Marine Shipping",
    "Airlines",                       # Lynch's own canonical cyclical

    # ── Semis: Lynch's other canonical cyclical (memory-chip cycle) ──
    "Semiconductors",
    "Semiconductor Equipment & Materials",

    # ── Real estate development: capex-financed, big-ticket demand ──
    "Real Estate - Development",
    "Real Estate — Development",      # em-dash variant, per _UNEVALUABLE precedent

    # ── Textiles: cotton/yarn price-taking, spinning-capacity cycles ──
    "Textile Manufacturing",

    # ── Borderline, RULED IN (multi-year cyclical downturn observed) ──
    "Lodging",                        # Indian hotel ARR/occupancy: 2008-13 supply glut
    "Resorts & Casinos",              # same demand driver as Lodging
    "Agricultural Inputs",            # global agrochem destocking 2023-24
    "Building Products & Equipment",  # housing-capex derived; tracked 2013-19
    "Real Estate - Diversified",      # same 2013-19 downturn; rental mix dampens
    "Real Estate — Diversified",      # em-dash variant
})

# Borderline RULED OUT, recorded so the reasoning is not re-litigated:
#   Aerospace & Defense  — Lynch's framing is US BUDGET cycles; Indian defence is
#                          a secular indigenisation order-book. Policy-reversal
#                          risk is real but is not a cycle.
#   Luxury Goods         — Indian jewellery volumes are wedding-driven and stable;
#                          gold is a pass-through. The 2016-17 dip was
#                          demonetisation + import curbs, i.e. policy shocks.
#   Electronic Components— Indian names are contract assembly on order books;
#                          EMS is in secular expansion.
#   Railroads            — listed names are wagon/component makers on government
#                          order books.
#   Trucking             — listed names are asset-light logistics, not fleet owners.
#   Consumer Electronics — secular penetration growth dominates any cycle.
#   Airports & Air Services — concession-based regulated tariff: annuity, not cycle.
# Explicitly out and not borderline: all "Utilities - *" (regulated), all
# financials (mostly is_unevaluable already), Packaging & Containers,
# Conglomerates, Infrastructure Operations, Specialty Chemicals.


# ─── Confidence gate (SPECIFICATION, not a fitted parameter) ───
# Deliberately NOT tuned against outcomes — that is the falsification-vs-tuning
# line the project holds.
#
# ONE gate, not two. An earlier design also rejected rows whose top two scores
# were within a margin. That was wrong: weak evidence and a close call are
# DIFFERENT situations. "No archetype fits" -> abstain. "Two archetypes fit
# equally well" -> that is not ignorance, it is a DUAL LABEL, which is precisely
# what the multi-archetype path exists for. Collapsing both into unclassified
# put a ~2-point band of the growth measure (g ~ 13.5-15.5, where the
# fast_grower and stalwart curves cross) into permanent abstention for no reason.
MIN_EVIDENCE = 0.50   # the winner must have real evidence, not merely be least-weak
SECONDARY_AT = 0.70   # runner-up is emitted as a dual label at >= this x top

ARCHETYPES = ("cyclical", "fast_grower", "stalwart",
              "slow_grower", "asset_play", "turnaround")

# archetype_primary -> lynch_category. Identity for the six; unclassified maps to
# "unknown", which (pending the abstention wiring) makes Lynch ABSTAIN rather
# than score 0. Scoring an unclassifiable stock 0 records it as having FAILED
# Lynch. It has not failed Lynch; Lynch cannot evaluate it.
LYNCH_CATEGORY_MAP = {a: a for a in ARCHETYPES}
LYNCH_CATEGORY_MAP["unclassified"] = "unknown"


# ═══════════════════════════════════════════
# EVIDENCE FUNCTIONS  ->  float in [0, 1]
# ═══════════════════════════════════════════

def _ni_series(data):
    """Net income y0..y3, most-recent-first. May contain None."""
    return [_sf(data.get(f"net_income_y{i}")) for i in range(4)]


def _direction_consistent(vals):
    """>= 2 of 3 consecutive steps rising (vals are most-recent-first, so a
    rising business has vals[i] > vals[i+1]). Fail-closed on missing data:
    fewer than 4 usable points -> False."""
    if any(v is None for v in vals):
        return False
    return sum(1 for i in range(3) if vals[i] > vals[i + 1]) >= 2


def _ev_cyclical(data):
    """DECLARED, not measured. See the estimability principle above.
    Binary and TERMINAL — no evidence-based label may override it."""
    ind = (data.get("industry") or "")
    return 1.0 if str(ind).strip() in CYCLICAL_INDUSTRIES else 0.0


def _ev_fast_grower(data):
    """Lynch: 20-25% is the sweet spot, 12-20% acceptable, >50% unsustainable.
    Thresholds are Lynch's own, already encoded at deep_metrics.py:483-493.

    REQUIRES BOTH revenue AND net-income growth. The old ladder used ni_cagr_3y
    alone; NI growth without revenue growth is margin expansion or one-offs, not
    a growth story. That is the substantive fix in this function."""
    rev = _sf(data.get("revenue_cagr_3y"))
    ni = _sf(data.get("ni_cagr_3y"))
    if rev is None or ni is None or rev < 12 or ni < 12:
        return 0.0
    # 0.5 at Lynch's "acceptable" floor, 1.0 at his "ideal" band.
    base = 0.5 + 0.5 * _ramp(min(rev, ni), 12, 20)
    if ni > 50:
        base *= 0.5                                   # Lynch: dangerous/unsustainable
    if not _direction_consistent(_ni_series(data)):
        base *= 0.7
    return round(base, 4)


def _ev_stalwart(data):
    """Lynch: large, established, ~10-12% growth, rides out recessions.
    BAND, not a ramp — a 19% grower is not more stalwart than a 12% grower;
    above ~20% it is a fast grower and the evidence should decay to zero.

    GROWTH MEASURE is min(rev_cagr, ni_cagr), not ni_cagr alone. NI growth above
    revenue growth is margin expansion — finite, cannot compound indefinitely —
    so the slower of the two is what the BUSINESS is doing. Gating on ni alone
    put mature compounders in a dead zone (ABBOTINDIA: ni 17.8 / rev 9.0 scored
    0.275 and fell to unclassified) — too fast for the stalwart band, not
    qualifying as fast_grower because revenue lags. Same measure fast_grower
    already ramps on."""
    rev = _sf(data.get("revenue_cagr_3y"))
    ni = _sf(data.get("ni_cagr_3y"))
    if rev is None or ni is None:
        return 0.0
    g = min(rev, ni)
    if g < 5 or g >= 20:
        return 0.0
    base = 0.5 + 0.5 * _ramp(g, 5, 10)                # rises into Lynch's band
    if g > 12:
        base *= _ramp_down(g, 12, 20)                 # decays out the top
    if not data.get("graham_adequate_size"):
        base *= 0.6                                   # Lynch's stalwarts are large
    if not data.get("graham_earnings_stable_4y"):
        base *= 0.7
    return round(base, 4)


def _ev_slow_grower(data):
    """Lynch: large, old, sluggish, generous dividend. The DIVIDEND is the
    thesis — a no-dividend sluggish company is not a slow grower in Lynch's
    sense, it is something he would not own. Halving without it is deliberate:
    such a stock should fall to unclassified and let Lynch abstain.

    Growth measure is min(rev_cagr, ni_cagr) — see _ev_stalwart."""
    rev = _sf(data.get("revenue_cagr_3y"))
    ni = _sf(data.get("ni_cagr_3y"))
    if rev is None or ni is None:
        return 0.0
    g = min(rev, ni)
    if g >= 5:
        return 0.0
    # Declining is not slow-growing: decays to 0 at -5% CAGR.
    base = _ramp(g, -5, 5)
    consec = _sf(data.get("dividend_consecutive_years"), 0) or 0
    if consec < 5:
        base *= 0.5
    return round(base, 4)


def _ev_asset_play(data):
    """UNCHANGED from the old ladder. Narrow (25 rows universe-wide) but honest:
    a hard book-value floor with genuine net cash behind it."""
    pb = _sf(data.get("pb"))
    net_cash = _sf(data.get("graham_net_cash"))
    return 1.0 if (pb is not None and pb <= 0.67
                   and net_cash is not None and net_cash > 0) else 0.0


def _ev_turnaround(data):
    """MATERIAL loss followed by a HOLDING recovery.

    The old rule fired on any loss of any magnitude in y2/y3 with NI_y0 > 0, and
    held top precedence in the ladder — hence 660 hits (14.7% of the universe),
    which is not a description of the Indian market. It was capturing noisy
    microcaps that happened to lose money three years ago, and stealing them
    from every other bucket.

    The 5%-of-revenue materiality floor and the NI_y0 >= NI_y1 sustained-recovery
    test are SPECIFICATIONS OF THIS PROJECT, not Lynch's numbers. Flagged as such
    rather than dressed up as book-sourced."""
    ni = _ni_series(data)
    rev = [_sf(data.get(f"revenue_y{i}")) for i in range(4)]

    material_loss = False
    for i in (2, 3):
        if ni[i] is not None and ni[i] < 0 and rev[i] is not None and rev[i] > 0:
            if abs(ni[i]) >= 0.05 * rev[i]:
                material_loss = True
                break
    if not material_loss:
        return 0.0

    # Recovery must be HOLDING, not a one-year blip.
    if ni[0] is None or ni[0] <= 0:
        return 0.0
    if ni[1] is not None and ni[0] < ni[1]:
        return 0.0
    return 1.0


_EVIDENCE = {
    "cyclical":    _ev_cyclical,
    "fast_grower": _ev_fast_grower,
    "stalwart":    _ev_stalwart,
    "slow_grower": _ev_slow_grower,
    "asset_play":  _ev_asset_play,
    "turnaround":  _ev_turnaround,
}


# ═══════════════════════════════════════════
# ASSIGNMENT
# ═══════════════════════════════════════════

def assign_archetype(data):
    """Pure function: dict of stored metrics -> archetype dict. No network, no
    Streamlit, no global state. Safe for the reconcile harness and for honest
    point-in-time backtesting.

    Returns:
        archetype_primary    : one of ARCHETYPES, or "unclassified"
        archetype_secondary  : dual label, or None
        archetype_confidence : min(top_evidence, margin), INFORMATIONAL ONLY
        archetype_basis      : deterministic human-readable trace

    NOTE ON THE GATE: archetype_confidence is a summary number for display and
    for the LLM to phrase. It is NOT the gate. The gate is the single explicit
    condition below (MIN_EVIDENCE on the top score). Confidence is
    min(top_ev, margin), so a DUAL-LABELLED row reports LOW confidence even
    though its evidence is strong. That is correct and deliberate: it reports
    confidence in the PRIMARY label specifically, not in the classification as a
    whole. Do NOT refactor this into a threshold on confidence — a near-tie
    would then be rejected again, which is the bug this design removed.
    """
    # ── Cyclical is TERMINAL ──
    # The cyclical label exists specifically to fire WHEN THE FIRM LOOKS LIKE A
    # FAST GROWER. Peak-cycle earnings are exactly when growth reads strong, PE
    # reads low and margins read best. Letting recent growth override the
    # cyclical lens would rebuild the value trap inside the classifier. The same
    # logic disposes of turnaround (a cyclical at the trough is a cyclical, not a
    # turnaround) and asset_play (a cyclical at the trough trades below book).
    #
    # Confidence 1.0: there is no ESTIMATION uncertainty in "is this industry on
    # the list". The uncertainty is whether the LIST is right, which is model
    # risk documented at the constant — not a per-row confidence. Emitting 0.8
    # here would be false precision.
    if _ev_cyclical(data) >= 1.0:
        ind = str(data.get("industry") or "").strip()
        return {
            "archetype_primary": "cyclical",
            "archetype_secondary": None,
            "archetype_confidence": 1.0,
            "archetype_basis": f"cyclical: declared industry '{ind}'",
        }

    ev = {name: fn(data) for name, fn in _EVIDENCE.items() if name != "cyclical"}
    ranked = sorted(ev.items(), key=lambda kv: (-kv[1], kv[0]))   # name breaks ties
    top_name, top_ev = ranked[0]
    second_name, second_ev = ranked[1]
    margin = round(top_ev - second_ev, 4)

    if top_ev < MIN_EVIDENCE:
        return {
            "archetype_primary": "unclassified",
            "archetype_secondary": None,
            "archetype_confidence": round(top_ev, 4),
            "archetype_basis": (f"unclassified: no evidence -- every archetype scored 0.00" if top_ev == 0 else f"unclassified: best={top_name} ev={top_ev:.2f} < MIN_EVIDENCE {MIN_EVIDENCE}"),
        }

    # Dual label. Used ONLY to SUPPRESS sign flips where primary and secondary
    # disagree — never to add an interpretation. Conservative by construction.
    secondary = second_name if second_ev >= SECONDARY_AT * top_ev else None

    basis = f"{top_name} ev={top_ev:.2f}"
    if secondary:
        basis += f" | dual {secondary} ev={second_ev:.2f}"
    else:
        basis += f" | runner-up {second_name} ev={second_ev:.2f} margin={margin:.2f}"

    return {
        "archetype_primary": top_name,
        "archetype_secondary": secondary,
        "archetype_confidence": round(min(top_ev, margin), 4),
        "archetype_basis": basis,
    }


def lynch_category_from(primary):
    """archetype_primary -> lynch_category. Single site for the derivation."""
    return LYNCH_CATEGORY_MAP.get(primary, "unknown")
