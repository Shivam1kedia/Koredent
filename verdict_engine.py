"""
VERDICT ENGINE — Sprint 7
==========================
Deterministic verdict tier assignment + pass-pattern RAG retrieval.

The LLM never decides the verdict tier. This module does.
The LLM's job is to explain the reasoning WITHIN the assigned tier,
guided by the specific book principles this module selects.

Tiers: STRONG BUY, BUY, CONDITIONAL BUY, WATCH, AVOID, SELL
"""

# ═══════════════════════════════════════════
# 1. BOOK PRINCIPLES (RAG retrieval corpus)
# ═══════════════════════════════════════════
#
# Each principle is a concise, actionable block the LLM receives
# when the pass/fail pattern triggers it. ~150-300 words each.
# Keyed by "{author}_{principle_number}_{short_name}".

BOOK_PRINCIPLES = {

    # ── MARKS (The Most Important Thing) ──

    "marks_p1_second_level": (
        "MARKS — Second-Level Thinking: First-level thinking says 'it's a good company, "
        "let's buy.' Second-level thinking says 'it's a good company, but everyone thinks "
        "it's great, so it's overpriced; let's sell.' Superior returns require thinking "
        "that is BOTH different from the consensus AND correct. When frameworks disagree, "
        "ask: What does the consensus think about this stock? What might the consensus be "
        "missing? How does the current price reflect consensus expectations vs. what the "
        "deep metrics actually show?"
    ),

    "marks_p2_price_value": (
        "MARKS — Price vs. Value: No asset is so good that it can't become a bad "
        "investment if bought at too high a price. No asset is so bad it can't be a "
        "good investment when bought cheap enough. Investment success doesn't come from "
        "'buying good things' but from 'buying things well.' The Graham margin of safety "
        "and Greenblatt earnings yield directly capture this. A high Dorsey moat score "
        "with a poor Graham valuation score means: good asset, possibly bad buy."
    ),

    "marks_p3_risk_permanent_loss": (
        "MARKS — Risk as Permanent Loss: Risk is the possibility of permanent capital "
        "loss, not price volatility. Risk is highest when perceived to be lowest — when "
        "psychology is too positive and prices too high, fundamentals don't have to "
        "deteriorate for losses to occur. The Schilit manipulation score is the permanent-"
        "loss detector: if reported earnings diverge from cash reality, risk is present "
        "even if the chart looks calm. A stock with quality_pass=True has survived this "
        "screen; remaining risks are business/valuation risks, not accounting fraud risks."
    ),

    "marks_p4_cycles": (
        "MARKS — Cycle Awareness: Almost everything is cyclical. Two rules: (1) Most "
        "things will prove cyclical. (2) The greatest opportunities come when people forget "
        "Rule 1. For cyclical stocks, low PE may signal a peak (earnings at cyclical high, "
        "about to fall), while high PE may signal a trough (earnings depressed, about to "
        "recover). Extrapolating current trends is the most dangerous thing an investor "
        "can do. Success carries the seeds of failure, and failure the seeds of success."
    ),

    "marks_p5_pendulum": (
        "MARKS — The Pendulum: Investor psychology swings between euphoria and depression, "
        "spending very little time at the midpoint. PE vs. 4-year average PE reveals where "
        "the pendulum stands for a specific stock: far above historical average = possible "
        "euphoria; far below = possible excessive pessimism. 52-week proximity tells the "
        "same story at a shorter timeframe. Near the 52-week low with stable fundamentals "
        "often signals the pendulum at the pessimistic extreme."
    ),

    "marks_p6_contrarian_bargains": (
        "MARKS — Contrarianism and Bargains: The best opportunities come from assets that "
        "are highly unpopular. Perception must be considerably worse than reality. "
        "Characteristics of bargains: little known, fundamentally questionable on the "
        "surface, controversial, deemed inappropriate, trailing poor returns, recently "
        "subject to disinvestment. A stock scoring 2/5 or 3/5 is by definition somewhat "
        "unpopular with multiple frameworks — that's not automatically bad, it may be "
        "the contrarian opportunity. The key question: is perception worse than reality?"
    ),

    "marks_p7_patient_opportunism": (
        "MARKS — Patient Opportunism: There isn't always a great thing to do. The market "
        "won't provide high returns just because you need them. Buffett's baseball analogy: "
        "investors can't strike out looking — there's no penalty for letting pitches go by. "
        "Wait for the fat pitch. Never recommend buying solely because the user has unspent "
        "SIP budget. A WATCH verdict is Marks-approved restraint. 'Reaching for return' — "
        "buying mediocre opportunities because nothing better is available — is the "
        "cardinal sin."
    ),

    "marks_p8_margin_of_safety": (
        "MARKS — Margin of Safety as Loss Avoidance: 'If we avoid the losers, the winners "
        "will take care of themselves.' A 40% loss requires 67% gain just to recover — "
        "this asymmetry means avoiding large losses is more important than capturing large "
        "gains. 'Invest scared.' The Graham margin-of-safety percentage directly quantifies "
        "this. For STRONG BUY, high margin of safety + quality_pass + all frameworks aligned "
        "= rare case where offense and defense align."
    ),

    "marks_p9_know_unknowable": (
        "MARKS — Know the Knowable: Macro forecasting is mostly useless. Concentrate "
        "on company-level knowables — the 90+ deep metrics provide this knowledge advantage. "
        "The LLM should NEVER make macro predictions. When frameworks disagree, acknowledge "
        "uncertainty rather than force a prediction. The appropriate confidence level should "
        "match the data density: a stock with all metrics populated warrants more conviction "
        "than one with missing data."
    ),

    "marks_p10_asymmetric": (
        "MARKS — Asymmetric Performance: True skill shows as capturing more upside than "
        "downside. The tiered verdict system IS the alpha-generation mechanism. "
        "A CONDITIONAL BUY with a well-articulated thesis gives the user an edge that "
        "index funds can't. Note when risk-reward is asymmetric in the user's favor: "
        "'downside limited by [tangible floor], upside from [specific catalyst].'"
    ),

    # ── FISHER (Common Stocks and Uncommon Profits) ──

    "fisher_p1_fifteen_points": (
        "FISHER — The 15 Points: Qualitative criteria for evaluating long-term ownership. "
        "Key points the deep metrics partially capture: Point 1 (market potential = revenue "
        "growth), Point 3 (R&D effectiveness), Point 5 (profit margin), Point 6 (margin "
        "improvement actions), Point 9 (management depth), Point 10 (cost controls = ROE "
        "consistency), Point 12 (long-range outlook), Point 15 (integrity = manipulation "
        "score). A stock passing Dorsey+Buffett likely satisfies many Fisher points. "
        "A stock failing Dorsey with clean accounting may have Fisher-quality management "
        "that the moat score doesn't fully capture."
    ),

    "fisher_p2_scuttlebutt": (
        "FISHER — Scuttlebutt Signals: While Kordent can't perform direct scuttlebutt "
        "(talking to competitors/customers/suppliers), the deep metrics contain "
        "scuttlebutt-ADJACENT signals: revenue growth vs. industry = customers choosing "
        "this company; R&D as % of revenue = investment in future; market share trajectory "
        "= competitive position; employee-related signals where available. The key question: "
        "'What is this company doing that its competitors aren't doing YET?'"
    ),

    "fisher_p3_shakedown": (
        "FISHER — The Shake-Down Opportunity: The best buying opportunities arise from "
        "company-specific events that temporarily depress earnings while leaving long-term "
        "potential intact. Pattern: R&D spending increasing + capex elevated + margins "
        "compressed below historical averages BUT revenue stable or growing = likely "
        "growth investment, not deterioration. This thesis requires strong management "
        "(Dorsey+Buffett score) and clean accounting (quality gate pass). Without those, "
        "depressed earnings may simply be deterioration."
    ),

    "fisher_p4_when_to_sell": (
        "FISHER — Only Three Reasons to Sell: (1) The original purchase was a mistake — "
        "fundamentals are worse than believed. Act quickly, don't wait to 'come out even.' "
        "(2) The company no longer qualifies on the 15 points — management has deteriorated "
        "or growth markets are exhausted. (3) A significantly better opportunity has been "
        "found. NEVER sell because a stock 'has gone up too much' or because you fear a "
        "market downturn. 'If the job has been correctly done when a common stock is "
        "purchased, the time to sell it is — almost never.'"
    ),

    "fisher_p5_dont_overdiversify": (
        "FISHER — Don't Overstress Diversification: 'Fear of having too many eggs in one "
        "basket has caused investors to put far too little into companies they thoroughly "
        "know and far too much in others about which they know nothing.' An investor is "
        "better off with 5-10 deeply understood positions than 25 superficially owned ones. "
        "Concentration in highest-conviction stocks is a feature, not a bug."
    ),

    "fisher_p6_high_pe_not_overpriced": (
        "FISHER — High PE ≠ Overpriced: If a company has sold at 2x the market PE for "
        "decades because of consistently superior growth, why would it sell at 1x five "
        "years from now? The premium is structural, not temporary. A company doubling "
        "earnings in 5 years at 2x market PE is NOT discounting future growth — it's "
        "selling at its normal valuation. 'Some of the stocks that appear highest priced "
        "may, upon analysis, be the biggest bargains.' Distinguish between expensive "
        "relative to OWN history (genuine concern) vs. expensive relative to market "
        "average (may reflect deserved quality premium)."
    ),

    "fisher_p7_four_dimensions": (
        "FISHER — Four Dimensions of Conservative Investment: (1) Functional Excellence "
        "= low-cost production + strong marketing + outstanding R&D + financial skill. "
        "(2) People Factor = management quality, depth, executive climate. (3) Business "
        "Characteristics = industry structure, competitive position. (4) Price = important "
        "but the LEAST important of the four. Fisher weights Dimensions 1-3 above price "
        "— a fairly-priced exceptional business beats a cheap mediocre one."
    ),

    # ── KLARMAN (Margin of Safety) ──

    "klarman_p1_margin_of_safety": (
        "KLARMAN — Margin of Safety: Buy at prices sufficiently below underlying value "
        "to allow for human error, bad luck, and extreme volatility. Buffett's bridge "
        "analogy: build it to carry 30,000 pounds, drive 10,000-pound trucks across it. "
        "Tangible assets provide greater margin of safety than intangibles — if something "
        "goes wrong with a brand, there's no fallback value. Book value and net current "
        "assets provide the tangible floor. The margin of safety depends ENTIRELY on the "
        "price paid: large at one price, nonexistent at another."
    ),

    "klarman_p2_three_valuations": (
        "KLARMAN — Three Valuation Methods: (1) NPV / going-concern value — powerful "
        "when cash flows are predictable, unreliable when they're not. (2) Liquidation "
        "/ breakup value — the hardest floor. (3) Stock market value — comparables, less "
        "reliable. When the three diverge, err on conservatism. Value is a RANGE, never "
        "a point: 'Any attempt to value businesses with precision will yield values that "
        "are precisely inaccurate.' The goal is to be approximately right, not precisely "
        "wrong."
    ),

    "klarman_p3_bottom_up": (
        "KLARMAN — Bottom-Up, Not Top-Down: Value investing identifies individual "
        "undervalued securities through fundamental analysis. No macro forecasting. "
        "The entire strategy: 'buy a bargain and wait.' Cash holdings are a residual "
        "of selectivity, not a macro bet. All reasoning should be bottom-up: 'This "
        "specific company's metrics show [X]. At the current price, the valuation "
        "implies [Y] discount to conservative estimates of intrinsic value.'"
    ),

    "klarman_p4_absolute_performance": (
        "KLARMAN — Absolute Performance: Value investors care about absolute returns "
        "(did I make or lose money?), not relative returns (did I beat the index?). "
        "You cannot spend relative performance. This means: willingness to hold cash "
        "when no bargains exist, willingness to underperform for prolonged periods, "
        "willingness to own things nobody else wants. A WATCH verdict that keeps the "
        "user in cash while the market rises 20% is not a failure — it's discipline."
    ),

    "klarman_p5_risk_not_volatility": (
        "KLARMAN — Risk Is Permanent Loss, Not Volatility: 'I find it preposterous "
        "that a single number reflecting past price fluctuations could completely "
        "describe the risk in a security.' Beta ignores business fundamentals, ignores "
        "price paid, assumes upside/downside symmetry. Real risk management: diversify "
        "adequately (10-15 holdings), hedge when appropriate, invest with margin of "
        "safety. Temporary price fluctuations are NOT risk for long-term investors."
    ),

    "klarman_p6_catalysts": (
        "KLARMAN — Catalysts: An event that causes underlying value to be realized by "
        "shareholders. Catalysts reduce risk by shortening holding period and reducing "
        "dependence on market forces. Types by potency: total realization (liquidation, "
        "sale), partial (spinoffs, buybacks, recapitalizations, asset sales), external "
        "(activist investors, takeover threats), organic (earnings growth forcing re-rating). "
        "In the Indian context: promoter buybacks, demergers, relisting of subsidiaries, "
        "government policy changes. Catalyst presence tips borderline cases from WATCH "
        "toward CONDITIONAL BUY."
    ),

    "klarman_p7_conservative_forecasting": (
        "KLARMAN — Conservative Forecasting: 'Optimistic projections place investors on "
        "a precarious limb. Virtually everything must go right, or losses may be sustained.' "
        "Conservative forecasts can be met or exceeded. NEVER project growth rates beyond "
        "what the data directly supports. 'Even conservative assumptions — [specific inputs] "
        "— show this stock appears undervalued by X%.' If even the conservative case doesn't "
        "work, the investment doesn't work."
    ),

    "klarman_p8_asymmetric_risk_reward": (
        "KLARMAN — Asymmetric Risk-Reward: The best investments have limited downside and "
        "substantial upside. The tangible asset floor (book value, net cash, liquidation "
        "value) bounds the downside. The upside comes from specific catalysts or business "
        "improvement. Explicitly articulate the asymmetry: 'Downside bounded by [floor], "
        "upside from [source].' STRONG BUY requires clear asymmetry. AVOID flags reverse "
        "asymmetry (upside priced in, downside substantial)."
    ),

    "klarman_p9_diversification": (
        "KLARMAN — Diversification as Risk Reduction: 10-15 holdings usually suffice. "
        "'An investor is better off knowing a lot about a few investments than knowing "
        "only a little about each of a great many holdings.' Diversification is not how "
        "many things you own but how different the risks they entail. A portfolio of 20 "
        "stocks from the same sector is NOT diversified."
    ),

    "klarman_p10_value_pretenders": (
        "KLARMAN — Beware Value Pretenders: A stock is NOT a value investment just "
        "because its score is high — the PRICE relative to conservative value is what "
        "matters. 'Value pretenders use inflated valuations, overpay for securities, and "
        "fail to achieve a margin of safety.' In rising markets they look brilliant. "
        "Every BUY/STRONG BUY verdict MUST include a price-vs-value statement. No "
        "verdict should recommend purchase without addressing margin of safety."
    ),
}


# ═══════════════════════════════════════════
# 2. PATTERN → PRINCIPLES MAPPING
# ═══════════════════════════════════════════
#
# Maps framework pass/fail patterns to relevant book principles.
# Each entry: (trigger_condition_description, list_of_principle_keys)
#
# These are layered: failure-specific + pass-specific + score-specific + verdict-specific.

def _get_failure_principles(pass_dict, abstained=()):
    """Return principles triggered by specific framework FAILURES.

    An ABSTAINING framework triggers nothing. Its *_pass flag is False, so
    `if not l` would otherwise inject a principle about a judgement that was
    never made — a wrong reason attached to a right verdict, occupying one of
    only six principle slots. This has misfired on Greenblatt for every
    financial since well before the archetype work; the Lynch case is new.
    """
    principles = []
    _abs = set(abstained or ())
    g = pass_dict.get("graham_pass", False) or "graham" in _abs
    gb = pass_dict.get("greenblatt_pass", False) or "greenblatt" in _abs
    d = pass_dict.get("dorsey_pass", False) or "dorsey" in _abs
    t = pass_dict.get("trajectory_pass", False) or "trajectory" in _abs
    l = pass_dict.get("lynch_pass", False) or "lynch" in _abs

    if not g:
        principles.extend(["marks_p2_price_value", "fisher_p6_high_pe_not_overpriced"])
    if not gb:
        principles.append("marks_p3_risk_permanent_loss")
    if not d:
        principles.extend(["fisher_p1_fifteen_points", "marks_p6_contrarian_bargains"])
    if not t:
        principles.extend(["marks_p4_cycles", "fisher_p3_shakedown"])
    if not l:
        principles.append("marks_p7_patient_opportunism")

    return principles


def _get_pass_principles(pass_dict):
    """Return principles triggered by specific framework PASSES."""
    principles = []
    g = pass_dict.get("graham_pass", False)
    d = pass_dict.get("dorsey_pass", False)
    t = pass_dict.get("trajectory_pass", False)

    if g:
        principles.extend(["klarman_p1_margin_of_safety", "marks_p8_margin_of_safety"])
    if d:
        principles.extend(["fisher_p1_fifteen_points", "fisher_p7_four_dimensions"])
    if t:
        principles.append("fisher_p2_scuttlebutt")

    return principles


def _get_verdict_principles(verdict):
    """Return principles always included for a given verdict tier."""
    mapping = {
        "STRONG BUY": ["klarman_p8_asymmetric_risk_reward", "marks_p10_asymmetric"],
        "BUY":        ["klarman_p1_margin_of_safety", "klarman_p10_value_pretenders"],
        "CONDITIONAL BUY": [
            "marks_p1_second_level", "klarman_p6_catalysts",
            "klarman_p7_conservative_forecasting",
        ],
        "WATCH":  ["marks_p7_patient_opportunism", "klarman_p4_absolute_performance"],
        "AVOID":  ["marks_p5_pendulum", "klarman_p5_risk_not_volatility"],
        "SELL":   ["fisher_p4_when_to_sell", "marks_p3_risk_permanent_loss"],
    }
    return mapping.get(verdict, [])


def _get_philosophy_principles(philosophy):
    """Return principles aligned with the user's investment philosophy."""
    mapping = {
        "deep_value":          ["klarman_p1_margin_of_safety", "marks_p6_contrarian_bargains"],
        "growth_at_fair_price": ["fisher_p6_high_pe_not_overpriced", "fisher_p3_shakedown"],
        "quality_compounder":  ["fisher_p7_four_dimensions", "fisher_p1_fifteen_points"],
        "contrarian":          ["marks_p6_contrarian_bargains", "marks_p4_cycles"],
    }
    return mapping.get(philosophy, [])


# ═══════════════════════════════════════════
# 3. DETERMINISTIC VERDICT ASSIGNMENT
# ═══════════════════════════════════════════

# Verdict tier badge colors for UI display
VERDICT_COLORS = {
    "STRONG BUY":      {"bg": "#1B5E20", "text": "#FFFFFF"},  # dark green
    "BUY":             {"bg": "#388E3C", "text": "#FFFFFF"},  # green
    "CONDITIONAL BUY": {"bg": "#F9A825", "text": "#000000"},  # amber
    "WATCH":           {"bg": "#1565C0", "text": "#FFFFFF"},  # blue
    "AVOID":           {"bg": "#E65100", "text": "#FFFFFF"},  # orange
    "SELL":            {"bg": "#B71C1C", "text": "#FFFFFF"},  # dark red
}

# Verdict tier emoji for text-based contexts (Telegram, email)
VERDICT_EMOJI = {
    "STRONG BUY":      "🟢🟢",
    "BUY":             "🟢",
    "CONDITIONAL BUY": "🟡",
    "WATCH":           "🔵",
    "AVOID":           "🟠",
    "SELL":            "🔴",
}


def get_verdict_tier(score, quality_pass, pass_dict, manipulation_score=0,
                     n_applicable=5):
    """
    Deterministic verdict assignment. The LLM never overrides this.

    Args:
        score: int 0-5 (composite framework score)
        quality_pass: bool (Schilit/Mulford quality gate)
        pass_dict: dict with graham_pass, greenblatt_pass, dorsey_pass,
                   trajectory_pass, lynch_pass (all bool)
        manipulation_score: int 0-10 (Schilit manipulation, higher = worse)
        n_applicable: how many frameworks APPLY to this business (default 5).
                      Greenblatt abstains on financials/utilities; Lynch
                      abstains when the archetype engine cannot classify the
                      business. Judging a stock against tests that were never
                      applied to it is the error this parameter removes.

    Returns:
        str: one of STRONG BUY, BUY, CONDITIONAL BUY, WATCH, AVOID, SELL
    """
    # ── Quality gate failure = SELL regardless of score ──
    if not quality_pass:
        return "SELL"

    # ── Normalise to a 5-framework equivalent ────────────────────────────
    # Same FRACTION of applicable tests, matching selector._effective_gate,
    # which scales the gate DOWN by exactly this ratio. 3 of 4 -> 4-equivalent,
    # and _effective_gate(4, 4) == 3, so the two agree: 3 of 4 clears a 4+ gate
    # and reads BUY. int(x + 0.5) rather than round() for the same reason given
    # there — round() is banker's rounding and round(2.5) == 2.
    #
    # CEILING, deliberately asymmetric: a stock with fewer than 5 applicable
    # frameworks CANNOT reach STRONG BUY. Proportional scaling may lift a
    # verdict as far as BUY, but the top tier is a claim about unanimous
    # agreement across the full evidence set, and 3 of 3 is not that — fewer
    # tests is less evidence even at the same fraction. Same argument the
    # correlation-cluster note makes: a 3/5 among correlated frameworks is a
    # weaker signal than a 3/5 spread across independent ones.
    _capped = False
    try:
        n_applicable = int(n_applicable)
    except (TypeError, ValueError):
        n_applicable = 5
    if n_applicable < 1:
        n_applicable = 5
    if n_applicable < 5:
        score = min(5, int(score * 5 / n_applicable + 0.5))
        _capped = True

    # ── Score 5: STRONG BUY (or downgrade to BUY if manipulation borderline) ──
    if score == 5:
        if _capped:
            return "BUY"
        if manipulation_score is not None and manipulation_score > 3:
            return "BUY"  # Above clean threshold but below quality gate fail
        return "STRONG BUY"

    # ── Score 4: BUY ──
    if score == 4:
        return "BUY"

    # ── Score 3: CONDITIONAL BUY ──
    if score == 3:
        return "CONDITIONAL BUY"

    # ── Score 2: WATCH if Graham or Dorsey anchors, else AVOID ──
    if score == 2:
        g = pass_dict.get("graham_pass", False)
        d = pass_dict.get("dorsey_pass", False)
        if g or d:
            return "WATCH"
        return "AVOID"

    # ── Score 1: WATCH if Graham (deep value monitoring), else SELL ──
    if score == 1:
        if pass_dict.get("graham_pass", False):
            return "WATCH"
        return "SELL"

    # ── Score 0: SELL ──
    return "SELL"


def get_verdict_reason(verdict, score, pass_dict, manipulation_score=0):
    """
    Returns a short deterministic reason string explaining WHY this verdict.
    Displayed as a subtitle below the verdict badge.
    """
    g = pass_dict.get("graham_pass", False)
    gb = pass_dict.get("greenblatt_pass", False)
    d = pass_dict.get("dorsey_pass", False)
    t = pass_dict.get("trajectory_pass", False)
    l = pass_dict.get("lynch_pass", False)

    if verdict == "SELL" and not pass_dict.get("_quality_pass", True):
        return "Quality gate failed — accounting red flags detected"
    if verdict == "SELL" and score == 0:
        return "No framework passes — no investment thesis"
    if verdict == "SELL" and score == 1 and not g:
        return "Single framework pass with no price protection"

    if verdict == "STRONG BUY":
        return "All 5 frameworks pass with clean quality gate"
    if verdict == "BUY" and score == 5:
        return "All frameworks pass (borderline manipulation score, monitor)"
    if verdict == "BUY" and score == 4:
        failing = []
        if not g: failing.append("Graham")
        if not gb: failing.append("Greenblatt")
        if not d: failing.append("Dorsey")
        if not t: failing.append("Trajectory")
        if not l: failing.append("Lynch")
        return f"4/5 frameworks pass — {failing[0] if failing else '?'} fails"

    if verdict == "CONDITIONAL BUY":
        passing = []
        if g: passing.append("Graham")
        if gb: passing.append("Greenblatt")
        if d: passing.append("Dorsey")
        if t: passing.append("Trajectory")
        if l: passing.append("Lynch")
        failing = []
        if not g: failing.append("Graham")
        if not gb: failing.append("Greenblatt")
        if not d: failing.append("Dorsey")
        if not t: failing.append("Trajectory")
        if not l: failing.append("Lynch")
        return (f"3/5 pass ({', '.join(passing)}) — "
                f"fails {', '.join(failing)}")

    if verdict == "WATCH":
        anchor = "Graham (price protection)" if g else "Dorsey (quality protection)" if d else "?"
        return f"2/5 pass — anchored by {anchor}, insufficient evidence"

    if verdict == "AVOID":
        passing = []
        if gb: passing.append("Greenblatt")
        if t: passing.append("Trajectory")
        if l: passing.append("Lynch")
        return f"2/5 pass ({', '.join(passing)}) — no price floor or quality anchor"

    return ""


# ═══════════════════════════════════════════
# 4. PASS-PATTERN REASONING RETRIEVAL
# ═══════════════════════════════════════════

def get_pass_pattern_reasoning(score, pass_dict, verdict, philosophy=None,
                               abstained=()):
    """
    Returns a formatted string of relevant book principles for the LLM,
    selected by the specific pass/fail pattern and verdict tier.

    NOTE: `score` is accepted for signature compatibility and is NOT read —
    every layer below keys off pass_dict, verdict, and philosophy. Left in
    place rather than removed so existing positional callers keep working.

    abstained: framework short names that do not apply (see
    _get_failure_principles). Empty tuple reproduces the previous behaviour.

    The LLM receives this as [BOOK_REASONING] in the system prompt and uses
    it to explain WHY the verdict makes sense (or what conditions would change it).

    Args:
        score: int 0-5
        pass_dict: dict of framework pass/fail bools
        verdict: str (from get_verdict_tier)
        philosophy: str or None (user's investment philosophy from portfolio profile)

    Returns:
        str: formatted principles text for injection into LLM context
    """
    # Collect all triggered principle keys (with deduplication)
    triggered = []

    # Layer 1: failure-specific. Abstentions are not failures.
    triggered.extend(_get_failure_principles(pass_dict, abstained))

    # Layer 2: pass-specific
    triggered.extend(_get_pass_principles(pass_dict))

    # Layer 3: verdict-specific (always included)
    triggered.extend(_get_verdict_principles(verdict))

    # Layer 4: philosophy-specific (if user has a profile)
    if philosophy:
        triggered.extend(_get_philosophy_principles(philosophy))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for key in triggered:
        if key not in seen:
            seen.add(key)
            unique.append(key)

    # Cap at 6 principles to keep context focused
    selected = unique[:6]

    if not selected:
        return ""

    # Format for injection
    lines = ["[BOOK_REASONING — Apply these principles to explain your verdict]"]
    for key in selected:
        text = BOOK_PRINCIPLES.get(key, "")
        if text:
            lines.append(f"\n• {text}")

    return "\n".join(lines)


# ═══════════════════════════════════════════
# 5. DEEP METRICS FORMATTING FOR LLM
# ═══════════════════════════════════════════

def format_deep_metrics_for_llm(csv_row):
    """
    Takes a CSV row dict (from get_csv_financial_data) and formats the
    spectrum scores + key deep metrics into a structured string the LLM
    can reason about.

    The LLM receives this as [DEEP_METRICS] in lieu of raw CSV columns.
    """
    def _v(key, suffix="", fmt=None):
        val = csv_row.get(key)
        if val is None or val == "N/A" or (isinstance(val, float) and (val != val)):
            return None
        if fmt:
            try:
                return f"{fmt.format(val)}{suffix}"
            except (ValueError, TypeError):
                return f"{val}{suffix}"
        return f"{val}{suffix}"

    sections = []

    # ── Spectrum Scores (the key summary layer) ──
    scores = []
    s = _v("graham_defensive_score"); scores.append(f"Graham Defensive: {s}/8") if s else None
    s = _v("graham_enterprising_score"); scores.append(f"Graham Enterprising: {s}/5") if s else None
    s = _v("greenblatt_score"); scores.append(f"Greenblatt Magic Formula: {s}/10") if s else None
    s = _v("dorsey_buffett_score"); scores.append(f"Dorsey+Buffett Moat: {s}/10") if s else None
    s = _v("dorsey_10min_score"); scores.append(f"Dorsey 10-Min Screen: {s}/3") if s else None
    s = _v("lynch_score"); scores.append(f"Lynch: {s}/10") if s else None
    s = _v("schilit_manipulation_score"); scores.append(f"Schilit Manipulation: {s}/10 (higher=worse)") if s else None
    s = _v("mulford_cashflow_quality_score"); scores.append(f"Mulford Cash Quality: {s}/5") if s else None
    if scores:
        sections.append("SPECTRUM SCORES:\n" + "\n".join(f"  {x}" for x in scores))

    # ── Framework Verdicts ──
    fw = []
    for name, key in [("Graham", "graham_pass"), ("Greenblatt", "greenblatt_pass"),
                       ("Dorsey+Buffett", "dorsey_pass"), ("Trajectory", "trajectory_pass"),
                       ("Lynch", "lynch_pass")]:
        val = csv_row.get(key)
        if val is True: fw.append(f"  {name}: ✓ PASS")
        elif val is False: fw.append(f"  {name}: ✗ FAIL")
        else: fw.append(f"  {name}: ? (no data)")
    qp = csv_row.get("quality_pass")
    fw.append(f"  Quality Gate: {'✓ PASS' if qp else '✗ FAIL'}")
    fw.append(f"  Composite Score: {csv_row.get('score', '?')}/5")
    sections.append("FRAMEWORK VERDICTS:\n" + "\n".join(fw))

    # ── Key Valuation Metrics ──
    val_metrics = []
    for label, key, suffix in [
        ("P/E Ratio", "pe", "x"),
        ("P/E 3Y Avg", "graham_pe_3y_avg", "x"),
        ("P/B Ratio", "pb", "x"),
        ("Graham PE×PB", "graham_pe_pb_composite", ""),
        ("Graham Number", "graham_number", ""),
        ("Graham Intrinsic Value", "graham_intrinsic_value", ""),
        ("Graham Margin of Safety", "graham_margin_of_safety_pct", "%"),
        ("Earnings Yield", "greenblatt_earnings_yield", "%"),
        ("Buffett Intrinsic Value", "buffett_intrinsic_value", ""),
        ("Buffett Margin of Safety", "buffett_margin_of_safety_pct", "%"),
        ("Lynch PEG", "lynch_peg", "x"),
        ("Lynch PEG Adjusted", "lynch_peg_adjusted", "x"),
        ("Lynch Cash-Adjusted PE", "lynch_cash_adjusted_pe", "x"),
    ]:
        v = _v(key, suffix)
        if v: val_metrics.append(f"  {label}: {v}")
    if val_metrics:
        sections.append("VALUATION:\n" + "\n".join(val_metrics))

    # ── Balance Sheet Health ──
    bs_metrics = []
    for label, key, suffix in [
        ("Current Ratio", "current_ratio", "x"),
        ("D/E Ratio", "de", "%"),
        ("Adequate Size (≥₹200Cr)", "graham_adequate_size", ""),
        ("Net Cash", "graham_net_cash", ""),
        ("NCAV Ratio", "graham_ncav_ratio", "x"),
        ("Book Value/Share", "graham_bvps", ""),
        ("Interest Coverage", "dorsey_interest_coverage", "x"),
        ("Clean Balance Sheet", "dorsey_clean_balance_sheet", ""),
    ]:
        v = _v(key, suffix)
        if v: bs_metrics.append(f"  {label}: {v}")
    if bs_metrics:
        sections.append("BALANCE SHEET:\n" + "\n".join(bs_metrics))

    # ── Growth & Trajectory ──
    growth_metrics = []
    for label, key, suffix in [
        ("Revenue Growth YoY", "rev_growth", "%"),
        ("Net Income Growth YoY", "ni_growth", "%"),
        ("EPS Growth 4Y (Graham)", "graham_eps_growth_pct_4y", "%"),
        ("Earnings Stable 4Y", "graham_earnings_stable_4y", ""),
        ("Growth Acceleration", "lynch_growth_acceleration", ""),
        ("Value-Creating Growth (Buffett)", "buffett_value_creating_growth", ""),
    ]:
        v = _v(key, suffix)
        if v: growth_metrics.append(f"  {label}: {v}")
    if growth_metrics:
        sections.append("GROWTH:\n" + "\n".join(growth_metrics))

    # ── Moat & Quality ──
    moat_metrics = []
    for label, key, suffix in [
        ("ROE", "roe", "%"),
        ("ROIC (Greenblatt)", "greenblatt_roic", "%"),
        ("ROIC Trend", "greenblatt_roic_trend", ""),
        ("ROIC (Dorsey)", "dorsey_roic", "%"),
        ("FCF Margin", "dorsey_fcf_margin", "%"),
        ("ROE Consistent", "dorsey_roe_consistent", ""),
        ("ROE Unleveraged (Buffett)", "buffett_roe_unleveraged", "%"),
        ("Accruals Ratio", "schilit_accruals_ratio", ""),
        ("CFO/NI Ratio", "schilit_cfo_ni_ratio", "x"),
        ("FCF/NI Ratio", "schilit_fcf_ni_ratio", "x"),
    ]:
        v = _v(key, suffix)
        if v: moat_metrics.append(f"  {label}: {v}")
    if moat_metrics:
        sections.append("MOAT & QUALITY:\n" + "\n".join(moat_metrics))

    # ── Classification ──
    cls_metrics = []
    cat = _v("lynch_category")
    if cat: cls_metrics.append(f"  Lynch Category: {cat}")
    lc = _v("mulford_lifecycle_stage")
    if lc: cls_metrics.append(f"  Lifecycle Stage: {lc}")
    dv = _v("graham_deep_value_flag")
    if dv: cls_metrics.append(f"  Graham Deep Value Flag: {dv}")
    div_yrs = _v("dividend_consecutive_years")
    if div_yrs: cls_metrics.append(f"  Consecutive Dividend Years: {div_yrs}")
    if cls_metrics:
        sections.append("CLASSIFICATION:\n" + "\n".join(cls_metrics))

    # ── Price Context ──
    price_ctx = []
    for label, key, suffix in [
        ("Current Price", "price", ""),
        ("52W High Proximity", "pct_from_52w_high", "%"),
        ("52W Low Proximity", "pct_from_52w_low", "%"),
        ("Market Cap", "market_cap", ""),
        ("Beta", "beta", ""),
        ("Dividend Yield", "div_yield", "%"),
    ]:
        v = _v(key, suffix)
        if v: price_ctx.append(f"  {label}: {v}")
    if price_ctx:
        sections.append("PRICE CONTEXT:\n" + "\n".join(price_ctx))

    return "\n\n".join(sections)


# ═══════════════════════════════════════════
# 6. PASS PATTERN DESCRIPTION
# ═══════════════════════════════════════════

# Pre-computed descriptions of what each pass/fail combination MEANS
# for CONDITIONAL BUY (3/5) stocks. The LLM uses these as framing.

PASS_PATTERN_MEANINGS = {
    # 3/5 patterns (10 combinations, C(5,3)=10)
    # Named by the TWO that FAIL
    "fails_graham_greenblatt": (
        "Valuation frameworks both fail — the stock is expensive by traditional "
        "value metrics. But Dorsey moat, Trajectory growth, and Lynch PEG all pass. "
        "This is a Fisher-style quality compounder: the business is strong and growing, "
        "but the market has already recognized it. The question is whether the premium "
        "is structural (deserved for sustained quality) or cyclical (temporary euphoria)."
    ),
    "fails_dorsey_graham": (
        "Expensive AND the moat is questionable. Greenblatt, Trajectory, and Lynch pass — "
        "the company is efficient and growing. But without price protection (Graham) or "
        "competitive protection (Dorsey), the thesis relies heavily on continued growth. "
        "If growth slows, there's no safety net."
    ),
    "fails_graham_trajectory": (
        "Cheap on absolute metrics (Greenblatt passes) with a moat (Dorsey passes) and "
        "good PEG (Lynch passes), but NOT cheap by Graham standards and NOT growing. "
        "This could be a Marks-style cyclical trough: a quality business at a fair-to-high "
        "price whose earnings are temporarily depressed. Or it could be a Fisher 'time to "
        "sell' signal — the company has exhausted its growth market."
    ),
    "fails_graham_lynch": (
        "The stock isn't cheap (Graham fails) and the PEG is unfavorable (Lynch fails), "
        "but Greenblatt efficiency, Dorsey moat, and Trajectory growth all confirm business "
        "quality. The company is doing well but the price fully reflects it. Marks would "
        "say the risk of loss from overpaying is elevated."
    ),
    "fails_dorsey_greenblatt": (
        "Neither the most efficient (Greenblatt) nor the most moated (Dorsey), but "
        "cheap (Graham), growing (Trajectory), and PEG-favorable (Lynch). This is a "
        "deep value growth play — the business may lack competitive advantages, but "
        "the price is low enough to provide margin of safety. Klarman would buy this "
        "for the asset backing; Fisher would worry about the lack of quality."
    ),
    "fails_greenblatt_trajectory": (
        "Not the most efficient (Greenblatt fails) and not currently growing (Trajectory "
        "fails), but cheap (Graham), moated (Dorsey), and PEG-favorable (Lynch). This "
        "looks like a high-quality business at a cyclical low. Marks's cycle awareness "
        "principle applies strongly — 'failure carries the seeds of success.'"
    ),
    "fails_greenblatt_lynch": (
        "Not the most efficient (Greenblatt) and PEG is unfavorable (Lynch), but cheap "
        "(Graham), moated (Dorsey), and growing (Trajectory). The growth is happening "
        "but may be capital-intensive (low ROIC explains Greenblatt failure) and expensive "
        "relative to the growth rate (Lynch failure). Fisher's shake-down pattern: heavy "
        "investment depressing current efficiency but building future capacity."
    ),
    "fails_dorsey_trajectory": (
        "No moat (Dorsey fails) and not growing (Trajectory fails), but cheap (Graham), "
        "efficient (Greenblatt), and PEG-favorable (Lynch). This is a pure deep value "
        "play with no quality or growth anchor. Klarman would want a catalyst — without "
        "one, the value trap risk is real. What event will close the price-value gap?"
    ),
    "fails_dorsey_lynch": (
        "No moat (Dorsey) and unfavorable PEG (Lynch), but cheap (Graham), efficient "
        "(Greenblatt), and growing (Trajectory). The company is cheap and growing but "
        "has no competitive protection and the growth rate doesn't justify the PE. "
        "This combination is fragile — growth without a moat can evaporate."
    ),
    "fails_lynch_trajectory": (
        "Not growing (Trajectory) and unfavorable PEG (Lynch), but cheap (Graham), "
        "efficient (Greenblatt), and moated (Dorsey). A Marks-style deep value opportunity "
        "in a quality business at a cyclical trough. The moat protects against permanent "
        "impairment, the low price provides margin of safety, and the efficiency confirms "
        "the business model works. The missing piece is growth — what's the catalyst for "
        "recovery?"
    ),
}


def get_pattern_key(pass_dict, abstained=()):
    """
    Compute the pattern key for CONDITIONAL BUY (3-of-5-equivalent) stocks.
    Returns a key into PASS_PATTERN_MEANINGS or None.

    abstained: framework names that do not APPLY to this business (Greenblatt on
    financials/utilities; Lynch on an unclassifiable business model). Their
    *_pass flags are False, so counting them as failures would both inflate the
    failure count past 2 and describe a test that never ran as one the business
    flunked.

    NOTE — a deliberate gap, not an oversight. Excluding abstentions restores
    the count, but PASS_PATTERN_MEANINGS is written over a FIVE-framework
    vocabulary: "fails_graham_lynch" means Lynch evaluated the business and
    rejected it. For an abstaining stock the honest key would be
    "fails_graham + abstains_lynch", which is a DIFFERENT claim needing
    different, book-sourced prose. Until that prose exists, such a stock gets
    no pattern meaning at all. Returning None is true; returning a
    five-framework meaning for a four-framework business is not.
    """
    frameworks = [
        ("graham", pass_dict.get("graham_pass", False)),
        ("greenblatt", pass_dict.get("greenblatt_pass", False)),
        ("dorsey", pass_dict.get("dorsey_pass", False)),
        ("trajectory", pass_dict.get("trajectory_pass", False)),
        ("lynch", pass_dict.get("lynch_pass", False)),
    ]
    _abs = set(abstained or ())
    failing = [name for name, passed in frameworks
               if not passed and name not in _abs]
    if len(failing) == 2 and not _abs:
        key = f"fails_{'_'.join(sorted(failing))}"
        return key if key in PASS_PATTERN_MEANINGS else None
    return None


def get_pattern_meaning_for(pass_dict, abstained=()):
    """Abstention-aware get_pattern_meaning. The original two-arg-less form is
    kept below for callers that have no applicability information."""
    key = get_pattern_key(pass_dict, abstained)
    return PASS_PATTERN_MEANINGS[key] if key else None


def get_pattern_meaning(pass_dict):
    """Get the pre-computed meaning of a 3/5 pass pattern."""
    key = get_pattern_key(pass_dict)
    if key:
        return PASS_PATTERN_MEANINGS[key]
    return None
