"""Synthetic Peacock subscriber + engagement dataset generator.

Produces a daily ``tier x acquisition_channel`` time series that mimics the
mechanics of a streaming subscription business:

* **Gross adds** driven by day-of-week, seasonality, the content calendar
  (NFL, Premier League, NBA, Olympics, Bravo / Love Island premieres), promo
  offers, and price (acquisition elasticity).
* **Churn** from two pools: a core base with a low hazard and a "tentpole /
  promo cohort" that churns much faster once the event that brought it in is
  over. List-price increases add a temporary churn bump.
* **Usage** (hours watched, daily actives) scaled by base size, tier, channel,
  weekends, seasonality and tentpole engagement.

The subscriber base identity ``paid_subs_eod = paid_subs_bod + gross_adds -
churned_subs`` holds exactly on every row, and ``paid_subs_bod`` of day t equals
``paid_subs_eod`` of day t-1 within a segment.

Outputs (schema matches ``retail_forecasting_optimization/config/config.yaml``):
* the daily history CSV (``--out``)
* a content calendar CSV (``--calendar-out``) that extends past the history so
  forecasts can use known future tentpoles.

The tentpole calendar is loosely based on real Peacock programming but the
dates, intensities, prices for the ad tier and all subscriber volumes are
synthetic.

Usage:
    python generate_dataset.py --out sample_input.csv --seed 42
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Static dimension definitions
# ---------------------------------------------------------------------------

# Tier: (base_price, hours_per_sub_per_day, churn_hazard_multiplier,
#        acquisition_elasticity, churn_elasticity)
TIERS = {
    "Premium": (5.99, 0.80, 1.00, -1.0, 1.8),
    "Premium Plus": (11.99, 1.10, 0.80, -0.8, 1.4),
    "Ad Tier": (2.99, 0.55, 1.35, -1.3, 2.2),
}

# List-price history per tier: (effective_date, new_price).
PRICE_HISTORY = {
    "Premium": [("2024-07-18", 7.99), ("2025-07-23", 10.99)],
    "Premium Plus": [("2024-07-18", 13.99), ("2025-07-23", 16.99)],
    "Ad Tier": [("2025-07-23", 4.99)],
}

# Channel: (distribution_partner, base_share, daily_core_churn_hazard,
#           tentpole_sensitivity, price_sensitivity, usage_multiplier,
#           promo_eligible)
CHANNELS = {
    "direct": ("Peacock", 0.35, 0.0019, 1.00, 1.00, 1.00, True),
    "app_store": ("Apple/Google", 0.20, 0.0022, 1.10, 1.00, 1.05, True),
    "mvpd_partner": ("Xfinity/Spectrum", 0.30, 0.0007, 0.35, 0.30, 0.70, False),
    "retail_bundle": ("Retail Bundle", 0.15, 0.0012, 0.25, 0.30, 0.60, False),
}

# Starting paid subscribers per tier (split across channels by base_share).
TIER_START_SUBS = {
    "Premium": 22_000_000,
    "Premium Plus": 7_000_000,
    "Ad Tier": 3_000_000,
}

# Organic growth bias per tier: gross adds are set to churn x this factor at
# baseline, so Ad Tier slowly shrinks while Premium Plus grows.
TIER_GROWTH_BIAS = {
    "Premium": 1.04,
    "Premium Plus": 1.08,
    "Ad Tier": 0.97,
}


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth given weekday (Mon=0..Sun=6) in a month (n starts at 1)."""
    d = date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_tentpole_map(start: date, end: date) -> dict[date, tuple[str, str, float]]:
    """Map date -> (tentpole_name, tentpole_type, intensity).

    ``intensity`` is the peak gross-adds multiplier for direct channels. When
    events overlap the highest-intensity event wins the day.
    """
    events: dict[date, tuple[str, str, float]] = {}

    def put(d: date, name: str, kind: str, intensity: float) -> None:
        if d < start or d > end:
            return
        if d not in events or events[d][2] < intensity:
            events[d] = (name, kind, intensity)

    for y in range(start.year, end.year + 1):
        # NFL Sunday Night Football: Sundays from early Sep through early Jan.
        kickoff = _nth_weekday_of_month(y, 9, 3, 1)  # first Thursday of Sep
        for d in _daterange(kickoff, date(y + 1, 1, 7)):
            if d.weekday() == 6:
                put(d, "Sunday Night Football", "sports", 1.45)
        put(kickoff, "NFL Kickoff", "sports", 1.6)

        # Premier League: Saturdays Aug through May.
        for d in _daterange(date(y, 8, 15), date(y + 1, 5, 25)):
            if d.weekday() == 5:
                put(d, "Premier League", "sports", 1.15)

        # NBA on Peacock from the 2025-26 season: Tue and Sun, late Oct - Apr.
        if y >= 2025:
            for d in _daterange(date(y, 10, 21), date(y + 1, 4, 15)):
                if d.weekday() in (1, 6):
                    put(d, "NBA", "sports", 1.25)

        # Love Island USA: daily episodes through June-July.
        for d in _daterange(date(y, 6, 3), date(y, 7, 21)):
            put(d, "Love Island USA", "entertainment", 1.30)

        # The Traitors: January premiere window.
        for d in _daterange(date(y, 1, 9), date(y, 2, 27)):
            if d.weekday() == 3:
                put(d, "The Traitors", "entertainment", 1.25)

        # Bravo fall premiere week (Real Housewives et al.).
        for d in _daterange(date(y, 10, 1), date(y, 10, 7)):
            put(d, "Bravo Fall Premieres", "entertainment", 1.20)

    # One-off tentpoles.
    put(date(2024, 1, 13), "NFL Wild Card Exclusive", "sports", 4.0)
    put(date(2024, 9, 6), "NFL Brazil Exclusive", "sports", 3.0)
    for d in _daterange(date(2024, 7, 26), date(2024, 8, 11)):
        put(d, "Paris Olympics", "sports", 2.6)
    for d in _daterange(date(2026, 2, 6), date(2026, 2, 22)):
        put(d, "Milan Cortina Olympics", "sports", 2.3)
    put(date(2026, 2, 8), "Super Bowl LX", "sports", 3.5)
    return events


def build_holidays(start: date, end: date) -> set[date]:
    out: set[date] = set()
    for y in range(start.year, end.year + 1):
        thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)
        out.update({date(y, 1, 1), date(y, 7, 4), thanksgiving, date(y, 12, 25)})
    return {d for d in out if start <= d <= end}


def build_promo_windows(start: date, end: date) -> dict[date, tuple[str, float]]:
    """Black Friday style deals: (promo_name, discount_pct)."""
    promos: dict[date, tuple[str, float]] = {}
    for y in range(start.year, end.year + 1):
        thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)
        for d in _daterange(thanksgiving - timedelta(days=7), thanksgiving + timedelta(days=5)):
            promos[d] = ("Black Friday Deal", 0.70)
        for d in _daterange(date(y, 7, 1), date(y, 7, 7)):
            promos[d] = ("Summer Sale", 0.40)
    return {d: v for d, v in promos.items() if start <= d <= end}


def build_content_calendar(start: date, end: date) -> pd.DataFrame:
    """Daily known-in-advance calendar (tentpoles + holidays)."""
    tentpoles = build_tentpole_map(start, end)
    holidays = build_holidays(start, end)
    rows = []
    for d in _daterange(start, end):
        ev = tentpoles.get(d)
        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "tentpole_flag": 1 if ev else 0,
            "tentpole_name": ev[0] if ev else None,
            "tentpole_type": ev[1] if ev else "none",
            "tentpole_intensity": round(ev[2], 3) if ev else 1.0,
            "holiday_flag": 1 if d in holidays else 0,
        })
    return pd.DataFrame(rows)


def list_price_on(tier: str, d: date) -> float:
    price = TIERS[tier][0]
    for eff, new_price in PRICE_HISTORY.get(tier, []):
        if d >= date.fromisoformat(eff):
            price = new_price
    return price


def season_of(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


# ---------------------------------------------------------------------------
# Behavior components
# ---------------------------------------------------------------------------

def weekly_adds_multiplier(weekday: int) -> float:
    """Signups skew to the weekend when live sports air."""
    return [0.90, 0.85, 0.88, 0.92, 1.00, 1.15, 1.30][weekday]


def monthly_adds_multiplier(month: int) -> float:
    return {
        1: 1.10, 2: 1.00, 3: 0.92, 4: 0.90, 5: 0.90, 6: 0.95,
        7: 1.00, 8: 1.00, 9: 1.10, 10: 1.08, 11: 1.12, 12: 1.05,
    }[month]


def weekly_usage_multiplier(weekday: int) -> float:
    return [0.92, 0.90, 0.90, 0.93, 1.00, 1.15, 1.20][weekday]


def monthly_usage_multiplier(month: int) -> float:
    return {
        1: 1.10, 2: 1.08, 3: 1.00, 4: 0.96, 5: 0.94, 6: 0.92,
        7: 0.92, 8: 0.94, 9: 1.00, 10: 1.03, 11: 1.06, 12: 1.10,
    }[month]


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    dates = list(_daterange(start_d, end_d))

    tentpoles = build_tentpole_map(start_d, end_d)
    holidays = build_holidays(start_d, end_d)
    promos = build_promo_windows(start_d, end_d)
    price_change_days = {
        (tier, date.fromisoformat(eff)) for tier, hist in PRICE_HISTORY.items() for eff, _ in hist
    }

    rows = []
    for tier, (base_price, hours_per_sub, tier_hazard, acq_el, churn_el) in TIERS.items():
        for channel, (partner, share, core_hazard, tp_sens, price_sens, usage_mult,
                      promo_ok) in CHANNELS.items():
            core = float(TIER_START_SUBS[tier] * share)
            cohort = 0.0
            hazard = core_hazard * tier_hazard
            base_adds = core * hazard * TIER_GROWTH_BIAS[tier]
            price_bump_days_left = 0
            price_bump_size = 0.0
            last_price = list_price_on(tier, start_d)

            for day_index, d in enumerate(dates):
                list_price = list_price_on(tier, d)
                price_increase = 1 if (tier, d) in price_change_days else 0
                if price_increase:
                    pct = (list_price - last_price) / last_price
                    price_bump_size = churn_el * pct * price_sens
                    price_bump_days_left = 45
                last_price = list_price

                promo = promos.get(d) if promo_ok else None
                discount = promo[1] if promo else 0.0
                effective_price = round(list_price * (1.0 - discount), 2)

                ev = tentpoles.get(d)
                intensity = ev[2] if ev else 1.0
                tp_lift = 1.0 + (intensity - 1.0) * tp_sens
                promo_lift = 1.0 + discount * 1.6 if promo else 1.0
                # Long-run acquisition response to list price is damped: content
                # investment offsets most of the permanent hit from increases.
                price_ratio = list_price / base_price
                price_mult = price_ratio ** (acq_el * price_sens * 0.3)
                trend = 1.0 + 0.15 * day_index / len(dates)

                mean_adds = (
                    base_adds
                    * trend
                    * weekly_adds_multiplier(d.weekday())
                    * monthly_adds_multiplier(d.month)
                    * tp_lift
                    * promo_lift
                    * price_mult
                )
                gross_adds = int(rng.poisson(max(mean_adds, 1.0)))

                # Event- and deal-driven signups land in the fast-churn cohort.
                lift = tp_lift * promo_lift
                cohort_share = (lift - 1.0) / lift if lift > 1.0 else 0.0
                cohort_adds = gross_adds * cohort_share

                bod = core + cohort
                bump = price_bump_size * (price_bump_days_left / 45.0) if price_bump_days_left > 0 else 0.0
                core_h = hazard * (1.0 + bump)
                cohort_h = hazard * (1.0 if ev else 4.0)
                churn = int(
                    rng.poisson(max(core * core_h, 0.0)) + rng.poisson(max(cohort * cohort_h, 0.0))
                )
                churn = min(churn, int(round(bod)))
                if price_bump_days_left > 0:
                    price_bump_days_left -= 1

                # Split churn across pools proportionally to expected hazard.
                exp_core = core * core_h
                exp_cohort = cohort * cohort_h
                tot = exp_core + exp_cohort
                churn_cohort = churn * (exp_cohort / tot) if tot > 0 else 0.0
                churn_core = churn - churn_cohort

                bod_int = int(round(bod))
                eod_int = bod_int + gross_adds - churn

                core = core - churn_core + (gross_adds - cohort_adds)
                cohort = cohort - churn_cohort + cohort_adds
                # Survivors of the event cohort graduate into the core base.
                graduate = cohort / 60.0
                cohort -= graduate
                core += graduate
                # Re-anchor floats to the integer ledger so the identity holds.
                drift = eod_int - (core + cohort)
                core += drift

                usage_lift = 1.0 + (intensity - 1.0) * 0.6
                hours = (
                    bod_int
                    * hours_per_sub
                    * usage_mult
                    * weekly_usage_multiplier(d.weekday())
                    * monthly_usage_multiplier(d.month)
                    * usage_lift
                    * rng.normal(1.0, 0.03)
                )
                active_rate = min(0.95, 0.40 * usage_mult * weekly_usage_multiplier(d.weekday()) * usage_lift)
                dau = int(min(bod_int, bod_int * active_rate * rng.normal(1.0, 0.02)))

                rows.append((
                    d.strftime("%Y-%m-%d"),
                    tier,
                    channel,
                    partner,
                    gross_adds,
                    churn,
                    bod_int,
                    eod_int,
                    round(max(hours, 0.0), 1),
                    max(dau, 0),
                    round(list_price, 2),
                    effective_price,
                    round(discount, 3),
                    1 if promo else 0,
                    promo[0] if promo else None,
                    1 if ev else 0,
                    ev[0] if ev else None,
                    ev[1] if ev else "none",
                    round(intensity, 3),
                    price_increase,
                    1 if d in holidays else 0,
                    int(pd.Timestamp(d).isocalendar().week),
                    d.month,
                    (d.month - 1) // 3 + 1,
                    season_of(d.month),
                ))

    columns = [
        "date", "tier", "acquisition_channel", "distribution_partner",
        "gross_adds", "churned_subs", "paid_subs_bod", "paid_subs_eod",
        "hours_watched", "daily_active_subs", "list_price", "effective_price",
        "discount_pct", "promo_flag", "promo_name", "tentpole_flag",
        "tentpole_name", "tentpole_type", "tentpole_intensity",
        "price_increase_flag", "holiday_flag", "fiscal_week", "fiscal_month",
        "fiscal_quarter", "season",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, n_days: int) -> None:
    """Assert data-quality guarantees and print a summary."""
    keys = ["tier", "acquisition_channel"]
    n_seg = df.groupby(keys).ngroups

    dup = df.duplicated(subset=["date"] + keys).sum()
    assert dup == 0, f"Found {dup} duplicate (date, segment) rows"
    assert len(df) == n_seg * n_days, f"Row count {len(df)} != {n_seg * n_days}"

    for col in ["gross_adds", "churned_subs", "paid_subs_bod", "paid_subs_eod",
                "hours_watched", "daily_active_subs"]:
        assert (df[col] >= 0).all(), f"Negative {col}"
    assert (df["effective_price"] > 0).all(), "Non-positive effective_price"
    assert (df["effective_price"] <= df["list_price"] + 1e-6).all(), \
        "effective_price exceeds list_price"
    assert df["discount_pct"].between(0, 1).all(), "discount_pct out of [0, 1]"
    assert (df["daily_active_subs"] <= df["paid_subs_bod"]).all(), "DAU exceeds base"

    identity = df["paid_subs_bod"] + df["gross_adds"] - df["churned_subs"]
    assert (identity == df["paid_subs_eod"]).all(), "Base identity eod = bod + adds - churn broken"
    ordered = df.sort_values(keys + ["date"])
    prev_eod = ordered.groupby(keys)["paid_subs_eod"].shift(1)
    mask = prev_eod.notna()
    assert (ordered.loc[mask, "paid_subs_bod"] == prev_eod[mask]).all(), \
        "paid_subs_bod does not chain from prior paid_subs_eod"

    last = ordered.groupby(keys).tail(1)
    first = ordered.groupby(keys).head(1)
    print("Validation passed.")
    print(f"  Rows:              {len(df):,}")
    print(f"  Segments:          {n_seg}")
    print(f"  Days:              {n_days} ({df['date'].min()} -> {df['date'].max()})")
    print(f"  Paid subs start:   {first['paid_subs_bod'].sum():,}")
    print(f"  Paid subs end:     {last['paid_subs_eod'].sum():,}")
    print(f"  Tentpole day rate: {df['tentpole_flag'].mean():.2%}")
    print(f"  Promo day rate:    {df['promo_flag'].mean():.2%}")
    hours_per_sub = df["hours_watched"].sum() / df["paid_subs_bod"].sum()
    print(f"  Hours/sub/day:     {hours_per_sub:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Peacock subscriber dataset")
    parser.add_argument("--out", default="sample_input.csv", help="History CSV path")
    parser.add_argument("--calendar-out", default=None,
                        help="Content calendar CSV path (default: content_calendar.csv next to --out)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-30", help="End date (YYYY-MM-DD)")
    parser.add_argument("--calendar-end", default="2026-12-31",
                        help="Last date in the content calendar (should cover the forecast horizon)")
    args = parser.parse_args()

    print(f"Generating dataset {args.start} -> {args.end} (seed={args.seed}) ...")
    df = generate(args.start, args.end, args.seed)
    n_days = (date.fromisoformat(args.end) - date.fromisoformat(args.start)).days + 1
    validate(df, n_days)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")

    cal_path = Path(args.calendar_out) if args.calendar_out else Path(args.out).with_name("content_calendar.csv")
    cal = build_content_calendar(date.fromisoformat(args.start), date.fromisoformat(args.calendar_end))
    cal.to_csv(cal_path, index=False)
    print(f"Wrote {len(cal):,} calendar days to {cal_path}")


if __name__ == "__main__":
    main()
