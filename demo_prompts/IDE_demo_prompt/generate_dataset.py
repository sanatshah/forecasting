"""Synthetic retail forecasting dataset generator.

Produces a daily, SKU x location time series that mimics realistic retail demand
behavior (seasonality, holidays, promotions, price elasticity, inventory
stockouts, product lifecycle, and markdowns).

The output schema matches the "Expected columns" of the companion forecasting
system (see forecasting_prompt.txt), so the CSV can feed that pipeline directly.

Grain: date x sku_id x location_id (channel is a per-location attribute, not an
extra multiplier). Every calendar day in the range is present for every
(sku, location) pair -> no missing dates, no duplicate keys.

Usage:
    python generate_dataset.py --out retail_demand_dataset.csv --seed 42
"""

from __future__ import annotations

import argparse
import calendar
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Static dimension definitions
# ---------------------------------------------------------------------------

DEPARTMENTS = ["Womens Apparel", "Mens Apparel", "Home"]

# 4 SKUs per department -> 12 SKUs total. Each SKU carries product/class/subclass
# metadata plus a base demand level, price, and a lifecycle profile.
# Fields: (sku_id, department, class, subclass, regular_price,
#          base_daily_units, lifecycle_status, launch_offset, eol_offset)
# base_daily_units = average units/day at a size-1.0 location before any
#   seasonality/promo/elasticity effects.
# lifecycle: (status, launch_offset_days, eol_offset_days_from_end)
#   - "New":  launches after start, ramps up
#   - "Core": available whole horizon, stable
#   - "End of Life": declines and is discontinued before the end
SKU_CATALOG = [
    # Womens Apparel
    ("SKU0001", "Womens Apparel", "Dresses", "Casual Dress", 34.0, 18.0, "Core", 0, 0),
    ("SKU0002", "Womens Apparel", "Tops", "Blouse", 22.0, 26.0, "New", 210, 0),
    ("SKU0003", "Womens Apparel", "Denim", "Skinny Jeans", 48.0, 14.0, "Core", 0, 0),
    ("SKU0004", "Womens Apparel", "Outerwear", "Wool Coat", 89.0, 7.0, "End of Life", 0, 240),
    # Mens Apparel
    ("SKU0005", "Mens Apparel", "Tops", "Polo Shirt", 26.0, 22.0, "Core", 0, 0),
    ("SKU0006", "Mens Apparel", "Denim", "Straight Jeans", 52.0, 12.0, "Core", 0, 0),
    ("SKU0007", "Mens Apparel", "Activewear", "Track Jacket", 44.0, 10.0, "New", 150, 0),
    ("SKU0008", "Mens Apparel", "Outerwear", "Puffer Jacket", 99.0, 6.0, "End of Life", 0, 300),
    # Home
    ("SKU0009", "Home", "Bedding", "Sheet Set", 59.0, 11.0, "Core", 0, 0),
    ("SKU0010", "Home", "Kitchen", "Cookware Set", 79.0, 8.0, "Core", 0, 0),
    ("SKU0011", "Home", "Decor", "Throw Pillow", 18.0, 30.0, "New", 300, 0),
    ("SKU0012", "Home", "Bath", "Towel Bundle", 24.0, 16.0, "End of Life", 0, 200),
]

# 5 store locations, each with a size multiplier and a sales channel attribute.
LOCATIONS = [
    ("LOC01", 1.30, "store"),   # flagship
    ("LOC02", 1.00, "store"),
    ("LOC03", 0.80, "store"),
    ("LOC04", 0.65, "store"),
    ("LOC05", 1.10, "online"),  # e-commerce fulfillment node
]

# Per-department price elasticity (higher => more sensitive to discounts).
DEPT_ELASTICITY = {
    "Womens Apparel": 1.8,
    "Mens Apparel": 1.5,
    "Home": 1.2,
}


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last given weekday (Mon=0..Sun=6) in a month."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


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


def build_holiday_map(years: list[int]) -> dict[date, tuple[str, float]]:
    """Map specific dates -> (event_name, demand_multiplier).

    Multipliers are peak-day values; a small ramp is applied around some events
    inside the demand loop for a more realistic build-up.
    """
    holidays: dict[date, tuple[str, float]] = {}
    for y in years:
        # Valentine's Day - Feb 14
        holidays[date(y, 2, 14)] = ("Valentines Day", 1.8)
        # Memorial Day - last Monday of May
        holidays[_last_weekday_of_month(y, 5, 0)] = ("Memorial Day", 1.7)
        # Back-to-School - a mid-August window (peak Aug 15)
        for offset in range(-7, 8):
            d = date(y, 8, 15) + timedelta(days=offset)
            mult = 2.0 - abs(offset) * 0.06
            holidays[d] = ("Back to School", round(mult, 3))
        # Black Friday - day after the 4th Thursday of November
        thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)  # Thursday=3
        black_friday = thanksgiving + timedelta(days=1)
        holidays[black_friday] = ("Black Friday", 3.2)
        # Cyber Monday - Monday after Black Friday
        cyber_monday = black_friday + timedelta(days=3)
        holidays[cyber_monday] = ("Cyber Monday", 2.8)
        # Christmas - December ramp toward Dec 25
        for day in range(1, 26):
            d = date(y, 12, day)
            mult = 1.2 + (day / 25.0) * 1.6  # ~1.2 -> ~2.8 by Dec 25
            name = "Christmas" if day >= 20 else "Holiday Season"
            holidays[d] = (name, round(mult, 3))
    return holidays


def fiscal_fields(d: pd.Timestamp) -> tuple[int, int, int, str]:
    """Return (fiscal_week, fiscal_month, fiscal_quarter, season).

    Uses a simple calendar-aligned fiscal calendar (fiscal year == calendar
    year) which is sufficient for synthetic modeling.
    """
    fiscal_week = int(d.isocalendar().week)
    fiscal_month = d.month
    fiscal_quarter = (d.month - 1) // 3 + 1
    if d.month in (12, 1, 2):
        season = "Winter"
    elif d.month in (3, 4, 5):
        season = "Spring"
    elif d.month in (6, 7, 8):
        season = "Summer"
    else:
        season = "Fall"
    return fiscal_week, fiscal_month, fiscal_quarter, season


# ---------------------------------------------------------------------------
# Behavior components
# ---------------------------------------------------------------------------

def weekly_multiplier(weekday: int) -> float:
    """Day-of-week shape: retail lifts Thu-Sat, dips mid-week."""
    # Mon..Sun
    curve = [0.85, 0.80, 0.85, 1.00, 1.25, 1.45, 1.15]
    return curve[weekday]


def monthly_multiplier(month: int) -> float:
    """Annual shape independent of the specific holiday spikes."""
    curve = {
        1: 0.80, 2: 0.85, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,
        7: 1.05, 8: 1.15, 9: 1.00, 10: 1.05, 11: 1.25, 12: 1.40,
    }
    return curve[month]


def lifecycle_factor(status: str, day_index: int, n_days: int,
                     launch_offset: int, eol_offset: int) -> tuple[float, str]:
    """Return (demand_multiplier, lifecycle_status_label) for a given day.

    - New: zero until launch, then a ramp to full over ~90 days.
    - Core: flat ~1.0 the whole horizon.
    - End of Life: full until EOL window, then linear decline to ~0, then
      discontinued (multiplier 0).
    """
    end_index = n_days - 1
    if status == "New":
        if day_index < launch_offset:
            return 0.0, "New"
        ramp_days = 90
        progress = (day_index - launch_offset) / ramp_days
        if progress < 1.0:
            return 0.2 + 0.8 * progress, "New"
        return 1.0, "Core"
    if status == "End of Life":
        eol_start = end_index - eol_offset
        if day_index < eol_start:
            return 1.0, "Core"
        decline_days = max(eol_offset, 1)
        progress = (day_index - eol_start) / decline_days
        mult = max(0.0, 1.0 - progress)
        if mult <= 0.05:
            return 0.0, "End of Life"
        return mult, "End of Life"
    # Core
    return 1.0, "Core"


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)
    years = sorted({d.year for d in dates})
    holiday_map = build_holiday_map(years)

    # Pre-compute calendar-level fields once (shared across sku/location).
    cal = []
    for d in dates:
        fw, fm, fq, season = fiscal_fields(d)
        event = holiday_map.get(d.date())
        cal.append({
            "weekday": d.weekday(),
            "month": d.month,
            "fiscal_week": fw,
            "fiscal_month": fm,
            "fiscal_quarter": fq,
            "season": season,
            "holiday_flag": 1 if event else 0,
            "holiday_name": event[0] if event else None,
            "holiday_mult": event[1] if event else 1.0,
        })
    cal_df = pd.DataFrame(cal)

    rows = []

    for (sku_id, dept, klass, subclass, reg_price, base_units,
         status, launch_offset, eol_offset) in SKU_CATALOG:
        product_id = "PID" + sku_id[-4:]
        elasticity = DEPT_ELASTICITY[dept]

        # --- Promo & markdown calendars (per SKU, shared across locations) ---
        promo_active = np.zeros(n_days, dtype=bool)
        promo_name = np.array([None] * n_days, dtype=object)
        promo_lift = np.ones(n_days)
        markdown_pct = np.zeros(n_days)

        # Random promo windows (~ every 3-5 weeks, lasting 3-7 days).
        cursor = int(rng.integers(7, 25))
        while cursor < n_days:
            length = int(rng.integers(3, 8))
            lift = 1.0 + rng.uniform(0.10, 0.40)  # 10%-40% unit lift
            ev = holiday_map.get(dates[cursor].date())
            name = ev[0] if ev else "Weekly Promo"
            for i in range(cursor, min(cursor + length, n_days)):
                promo_active[i] = True
                promo_lift[i] = lift
                promo_name[i] = name
            cursor += length + int(rng.integers(15, 30))

        # Random markdown events (5%-50%), lasting 7-21 days, clearance-style.
        cursor = int(rng.integers(30, 90))
        while cursor < n_days:
            length = int(rng.integers(7, 22))
            depth = rng.uniform(0.05, 0.50)
            for i in range(cursor, min(cursor + length, n_days)):
                markdown_pct[i] = max(markdown_pct[i], depth)
            cursor += length + int(rng.integers(30, 75))

        # EOL SKUs get deeper clearance markdowns toward the end.
        if status == "End of Life":
            eol_start = n_days - 1 - eol_offset
            for i in range(max(eol_start, 0), n_days):
                markdown_pct[i] = max(markdown_pct[i], rng.uniform(0.30, 0.50))

        for loc_id, loc_mult, channel in LOCATIONS:
            # Inventory state for this (sku, location).
            base_daily = base_units * loc_mult
            # Replenish roughly to a target ~14 days of base demand.
            reorder_point = base_daily * 5
            order_up_to = base_daily * 18
            on_hand = order_up_to
            in_transit = 0.0
            transit_arrival = -1  # day index when in_transit lands

            for day_index, d in enumerate(dates):
                c = cal_df.iloc[day_index]

                # Inventory receipt.
                if transit_arrival == day_index:
                    on_hand += in_transit
                    in_transit = 0.0

                life_mult, life_status = lifecycle_factor(
                    status, day_index, n_days, launch_offset, eol_offset)

                # ----- Pricing -----
                md = float(markdown_pct[day_index])
                selling_price = round(reg_price * (1.0 - md), 2)
                selling_price = min(selling_price, reg_price)  # never above regular
                selling_price = max(selling_price, 0.01)       # never negative/zero

                # Price elasticity: cheaper -> more units.
                price_ratio = reg_price / selling_price
                elasticity_mult = price_ratio ** (elasticity * 0.5)

                # Holiday ramp with department weighting (apparel spikes harder).
                hol_mult = float(c["holiday_mult"])
                if hol_mult > 1.0 and dept != "Home":
                    hol_mult = 1.0 + (hol_mult - 1.0) * 1.15

                # ----- Latent (unconstrained) demand -----
                latent = (
                    base_daily
                    * weekly_multiplier(int(c["weekday"]))
                    * monthly_multiplier(int(c["month"]))
                    * hol_mult
                    * promo_lift[day_index]
                    * elasticity_mult
                    * life_mult
                )
                # Poisson-style integer noise around the latent mean.
                latent_demand = float(rng.poisson(max(latent, 0.01)))

                # ----- Inventory constraint -----
                available = max(on_hand, 0.0)
                units_sold = int(min(round(latent_demand), available))
                stockout = 1 if (latent_demand > available and life_mult > 0) else 0
                on_hand -= units_sold

                # Reorder when below reorder point and nothing inbound.
                if on_hand <= reorder_point and in_transit == 0.0 and life_mult > 0:
                    in_transit = order_up_to - on_hand
                    transit_arrival = day_index + int(rng.integers(2, 6))

                sales_revenue = round(units_sold * selling_price, 2)

                rows.append((
                    d.strftime("%Y-%m-%d"),
                    sku_id,
                    product_id,
                    loc_id,
                    dept,
                    klass,
                    subclass,
                    channel,
                    units_sold,
                    sales_revenue,
                    round(reg_price, 2),
                    selling_price,
                    round(md * 100, 2),
                    int(bool(promo_active[day_index])),
                    promo_name[day_index] if promo_active[day_index] else None,
                    int(round(on_hand + units_sold)),  # on-hand at start of day
                    int(round(in_transit)),
                    stockout,
                    int(c["holiday_flag"]),
                    int(c["fiscal_week"]),
                    int(c["fiscal_month"]),
                    int(c["fiscal_quarter"]),
                    c["season"],
                    life_status,
                ))

    columns = [
        "date", "sku_id", "product_id", "location_id", "department", "class",
        "subclass", "channel", "units_sold", "sales_revenue", "regular_price",
        "selling_price", "markdown_pct", "promo_flag", "promo_event_name",
        "inventory_on_hand", "inventory_in_transit", "stockout_flag",
        "holiday_flag", "fiscal_week", "fiscal_month", "fiscal_quarter",
        "season", "product_lifecycle_status",
    ]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    """Assert data-quality guarantees and print a summary."""
    n_sku = df["sku_id"].nunique()
    n_loc = df["location_id"].nunique()
    n_days = len(dates)

    # No duplicate keys.
    dup = df.duplicated(subset=["date", "sku_id", "location_id"]).sum()
    assert dup == 0, f"Found {dup} duplicate (date, sku, location) rows"

    # No missing dates: every (sku, loc) must have every date.
    expected = n_sku * n_loc * n_days
    assert len(df) == expected, f"Row count {len(df)} != expected {expected}"
    per_group = df.groupby(["sku_id", "location_id"])["date"].nunique()
    assert (per_group == n_days).all(), "Some (sku, loc) pairs miss dates"

    # Value bounds.
    assert (df["units_sold"] >= 0).all(), "Negative units_sold"
    assert (df["regular_price"] > 0).all(), "Non-positive regular_price"
    assert (df["selling_price"] > 0).all(), "Non-positive selling_price"
    assert (df["selling_price"] <= df["regular_price"] + 1e-6).all(), \
        "selling_price exceeds regular_price"
    assert df["markdown_pct"].between(0, 50).all(), "markdown_pct out of [0, 50]"
    assert (df["sales_revenue"] >= 0).all(), "Negative sales_revenue"

    print("Validation passed.")
    print(f"  Rows:            {len(df):,}")
    print(f"  SKUs:            {n_sku}")
    print(f"  Locations:       {n_loc}")
    print(f"  Days:            {n_days} ({df['date'].min()} -> {df['date'].max()})")
    print(f"  Departments:     {df['department'].nunique()}")
    print(f"  Stockout rate:   {df['stockout_flag'].mean():.2%}")
    print(f"  Promo rate:      {df['promo_flag'].mean():.2%}")
    print(f"  Holiday rate:    {df['holiday_flag'].mean():.2%}")
    print(f"  Mean units_sold: {df['units_sold'].mean():.2f}")
    print(f"  Mean markdown:   {df['markdown_pct'].mean():.2f}%")
    print("  Lifecycle mix:")
    for k, v in df["product_lifecycle_status"].value_counts().items():
        print(f"    {k:<12} {v:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic retail dataset")
    parser.add_argument("--out", default="retail_demand_dataset.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"Generating dataset {args.start} -> {args.end} (seed={args.seed}) ...")
    df = generate(args.start, args.end, args.seed)

    dates = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq="D")
    validate(df, dates)

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
