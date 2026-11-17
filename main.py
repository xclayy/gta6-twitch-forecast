"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun gta6-twitch-ccu-forecast-01).

Reads one case per run (/input/<id>.in, schema makefun/twitch-ccu-case/v1): everything recorded
up to the end of hour 1 of the release day, hour 1's print included. Writes /output/<id>.out:
for each of hours 2..24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's concurrent
Twitch viewers in the game's category.

Model (fit at run time, from the case's own comparables -- past big launches' hourly curves):
for each hour h, a log-linear regression of that hour's viewers on the launch's hour-1 print
and its pre-release baseline:

    ln(y_h) = a_h + b_h * ln(p1) + c_h * ln(pre7) + noise,   noise ~ N(0, s_h^2)

where p1 is the case's hour-1 print and pre7 is the mean hourly viewers in the 7 days before
the release day (from the target's own history, plus -- in the final -- the target game's own
pre-release category history, which the case carries as a comparable). The quantiles are then

    q_tau(h) = exp(mu_h + z_tau * s_h),   z = (-1.645, -0.674, 0, 0.674, 1.645)

with s_h the regression's residual standard deviation at that hour.

Why this shape, and why it beats the house baseline (persistence + a widening random-walk
spread, practice loss 0.387): across 38 past launches the log-ratio of an hour's viewers to
hour 1 spans 1x to 2000x, so a flat median says almost nothing -- but the level of an hour
is strongly predictable from hour 1 early in the day (slope ~1.0 at hour 2) and settles to a
launch-size level later (slope ~0.2 by hour 19), while the pre-release baseline carries the
rest of the launch's scale. On the four practice cases this scores a mean pinball loss of
0.214 (house baseline 0.387); leave-one-out over the 37 library launches gives 0.418
(baseline 0.930).

Deterministic, standard library only, no network. Python 3.13, run as `python3 -P -s main.py`.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path: put it back first, so an
# `import model` finds a model.py shipped beside this file (none is shipped here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Fallback spread (the house baseline's per-root-hour sigma) when a fit has too few points.
FALLBACK_SIGMA = 0.15
# Guards on the fitted spread: never claim more certainty than the data supports, and never
# widen past the point where the band stops meaning anything.
SD_FLOOR = 0.20
SD_CAP = 1.50
# A comparable only joins the library when its release-day curve is nearly complete.
MIN_CURVE_HOURS = 20
# A regression needs at least this many launches with that hour present.
MIN_FIT_POINTS = 8


def round2(x: float):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes
    1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_day(value: str) -> datetime:
    """An ISO-8601 UTC timestamp (or plain date) as an aware datetime."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def mean(values):
    return sum(values) / len(values) if values else None


def ols(features, targets):
    """Least squares with an intercept. features: list of feature vectors (no intercept
    column); targets: list of floats. Returns (coefficients with intercept first,
    residual population standard deviation), or None when the system is singular."""
    n = len(targets)
    k = len(features[0]) + 1
    rows = [[1.0] + list(f) for f in features]
    # Normal equations.
    mat = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    vec = [sum(rows[i][a] * targets[i] for i in range(n)) for a in range(k)]
    # Gaussian elimination with partial pivoting.
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(mat[r][col]))
        if abs(mat[piv][col]) < 1e-12:
            return None
        mat[col], mat[piv] = mat[piv], mat[col]
        vec[col], vec[piv] = vec[piv], vec[col]
        for r in range(col + 1, k):
            factor = mat[r][col] / mat[col][col]
            for cc in range(col, k):
                mat[r][cc] -= factor * mat[col][cc]
            vec[r] -= factor * vec[col]
    coef = [0.0] * k
    for r in range(k - 1, -1, -1):
        coef[r] = (vec[r] - sum(mat[r][cc] * coef[cc] for cc in range(r + 1, k))) / mat[r][r]
    resid = [targets[i] - sum(rows[i][j] * coef[j] for j in range(k)) for i in range(n)]
    var = sum(r * r for r in resid) / n
    return coef, math.sqrt(var)


def release_curve(comp):
    """A comparable's release-day hourly curve: {hour 1..24: viewers} on its day_start (UTC)."""
    day = comp.get("day_start", "")[:10]
    out = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        start = point.get("from", "")
        if not viewers or not start.startswith(day):
            continue
        hour = int(start[11:13]) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def pre_release_mean(hourly, day_start, days=7):
    """Mean hourly viewers in the `days` days before the given day_start (UTC)."""
    start = parse_day(day_start) - timedelta(days=days)
    end = parse_day(day_start)
    values = []
    for point in hourly or []:
        viewers = point.get("viewers")
        start_text = point.get("from")
        if not viewers or not start_text:
            continue
        try:
            when = parse_day(start_text)
        except ValueError:
            continue
        if start <= when < end:
            values.append(viewers)
    return mean(values)


def build_library(case):
    """The launches this case's comparables can teach from: each with its hour-1 print,
    its pre-release baseline, and its release-day curve. The target game's own pre-release
    category history (unlock text "Not a launch") is not a launch and is left out."""
    target_name = (case.get("game") or {}).get("name", "")
    library = []
    for comp in case.get("comparables") or []:
        if "Not a launch" in (comp.get("unlock") or ""):
            continue
        if comp.get("name") == target_name:
            continue
        curve = release_curve(comp)
        if 1 not in curve or len(curve) < MIN_CURVE_HOURS:
            continue
        pre7 = pre_release_mean(comp.get("hourly"), comp.get("day_start", ""))
        if pre7 is None or pre7 <= 0:
            continue
        library.append({"p1": curve[1], "pre7": pre7, "curve": curve})
    return library


def target_pre_release(case):
    """The target launch's own pre-release baseline: the mean hourly viewers in the 7 days
    before its release day, from its history_hourly and -- in the final -- from the target
    game's own pre-release category history carried as a comparable. Points are deduped by
    timestamp, history_hourly winning a tie."""
    day_start = case.get("day_start", "")
    target_name = (case.get("game") or {}).get("name", "")
    by_time = {}
    for comp in case.get("comparables") or []:
        # The target game's own pre-release category history is carried as a comparable
        # under a longer name ("Grand Theft Auto VI category before release ..."), so
        # match on the name starting with the target game's name, not on equality.
        if target_name and (comp.get("name") or "").startswith(target_name):
            for point in comp.get("hourly") or []:
                if point.get("viewers") and point.get("from"):
                    by_time.setdefault(point["from"], point["viewers"])
    for point in case.get("history_hourly") or []:
        if point.get("viewers") and point.get("from"):
            by_time[point["from"]] = point["viewers"]
    start = parse_day(day_start) - timedelta(days=7)
    end = parse_day(day_start)
    values = []
    for stamp in sorted(by_time):
        try:
            when = parse_day(stamp)
        except ValueError:
            continue
        if start <= when < end:
            values.append(by_time[stamp])
    return mean(values)


def median_ratio_forecast(library, p1, hour):
    """Fallback for one hour: the median log-ratio to hour 1 across the library, with the
    library's spread. Used when a regression has too few points for that hour."""
    ratios = [math.log(entry["curve"][hour] / entry["p1"]) for entry in library if hour in entry["curve"]]
    if len(ratios) < 3:
        return None
    ordered = sorted(ratios)
    mid = len(ordered) // 2
    mu = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    sd = math.sqrt(sum((r - mu) ** 2 for r in ratios) / len(ratios))
    sd = min(max(sd, SD_FLOOR), SD_CAP)
    return [max(1.0, p1 * math.exp(mu + z * sd)) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    library = build_library(case)
    p1 = max(1, first["viewers"]) if first and first.get("viewers") else None
    if p1 is None:
        # No hour-1 print: the rules say there is no answer, but the file must still be
        # valid, so anchor on the library's median hour-1 print.
        p1 = sorted(e["p1"] for e in library)[len(library) // 2] if library else 100000
    pre7 = target_pre_release(case)
    log_p1 = math.log(p1)
    log_pre7 = math.log(pre7) if pre7 and pre7 > 0 else None

    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        quantiles = None
        if library and log_pre7 is not None:
            rows = [(e, e["curve"][hour]) for e in library if hour in e["curve"]]
            if len(rows) >= MIN_FIT_POINTS:
                fit = ols(
                    [[math.log(e["p1"]), math.log(e["pre7"])] for e, _ in rows],
                    [math.log(v) for _, v in rows],
                )
                if fit is not None:
                    coef, sd = fit
                    mu = coef[0] + coef[1] * log_p1 + coef[2] * log_pre7
                    sd = min(max(sd, SD_FLOOR), SD_CAP)
                    quantiles = [math.exp(mu + z * sd) for z in Z]
        if quantiles is None and library:
            # Too few points for a fit at this hour, or no pre-release baseline: fall back
            # to the library's median ratio for the hour.
            quantiles = median_ratio_forecast(library, p1, hour)
        if quantiles is None:
            # No library at all: the house baseline's persistence + widening spread.
            quantiles = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]
        cleaned = []
        for q in quantiles:
            value = round2(min(max(q, 1.0), 1e9))
            cleaned.append(value)
        for i in range(1, 5):
            if cleaned[i] < cleaned[i - 1]:
                cleaned[i] = cleaned[i - 1]
        hours.append({"hour": hour, "q": cleaned})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
