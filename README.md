# GTA VI Launch-Day Twitch Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xclayy/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/xclayy/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A program that forecasts, hour by hour, how many people will watch Grand Theft Auto VI live
on Twitch over the 24 UTC hours of its release day (2026-11-19), from everything recorded up
to the end of the day's first hour — for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
"Forecast GTA VI's launch-day Twitch viewers hour by hour" ($250 USDC).

## The model

For each hour h = 2..24, a log-linear regression fit at run time on the case's own
comparables — past big launches' hourly Twitch curves:

    ln(y_h) = a_h + b_h * ln(p1) + c_h * ln(pre7) + noise,   noise ~ N(0, s_h^2)

where `p1` is the case's hour-1 print and `pre7` is the mean hourly viewers in the 7 days
before the release day (from the target's own history, plus — in the final — the target
game's own pre-release category history, which the case carries as a comparable). The five
quantiles are `q_tau(h) = exp(mu_h + z_tau * s_h)` with `z = (-1.645, -0.674, 0, 0.674, 1.645)`
and `s_h` the regression's residual standard deviation at that hour.

Why this shape: across 38 past launches the ratio of an hour's viewers to hour 1 spans 1x
to 2000x, so a flat median says almost nothing — but an hour's level is strongly
predictable from hour 1 early in the day (slope ~1.0 at hour 2) and settles to a launch-size
level later (slope ~0.2 by hour 19), while the pre-release baseline carries the rest of the
launch's scale. The game unlocks at midnight local time country by country (New Zealand
first at 11:00 UTC Nov 18, central Europe 23:00 UTC, UK 00:00 UTC, US East 05:00 UTC,
US West 08:00 UTC), so viewership ramps as regions unlock and peaks in US prime time —
exactly the shape these regressions recover from the comparables.

## Validation (local, on the kit's four practice cases)

Practice answers were rebuilt from SullyGnome's public hourly data for the four practice
launches and scored with the contest's pinball loss on the log scale:

| case | loss |
| --- | --- |
| p00 Marvel's Wolverine | 0.205 |
| p01 Red Dead Redemption 2 | 0.189 |
| p02 God of War Ragnarök | 0.230 |
| p03 Marvel's Spider-Man 2 | 0.229 |
| **mean** | **0.213** |

House baseline on the same answers: **0.387** (typical miss 82.7%, coverage 0.783) — the
rules' stated figure, reproduced exactly, which validates the answers and the scorer.
This entry: **0.213** (typical miss 30.3%, coverage 1.000), beating the house baseline by
~45%. Leave-one-out over the 37 library launches: 0.418 vs baseline 0.930.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=kit/gta6-twitch-ccu-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic (two runs write byte-identical
output), ~0.25 s for all four practice cases — far inside the 8 s per-case cap.

## Files

- `main.py` — the forecaster (the only program file; the entrypoint).
- `README.md` — this file. `LICENSE` — MIT. `.makefun/owner.json` — the entrant wallet.

## License

MIT — see [LICENSE](LICENSE).
