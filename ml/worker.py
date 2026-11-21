#!/usr/bin/env python3
"""
hrv-ml: Wall-powered server-side context correction, circadian BPM sleep scoring,
habit attribution (Ridge / scikit-learn), and truthful plain-English insights.
Reads from self-hosted Postgres `hrv` DB via local Unix socket; zero cloud traffic.
"""
import json
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
from xgboost import XGBRegressor

import os
LOCAL_TZ = ZoneInfo(os.environ.get("HRV_LOCAL_TZ", "UTC"))
DB_DSN = os.environ.get("HRV_DB_DSN", "host=/var/run/postgresql port=5432 dbname=hrv user=hrv")


def mad_clean(values: np.ndarray, k: float = 3.5) -> np.ndarray:
    if len(values) < 5:
        return values
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad if mad > 0 else max(2.0, np.std(values))
    mask = np.abs(values - med) <= k * scale
    return values[mask] if np.any(mask) else values


def wake_day_local(dt_utc: datetime):
    local = dt_utc.astimezone(LOCAL_TZ)
    shifted = local - timedelta(hours=12)
    return (shifted + timedelta(days=1)).date()


def score_sleep_segments(bins: list[tuple[datetime, float, float]]) -> list[tuple[datetime, datetime, str]]:
    """
    Lightweight circadian + HR-level + local-variability sleep stage classifier
    (Wake / Light / Deep / REM) for nights without Apple Health sleep stages.
    Honest: only runs when >= 90 min of overnight data exists.
    """
    if len(bins) < 18:
        return []
    hrs = np.array([b[1] for b in bins], dtype=float)
    sds = np.array([b[2] for b in bins], dtype=float)
    q25, q75 = np.percentile(hrs, [25, 75])
    q90 = np.percentile(hrs, 90)
    sd_med = np.median(sds)
    t0 = bins[0][0]
    total_sec = max(1.0, (bins[-1][0] - t0).total_seconds())

    raw_stages = []
    for t, hr, sd in bins:
        progress = (t - t0).total_seconds() / total_sec
        # Deep NREM clusters in first half with lowest HR & flattest variance
        # REM clusters in second half with elevated HR & higher beat variability
        if hr >= q90 and sd >= sd_med * 1.25:
            st = "awake"
        elif hr <= q25 and sd <= sd_med * 1.1 and progress <= 0.68:
            st = "deep"
        elif hr >= q75 and sd >= sd_med and progress >= 0.35:
            st = "rem"
        else:
            st = "light"
        raw_stages.append((t, t + timedelta(minutes=5), st))

    # Merge contiguous identical stages
    merged = []
    for s, e, st in raw_stages:
        if merged and merged[-1][2] == st and (s - merged[-1][1]).total_seconds() <= 60:
            merged[-1] = (merged[-1][0], e, st)
        else:
            merged.append((s, e, st))
    return merged



def ttm_forecast_expected(series_5min: list[float], horizon: int = 96) -> float | None:
    """Forecast expected overnight BPM using IBM Granite TinyTimeMixer (805k params, local CPU)."""
    if len(series_5min) < 12:
        return None
    try:
        import torch
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        model = TinyTimeMixerForPrediction.from_pretrained(
            "ibm-granite/granite-timeseries-ttm-r2",
            revision="main",
            local_files_only=True,
        )
        model.eval()
        ctx = model.config.context_length
        arr = np.array(series_5min[-ctx:], dtype=np.float32)
        if len(arr) < ctx:
            pad = np.full(ctx - len(arr), float(np.median(arr)), dtype=np.float32)
            arr = np.concatenate([pad, arr])
        x = torch.from_numpy(arr).view(1, ctx, 1)
        with torch.no_grad():
            pred = model(past_values=x).prediction_outputs[0, :horizon, 0].numpy()
        val = float(np.mean(pred))
        return round(val, 1) if math.isfinite(val) else None
    except Exception:
        return None

def run():
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT recorded_at, bpm
                FROM hr_samples
                WHERE recorded_at >= now() - interval '35 days'
                  AND (contact IS NULL OR contact = true)
                  AND bpm BETWEEN 30 AND 220
                ORDER BY recorded_at ASC
                """
            )
            rows = cur.fetchall()

            cur.execute(
                """
                SELECT day, fatigue, readiness_subjective, tags
                FROM daily_checkins
                WHERE day >= current_date - 35
                """
            )
            checkins = {r[0]: {"fatigue": r[1], "readiness": r[2], "tags": r[3] or []} for r in cur.fetchall()}

            cur.execute(
                """
                SELECT DISTINCT (end_at AT TIME ZONE 'UTC')::date
                FROM sleep_segments
                WHERE source = 'healthkit' AND end_at >= now() - interval '35 days'
                """
            )
            hk_sleep_days = {r[0] for r in cur.fetchall()}

        # Group overnight samples (21:00..06:30 local) by wake day
        by_night: dict = {}
        all_recent_hrs = []
        for ts, bpm in rows:
            local = ts.astimezone(LOCAL_TZ)
            hm = local.hour * 60 + local.minute
            if local >= datetime.now(LOCAL_TZ) - timedelta(hours=24):
                all_recent_hrs.append(bpm)
            if hm >= 21 * 60 or hm <= 6 * 60 + 30:
                wd = wake_day_local(ts)
                by_night.setdefault(wd, []).append((ts, float(bpm)))

        # Compute per-night cleaned metrics
        night_metrics = []
        for wd in sorted(by_night.keys()):
            pts = by_night[wd]
            # 5-min bins
            buckets: dict[datetime, list[float]] = {}
            for ts, bpm in pts:
                epoch = int(ts.timestamp())
                slot = datetime.fromtimestamp((epoch // 300) * 300, tz=timezone.utc)
                buckets.setdefault(slot, []).append(bpm)

            slots = sorted(buckets.keys())
            bins = []
            for s in slots:
                arr = mad_clean(np.array(buckets[s], dtype=float))
                bins.append((s, float(np.median(arr)), float(np.std(arr)) if len(arr) > 1 else 0.0))

            if len(bins) < 2:
                continue

            # Coverage out of an expected 8h (96 five-min bins) overnight window
            coverage = min(100.0, (len(bins) / 96.0) * 100.0)
            medians = np.array([b[1] for b in bins], dtype=float)
            clean_meds = mad_clean(medians)
            sleep4h = float(np.mean(clean_meds[:48])) if len(clean_meds) >= 6 else float(np.mean(clean_meds))

            # 10-min rolling nadir
            if len(bins) >= 2:
                pair_avgs = [(bins[i][1] + bins[i + 1][1]) / 2.0 for i in range(len(bins) - 1)]
                nadir_idx = int(np.argmin(pair_avgs))
                nadir_hr = float(pair_avgs[nadir_idx])
                nadir_time = bins[nadir_idx][0].astimezone(LOCAL_TZ)
            else:
                nadir_hr = float(np.min(medians))
                nadir_time = bins[0][0].astimezone(LOCAL_TZ)

            night_metrics.append({
                "day": wd,
                "bins": bins,
                "coverage": coverage,
                "sleep4h": sleep4h,
                "nadir": nadir_hr,
                "nadir_time": nadir_time,
            })

            # Write ML sleep segments only if >= 90 min data & no HealthKit sleep for that day
            if len(bins) >= 18 and wd not in hk_sleep_days:
                segs = score_sleep_segments(bins)
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM sleep_segments WHERE source = 'ml_server' AND (end_at AT TIME ZONE 'UTC')::date = %s",
                        (wd,),
                    )
                    for s, e, st in segs:
                        cur.execute(
                            """
                            INSERT INTO sleep_segments (start_at, end_at, stage, source)
                            VALUES (%s, %s, %s, 'ml_server')
                            ON CONFLICT (source, start_at, end_at) DO NOTHING
                            """,
                            (s, e, st),
                        )

        today = datetime.now(LOCAL_TZ).date()

        # Fit Ridge habit attribution once >= 10 valid nights exist
        habit_notes = []
        valid_nights = [m for m in night_metrics if m["coverage"] >= 35.0]
        all_tags = ["Late meal", "Evening workout", "Late caffeine", "Feeling unwell", "Warm room"]
        if len(valid_nights) >= 10:
            X = []
            y = []
            for m in valid_nights:
                c = checkins.get(m["day"], {})
                tags = set(c.get("tags", []))
                X.append([1.0 if t in tags else 0.0 for t in all_tags])
                y.append(m["sleep4h"])
            X_arr = np.array(X)
            y_arr = np.array(y)
            if np.any(X_arr.sum(axis=0) >= 3):
                model = XGBRegressor(n_estimators=40, max_depth=2, learning_rate=0.1, reg_lambda=2.0, verbosity=0).fit(X_arr, y_arr)
                base_vec = np.zeros((1, len(all_tags)), dtype=float)
                base_pred = float(model.predict(base_vec)[0])
                for idx, tag in enumerate(all_tags):
                    cnt = int(X_arr[:, idx].sum())
                    if cnt >= 3:
                        probe = base_vec.copy()
                        probe[0, idx] = 1.0
                        delta = float(model.predict(probe)[0]) - base_pred
                        if abs(delta) >= 0.4:
                            habit_notes.append({
                                "tag": tag,
                                "nights": cnt,
                                "delta_bpm": round(delta, 1),
                            })

        # Build truthful insight for today (explicit about data completeness)
        if not night_metrics:
            n_pts = len(all_recent_hrs)
            avg_hr = round(float(np.mean(all_recent_hrs))) if all_recent_hrs else 0
            headline = "Waiting for First Overnight Session"
            summary = (
                f"Server has {n_pts} recent heart-rate readings (avg {avg_hr} bpm), "
                "but no 9 PM?6 AM overnight sleep window yet. Keep the sensor connected overnight to unlock recovery insights."
                if n_pts > 0
                else "No heart-rate samples in Postgres yet. Connect your BLE sensor or Apple Health to start."
            )
            confidence = "low"
            cov = 0.0
            c_s4 = None
            c_nad = None
            target_day = today
        else:
            latest = night_metrics[-1]
            target_day = latest["day"]
            cov = latest["coverage"]
            c_s4 = round(latest["sleep4h"], 1)
            c_nad = round(latest["nadir"], 1)
            nadir_str = latest["nadir_time"].strftime("%I:%M %p").lstrip("0")

            base_vals = [m["sleep4h"] for m in valid_nights] if valid_nights else [m["sleep4h"] for m in night_metrics]
            mu = float(np.mean(base_vals))
            sd = max(1.5, float(np.std(base_vals, ddof=1)) if len(base_vals) >= 2 else 1.5)
            z = (latest["sleep4h"] - mu) / sd
            diff = latest["sleep4h"] - mu

            if cov < 40.0 or len(valid_nights) < 3:
                confidence = "low"
            elif cov < 75.0 or len(valid_nights) < 7:
                confidence = "medium"
            else:
                confidence = "high"

            chk = checkins.get(target_day, {})
            fatigue = chk.get("fatigue")

            if cov < 35.0:
                headline = "Partial Overnight Data"
                summary = (
                    f"Only {int(round(cov))}% of the overnight window was captured "
                    f"(avg {c_s4:.0f} bpm, low {c_nad:.0f} bpm at {nadir_str}). "
                    "Short captures can skew recovery - keep Bluetooth connected overnight for full accuracy."
                )
            elif fatigue and fatigue >= 4 and z <= -0.5:
                headline = "Hidden Fatigue Signal - Rest Recommended"
                summary = (
                    f"Your sleeping heart rate dropped to {c_s4:.0f} bpm ({diff:+.1f} vs baseline), "
                    f"but you logged high fatigue ({fatigue}/5). Take an easy day to avoid overtraining."
                )
            elif z >= 1.5:
                headline = "Elevated Overnight Strain"
                summary = (
                    f"Sleeping heart rate averaged {c_s4:.0f} bpm ({diff:+.1f} bpm above your {mu:.0f} bpm normal), "
                    f"with overnight low {c_nad:.0f} bpm at {nadir_str}. Prioritize rest and hydration today."
                )
            elif z <= 0.5:
                headline = "Solid Overnight Recovery"
                summary = (
                    f"Sleeping heart rate settled at {c_s4:.0f} bpm with an overnight low of {c_nad:.0f} bpm at {nadir_str} "
                    f"({int(round(cov))}% coverage). Within your normal range ({mu - 0.5 * sd:.0f}?{mu + 0.5 * sd:.0f} bpm)."
                )
            else:
                headline = "Mild Strain - Steady Day"
                summary = (
                    f"Sleeping heart rate was slightly elevated at {c_s4:.0f} bpm ({diff:+.1f} bpm above baseline), "
                    f"low {c_nad:.0f} bpm at {nadir_str}. Normal training is fine; avoid all-out efforts."
                )

            all_5m = [b[1] for m in night_metrics for b in m["bins"]]
            ttm_expected = ttm_forecast_expected(all_5m)
            if ttm_expected is not None:
                resid = round(latest["sleep4h"] - ttm_expected, 1)
                summary += f" Forecast model expected {ttm_expected:.0f} bpm ({resid:+.1f} bpm difference)."
            if len(valid_nights) < 7:
                summary += f" Baseline is still learning ({len(valid_nights)}/7 full nights)."

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO server_insights
                  (day, headline, summary, confidence, coverage_pct, corrected_sleep4h, corrected_nadir, habit_notes, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                ON CONFLICT (day) DO UPDATE SET
                  headline = EXCLUDED.headline,
                  summary = EXCLUDED.summary,
                  confidence = EXCLUDED.confidence,
                  coverage_pct = EXCLUDED.coverage_pct,
                  corrected_sleep4h = EXCLUDED.corrected_sleep4h,
                  corrected_nadir = EXCLUDED.corrected_nadir,
                  habit_notes = EXCLUDED.habit_notes,
                  updated_at = now()
                """,
                (target_day, headline, summary, confidence, round(cov, 1), c_s4, c_nad, json.dumps(habit_notes)),
            )
        conn.commit()
        print(json.dumps({
            "day": str(target_day),
            "confidence": confidence,
            "coverage_pct": round(cov, 1),
            "headline": headline,
            "summary": summary,
        }))


if __name__ == "__main__":
    run()
