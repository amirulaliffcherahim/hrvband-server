-- PostgreSQL time-series schema for HRVBand Server

CREATE TABLE IF NOT EXISTS hr_samples (
    recorded_at TIMESTAMPTZ NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'ble',
    device_name TEXT,
    bpm         SMALLINT    NOT NULL CHECK (bpm > 0 AND bpm < 300),
    rr_ms       REAL[],
    contact     BOOLEAN,
    PRIMARY KEY (source, recorded_at)
);
CREATE INDEX IF NOT EXISTS idx_hr_samples_brin ON hr_samples USING BRIN (recorded_at);

CREATE TABLE IF NOT EXISTS hrv_samples (
    recorded_at TIMESTAMPTZ NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'healthkit',
    sdnn_ms     REAL,
    rmssd_ms    REAL,
    PRIMARY KEY (source, recorded_at)
);
CREATE INDEX IF NOT EXISTS idx_hrv_samples_brin ON hrv_samples USING BRIN (recorded_at);

CREATE TABLE IF NOT EXISTS sleep_segments (
    start_at    TIMESTAMPTZ NOT NULL,
    end_at      TIMESTAMPTZ NOT NULL,
    stage       TEXT        NOT NULL,
    source      TEXT        NOT NULL DEFAULT 'healthkit',
    PRIMARY KEY (source, start_at, end_at)
);
CREATE INDEX IF NOT EXISTS idx_sleep_segments_start ON sleep_segments (start_at DESC);

CREATE TABLE IF NOT EXISTS daily_readiness (
    day             DATE PRIMARY KEY,
    score           SMALLINT NOT NULL,
    hrv_component   REAL,
    rhr_component   REAL,
    sleep_component REAL,
    basis           TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS daily_checkins (
    day                  DATE PRIMARY KEY,
    fatigue              SMALLINT CHECK (fatigue BETWEEN 1 AND 5),
    readiness_subjective SMALLINT CHECK (readiness_subjective BETWEEN 1 AND 5),
    tags                 TEXT[] DEFAULT '{}',
    supine_hr            REAL,
    standing_hr          REAL,
    hr_sl                REAL,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS server_insights (
    id                BIGSERIAL PRIMARY KEY,
    day               DATE UNIQUE NOT NULL,
    headline          TEXT NOT NULL,
    summary           TEXT NOT NULL,
    confidence        TEXT NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),
    coverage_pct      REAL NOT NULL DEFAULT 0,
    corrected_sleep4h REAL,
    corrected_nadir   REAL,
    model_version     TEXT NOT NULL DEFAULT 'hrv-ml-v3-xgboost-ttm-r2',
    habit_notes       JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
