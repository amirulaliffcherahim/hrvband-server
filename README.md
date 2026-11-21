# hrvband-server

Self-hosted, privacy-first backend for [hrvband-client](https://github.com/amirulaliffcherahim/hrvband-client):
- **Minimal-surface Node.js ingest API** (`ingest/server.js`): ONLY two authenticated endpoints (`POST /v1/ingest`, `GET /v1/history`), constant-time Bearer token comparison, per-IP rate limiting + brute-force ban, binds to `127.0.0.1:8795` behind a Cloudflare Tunnel (zero open inbound ports).
- **PostgreSQL time-series schema** (`db/schema.sql`): BRIN-indexed `hr_samples`, `hrv_samples`, `sleep_segments`, `daily_readiness`, `daily_checkins`, and `server_insights`.
- **Wall-powered Python ML & truthful-insight worker** (`ml/worker.py`):
  - MAD outlier filtering + overnight coverage audit
  - **IBM Granite TinyTimeMixer (`granite-timeseries-ttm-r2`)** (805k params, offline local CPU) for overnight BPM curve forecasting & recovery residuals
  - **XGBoost (`XGBRegressor`)** for personalized N-of-1 habit attribution once ≥ 10 valid nights + tags accumulate
  - Circadian BPM sleep-stage estimator when Apple Health sleep stages are absent

## Setup

1. Create Postgres database & schema:
   ```bash
   createdb hrv
   psql -d hrv -f db/schema.sql
   ```
2. Copy `.env.example` to `~/.config/hrv-ingest.env` (`chmod 600`) and generate a 48-char hex `HRV_API_TOKEN`.
3. Install & run the ingest API:
   ```bash
   cd ingest && npm install --omit=dev
   node server.js
   ```
4. Expose `http://localhost:8795` via Cloudflare Tunnel (HTTPS TLS, zero router port-forwarding).
5. Install & run the ML worker on a 30-min `systemd --user` timer (`ml/requirements.txt` + `ml/worker.py`).
