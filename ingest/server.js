import http from 'node:http';
import crypto from 'node:crypto';
import pg from 'pg';

const PORT = Number(process.env.PORT || 8795);
const HOST = process.env.HOST || '127.0.0.1';
const TOKEN = process.env.HRV_API_TOKEN || '';

if (!TOKEN || TOKEN.length < 32) {
  console.error('HRV_API_TOKEN (>= 32 chars) required');
  process.exit(1);
}
const TOKEN_BUF = Buffer.from(`Bearer ${TOKEN}`, 'utf8');

const pool = new pg.Pool({
  host: process.env.PGHOST || '/var/run/postgresql',
  port: Number(process.env.PGPORT || 5433),
  database: process.env.PGDATABASE || 'hrv',
  user: process.env.PGUSER || 'hrv',
  max: 6,
});

// Simple per-IP rate limit + brute-force ban (5 bad auths -> 15 min ban)
const ipState = new Map();
function clientIP(req) {
  const xff = req.headers['x-forwarded-for'];
  if (typeof xff === 'string' && xff.length > 0) return xff.split(',')[0].trim();
  return req.socket.remoteAddress || 'unknown';
}
function checkRate(ip) {
  const now = Date.now();
  let st = ipState.get(ip);
  if (!st) {
    st = { count: 0, windowStart: now, fails: 0, bannedUntil: 0 };
    ipState.set(ip, st);
  }
  if (now < st.bannedUntil) return false;
  if (now - st.windowStart > 60_000) {
    st.count = 0;
    st.windowStart = now;
  }
  st.count += 1;
  return st.count <= 30;
}
function recordAuthFail(ip) {
  const st = ipState.get(ip);
  if (!st) return;
  st.fails += 1;
  if (st.fails >= 5) {
    st.bannedUntil = Date.now() + 15 * 60_000;
    st.fails = 0;
  }
}
function checkAuth(req) {
  const h = req.headers.authorization;
  if (typeof h !== 'string') return false;
  const buf = Buffer.from(h, 'utf8');
  if (buf.length !== TOKEN_BUF.length) return false;
  return crypto.timingSafeEqual(buf, TOKEN_BUF);
}

function send(res, status, body) {
  const payload = body === null ? '' : JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
    'X-Content-Type-Options': 'nosniff',
    'Cache-Control': 'no-store',
  });
  res.end(payload);
}

function readBody(req, maxBytes = 2 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => {
      size += c.length;
      if (size > maxBytes) {
        reject(new Error('too_large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const ip = clientIP(req);
  if (!checkRate(ip)) return send(res, 429, null);

  // Require Bearer token on EVERY route ? zero unauthenticated info disclosure
  if (!checkAuth(req)) {
    recordAuthFail(ip);
    return send(res, 401, null);
  }

  try {
    const url = new URL(req.url || '/', 'http://localhost');

    // Minimal API surface: ONLY GET /v1/history and POST /v1/ingest
    if (req.method === 'GET' && url.pathname === '/v1/history') {
      const days = Math.min(90, Math.max(1, Number(url.searchParams.get('days') || 35)));
      const [buckets, checkins, sleep, ins] = await Promise.all([
        pool.query(
          `SELECT date_trunc('minute', recorded_at) AS t,
                  round(avg(bpm))::int AS avg_bpm,
                  min(bpm)::int AS min_bpm,
                  max(bpm)::int AS max_bpm
           FROM hr_samples
           WHERE recorded_at >= now() - ($1 || ' days')::interval
             AND (contact IS NULL OR contact = true)
           GROUP BY 1
           ORDER BY 1 ASC`,
          [days]
        ),
        pool.query(
          `SELECT to_char(day, 'YYYY-MM-DD') AS day,
                  fatigue, readiness_subjective, tags, supine_hr, standing_hr, hr_sl
           FROM daily_checkins
           WHERE day >= current_date - $1::int
           ORDER BY day ASC`,
          [days]
        ),
        pool.query(
          `SELECT start_at AS start, end_at AS end, stage, source
           FROM sleep_segments
           WHERE end_at >= now() - ($1 || ' days')::interval
           ORDER BY start_at ASC`,
          [days]
        ),
        pool.query(
          `SELECT to_char(day, 'YYYY-MM-DD') AS day,
                  headline, summary, confidence, coverage_pct,
                  corrected_sleep4h, corrected_nadir, habit_notes
           FROM server_insights
           ORDER BY day DESC
           LIMIT 1`
        ),
      ]);
      return send(res, 200, {
        ok: true,
        days,
        buckets: buckets.rows,
        checkins: checkins.rows,
        sleep: sleep.rows,
        insight: ins.rows[0] || null,
      });
    }

    if (req.method === 'POST' && url.pathname === '/v1/ingest') {
      const raw = await readBody(req);
      const body = JSON.parse(raw || '{}');
      const samples = Array.isArray(body.samples) ? body.samples : [];
      const hrv = Array.isArray(body.hrv) ? body.hrv : [];
      const sleep = Array.isArray(body.sleep) ? body.sleep : [];
      const checkins = Array.isArray(body.checkins) ? body.checkins : [];
      const readiness = body.readiness && typeof body.readiness === 'object' ? body.readiness : null;

      const client = await pool.connect();
      let insertedSamples = 0, insertedHrv = 0, insertedSleep = 0;
      try {
        await client.query('BEGIN');
        for (const s of samples) {
          if (!s || !s.t || !Number.isFinite(Number(s.bpm))) continue;
          const bpm = Math.round(Number(s.bpm));
          if (bpm <= 0 || bpm >= 300) continue;
          const rr = Array.isArray(s.rr_ms) && s.rr_ms.length > 0 ? s.rr_ms.map(Number) : null;
          const r = await client.query(
            `INSERT INTO hr_samples (recorded_at, source, device_name, bpm, rr_ms, contact)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (source, recorded_at) DO NOTHING`,
            [s.t, s.source || 'ble', s.device_name || null, bpm, rr, s.contact ?? null]
          );
          insertedSamples += r.rowCount || 0;
        }
        for (const h of hrv) {
          if (!h || !h.t) continue;
          const r = await client.query(
            `INSERT INTO hrv_samples (recorded_at, source, sdnn_ms, rmssd_ms)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (source, recorded_at) DO NOTHING`,
            [h.t, h.source || 'healthkit', h.sdnn_ms ?? null, h.rmssd_ms ?? null]
          );
          insertedHrv += r.rowCount || 0;
        }
        for (const seg of sleep) {
          if (!seg || !seg.start || !seg.end || !seg.stage) continue;
          const r = await client.query(
            `INSERT INTO sleep_segments (start_at, end_at, stage, source)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT (source, start_at, end_at) DO NOTHING`,
            [seg.start, seg.end, seg.stage, seg.source || 'healthkit']
          );
          insertedSleep += r.rowCount || 0;
        }
        for (const c of checkins) {
          if (!c || !c.day) continue;
          await client.query(
            `INSERT INTO daily_checkins (day, fatigue, readiness_subjective, tags, supine_hr, standing_hr, hr_sl, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, now())
             ON CONFLICT (day) DO UPDATE SET
               fatigue = COALESCE(EXCLUDED.fatigue, daily_checkins.fatigue),
               readiness_subjective = COALESCE(EXCLUDED.readiness_subjective, daily_checkins.readiness_subjective),
               tags = COALESCE(EXCLUDED.tags, daily_checkins.tags),
               supine_hr = COALESCE(EXCLUDED.supine_hr, daily_checkins.supine_hr),
               standing_hr = COALESCE(EXCLUDED.standing_hr, daily_checkins.standing_hr),
               hr_sl = COALESCE(EXCLUDED.hr_sl, daily_checkins.hr_sl),
               updated_at = now()`,
            [c.day, c.fatigue ?? null, c.readiness_subjective ?? null, Array.isArray(c.tags) ? c.tags : null, c.supine_hr ?? null, c.standing_hr ?? null, c.hr_sl ?? null]
          );
        }
        if (readiness && readiness.day && Number.isFinite(Number(readiness.score))) {
          await client.query(
            `INSERT INTO daily_readiness (day, score, hrv_component, rhr_component, sleep_component, basis, updated_at)
             VALUES ($1, $2, $3, $4, $5, $6, now())
             ON CONFLICT (day) DO UPDATE SET
               score = EXCLUDED.score,
               hrv_component = EXCLUDED.hrv_component,
               rhr_component = EXCLUDED.rhr_component,
               sleep_component = EXCLUDED.sleep_component,
               basis = EXCLUDED.basis,
               updated_at = now()`,
            [readiness.day, Math.round(Number(readiness.score)), readiness.hrv_component ?? null, readiness.rhr_component ?? null, readiness.sleep_component ?? null, readiness.basis || null]
          );
        }
        await client.query('COMMIT');
      } catch (err) {
        await client.query('ROLLBACK');
        throw err;
      } finally {
        client.release();
      }
      return send(res, 200, { ok: true, inserted: { samples: insertedSamples, hrv: insertedHrv, sleep: insertedSleep } });
    }

    return send(res, 404, null);
  } catch (err) {
    console.error(err);
    return send(res, 500, null);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`hrv-ingest (hardened) listening on http://${HOST}:${PORT}`);
});
