# Infusion Remaining-Volume Monitor

> A nurse films 10–20 seconds of an IV pole and uploads it; the service returns a remaining-volume curve per bag, an estimate of how much longer each bag will last, and a named "this one needs changing" call — and once the nurse confirms the change, the bag-change event is written back to the nursing execution platform.

## What it is, and who it is for

`infusion-volume-monitor` is a resident, headless HTTP API service (FastAPI + uvicorn; the OpenAPI 3.1 contract page lives at `/docs`) that takes over exactly one step of bedside infusion rounds: turning a short clip of an IV pole into a reviewable remaining-volume curve, a nurse-facing pickup call that carries its own reason, and — only after a nurse confirms — a write-back of the bag-change event to the nursing execution platform. Target users are the ward nurse and nursing assistant (film the pole on every 1–2 hour round), the head nurse (work the pickup list and get the reason behind every call), and the information-department engineer (watch the pending write-back queue, replay failed events, plan retention and disk capacity). Bedside rounds today rely on eyeballing the volume and saying it out loud at handover, which leaves nothing to check afterwards and no answer to "how far along is this order right now?" — this service offers one checkable path instead: clip in, curve out, call with a reason, write-back after confirmation.

## Core capabilities

- **A reviewable curve instead of a visual estimate**: one upload returns a cross-frame-consistent area-ratio series per bag, so the nurse reads the trend instead of re-estimating each bag from scratch or reciting it from memory.
- **Millilitres appear only when they are trustworthy**: with no bag-type calibration record, or when the computed volume lands in a wide uncertainty band, the response omits the millilitre field entirely (not `null`) and states the cause — no untrustworthy number is allowed to drive a change alert.
- **Millilitres come from the bag's real shape, not area ratio × capacity**: once the cavity geometry of a bag type is registered, the projected area is back-solved into the true liquid height and integrated over the cavity; the residual of a bag reported as empty is deducted from the nurse's confirmation, and left alone when the evidence is insufficient.
- **The "how much longer" estimate has a source, or no time point is given**: drip rate comes from the infusion order first and falls back to pump telemetry; with neither available, or when the order and the pump disagree by more than 20%, no time point is returned and that bag's pickup call is suppressed — both values and the discrepancy are handed over for a human to settle, and no midpoint is invented.
- **Fewer false calls, no wasted walks**: pickups pass three gates (hysteresis band, 15-minute minimum interval per bag, at most 3 bags per round truncated by ETA). Bags that were not called still carry a reason code, so the head nurse can answer "why was this one not flagged"; a single-frame glitch never raises an alert.
- **Occlusion loses no bag, and a swapped bag is never mixed up**: an occluded stretch is marked as occlusion rather than dropped and rejoins the same bag; after too long a gap the bag is marked "not found" and is only re-anchored when position, hook point and colour all agree at once — otherwise the service plainly states that a nurse must claim it. Whether the next round shows the same bag is decided by four pieces of evidence (colour, liquid level, label region, hook geometry), with a liquid column *higher* than the previous round vetoing the match outright.
- **Only a nurse confirmation counts**: bag-change state advances only when a nurse confirms on the terminal — the service advises, curves and writes events. The pickup list is read-only and never produces a write-back however often it is queried; confirming the same change twice yields one event row and one outbound message. Write-back happens only after confirmation, and never gets lost: the event carries an idempotency key, and 5xx/timeout failures are retried with class-based exponential backoff (≤2 attempts) before landing in the pending write-back list for manual replay, with how long it has been owed, how many attempts were made and what the last error was all written to disk.
- **Fallbacks for bad footage and broken networks**: a pre-upload check scores quality (sharpness, brightness, glare, resolution, skew) and returns one instruction the nurse can act on immediately — degrade, never reject — so a bad clip still returns a curve and a millilitre figure, flagged as doubtful. An upload cut off mid-way can be opened once and resumed chunk by frame; the service reports which frames it already has and which are missing, and the final commit runs the same computation path, so the resulting curve is point-for-point identical to a one-shot upload.

## Quick start

```bash
cd infusion-volume-monitor
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env          # fill in the credential slots; nothing is hard-coded
uvicorn infusion_monitor.api:app --host 0.0.0.0 --port 8000
```

This product ships **no graphical interface** — the contract page at <http://127.0.0.1:8000/docs> is the entry point. Upload a frame sequence and read the pickup list:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/infusion/tracks -F rack_id=R-07A -F infusion_order_no=<EXAMPLE_ORDER> \
     -F bed_no=07-12 -F captured_at=2026-01-01T09:00:00 -F slots='[{"bag_seq":"A03","slot_index":3,"bag_type_code":"NS1000"}]' -F frames=@rack_01.zip
curl -sS http://127.0.0.1:8000/api/v1/infusion/pickups   # read-only pickup list
```

Required environment variables (read from the environment, never hard-coded): `NURSE_EXEC_BASE_URL`, `NURSE_EXEC_API_TOKEN`, `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`. The service validates them by name at startup and names the missing one. The full configuration table, the complete endpoint list, the error/degradation table, the acceptance checklist and the architecture invariants are in the Chinese README.

## Repository layout

```
infusion-volume-monitor/
├── src/infusion_monitor/   HTTP API, ingest, core tracking kernel, calibration, identity & state machine, outbound adapters, model gateway, storage
├── tests/ scripts/         contract, end-to-end and stub fixtures; config self-check, retention cleanup, storage estimate, screenshot capture
├── docs/                   layered documentation, visual target card, real run screenshots
├── var/                    independent verification scripts (real process + real HTTP + real SQLite)
└── README.md INSTALL.md requirements.txt .env.example Dockerfile   full docs (Chinese), ward-side install, dependencies, credential sample, image
```

---

Full documentation: see README.md (Chinese)

---

## Follow Us

Scan the QR code to follow our official account for updates and community access:

![Follow Us](qrcode.jpg)