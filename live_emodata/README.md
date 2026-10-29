# live_emo_DATA

A small web server that reads emotion records (`mem:*` hashes) from Redis and shows them live in the browser.
Each record has VAD (Valence, Arousal, Dominance) values, emotion terms, and the conversation text.

## Layout

| File | Description |
| --- | --- |
| `server.py` | HTTP server. Queries Redis and serves the data as HTML, CSV, or JSON |
| `viewer.html` | Emotion parameter viewer page. The server injects the data into it on each request |
| `original_py/export_redis.py` | Redis value decoding, emotion record extraction, ID pseudonymization, `.env` loader |
| `original_py/view_emotions.py` | Helpers for injecting data into the viewer template |
| `VAD.json` | (Optional) Per-term VAD lexicon. Embedded in the viewer when present |

## Install

Requires Python 3.12. Development used the conda environment `Emo_HTML`.

```bash
conda create -n Emo_HTML python=3.12
conda activate Emo_HTML
pip install redis
```

`redis` is the only external dependency.

## Configuration

Create a `.env` file in the project root. It is not committed. Environment variables that are already set take precedence over `.env`.

```dotenv
REDIS_HOST=127.0.0.1
REDIS_PORT=8080
REDIS_DB=0
REDIS_USERNAME=
REDIS_PASSWORD=

SERVER_BIND=127.0.0.1
SERVER_PORT=8080
CACHE_SECONDS=5

# If empty, pseudonyms change every time the server restarts
EXPORT_SALT=
```

## Run

```bash
python server.py
# Serving on http://127.0.0.1:8080 (Redis 127.0.0.1:8080/0)
```

Command-line options override the `.env` values.

```bash
python server.py --host redis.example.com --port 6380 --tls --cache-seconds 10
```

| Option | Environment variable | Default |
| --- | --- | --- |
| `--bind` | `SERVER_BIND` | `127.0.0.1` |
| `--listen-port` | `SERVER_PORT` | `3786` |
| `--host` | `REDIS_HOST` | `127.0.0.1` |
| `--port` | `REDIS_PORT` | `6379` |
| `--db` | `REDIS_DB` | `0` |
| `--tls` | — | off |
| `--cache-seconds` | `CACHE_SECONDS` | `5` |

## Endpoints

| Path | Response |
| --- | --- |
| `/` | Viewer page. Falls back to a plain table if `viewer.html` is missing |
| `/emotions.csv` | All records as CSV |
| `/api/emotions` | All records as JSON |

Redis results are cached for `CACHE_SECONDS`. `viewer.html` is re-read on every request, so edits show up without restarting the server.

## Privacy

- Discord user, session, and channel IDs are replaced with 12-character pseudonyms (HMAC-SHA256 keyed with `EXPORT_SALT`). Records from the same user share a pseudonym, but the original IDs cannot be recovered without the salt.
- **User names and message text are shown as-is.** For that reason the server listens only on `127.0.0.1` by default. If you expose it (e.g. `--bind 0.0.0.0`), put access control in front of it.
