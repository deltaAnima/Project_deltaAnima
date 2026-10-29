import argparse
import csv
import html
import io
import json
import os
import pathlib
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import redis

from original_py.export_redis import decode, emotion_record, load_env
from original_py.view_emotions import MARKER, VAD_MARKER, js_literal

HERE = pathlib.Path(__file__).resolve().parent


class EmotionSource:
    """Reads mem:* hashes from Redis, cached for a few seconds so page reloads don't rescan every time."""

    def __init__(self, client, salt, ttl):
        self.client = client
        self.salt = salt
        self.ttl = ttl
        self.lock = threading.Lock()
        self.cached_at = 0.0
        self.records = []

    def get(self):
        with self.lock:
            if time.monotonic() - self.cached_at >= self.ttl:
                self.records = self.fetch()
                self.cached_at = time.monotonic()
            return self.records

    def fetch(self):
        keys = list(self.client.scan_iter(match='mem:*', count=500, _type='hash'))
        pipe = self.client.pipeline(transaction=False)
        for key in keys:
            pipe.hgetall(key)
        hashes = [{decode(k): decode(v) for k, v in h.items()} for h in pipe.execute()]
        records = [emotion_record(h, self.salt) for h in hashes if 'memory.emotion.current.V' in h]
        records.sort(key=lambda r: r['timestamp'] or 0)
        return records


def to_csv(records):
    columns = list(dict.fromkeys(k for r in records for k in r))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()


def find_file(name):
    return next((p for p in (HERE / name, HERE / 'original_py' / name) if p.exists()), None)


def render_viewer(template, records, vad):
    page = template.replace(MARKER, f'window.EMBEDDED_CSV = {js_literal(to_csv(records))}; window.EMBEDDED_NAME = "redis (live)";', 1)
    if vad is not None:
        page = page.replace(VAD_MARKER, f'window.EMBEDDED_VAD = {js_literal(vad)}; window.EMBEDDED_VAD_NAME = "VAD.json";', 1)
    return page


FALLBACK_COLUMNS = ['timestamp', 'user_name', 'persona', 'emotion_term', 'V', 'A', 'D', 'radius', 'user_text', 'persona_text']


def render_fallback(records):
    # Used when viewer.html isn't present: a plain table of the latest records.
    def cell(r, c):
        v = r.get(c)
        if c == 'timestamp' and v:
            v = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v))
        elif isinstance(v, float):
            v = f'{v:.3f}'
        return f'<td>{html.escape(str(v)) if v is not None else ""}</td>'

    head = ''.join(f'<th>{c}</th>' for c in FALLBACK_COLUMNS)
    rows = '\n'.join('<tr>' + ''.join(cell(r, c) for c in FALLBACK_COLUMNS) + '</tr>' for r in reversed(records))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live Emotions</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 16px; background: #fff; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }}
td:nth-child(n+5):nth-child(-n+8) {{ font-variant-numeric: tabular-nums; }}
td:nth-child(9), td:nth-child(10) {{ max-width: 32em; white-space: pre-wrap; }}
</style></head><body>
<h1>Live Emotions</h1>
<p>{len(records)} records, newest first · <a href="/emotions.csv">CSV</a> · <a href="/api/emotions">JSON</a></p>
<table><thead><tr>{head}</tr></thead><tbody>
{rows}
</tbody></table></body></html>'''


def make_handler(source, vad):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path not in ('/', '/api/emotions', '/emotions.csv'):
                self.send_error(404)
                return
            try:
                records = source.get()
            except redis.RedisError as exc:
                self.send_error(502, f'Redis error: {exc}')
                return
            if path == '/api/emotions':
                self.reply(json.dumps(records, ensure_ascii=False), 'application/json')
            elif path == '/emotions.csv':
                self.reply(to_csv(records), 'text/csv')
            else:
                # Re-read viewer.html per request so edits show up without a restart.
                viewer = find_file('viewer.html')
                if viewer:
                    page = render_viewer(viewer.read_text(encoding='utf-8'), records, vad)
                else:
                    page = render_fallback(records)
                self.reply(page, 'text/html')

        def reply(self, text, content_type):
            body = text.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', f'{content_type}; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    load_env(HERE / '.env')
    parser = argparse.ArgumentParser(description='Serve live emotion data from Redis as HTML')
    # The page shows user names and message text, so only listen on localhost unless told otherwise.
    parser.add_argument('--bind', default=os.getenv('SERVER_BIND', '127.0.0.1'))
    parser.add_argument('--listen-port', type=int, default=int(os.getenv('SERVER_PORT', '3786')))
    parser.add_argument('--host', default=os.getenv('REDIS_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('REDIS_PORT', '6379')))
    parser.add_argument('--db', type=int, default=int(os.getenv('REDIS_DB', '0')))
    parser.add_argument('--tls', action='store_true')
    parser.add_argument('--cache-seconds', type=float, default=float(os.getenv('CACHE_SECONDS', '5')))
    args = parser.parse_args()

    client = redis.Redis(host=args.host, port=args.port, db=args.db,
                         username=os.getenv('REDIS_USERNAME') or None,
                         password=os.getenv('REDIS_PASSWORD') or None,
                         ssl=args.tls, socket_timeout=15)
    # Set EXPORT_SALT to keep pseudonyms stable across restarts; otherwise they change each run.
    salt = os.getenv('EXPORT_SALT', '').encode('utf-8') or secrets.token_bytes(32)
    source = EmotionSource(client, salt, args.cache_seconds)

    vad_path = find_file('VAD.json')
    vad = json.loads(vad_path.read_text(encoding='utf-8-sig')) if vad_path else None

    server = ThreadingHTTPServer((args.bind, args.listen_port), make_handler(source, vad))
    print(f'Serving on http://{args.bind}:{args.listen_port} (Redis {args.host}:{args.port}/{args.db})')
    server.serve_forever()


if __name__ == '__main__':
    main()
