import argparse
import base64
import csv
import hashlib
import hmac
import json
import os
import secrets

import redis


def decode(value):
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8')
        except UnicodeDecodeError:
            return {'__base64__': base64.b64encode(value).decode('ascii')}
    return value


def export_value(client, key, kind):
    if kind == 'string':
        return decode(client.get(key))
    if kind == 'hash':
        return {decode(k): decode(v) for k, v in client.hgetall(key).items()}
    if kind == 'list':
        return [decode(v) for v in client.lrange(key, 0, -1)]
    if kind == 'set':
        return [decode(v) for v in client.smembers(key)]
    if kind == 'zset':
        return [{'member': decode(v), 'score': score} for v, score in client.zrange(key, 0, -1, withscores=True)]
    if kind == 'stream':
        return [{'id': decode(entry_id), 'fields': {decode(k): decode(v) for k, v in fields.items()}} for entry_id, fields in client.xrange(key)]
    if kind in ('ReJSON-RL', 'json'):
        return client.json().get(key)
    return {'__unsupported_type__': kind}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pseudonym(salt, value):
    # Salted HMAC: records from the same user/session stay groupable, but the
    # original Discord IDs cannot be recovered or brute-forced without the salt.
    if not value:
        return None
    return hmac.new(salt, str(value).encode('utf-8'), hashlib.sha256).hexdigest()[:12]


def flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f'{prefix}.{k}' if prefix else k, out)
    else:
        out[prefix] = obj


def emotion_record(fields, salt):
    """Keep emotion fields with names and message text; drop raw Discord IDs."""
    try:
        terms = json.loads(fields.get('memory.emotion.emotion_terms') or '[]')
    except json.JSONDecodeError:
        terms = []
    rec = {
        'user': pseudonym(salt, fields.get('memory.user.user_key')),
        'session': pseudonym(salt, fields.get('metadata.session_id')),
        # Emotion state carries over per channel, so keep a pseudonymous channel to group by.
        'scope': pseudonym(salt, fields.get('metadata.scope_key')),
        'timestamp': int(to_float(fields.get('metadata.timestamp')) or 0) or None,
        'user_name': fields.get('memory.user.user_name'),
        'persona': fields.get('memory.persona.persona_name'),
        # user_content / persona_content duplicate these two in every record.
        'user_text': fields.get('memory.content.user_input'),
        'persona_text': fields.get('memory.content.model_response'),
        'V': to_float(fields.get('memory.emotion.current.V')),
        'A': to_float(fields.get('memory.emotion.current.A')),
        'D': to_float(fields.get('memory.emotion.current.D')),
        'radius': to_float(fields.get('memory.emotion.current.radius')),
        'similarity': to_float(fields.get('memory.emotion.similarity')),
        'importance': to_float(fields.get('metadata.importance')),
        'emotion_term': terms[0] if terms else None,
        'emotion_terms': '|'.join(terms),
    }
    try:
        analysis = json.loads(fields.get('emotion_analysis.deltaEGO_analysis') or '{}').get('analysis', {})
    except json.JSONDecodeError:
        analysis = {}
    flatten(analysis, '', rec)
    return rec


def write_emotions(records, output):
    records.sort(key=lambda r: r['timestamp'] or 0)
    if output.lower().endswith('.csv'):
        columns = list(dict.fromkeys(k for r in records for k in r))
        with open(output, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    print(f'Exported {len(records)} emotion records to {output}')


def load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')):
    # Minimal .env reader so python-dotenv isn't required; real env vars take precedence.
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip().removeprefix('export ').strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
                value = value[1:-1]
            os.environ.setdefault(key, value)


def main():
    load_env()
    parser = argparse.ArgumentParser(description='Export one Redis logical database to JSON')
    parser.add_argument('--host', default=os.getenv('REDIS_HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.getenv('REDIS_PORT', '6379')))
    parser.add_argument('--db', type=int, default=int(os.getenv('REDIS_DB', '0')))
    parser.add_argument('--output', default='redis_export.json')
    parser.add_argument('--tls', action='store_true')
    parser.add_argument('--emotion-only', action='store_true',
                        help='Export emotion fields, names and message text without raw Discord IDs (.csv or .json by --output extension)')
    parser.add_argument('--from-json', help='With --emotion-only: read an existing raw export instead of Redis')
    args = parser.parse_args()

    if args.emotion_only:
        # Set EXPORT_SALT to keep pseudonyms stable across exports; otherwise a random one is used.
        salt = os.getenv('EXPORT_SALT', '').encode('utf-8') or secrets.token_bytes(32)
        if args.from_json:
            with open(args.from_json, encoding='utf-8') as f:
                raw = json.load(f)
            hashes = [e['value'] for e in raw.values() if e.get('type') == 'hash' and isinstance(e.get('value'), dict)]
        else:
            hashes = None
        if hashes is None:
            client = redis.Redis(host=args.host, port=args.port, db=args.db,
                                 username=os.getenv('REDIS_USERNAME') or None,
                                 password=os.getenv('REDIS_PASSWORD') or None,
                                 ssl=args.tls, socket_timeout=15)
            hashes = []
            for key in client.scan_iter(match='mem:*', count=500, _type='hash'):
                hashes.append({decode(k): decode(v) for k, v in client.hgetall(key).items()})
        records = [emotion_record(h, salt) for h in hashes if 'memory.emotion.current.V' in h]
        write_emotions(records, args.output)
        return

    client = redis.Redis(host=args.host, port=args.port, db=args.db,
                         username=os.getenv('REDIS_USERNAME') or None,
                         password=os.getenv('REDIS_PASSWORD') or None,
                         ssl=args.tls, socket_timeout=15)
    data = {}
    for key in client.scan_iter(count=500):
        kind = decode(client.type(key))
        try:
            data[decode(key)] = {'type': kind, 'ttl_seconds': client.ttl(key), 'value': export_value(client, key, kind)}
        except redis.ResponseError as exc:
            # Keys can expire or change type while an export is running.
            data[decode(key)] = {'type': kind, 'error': str(exc)}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'Exported {len(data)} keys to {args.output}')


if __name__ == '__main__':
    main()
