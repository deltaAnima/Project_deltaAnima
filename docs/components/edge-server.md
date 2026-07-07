[← Back to components](./README.md) · [← Back to root](../../README.md)

# delta-edge — routing plane

`delta-edge` is the Rust WebSocket hub at the center of the system. Every other service — the
Unity client, STT, the orchestrator, TTS — connects to the edge and is routed through it. The
edge holds no application logic; it is a **role-aware message router** built on `tokio` and
`tokio-tungstenite`.

Its job: accept connections, make each one declare a role, and forward messages to the right
role based on a static routing table (or an explicit target). See
[architecture.md](../architecture.md) for where it sits in a turn.

---

## Connection lifecycle

Each connection is handled on its own `tokio` task and goes through a strict sequence:

1. **Handshake.** The TCP stream is upgraded to WebSocket (`accept_async`).
2. **Registration (required first).** The peer must send a `register` message naming its `Role`
   within **5 seconds**, or it's dropped. This is enforced in `wait_for_registration` before any
   routing is possible — an unregistered peer can't send traffic into the hub.
3. **Peer bookkeeping.** On successful registration the peer gets a UUID and is stored under its
   role; an ack (`registered`) is sent back.
4. **Two concurrent halves.** An **outbound task** drains an mpsc channel into the WebSocket sink,
   while the **inbound loop** reads frames and routes them. The channel decouples "who wants to
   send to this peer" from the peer's actual socket.
5. **Cleanup.** On close/disconnect the peer is removed from its role bucket (and the bucket is
   dropped if empty), and the outbound task is aborted.

---

## Roles

Four roles are defined; every connection is exactly one:

| Role | Who |
|---|---|
| `Client` | Unity frontend |
| `Stt` | STT server |
| `Orchestrator` | Main inference |
| `Tts` | TTS server |

The hub stores `role → Vec<PeerHandle>`, i.e. **multiple peers per role** are supported (e.g.
more than one client). A `PeerHandle` is just an id, a role, and the mpsc sender — sending to a
role fans out to every peer registered under it.

Concurrency is handled with a `DashMap` (sharded concurrent map) behind an `Arc`, so the hub is
cloned cheaply into each connection task without a global lock.

---

## Routing

Routing is **static and role-based** — the edge decides destinations from a fixed table, not
from application state. There are two layers.

### Explicit routing (`to` field)

If an incoming envelope names a `to` role, the edge forwards straight to it. This is the escape
hatch for direct addressing.

### Implicit routing (routing table)

Otherwise the edge resolves the destination from `(sender_role, msg_type)`. The table is the
single source of truth for how a turn flows:

| From | Message | → To |
|---|---|---|
| `Client` | `audio_input` | `Stt` |
| `Client` | `user_input` | `Orchestrator` |
| `Stt` | `stt_result` | `Orchestrator` |
| `Orchestrator` | `tts_request` | `Tts` |
| `Orchestrator` | `inference_result` | `Client` |
| `Tts` | `tts_start` | `Client` |
| `Tts` | `tts_done` | `Client` |

Anything not in the table is logged as "no route" and dropped, rather than guessed.

### Binary routing

Audio moves as **binary** frames, routed by sender role alone (no msg_type needed):

| From | → To |
|---|---|
| `Client` (raw mic audio) | `Stt` |
| `Tts` (audio chunks) | `Client` |

Splitting text (control/metadata as JSON) from binary (audio payloads) keeps the hot audio path
free of per-message parsing — the edge just looks at the sender and forwards the bytes.

---

## Message protocol

Inbound messages are JSON envelopes:

```jsonc
{
  "type": "user_input",     // msg_type — drives implicit routing
  "role": "client",         // only on the initial register message
  "to":   "orchestrator",   // optional — forces explicit routing
  "payload": { ... }        // forwarded as-is
}
```

Forwarded messages are re-wrapped with a `from` field so the receiver knows the origin:

```jsonc
{ "type": "...", "from": "stt", "payload": { ... } }
```

The edge treats `payload` as opaque — it never inspects or transforms it. This is deliberate: the
routing plane stays ignorant of application semantics, so adding a new message type is a routing-
table entry, not a code change to the hub.

---

## Design notes

- **Why Rust:** the edge multiplexes many concurrent WebSocket connections, where concurrency
  correctness matters most. See [design-decisions ADR-06](../design-decisions.md).
- **Registration-first** means the hub never has to guess who a socket is; role is known before
  the first byte of traffic is routed.
- **Static routing** keeps the edge simple and predictable — it is infrastructure, not logic. All
  intelligence lives in the orchestrator; the edge only moves bytes to the right place.
- **Channel-per-peer** (mpsc) decouples senders from sockets, so a slow or closed socket can't
  block the router — a failed send is logged and the peer cleaned up.

---

## Related

- Turn flow: [architecture.md](../architecture.md#request-lifecycle)
- Deployment / node placement: [infrastructure.md](../infrastructure.md)
- Orchestrator (the main routing counterpart): [orchestrator.md](./orchestrator.md)