[← Back to README](../README.md)

# Hardware

This document is the **physical parts catalog** for the deltaAnima homelab and the reasoning
behind the purchases. For how compute is *placed and used* across this hardware, see
[infrastructure.md](./infrastructure.md#placement-rationale) — this file is about the parts
themselves, not the deployment.

---

## Main node — inference workhorse

| Component | Spec |
|---|---|
| CPU | Ryzen 9 9950X (dual-CCD) |
| GPU | RTX 5090 + RTX 3090 |
| RAM | 96 GB DDR5 |
| Storage | 4 TB NVMe + 6 TB HDD |
| OS | Windows 11 (WSL2 + Docker) |

### GPU choice — mixed architecture, on a budget

The 5090 is the new, fast card for inference. A second new 5090-class card was hard to justify
on cost, so the second slot is a **used RTX 3090**, which is comparatively cheap on the second-
hand market. This results in a deliberately **mixed-architecture** GPU setup — the cost-driven
choice that, in turn, motivates the per-model GPU assignment described in
[infrastructure.md](./infrastructure.md#placement-rationale).

### Storage layout

The main node carries a **4 TB NVMe** and a **6 TB HDD**, both attached to Windows. They are
**partitioned and the partitions are mounted into the WSL2/Docker container**, so the inference
environment gets fast NVMe for hot paths (model weights, active data) with HDD capacity for bulk
storage — without giving the container the whole host disk.

---

## Subnode — services + dev + observability

| Component | Spec |
|---|---|
| CPU | Ryzen 9 7900X |
| GPU | RTX 3050 + RTX 3090 |
| RAM | 80 GB |
| Hypervisor | ESXi |

The 7900X/80 GB box runs ESXi and hosts the non-inference services (edge, GitLab, dev
environments, Grafana — see [infrastructure.md](./infrastructure.md)).

- **RTX 3090** — GPU capacity for sub-project workloads / future DB-server needs.
- **RTX 3050** — backup/utility card pulled in on demand for **CUDA-kernel or OpenVINO**
  sub-project work; not part of the always-on path.

---

## Travel shuttle — Framework Laptop 13

| Component | Spec |
|---|---|
| Model | Framework Laptop 13 (Ryzen 7 7840U) |

A thin client for **Vancouver ⇄ Korea travel**, used to SSH back into the homelab through the
reverse tunnel (see [infrastructure.md](./infrastructure.md#networking)). The heavy compute
always stays on the homelab nodes; the laptop is just the access terminal. The Framework choice
fits the broader ownership philosophy — repairable, upgradeable hardware the user controls end
to end.

---

## Networking gear

- **Switched LAN**, CAT6e-or-better cabling between nodes to keep inter-node latency low.
- **FortiGate firewall** + a **Raspberry Pi** hosting the reverse SSH tunnel for remote access.

Details on how these are used are in [infrastructure.md](./infrastructure.md#networking).

---

## Why this shape

The homelab is built as a **deliberate infrastructure-learning environment**, not just a
compute box: heterogeneous GPUs to force smart placement, a hypervisor subnode to practice
service isolation, self-owned networking gear to learn the stack end to end, and full
observability with phone alerts to operate it like a real system. The "own everything end to
end" philosophy is the through-line — every layer is something to understand and control rather
than abstract away.
