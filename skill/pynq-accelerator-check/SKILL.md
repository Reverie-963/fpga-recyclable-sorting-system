---
name: pynq-accelerator-check
description: Inspect paired PYNQ bit/HWH artifacts, validate binary image accelerator outputs against an independent CPU reference, and diagnose full-call latency using offline recordings or an existing statistics service. Use for DMA length, stale job, mask geometry, output mismatch, or CPU/FPGA comparison questions. Does not build hardware or control a robot.
---

# PYNQ accelerator check

Use the bundled `scripts/accelerator_check.py` with Python 3.11 or newer. It is
self-contained and needs only the standard library. Resolve paths relative to
this skill, not to a particular checkout. Read [contracts](references/contracts.md)
when preparing hardware profiles, importing recordings, or adapting another core.

## Choose the evidence route

- No device: run `demo --output <new-directory>` to exercise the tool; then run
  `replay --recording <recording.json> --output <new-directory>` on actual records.
  The demo also writes `deliberate-failure.json`; replay must fail on its stale job.
- Existing bit/HWH files: prepare a profile using `profile --bit <file> --hwh <file>
  --build-id <actual-build-id> --contract <contract.json> --output <new-profile.json>`,
  then `inspect --profile <profile.json> --output <new-directory>`. Confirm the
  build identity from the owning build record. Hashing two files does not establish
  that they came from one build, or that either file is loaded on a device.
- A running compatible service: use `rpc` only after verifying that the service
  owns the intended hardware exclusively and no camera/controller/other user is
  using that overlay. Reuse authorization already covering this check. Do not stop
  another service or load an overlay merely to make this diagnostic run succeed.
  Use `--allow-run --exclusive-device-confirmed --service-kind hardware`, the
  frozen profile, endpoint and a token supplied via an environment variable.
  Mark local test doubles as `--service-kind simulated`.

For a quick invocation:

```text
python scripts/accelerator_check.py demo --output check-demo
python scripts/accelerator_check.py replay --recording check-demo/deliberate-failure.json --output check-failure
```

Full commands, reuse validation and current limitations are in [README.md](README.md).

## Interpret and respond

Inspect `report.json` and `report.md`, including failure cases, before concluding.
For RPC runs, retain `measurements.jsonl` even on failure. Report exit status:
0 means all selected checks passed; 1 means a failed check or interruption; 2 is
an argument error. Output locations must be new, preserving previous evidence.

Explain which properties were actually checked: declared artifacts, HWH fields,
CPU-versus-recorded/service statistics, and/or normalized output hashes. Supported
algorithms are binary geometry without morphology and binary before/after set
comparison. Bounds are inclusive pixel indices; empty outputs have zero bounds.
Neither output masks nor regional change proves material classification or a grasp.

Separate reference computation, packing, service round trip and validation times.
The service call includes HTTP, PS work, DMA and PL; it is not kernel-only time.
The portable Python oracle is for correctness, not an optimized CPU performance
baseline. Compare equal roles, sizes and execution scopes; do not infer causes
from aggregate time alone. Keep warmup out of measured groups and omit P99 for
fewer than 100 samples. Larger samples still need an appropriate experiment.

Map errors to concrete next checks rather than inventing a root cause:

- `dma_length_exceeded`: compare `width * height * bytes_per_pixel` with the actual
  HWH length field. Never truncate a transfer to pass a check.
- `statistics_mismatch:job_id` / identity or session mismatch: investigate stale
  results, restarted services and ownership before another request.
- Statistics/output mismatch: check binary normalization, inclusive bounds,
  register offsets, image order, and the reference algorithm's scope.
- RPC transport failure: record the unknown outcome and stop. The tool makes no
  automatic retry, device reset, recovery, overlay load or firmware change.

The service's identity is a remote declaration, not independent proof of physical
PL execution. Never promote replay or simulated RPC to board acceptance. State
hardware, timing, cross-board and physical-task checks that remain unperformed.
Do not copy credentials or machine connection settings into reports or packages.
