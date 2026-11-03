# Validation snapshot - 2026-10-31

This is a dated validation statement, not a live device status or a hardware
acceptance certificate. Test commands are provided in README.md for reproduction.

- 44 behavioral tests, standard library only: hand-computed geometry and visual
  results; RLE/shape validation; actual DMA length field; missing IP/direction/
  register/hash errors; hierarchical renamed and namespaced HWH; nested field
  offsets; simulated HTTP identity, warmup, budget, dropped reply and no retry;
  redirect refusal; error reports and previous-evidence preservation.
- 40 additional comparisons against the project's existing independently coded
  CPU reference across both roles and five sizes, including 640x480.
- Local real paired bit/HWH files match the 2026-10-27 acceptance manifest. Their
  hashes and declared register/DMA interfaces pass the new file inspector.
- Package copied outside the repository and run with a separate Python runtime,
  `-I -S` (isolated mode, no site-packages). Standard-library tests, synthetic
  demo, good replay, deliberate stale-job rejection and actual-file inspection
  pass. This is same-host portability evidence, not a second physical machine.

Actual local artifact identity:

| Artifact | SHA-256 |
| --- | --- |
| vision.bit | 0015a6eabf7d8490ffa4873385b39cfb77cee85d88782749d2a59eebf4bf2039 |
| vision.hwh | bcfe6c53e1bf3c4c85cebce993ab2831b70cf075503199e719bec12adba043ae |

No new hardware build, overlay load, hardware RPC execution, pin/clock change,
camera, serial, mechanical motion, classification accuracy experiment or
cross-board verification was performed for this skill. Existing board acceptance
results describe their own historical source/driver, not this new checker.
The current station has ongoing commissioning; this task does not claim its
overlay or execution services. A future exclusive board check and another
operator/environment's reuse record remain useful competition evidence.
