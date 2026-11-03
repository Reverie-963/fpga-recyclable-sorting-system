# Profiles, records and service contract

## Frozen profile

`profile` writes this shape using actual files and an explicitly supplied build ID:

```json
{
  "schema": "pynq-accelerator-check-v1",
  "build_id": "build-identifier-from-your-build-record",
  "assets": {
    "bit": {"path": "vision.bit", "sha256": "64 lowercase hexadecimal characters"},
    "hwh": {"path": "vision.hwh", "sha256": "64 lowercase hexadecimal characters"}
  },
  "contract": {
    "image_size": [640, 480],
    "bytes_per_pixel": 4,
    "rpc_contract": "sorting-statistics-v1",
    "modules": [
      {"instance": "dma_mask", "dma": true, "parameters": {"c_include_mm2s": 1, "c_include_s2mm": 1}},
      {"instance": "mask_0", "registers": {"width": "0x10", "height": "0x18", "job_id": "0x30"}}
    ]
  }
}
```

The hash text in this schema illustration is explanatory, not an executable
profile. Generate profiles from real files. Paths resolve relative to the profile
file; module names are HWH FULLNAME without a leading slash (INSTANCE fallback).
HWH PARAMETERS/PARAMETER and REGISTERS/REGISTER/PROPERTY ADDRESS_OFFSET are read.
XML namespaces are supported; DTD/entity declarations are rejected. Decimal and
`0x` offsets are accepted. Parameter names are case insensitive, register names
are exact. An optional exact `vlnv` constrains the IP type.

The built-in `dual-core-contract.json` is for the project's mask/visual kernels,
not the current station RGB/frame-pipeline/motion overlay. Do not interchange them.
For another design, use actual HWH names and expected interface values from its
own source contract. The tool checks only explicitly listed properties. It does
not prove graph connectivity or source-to-bitstream equivalence.

## Binary cases and recordings

Suite shape: `{"schema":"pynq-accelerator-check-v1", "cases":[...]}`.
Case shape:

```json
{
  "name": "two-pixels",
  "role": "geometry",
  "image_size": [2, 2],
  "job_id": 7,
  "masks_rle": [[0, 1, 2, 1]],
  "observed": {
    "stats": {"m00":2,"m10":1,"m01":1,"m20":1,"m02":1,"m11":1,
              "min_row":0,"max_row":1,"min_col":0,"max_col":1,"job_id":7,"status":1},
    "output_sha256": "digest of the four normalized bytes 1,0,0,1"
  },
  "timings_ms": {"reference_ms": 0.4, "service_call_ms": 1.2}
}
```

Generate runnable examples with `demo`; its hashes are actual SHA-256 digests.
`replay` requires suite-level `provenance` = `synthetic` or `recorded-hardware`,
and `observed` for every case. Timings are optional and must be finite/nonnegative.
Timing values remain attributed to the recording; replay does not remeasure them.
When importing RPC journals, take `case`, attach `observed` and `timings_ms` from
the same row, and keep warmup out of the measurement record. Retain the original
journal/manifest separately; copying a row cannot prove its hardware provenance.

`image_size` is `[width,height]`, up to 640x480. RLE is full-image row-major with a
background-first run; foreground first requires an initial zero. It is not COCO
compressed RLE or network bit packing. Sum of runs must equal width*height.
Geometry takes one mask, visual takes two equal-sized masks. Stats are exact
integers (booleans and floats are rejected); status must equal 1, job ID must match.
Bounds are inclusive, empty bounds all zero. Output SHA-256 covers one normalized
uint8 byte (0/1) per pixel. It never hashes uint32 DMA padding or packed bits.

## Existing `sorting-statistics-v1` service

The client uses Bearer authentication, no environment proxy, redirects, discovery,
retry, overlay loading or service management. Endpoint is an http(s) origin without
credentials/query/path. Plain HTTP is for an existing trusted local connection;
the checker does not create a secure transport. Authentication is a token from the
named environment variable; do not bundle tokens or machine endpoints.

`GET /health` returns `contract`, 32-character `session_id`, `faulted:false`, and
`assets:{"vision.bit":<sha>,"vision.hwh":<sha>}`. The names are fixed by this service
protocol even if local filenames differ. Profile hashes must exactly match.

`POST /run` receives `role`, `session_id`, random 32-character `request_id`,
`image_size`, uint32 `job_id`, and `masks`: base64 strings of little-bit-order
packed binary masks with zero padding. This packing is distinct from both RLE and
the uint32 hardware AXIS/DMA format, which is the service driver's responsibility.
The reply returns identical session/assets/request ID plus `stats` and
`output_sha256`. The output digest is checked against the independent reference.

The service does not expose whole-station ownership; the operator/agent must
verify exclusive use through the actual owning controller before setting the
explicit run flags. Health is checked before each case and after the run. A failed
response is terminal; the tool does not infer that the request never reached PL.
The service's own driver may have a recovery policy; review it before invocation.

## Extending the skill

For a new HWH-only design, supply a new contract; no algorithm changes needed.
For another binary geometry/visual core with the same semantics, provide a service
adapter obeying the normalized protocol and use its actual artifact hashes.
For another algorithm, add an independently implemented reference, input validator
and transport adapter. Test hand-calculated positives and negatives before use.
Do not merely return reference values from a mock and call that hardware evidence.
Keep client wall time, driver time and hardware counters separate; document each
counter's clock domain, units and wrap behavior before using kernel-time claims.
