#!/usr/bin/env python3
"""Portable artifact inspection, binary-kernel replay and bounded RPC checks.

Python 3.11+, standard library only. No PYNQ import, overlay loading, MMIO,
service lifecycle, camera, serial or mechanical control.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid
import xml.etree.ElementTree as ET

SCHEMA = "pynq-accelerator-check-v1"
MAX_JSON = 16 * 1024 * 1024


class CheckError(ValueError):
    pass


def need(condition, code):
    if not condition:
        raise CheckError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    need(len(raw) <= MAX_JSON, "json_too_large")
    return json.loads(raw.decode("utf-8-sig"),
                      parse_constant=lambda _: (_ for _ in ()).throw(CheckError("nonfinite_json")))


def integer(value, minimum, maximum, code):
    need(type(value) is int and minimum <= value <= maximum, code)
    return value


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def hex_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def parse_int(value):
    if isinstance(value, str):
        return int(value, 16 if value.lower().startswith("0x") else 10)
    need(type(value) is int, "invalid_integer")
    return value


def decode_rle(runs, count):
    need(isinstance(runs, list) and 0 < len(runs) <= count * 2 + 1, "invalid_rle")
    need(all(type(n) is int and n >= 0 for n in runs) and sum(runs) == count,
         "invalid_rle_length")
    out = bytearray()
    for i, length in enumerate(runs):
        out.extend(bytes([i % 2]) * length)
    return bytes(out)


def encode_rle(pixels):
    runs, current, length = [], 0, 0
    for value in pixels:
        if value == current:
            length += 1
        else:
            runs.append(length)
            current, length = value, 1
    return runs + [length]


def case_inputs(case):
    need(isinstance(case, dict), "invalid_case")
    need(case.get("role") in ("geometry", "visual"), "unsupported_role")
    size = case.get("image_size")
    need(isinstance(size, list) and len(size) == 2, "invalid_image_size")
    w = integer(size[0], 1, 640, "invalid_width")
    h = integer(size[1], 1, 480, "invalid_height")
    integer(case.get("job_id"), 0, 2**32 - 1, "invalid_job_id")
    masks = case.get("masks_rle")
    need(isinstance(masks, list) and len(masks) == (1 if case["role"] == "geometry" else 2),
         "invalid_mask_count")
    return [decode_rle(runs, w * h) for runs in masks]


def reference(case):
    """Independent integer oracle; inclusive HLS bounds, no morphology."""
    masks = case_inputs(case)
    w = case["image_size"][0]
    if case["role"] == "geometry":
        output = masks[0]
        stats = dict(m00=0, m10=0, m01=0, m20=0, m02=0, m11=0)
        for i, bit in enumerate(output):
            if bit:
                y, x = divmod(i, w)
                stats["m00"] += 1
                stats["m10"] += x
                stats["m01"] += y
                stats["m20"] += x*x
                stats["m02"] += y*y
                stats["m11"] += x*y
    else:
        a, b = masks
        output = bytes(x ^ y for x, y in zip(a, b))
        stats = dict(inter=0, erased=0, beyond=0, union=0)
        for x, y in zip(a, b):
            stats["inter"] += x & y
            stats["erased"] += x & (1-y)
            stats["beyond"] += (1-x) & y
            stats["union"] += x | y
    points = [i for i, bit in enumerate(output) if bit]
    stats.update(min_row=min((i//w for i in points), default=0),
                 max_row=max((i//w for i in points), default=0),
                 min_col=min((i % w for i in points), default=0),
                 max_col=max((i % w for i in points), default=0),
                 job_id=case["job_id"], status=1)
    return dict(stats=stats, output_sha256=sha(output))


def compare(observed, expected):
    need(isinstance(observed, dict), "invalid_observation")
    stats = observed.get("stats")
    need(isinstance(stats, dict), "missing_statistics")
    for key, value in expected["stats"].items():
        need(type(stats.get(key)) is int and stats[key] == value, "statistics_mismatch:" + key)
    need(observed.get("output_sha256") == expected["output_sha256"], "output_hash_mismatch")


def summary(values):
    if not values:
        return None
    need(all(number(v) for v in values), "invalid_timing")
    ordered = sorted(values)
    # Linear interpolation, same convention as numpy.percentile default.
    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return dict(count=len(values), median_ms=statistics.median(values),
                p95_ms=percentile(.95),
                p99_ms=percentile(.99) if len(values) >= 100 else None,
                max_ms=max(values),
                tail_note="P99 omitted below 100 samples; 100 is a reporting guard, not statistical assurance.")


def hwh_modules(path):
    raw = Path(path).read_bytes()
    need(len(raw) <= MAX_JSON and b"<!DOCTYPE" not in raw.upper() and
         b"<!ENTITY" not in raw.upper(), "unsupported_hwh_xml")
    root = ET.fromstring(raw)
    modules = {}
    for node in root.iter():
        node.tag = node.tag.rsplit("}", 1)[-1]
    for module in root.iter("MODULE"):
        name = module.get("FULLNAME", module.get("INSTANCE", "")).strip("/")
        need(name and name not in modules, "duplicate_hwh_module")
        params = {p.get("NAME", "").lower(): p.get("VALUE")
                  for p in module.findall("./PARAMETERS/PARAMETER")}
        registers = {}
        for register in module.iter("REGISTER"):
            # Field-level offsets also use ADDRESS_OFFSET; they are relative to
            # the register and must never overwrite the register's AXI offset.
            for prop in register.findall("./PROPERTY"):
                if prop.get("NAME", "").upper() == "ADDRESS_OFFSET":
                    registers[register.get("NAME")] = parse_int(prop.get("VALUE"))
        modules[name] = dict(parameters=params, registers=registers, vlnv=module.get("VLNV"))
    return modules


def inspect_profile(path):
    path = Path(path).resolve()
    profile = read_json(path)
    need(isinstance(profile, dict) and profile.get("schema") == SCHEMA and isinstance(profile.get("build_id"), str)
         and profile["build_id"].strip(), "invalid_profile")
    assets = profile.get("assets")
    need(isinstance(assets, dict) and set(assets) == {"bit", "hwh"}, "invalid_assets")
    files, hashes = {}, {}
    for role, asset in assets.items():
        need(isinstance(asset, dict) and hex_sha(asset.get("sha256")), "invalid_asset_hash")
        need(isinstance(asset.get("path"), str) and asset["path"], "invalid_asset_path")
        target = (path.parent / asset["path"]).resolve()
        need(target.is_file(), "missing_asset:" + role)
        hashes[role] = file_sha(target)
        need(hashes[role] == asset["sha256"], "asset_hash_mismatch:" + role)
        files[role] = target
    contract = profile.get("contract")
    need(isinstance(contract, dict), "missing_contract")
    size = contract.get("image_size")
    need(isinstance(size, list) and len(size) == 2, "invalid_contract_size")
    w = integer(size[0], 1, 640, "invalid_contract_width")
    h = integer(size[1], 1, 480, "invalid_contract_height")
    bpp = integer(contract.get("bytes_per_pixel"), 1, 16, "invalid_bytes_per_pixel")
    byte_count = w*h*bpp
    modules = hwh_modules(files["hwh"])
    requirements = contract.get("modules")
    need(isinstance(requirements, list) and requirements, "missing_module_requirements")
    checks, instances = [], set()
    for requirement in requirements:
        need(isinstance(requirement, dict) and isinstance(requirement.get("instance"), str),
             "invalid_module_requirement")
        need(isinstance(requirement.get("parameters", {}), dict) and
             isinstance(requirement.get("registers", {}), dict), "invalid_module_fields")
        name = requirement["instance"]
        need(name not in instances, "duplicate_contract_module")
        instances.add(name)
        need(name in modules, "missing_ip:" + name)
        module = modules[name]
        if requirement.get("vlnv"):
            need(module["vlnv"] == requirement["vlnv"], "ip_type_mismatch:" + name)
        for key, value in requirement.get("parameters", {}).items():
            actual = module["parameters"].get(key.lower())
            need(actual is not None and parse_int(actual) == parse_int(value),
                 "parameter_mismatch:" + name + ":" + key)
        for key, value in requirement.get("registers", {}).items():
            need(module["registers"].get(key) == parse_int(value),
                 "register_mismatch:" + name + ":" + key)
        item = dict(instance=name, checked_parameters=requirement.get("parameters", {}),
                    checked_registers=requirement.get("registers", {}))
        if requirement.get("dma") is True:
            bits = parse_int(module["parameters"].get("c_sg_length_width", "0"))
            need(1 <= bits <= 64, "invalid_dma_length_width:" + name)
            need(byte_count <= (1 << bits)-1, "dma_length_exceeded:" + name)
            item.update(dma_length_bits=bits, transfer_bytes=byte_count,
                        maximum_transfer_bytes=(1 << bits)-1)
        checks.append(item)
    return dict(build_id=profile["build_id"], asset_sha256=hashes, checks=checks,
                contract=contract,
                pairing_proof="User-declared common build ID plus frozen hashes; hashes alone do not prove a common build.",
                loaded_overlay_verified=False,
                unchecked=["AXI connectivity and TLAST waveform", "cache synchronization at runtime",
                           "timing closure, physical pins and clocks", "current device ownership"])


def create_profile(bit, hwh, build_id, contract):
    need(isinstance(build_id, str) and build_id.strip(), "missing_build_id")
    need(Path(bit).resolve() != Path(hwh).resolve(), "assets_must_be_distinct")
    return dict(schema=SCHEMA, build_id=build_id,
                assets={role: dict(path=str(Path(p).resolve()), sha256=file_sha(p))
                        for role, p in (("bit", bit), ("hwh", hwh))}, contract=read_json(contract))


def demo_cases():
    cases = []
    for w, h in ((1, 1), (9, 7), (65, 1), (32, 24)):
        patterns = (bytes(w*h), bytes([1])*(w*h), bytes(w*h-1)+b"\x01",
                    bytes(int((i % w + i//w) % 3 == 0) for i in range(w*h)))
        for index, a in enumerate(patterns):
            b = bytes([0]) + a[:-1]
            for role, masks in (("geometry", [a]), ("visual", [a, b])):
                cases.append(dict(name=f"{role}-{w}x{h}-pattern{index}", role=role,
                                  image_size=[w, h], job_id=len(cases)+1,
                                  masks_rle=[encode_rle(m) for m in masks]))
    return cases


def check_suite(value):
    need(isinstance(value, dict) and value.get("schema") == SCHEMA, "invalid_suite")
    cases = value.get("cases")
    need(isinstance(cases, list) and 1 <= len(cases) <= 256, "invalid_case_count")
    names = set()
    for case in cases:
        need(isinstance(case, dict), "invalid_case")
        need(isinstance(case.get("name"), str) and
             re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", case["name"]), "invalid_case_name")
        need(case["name"] not in names, "duplicate_case_name")
        names.add(case["name"])
        case_inputs(case)
    return cases


def replay(recording, result):
    need(isinstance(recording, dict) and recording.get("provenance") in ("synthetic", "recorded-hardware"),
         "invalid_recording_provenance")
    cases = check_suite(recording)
    result["provenance"] = recording["provenance"]
    result["verification_scope"] = "Recorded outputs compared with an independent integer reference; no live hardware access."
    for case in cases:
        row = dict(name=case["name"], role=case["role"], image_size=case["image_size"], passed=False)
        result["records"].append(row)
        try:
            expected = reference(case)
            compare(case.get("observed"), expected)
            timings = case.get("timings_ms", {})
            need(isinstance(timings, dict) and all(number(v) for v in timings.values()), "invalid_timing")
            # Only a full service call is accepted; component names are never silently summed.
            row.update(passed=True, expected=expected, timings_ms=timings)
        except (CheckError, KeyError, TypeError, ValueError) as exc:
            row["error_code"] = str(exc) if isinstance(exc, CheckError) else "invalid_record"
    result["passed"] = all(r["passed"] for r in result["records"])
    result["timing_groups"] = timing_groups(result["records"])


def timing_groups(records):
    groups = {}
    for row in records:
        if not row["passed"] or row.get("warmup"):
            continue
        key = row["role"] + ":" + "x".join(map(str, row["image_size"]))
        groups.setdefault(key, []).append(row)
    return {key: {name: summary([r["timings_ms"][name] for r in rows
                                if name in r.get("timings_ms", {})])
                  for name in ("reference_ms", "packing_ms", "service_call_ms", "validation_ms")}
            for key, rows in groups.items()}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Rpc:
    def __init__(self, endpoint, token, timeout):
        parts = urlsplit(endpoint)
        need(parts.scheme in ("http", "https") and parts.hostname and not parts.username
             and not parts.password and not parts.query and not parts.fragment
             and parts.path in ("", "/"), "invalid_endpoint")
        need(isinstance(token, str) and len(token) >= 32 and "\n" not in token and "\r" not in token,
             "invalid_token")
        need(number(timeout) and 0 < timeout <= 30, "invalid_timeout")
        self.endpoint, self.token, self.timeout = endpoint.rstrip("/"), token, timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, route, body=None):
        need(route in ("health", "run"), "unsupported_route")
        request = urllib.request.Request(self.endpoint + "/" + route,
            data=None if body is None else json.dumps(body, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read(100001)
            need(len(raw) <= 100000, "oversized_rpc_response")
            return json.loads(raw.decode(), parse_constant=lambda _: None)
        except urllib.error.HTTPError as exc:
            raise CheckError("rpc_http_" + str(exc.code)) from None
        except CheckError:
            raise
        except (OSError, ValueError, http.client.HTTPException):
            # Never copy response bodies, URLs, bearer tokens or arbitrary remote errors into reports.
            raise CheckError("rpc_transport_or_json_error") from None


def pack_bits(pixels):
    packed = bytearray((len(pixels)+7)//8)
    for i, value in enumerate(pixels):
        packed[i//8] |= value << (i % 8)
    return base64.b64encode(packed).decode("ascii")


def validate_health(health, assets, session=None):
    need(isinstance(health, dict) and health.get("contract") == "sorting-statistics-v1"
         and health.get("faulted") is False and health.get("assets") == assets,
         "health_contract_assets_or_fault_mismatch")
    need(isinstance(health.get("session_id"), str) and len(health["session_id"]) == 32,
         "invalid_service_session")
    if session is not None:
        need(health["session_id"] == session, "service_session_changed")
    return health["session_id"]


def rpc_run(args, result):
    need(args.allow_run and args.exclusive_device_confirmed, "rpc_requires_run_and_exclusive_confirmation")
    need(args.service_kind in ("hardware", "simulated"), "missing_service_kind")
    inspected = inspect_profile(args.profile)
    result["inspection"] = inspected
    contract = inspected["contract"]
    need(contract.get("rpc_contract") == "sorting-statistics-v1" and
         contract.get("bytes_per_pixel") == 4, "unsupported_rpc_contract")
    need(args.token_env in os.environ, "token_environment_variable_missing")
    client = Rpc(args.endpoint, os.environ[args.token_env], args.timeout)
    suite = read_json(args.cases) if args.cases else dict(schema=SCHEMA, cases=demo_cases())
    cases = check_suite(suite)
    need(len(cases) * (args.repeats + 1) <= 512, "rpc_case_budget_exceeded")
    need(number(args.budget_seconds) and 1 <= args.budget_seconds <= 300, "invalid_time_budget")
    deadline = time.monotonic() + args.budget_seconds
    for case in cases:
        need(all(n <= limit for n, limit in zip(case["image_size"], contract["image_size"])),
             "case_exceeds_inspected_transfer_size")
    assets = {"vision.bit": inspected["asset_sha256"]["bit"],
              "vision.hwh": inspected["asset_sha256"]["hwh"]}
    session = validate_health(client.request("health"), assets)
    result.update(provenance="live-rpc-" + args.service_kind, service_session=session,
                  verification_scope="Client integer reference versus existing service outputs; includes HTTP/PS/DMA/PL round trip, not isolated kernel timing.")
    job = 0
    # One warmup per case; a failure is terminal even during warmup.
    for repetition in range(args.repeats + 1):
        for original in cases:
            job += 1
            case = dict(original, job_id=job)
            row = dict(name=case["name"], role=case["role"], image_size=case["image_size"],
                       warmup=repetition == 0, repetition=repetition, passed=False)
            result["records"].append(row)
            try:
                need(time.monotonic() < deadline, "rpc_time_budget_exceeded")
                validate_health(client.request("health"), assets, session)
                started = time.perf_counter_ns()
                expected = reference(case)
                reference_ms = (time.perf_counter_ns()-started)/1e6
                started = time.perf_counter_ns()
                packed = [pack_bits(p) for p in case_inputs(case)]
                packing_ms = (time.perf_counter_ns()-started)/1e6
                nonce = uuid.uuid4().hex
                body = dict(role=case["role"], session_id=session, request_id=nonce,
                            image_size=case["image_size"], job_id=job, masks=packed)
                need(time.monotonic() < deadline, "rpc_time_budget_exceeded")
                row.update(case=case, expected=expected, request_id=nonce,
                           request_outcome="unknown")
                append_journal(args.output, dict(row, event="before_submission"))
                started = time.perf_counter_ns()
                observed = client.request("run", body)
                service_ms = (time.perf_counter_ns()-started)/1e6
                started = time.perf_counter_ns()
                need(isinstance(observed, dict) and observed.get("assets") == assets and
                     observed.get("session_id") == session and observed.get("request_id") == nonce,
                     "response_identity_mismatch")
                row["observed"] = dict(session_id=session, assets=assets, request_id=nonce,
                    stats={k: v for k, v in observed.get("stats", {}).items()
                           if k in expected["stats"] and type(v) is int}
                          if isinstance(observed.get("stats"), dict) else {},
                    output_sha256=observed.get("output_sha256")
                                  if hex_sha(observed.get("output_sha256")) else "invalid")
                compare(observed, expected)
                validation_ms = (time.perf_counter_ns()-started)/1e6
                # Journal enough raw, correlated data to replay without access to the service.
                row.update(passed=True, request_outcome="validated",
                           timings_ms=dict(reference_ms=reference_ms, packing_ms=packing_ms,
                                           service_call_ms=service_ms, validation_ms=validation_ms))
                append_journal(args.output, row)
            except CheckError as exc:
                row["error_code"] = str(exc)
                append_journal(args.output, row)
                raise
    need(time.monotonic() < deadline, "rpc_time_budget_exceeded")
    validate_health(client.request("health"), assets, session)
    result["passed"] = True
    result["timing_groups"] = timing_groups(result["records"])


def append_journal(output, row):
    with (Path(output) / "measurements.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def recommendations(result):
    advice = []
    codes = [r.get("error_code", "") for r in result["records"]] + [result.get("error_code", "")]
    if any("dma_length" in c for c in codes):
        advice.append("Frame bytes exceed the HWH DMA length field. Reduce the frame or rebuild/revalidate the interface; do not truncate transfers.")
    if any("hash_mismatch" in c or "assets_or_fault" in c for c in codes):
        advice.append("Check the build manifest, bit/HWH pair and active service identity before another run.")
    if any("job_id" in c or "identity" in c or "session" in c for c in codes):
        advice.append("Check stale replies, session replacement and shared ownership. Do not retry the hardware request automatically.")
    if any("statistics_mismatch" in c or "output_hash" in c for c in codes):
        advice.append("Check inclusive bounds, binary normalization, register offsets and pixel order against the actual kernel contract.")
    if any("rpc_" in c for c in codes):
        advice.append("Keep the first failed request and current device state. Inspect service/DMA logs; this tool does not reset or reload hardware.")
    if result.get("provenance") == "synthetic":
        advice.append("Synthetic fixtures validate the checker only; their supplied times are not hardware performance measurements.")
    for key, group in result.get("timing_groups", {}).items():
        ref, call = group.get("reference_ms"), group.get("service_call_ms")
        if ref and call and call["median_ms"] >= ref["median_ms"]:
            advice.append(key + ": complete service call is no faster than this reference. Measure packing, transport and PS/DMA separately before attributing the cause to the PL kernel.")
    advice.append("The built-in pure Python oracle establishes correctness, not an optimized CPU baseline. No general CPU/FPGA speedup claim follows from these timings.")
    return advice


def write_report(output, result):
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    result["recommendations"] = recommendations(result)
    Path(output, "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# PYNQ accelerator check", "", "- Result: " + ("PASS" if result["passed"] else "FAIL"),
             "- Mode: " + result["mode"], "- Evidence: " + result.get("provenance", "offline-files"),
             "- Scope: " + result.get("verification_scope", "File hashes and declared HWH interface only."),
             "- Physical sorting verified: false", "- Hardware execution independently verified: false", ""]
    if result.get("error_code"):
        lines.extend(["Error: `" + result["error_code"] + "`", ""])
    if result.get("inspection"):
        lines.extend(["## Artifact checks", "", result["inspection"]["pairing_proof"], "",
                      "| Asset | SHA-256 |", "| --- | --- |"])
        lines.extend("| " + k + " | " + v + " |" for k, v in result["inspection"]["asset_sha256"].items())
        lines.append("")
    lines.extend(["## Case results", "", "| Case | Warmup | Result | Reason |",
                  "| --- | --- | --- | --- |"])
    lines.extend(f"| {r['name']} | {r.get('warmup', False)} | {'PASS' if r['passed'] else 'FAIL'} | {r.get('error_code', '')} |"
                 for r in result["records"])
    lines.extend(["", "## Timing", "", "Times are milliseconds. Warmup is excluded. Different roles/sizes remain separate.", "",
                  "| Group | Measurement | n | Median | P95 | P99 | Max |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for key, group in result.get("timing_groups", {}).items():
        for name, values in group.items():
            if values:
                p99 = "not reported" if values["p99_ms"] is None else f"{values['p99_ms']:.6f}"
                lines.append(f"| {key} | {name} | {values['count']} | {values['median_ms']:.6f} | {values['p95_ms']:.6f} | {p99} | {values['max_ms']:.6f} |")
    lines.extend(["", "P99 is omitted below 100 samples; a larger sample does not by itself ensure a reliable tail estimate.",
                  "", "## Findings and next checks", ""])
    lines.extend("- " + item for item in result["recommendations"])
    Path(output, "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic success and deliberate failure, offline")
    demo.add_argument("--output", required=True, type=Path)
    inspect = commands.add_parser("inspect", help="Read paired files and HWH without device access")
    inspect.add_argument("--profile", required=True, type=Path)
    inspect.add_argument("--output", required=True, type=Path)
    record = commands.add_parser("replay", help="Compare recorded outputs; never contacts a device")
    record.add_argument("--recording", required=True, type=Path)
    record.add_argument("--output", required=True, type=Path)
    profile = commands.add_parser("profile", help="Freeze declared build identity and existing file hashes")
    for key in ("bit", "hwh", "contract", "output"):
        profile.add_argument("--" + key, type=Path, required=True)
    profile.add_argument("--build-id", required=True)
    rpc = commands.add_parser("rpc", help="Check an existing exclusive statistics service, no overlay loading")
    rpc.add_argument("--profile", required=True, type=Path)
    rpc.add_argument("--endpoint", required=True)
    rpc.add_argument("--token-env", default="PYNQ_CHECK_TOKEN")
    rpc.add_argument("--cases", type=Path)
    rpc.add_argument("--output", required=True, type=Path)
    rpc.add_argument("--timeout", type=float, default=5)
    rpc.add_argument("--repeats", type=int, default=3)
    rpc.add_argument("--budget-seconds", type=float, default=60)
    rpc.add_argument("--service-kind", choices=("hardware", "simulated"), required=True)
    rpc.add_argument("--allow-run", action="store_true")
    rpc.add_argument("--exclusive-device-confirmed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "profile":
        value = create_profile(args.bit, args.hwh, args.build_id, args.contract)
        # 'x' protects existing profiles and their frozen identities.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        print("Profile written; pairing remains a user-declared build identity.")
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema=SCHEMA, mode=args.command, passed=False, records=[],
                  started_at=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), platform=platform.platform()),
                  checker_sha256=file_sha(__file__), physical_sorting_verified=False,
                  hardware_execution_independently_verified=False)
    try:
        if args.command == "inspect":
            result["inspection"] = inspect_profile(args.profile)
            result["passed"] = True
        elif args.command == "replay":
            replay(read_json(args.recording), result)
        elif args.command == "demo":
            cases = demo_cases()
            for case in cases:
                case["observed"] = reference(case)
                case["timings_ms"] = dict(reference_ms=.4, service_call_ms=1.2)
            recording = dict(schema=SCHEMA, provenance="synthetic", cases=cases)
            Path(args.output, "synthetic-recording.json").write_text(json.dumps(recording, indent=2) + "\n", encoding="utf-8")
            replay(recording, result)
            bad = json.loads(json.dumps(recording))
            bad["cases"][0]["observed"]["stats"]["job_id"] += 1
            Path(args.output, "deliberate-failure.json").write_text(json.dumps(bad, indent=2) + "\n", encoding="utf-8")
            Path(args.output, "cases.json").write_text(json.dumps(dict(schema=SCHEMA, cases=demo_cases()), indent=2) + "\n", encoding="utf-8")
        elif args.command == "rpc":
            integer(args.repeats, 1, 10, "invalid_repeats")
            rpc_run(args, result)
    except KeyboardInterrupt:
        result.update(error_code="interrupted_no_retry", passed=False)
    except (CheckError, OSError, ValueError, KeyError, TypeError, ET.ParseError) as exc:
        result.update(error_code=str(exc) if isinstance(exc, CheckError) else "invalid_input_or_io_error",
                      passed=False)
    finally:
        if args.command == "rpc":
            result["timing_groups"] = timing_groups(result["records"])
        write_report(args.output, result)
    print(("PASS" if result["passed"] else "FAIL") + " - see report.md and report.json")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
