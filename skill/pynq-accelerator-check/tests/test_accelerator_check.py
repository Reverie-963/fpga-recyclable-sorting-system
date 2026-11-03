"""Portable behavioral tests; all device responses here are explicitly simulated."""
import argparse
import base64
from copy import deepcopy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import accelerator_check as check


def hand_case(role="geometry"):
    case = dict(name="hand-2x2-" + role, role=role, image_size=[2, 2], job_id=7,
                masks_rle=[[0, 1, 2, 1]])
    if role == "visual":
        case["masks_rle"].append([1, 1, 1, 1])
    return case


def hand_result(role="geometry", job=7):
    if role == "geometry":
        stats = dict(m00=2, m10=1, m01=1, m20=1, m02=1, m11=1,
                     min_row=0, max_row=1, min_col=0, max_col=1)
        output = b"\x01\x00\x00\x01"
    else:
        stats = dict(inter=1, erased=1, beyond=1, union=3,
                     min_row=0, max_row=0, min_col=0, max_col=1)
        output = b"\x01\x01\x00\x00"
    stats.update(job_id=job, status=1)
    return dict(stats=stats, output_sha256=hashlib.sha256(output).hexdigest())


def fixture(directory, bits=26, namespace=False, renamed=False):
    directory = Path(directory)
    bit, hwh, contract_path = (directory / p for p in ("test-payload.bin", "test.hwh", "contract.json"))
    bit.write_bytes(b"SYNTHETIC TEST PAYLOAD - NOT A BITSTREAM")
    dma = "image_pipeline/stream_dma" if renamed else "dma_mask"
    core = "image_pipeline/geometry_accel" if renamed else "mask_0"
    xmlns = ' xmlns="urn:test"' if namespace else ""
    hwh.write_text(f'''<SYSTEM{xmlns}><MODULES>
    <MODULE FULLNAME="/{dma}" INSTANCE="stream_dma" VLNV="test:dma:1">
      <PARAMETERS><PARAMETER NAME="C_SG_LENGTH_WIDTH" VALUE="{bits}"/>
      <PARAMETER NAME="C_INCLUDE_MM2S" VALUE="1"/>
      <PARAMETER NAME="C_INCLUDE_S2MM" VALUE="1"/></PARAMETERS></MODULE>
    <MODULE FULLNAME="/{core}" INSTANCE="geometry_accel" VLNV="test:geometry:1">
      <REGISTERS><REGISTER NAME="width"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x10"/>
      <FIELDS><FIELD NAME="width"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0"/></FIELD></FIELDS></REGISTER>
      <REGISTER NAME="height"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x18"/></REGISTER>
      <REGISTER NAME="job_id"><PROPERTY NAME="ADDRESS_OFFSET" VALUE="0x30"/></REGISTER></REGISTERS>
    </MODULE></MODULES></SYSTEM>''', encoding="utf-8")
    contract = dict(image_size=[2, 2], bytes_per_pixel=4, rpc_contract="sorting-statistics-v1",
                    modules=[dict(instance=dma, dma=True, parameters=dict(c_include_mm2s=1, c_include_s2mm=1)),
                             dict(instance=core, registers=dict(width="0x10", height="0x18", job_id="0x30"))])
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    profile = check.create_profile(bit, hwh, "synthetic-test-build", contract_path)
    profile_path = directory / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path


class ReferenceTests(unittest.TestCase):
    def test_geometry_independent_hand_computation(self):
        self.assertEqual(check.reference(hand_case()), hand_result())

    def test_visual_independent_hand_computation(self):
        self.assertEqual(check.reference(hand_case("visual")), hand_result("visual"))

    def test_empty_has_zero_bounds(self):
        case = dict(hand_case(), masks_rle=[[4]])
        stats = check.reference(case)["stats"]
        self.assertTrue(all(v == 0 for k, v in stats.items() if k not in ("job_id", "status")))

    def test_last_pixel_inclusive_bounds(self):
        stats = check.reference(dict(hand_case(), image_size=[65, 1], masks_rle=[[64, 1]]))["stats"]
        self.assertEqual((stats["min_col"], stats["max_col"], stats["m20"]), (64, 64, 4096))

    def test_maximum_supported_frame(self):
        case = dict(hand_case(), image_size=[640, 480], masks_rle=[[307199, 1]])
        stats = check.reference(case)["stats"]
        self.assertEqual((stats["m00"], stats["max_col"], stats["max_row"]), (1, 639, 479))

    def test_rle_validation(self):
        for runs in ([3], [5], [-1, 5], [True, 3], [], "bad"):
            with self.subTest(runs=runs), self.assertRaises(check.CheckError):
                check.reference(dict(hand_case(), masks_rle=[runs]))

    def test_dimensions_and_job_types_rejected(self):
        for size in ([0, 2], [641, 1], [2, 481], [True, 2], [2.0, 2]):
            with self.subTest(size=size), self.assertRaises(check.CheckError):
                check.reference(dict(hand_case(), image_size=size))
        with self.assertRaises(check.CheckError):
            check.reference(dict(hand_case(), job_id=True))

    def test_mismatched_visual_count_rejected(self):
        with self.assertRaises(check.CheckError):
            check.reference(dict(hand_case("visual"), masks_rle=[[4]]))

    def test_noninteger_stats_rejected(self):
        for value in (True, 2.0, None):
            observed = hand_result()
            observed["stats"]["m00"] = value
            with self.assertRaisesRegex(check.CheckError, "statistics_mismatch:m00"):
                check.compare(observed, hand_result())

    def test_bad_status_stale_job_and_pixel_digest_rejected(self):
        for field in ("status", "job_id", "output_sha256"):
            observed = hand_result()
            if field == "output_sha256":
                observed[field] = "0" * 64
            else:
                observed["stats"][field] = 99
            with self.subTest(field=field), self.assertRaises(check.CheckError):
                check.compare(observed, hand_result())

    def test_little_bit_packing_with_zero_padding(self):
        pixels = b"\x01\x00\x01\x00\x00\x00\x00\x01\x01"
        self.assertEqual(base64.b64decode(check.pack_bits(pixels)), bytes([133, 1]))

    def test_percentile_guard_and_finite_values(self):
        self.assertIsNone(check.summary([1, 3])["p99_ms"])
        self.assertEqual(check.summary([1, 3])["p95_ms"], 2.9)
        self.assertIsNotNone(check.summary(list(range(100)))["p99_ms"])
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(check.CheckError):
                check.summary([value])


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = fixture(self.root)

    def rewrite(self, fn):
        value = check.read_json(self.profile)
        fn(value)
        self.profile.write_text(json.dumps(value), encoding="utf-8")

    def test_profile_hashes_and_capacity(self):
        value = check.inspect_profile(self.profile)
        self.assertEqual(value["checks"][0]["transfer_bytes"], 16)
        self.assertFalse(value["loaded_overlay_verified"])

    def test_namespaced_and_renamed_modules(self):
        profile = fixture(self.root, namespace=True, renamed=True)
        self.assertEqual(check.inspect_profile(profile)["checks"][0]["instance"], "image_pipeline/stream_dma")

    def test_hash_change_rejected(self):
        (self.root / "test-payload.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(check.CheckError, "asset_hash_mismatch:bit"):
            check.inspect_profile(self.profile)

    def test_dma_capacity_uses_actual_hwh_bits(self):
        profile = fixture(self.root, bits=4)
        with self.assertRaisesRegex(check.CheckError, "dma_length_exceeded"):
            check.inspect_profile(profile)

    def test_register_mismatch_rejected(self):
        self.rewrite(lambda p: p["contract"]["modules"][1]["registers"].update(width="0x14"))
        with self.assertRaisesRegex(check.CheckError, "register_mismatch"):
            check.inspect_profile(self.profile)

    def test_missing_ip_rejected(self):
        self.rewrite(lambda p: p["contract"]["modules"][0].update(instance="not_present"))
        with self.assertRaisesRegex(check.CheckError, "missing_ip"):
            check.inspect_profile(self.profile)

    def test_dma_direction_mismatch_rejected(self):
        self.rewrite(lambda p: p["contract"]["modules"][0]["parameters"].update(c_include_s2mm=0))
        with self.assertRaisesRegex(check.CheckError, "parameter_mismatch"):
            check.inspect_profile(self.profile)

    def test_dtd_rejected(self):
        (self.root / "test.hwh").write_text('<!DOCTYPE SYSTEM><SYSTEM/>', encoding="utf-8")
        self.rewrite(lambda p: p["assets"]["hwh"].update(sha256=check.file_sha(self.root / "test.hwh")))
        with self.assertRaisesRegex(check.CheckError, "unsupported_hwh_xml"):
            check.inspect_profile(self.profile)

    def test_relative_asset_paths(self):
        self.rewrite(lambda p: [a.update(path=Path(a["path"]).name) for a in p["assets"].values()])
        self.assertEqual(check.inspect_profile(self.profile)["build_id"], "synthetic-test-build")

    def test_nested_field_offset_does_not_override_register(self):
        modules = check.hwh_modules(self.root / "test.hwh")
        self.assertEqual(modules["mask_0"]["registers"]["width"], 0x10)


class ReplayTests(unittest.TestCase):
    def recording(self):
        return dict(schema=check.SCHEMA, provenance="synthetic",
                    cases=[dict(hand_case(), observed=hand_result(),
                                timings_ms=dict(reference_ms=1, service_call_ms=2))])

    def result(self):
        return dict(mode="replay", records=[], passed=False)

    def test_recorded_times_remain_attributed(self):
        result = self.result()
        check.replay(self.recording(), result)
        self.assertTrue(result["passed"])
        self.assertEqual(result["provenance"], "synthetic")
        self.assertEqual(result["timing_groups"]["geometry:2x2"]["service_call_ms"]["median_ms"], 2)

    def test_failed_row_preserved_and_not_in_timing_summary(self):
        recording = self.recording()
        recording["cases"][0]["observed"]["stats"]["job_id"] = 8
        result = self.result()
        check.replay(recording, result)
        self.assertFalse(result["passed"])
        self.assertIn("job_id", result["records"][0]["error_code"])
        self.assertEqual(result["timing_groups"], {})

    def test_unlabelled_recording_rejected(self):
        recording = self.recording()
        del recording["provenance"]
        with self.assertRaises(check.CheckError):
            check.replay(recording, self.result())

    def test_invalid_recording_time_is_failure(self):
        recording = self.recording()
        recording["cases"][0]["timings_ms"]["service_call_ms"] = -1
        result = self.result()
        check.replay(recording, result)
        self.assertFalse(result["passed"])

    def test_duplicate_names_rejected(self):
        recording = self.recording()
        recording["cases"] *= 2
        with self.assertRaises(check.CheckError):
            check.replay(recording, self.result())

    def test_malformed_case_is_controlled_failure(self):
        recording = self.recording()
        recording["cases"] = ["not-an-object"]
        with self.assertRaisesRegex(check.CheckError, "invalid_case"):
            check.replay(recording, self.result())

    def test_roles_and_sizes_not_pooled(self):
        rows = [dict(name="a", role="geometry", image_size=[1, 1], passed=True,
                     timings_ms=dict(service_call_ms=2)),
                dict(name="b", role="visual", image_size=[1, 1], passed=True,
                     timings_ms=dict(service_call_ms=4)),
                dict(name="c", role="geometry", image_size=[2, 2], passed=True,
                     timings_ms=dict(service_call_ms=6)),
                dict(name="warm", role="geometry", image_size=[1, 1], passed=True, warmup=True,
                     timings_ms=dict(service_call_ms=900))]
        groups = check.timing_groups(rows)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups["geometry:1x1"]["service_call_ms"]["count"], 1)


class RpcTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = fixture(self.root)
        inspected = check.inspect_profile(self.profile)
        self.assets = {"vision.bit": inspected["asset_sha256"]["bit"],
                       "vision.hwh": inspected["asset_sha256"]["hwh"]}
        self.state = dict(mode="ok", requests=0, healths=0)
        state, assets = self.state, self.assets

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, code, value):
                raw = json.dumps(value).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                state["healths"] += 1
                session = "b" * 32 if state["mode"] == "restart" and state["healths"] > 1 else "a" * 32
                health_assets = dict(assets)
                if state["mode"] == "wrong_assets":
                    health_assets["vision.bit"] = "0" * 64
                self.respond(200, dict(contract="sorting-statistics-v1", session_id=session,
                                       assets=health_assets, faulted=False))

            def do_POST(self):
                state["requests"] += 1
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if state["mode"] == "drop_reply":
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True
                    return
                if state["mode"] == "http_error":
                    self.respond(503, dict(error="SECRET-REMOTE-DETAIL"))
                    return
                response = hand_result(request["role"], request["job_id"])
                response.update(session_id="a"*32, assets=assets, request_id=request["request_id"])
                if state["mode"] == "stale_job":
                    response["stats"]["job_id"] += 1
                if state["mode"] == "wrong_nonce":
                    response["request_id"] = "c"*32
                response["extra_secret"] = "SECRET-REMOTE-DETAIL"
                self.respond(200, response)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = "http://127.0.0.1:" + str(self.server.server_port)
        cases_path = self.root / "cases.json"
        cases_path.write_text(json.dumps(dict(schema=check.SCHEMA,
            cases=[hand_case(), hand_case("visual")])), encoding="utf-8")
        output = self.root / "out"
        output.mkdir()
        self.args = argparse.Namespace(allow_run=True, exclusive_device_confirmed=True,
            service_kind="simulated", profile=self.profile, endpoint=self.endpoint,
            token_env="TEST_PYNQ_CHECK_TOKEN", timeout=1, cases=cases_path,
            repeats=1, output=output, budget_seconds=10)
        self.env = patch.dict(os.environ, TEST_PYNQ_CHECK_TOKEN="test-only-" + "x"*32)
        self.env.start()
        self.addCleanup(self.env.stop)

    def result(self):
        return dict(mode="rpc", records=[], passed=False)

    def test_simulated_http_two_roles_and_warmup(self):
        result = self.result()
        check.rpc_run(self.args, result)
        self.assertTrue(result["passed"])
        self.assertEqual(result["provenance"], "live-rpc-simulated")
        self.assertEqual(self.state["requests"], 4)
        self.assertEqual(result["timing_groups"]["geometry:2x2"]["service_call_ms"]["count"], 1)
        self.assertNotIn("SECRET-REMOTE-DETAIL", json.dumps(result))

    def test_no_run_flag_never_contacts_service(self):
        self.args.allow_run = False
        with self.assertRaises(check.CheckError):
            check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["healths"], 0)

    def test_nonexclusive_never_contacts_service(self):
        self.args.exclusive_device_confirmed = False
        with self.assertRaises(check.CheckError):
            check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["healths"], 0)

    def test_bad_job_stops_after_first_request_with_journal(self):
        self.state["mode"] = "stale_job"
        result = self.result()
        with self.assertRaisesRegex(check.CheckError, "job_id"):
            check.rpc_run(self.args, result)
        self.assertEqual(self.state["requests"], 1)
        self.assertFalse(result["records"][0]["passed"])
        self.assertIn("before_submission", (self.args.output / "measurements.jsonl").read_text())

    def test_wrong_nonce_stops(self):
        self.state["mode"] = "wrong_nonce"
        with self.assertRaisesRegex(check.CheckError, "response_identity_mismatch"):
            check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["requests"], 1)

    def test_session_restart_stops_before_submission(self):
        self.state["mode"] = "restart"
        with self.assertRaisesRegex(check.CheckError, "service_session_changed"):
            check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["requests"], 0)

    def test_http_error_not_retried_or_exposed(self):
        self.state["mode"] = "http_error"
        result = self.result()
        with self.assertRaisesRegex(check.CheckError, "rpc_http_503"):
            check.rpc_run(self.args, result)
        self.assertEqual(self.state["requests"], 1)
        text = (self.args.output / "measurements.jsonl").read_text()
        self.assertNotIn("SECRET-REMOTE-DETAIL", text)
        self.assertNotIn(os.environ[self.args.token_env], text)

    def test_time_budget_expires_without_hardware_submission(self):
        clock = iter([0, 11])
        with patch.object(check.time, "monotonic", side_effect=lambda: next(clock)):
            with self.assertRaisesRegex(check.CheckError, "rpc_time_budget_exceeded"):
                check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["requests"], 0)

    def test_case_budget_rejected_before_network(self):
        self.args.repeats = 300
        with self.assertRaisesRegex(check.CheckError, "rpc_case_budget_exceeded"):
            check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["healths"], 0)

    def test_dropped_response_preserves_unknown_outcome_without_retry(self):
        self.state["mode"] = "drop_reply"
        result = self.result()
        with self.assertRaisesRegex(check.CheckError, "rpc_transport_or_json_error"):
            check.rpc_run(self.args, result)
        self.assertEqual(self.state["requests"], 1)
        row = result["records"][0]
        self.assertEqual(row["request_outcome"], "unknown")
        self.assertEqual(len(row["request_id"]), 32)
        self.assertIn("case", row)

    def test_asset_mismatch_stops_before_submission(self):
        self.state["mode"] = "wrong_assets"
        with self.assertRaisesRegex(check.CheckError, "health_contract_assets_or_fault_mismatch"):
            check.rpc_run(self.args, self.result())
        self.assertEqual(self.state["requests"], 0)

    def test_redirect_does_not_forward_authorization(self):
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", self.server.target + "/health")
                self.end_headers()
        other = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        other.target = self.endpoint
        thread = threading.Thread(target=other.serve_forever, daemon=True)
        thread.start()
        try:
            client = check.Rpc("http://127.0.0.1:" + str(other.server_port), "x"*32, 1)
            with self.assertRaisesRegex(check.CheckError, "rpc_http_302"):
                client.request("health")
            self.assertEqual(self.state["healths"], 0)
        finally:
            other.shutdown()
            other.server_close()


class CliTests(unittest.TestCase):
    def test_demo_success_and_deliberate_failure_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(check.main(["demo", "--output", str(root / "demo")]), 0)
            self.assertEqual(check.main(["replay", "--recording", str(root / "demo/deliberate-failure.json"),
                                         "--output", str(root / "failed")]), 1)
            report = check.read_json(root / "failed/report.json")
            self.assertFalse(report["hardware_execution_independently_verified"])
            self.assertFalse(report["physical_sorting_verified"])
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["records"]), 32)

    def test_outputs_cannot_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                check.main(["demo", "--output", directory])

    def test_bad_json_still_writes_failure_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.json").write_text("{", encoding="utf-8")
            self.assertEqual(check.main(["replay", "--recording", str(root / "bad.json"),
                                         "--output", str(root / "report")]), 1)
            self.assertFalse(check.read_json(root / "report/report.json")["passed"])


if __name__ == "__main__":
    unittest.main()
