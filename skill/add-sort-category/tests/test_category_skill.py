"""Independent mask metrics, packaging and state boundaries; no ML dependencies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import category_core as core
import add_sort_category as cli


def spec():
    return dict(schema=core.SCHEMA, package_id="test-category", mode="text", confidence=.3, nms_iou=.5,
        categories=[dict(category_id="part", name="part", prompts=["small part"],
                         bin_id="paper", grasp_profile_id="known-profile")], references=[],
        routing_catalog=dict(bins=["paper"], grasp_profiles=["known-profile"]),
        evaluation_policy=dict(mask_iou=.5, min_precision=.9, min_recall=.8,
                               min_positive_images=1, min_negative_images=1))


def rows():
    obj = dict(category_id="part", mask_rle=[0, 2, 2])
    return [dict(image_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
        decoded_bgr_sha256=hashlib.sha256(("decoded"+str(i)).encode()).hexdigest(),
        image_size=[2, 2], group_id="group-"+str(i), split="validation", inference_ms=2+i,
        truth=[deepcopy(obj)] if i == 0 else [],
        predictions=[dict(obj, confidence=.8)] if i == 0 else []) for i in range(2)]


class ContractTests(unittest.TestCase):
    def test_valid_text_and_explicit_mapping(self):
        self.assertEqual(core.prompt_map(core.validate_spec(spec()))[0]["category_id"], "part")

    def test_duplicate_categories_rejected(self):
        s = spec(); s["categories"] *= 2
        with self.assertRaisesRegex(core.CategoryError, "duplicate_category"):
            core.validate_spec(s)

    def test_unknown_bin_or_grasp_rejected(self):
        for key in ("bin_id", "grasp_profile_id"):
            s = spec(); s["categories"][0][key] = "unknown"
            with self.subTest(key=key), self.assertRaisesRegex(core.CategoryError, "unknown_candidate_route"):
                core.validate_spec(s)

    def test_unknown_metadata_not_silently_exported(self):
        s = spec(); s["password"] = "not-a-secret-test-value"
        with self.assertRaisesRegex(core.CategoryError, "unsupported_spec_fields"):
            core.validate_spec(s)

    def test_unknown_category_metadata_not_exported(self):
        s = spec(); s["categories"][0]["machine_setting"] = "local-only-value"
        with self.assertRaisesRegex(core.CategoryError, "unsupported_category_fields"):
            core.validate_spec(s)

    def test_confidence_cannot_be_nan_or_boolean(self):
        for value in (True, float("nan"), 0, 1):
            s = spec(); s["confidence"] = value
            with self.subTest(value=value), self.assertRaises(core.CategoryError):
                core.validate_spec(s)

    def test_duplicate_prompts_rejected(self):
        s = spec(); s["categories"][0]["prompts"] *= 2
        with self.assertRaisesRegex(core.CategoryError, "duplicate_prompt"):
            core.validate_spec(s)

    def test_visual_reference_requires_group_and_development_split(self):
        s = spec(); s["mode"] = "visual"; del s["categories"][0]["prompts"]
        ref = dict(reference_id="ref", category_id="part", group_id="object-a", split="reference",
                   bbox_xyxy=[0, 0, 2, 2], image="example.png", image_sha256="a"*64)
        s["references"] = [ref]
        core.validate_spec(s)
        for field, value in (("group_id", ""), ("split", "validation"), ("bbox_xyxy", [2, 0, 1, 2])):
            bad = deepcopy(s); bad["references"][0][field] = value
            with self.subTest(field=field), self.assertRaises(core.CategoryError):
                core.validate_spec(bad)


class MaskMetricTests(unittest.TestCase):
    def test_hand_calculated_iou(self):
        self.assertEqual(core.mask_iou([0, 2, 2], [1, 2, 1], [2, 2]), 1/3)
        self.assertEqual(core.mask_iou([4], [4], [2, 2]), 0)

    def test_invalid_rle_and_dimensions(self):
        for mask in ([3], [5], [-1, 5], [True, 3], []):
            with self.subTest(mask=mask), self.assertRaises(core.CategoryError):
                core.mask_intervals(mask, [2, 2])
        with self.assertRaises(core.CategoryError):
            core.mask_intervals([1], [641, 1])

    def test_maximum_matching_avoids_greedy_undercount(self):
        preds = [dict(mask_rle=[0, 4]), dict(mask_rle=[0, 2, 2])]
        truth = [dict(mask_rle=[0, 2, 2]), dict(mask_rle=[2, 2])]
        count, _ = core.matching(preds, truth, [2, 2], .5)
        self.assertEqual(count, 2)

    def test_true_positive_and_negative_counts(self):
        report = core.evaluate_rows(rows(), spec(), "live-yoloe", "validation")
        m = report["metrics"]["part"]
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 0))
        self.assertEqual((m["positive_images"], m["negative_images"]), (1, 1))
        self.assertTrue(report["quality_gate_passed"])
        self.assertFalse(report["automatic_motion_allowed"])
        self.assertFalse(report["physical_acceptance_verified"])

    def test_false_positive_counts_on_negative_image(self):
        rr = rows(); rr[1]["predictions"] = deepcopy(rr[0]["predictions"])
        report = core.evaluate_rows(rr, spec(), "live-yoloe", "validation")
        self.assertEqual(report["metrics"]["part"]["precision"], .5)
        self.assertIn("part:min_precision", report["blockers"])

    def test_miss_counts_and_low_confidence_not_scored_as_detection(self):
        rr = rows(); rr[0]["predictions"][0]["confidence"] = .1
        report = core.evaluate_rows(rr, spec(), "live-yoloe", "validation")
        self.assertEqual((report["metrics"]["part"]["tp"], report["metrics"]["part"]["fn"]), (0, 1))

    def test_recorded_and_synthetic_never_pass(self):
        for source in ("recorded", "synthetic"):
            report = core.evaluate_rows(rows(), spec(), source, "validation")
            self.assertFalse(report["quality_gate_passed"])
            self.assertIn("no_live_inference_evidence", report["blockers"])

    def test_development_cannot_pass_even_with_perfect_scores(self):
        report = core.evaluate_rows(rows(), spec(), "live-yoloe", "development")
        self.assertFalse(report["quality_gate_passed"])
        self.assertIn("development_only", report["blockers"])

    def test_insufficient_positive_and_negative_samples(self):
        s = spec(); s["evaluation_policy"]["min_positive_images"] = 30
        s["evaluation_policy"]["min_negative_images"] = 30
        report = core.evaluate_rows(rows(), s, "live-yoloe", "validation")
        self.assertIn("part:min_positive_images", report["blockers"])
        self.assertIn("part:min_negative_images", report["blockers"])

    def test_duplicate_bytes_or_decoded_pixels_rejected(self):
        for key in ("image_sha256", "decoded_bgr_sha256"):
            rr = rows(); rr[1][key] = rr[0][key]
            with self.subTest(key=key), self.assertRaisesRegex(core.CategoryError, "duplicate_evaluation"):
                core.evaluate_rows(rr, spec(), "live-yoloe", "validation")

    def test_unknown_category_and_empty_instance_rejected(self):
        for field, value in (("category_id", "unknown"), ("mask_rle", [4])):
            rr = rows(); rr[0]["predictions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(core.CategoryError):
                core.evaluate_rows(rr, spec(), "live-yoloe", "validation")

    def test_sealed_test_rejected(self):
        rr = rows(); rr[0]["split"] = "test"
        with self.assertRaisesRegex(core.CategoryError, "sealed_or_test"):
            core.evaluate_rows(rr, spec(), "live-yoloe", "validation")

    def test_invalid_negative_scene_size_is_controlled_failure(self):
        rr = rows(); rr[1]["image_size"] = []
        with self.assertRaisesRegex(core.CategoryError, "invalid_image_size"):
            core.evaluate_rows(rr, spec(), "live-yoloe", "validation")

    def test_reference_group_leakage_rejected(self):
        s = spec(); s["mode"] = "visual"; del s["categories"][0]["prompts"]
        s["references"] = [dict(reference_id="ref", category_id="part", image="ref.png", image_sha256="a"*64,
            decoded_bgr_sha256="b"*64, bbox_xyxy=[0, 0, 1, 1], group_id="group-0", split="reference")]
        with self.assertRaisesRegex(core.CategoryError, "leakage"):
            core.evaluate_rows(rows(), s, "live-yoloe", "validation")
        report = core.evaluate_rows(rows(), s, "live-yoloe", "development")
        self.assertTrue(report["reference_overlap"])
        self.assertFalse(report["quality_gate_passed"])


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.demo = self.root/"demo"; self.demo.mkdir()
        cli.demo(self.demo)
        self.package = self.demo/"candidate"

    def test_demo_metrics_are_independently_expected(self):
        report = core.read(self.demo/"evaluation/report.json")
        self.assertEqual((report["metrics"]["demo-part"]["tp"], report["metrics"]["demo-part"]["fp"]), (2, 1))
        self.assertAlmostEqual(report["metrics"]["demo-part"]["precision"], 2/3)
        self.assertFalse(report["quality_gate_passed"])

    def test_integrity_and_motion_boundary(self):
        meta, _ = core.verify(self.package)
        self.assertFalse(meta["automatic_motion_allowed"])
        release = core.read(self.demo/"portable-preview/release.json")
        self.assertEqual(release["recognition_state"], "preview_only")
        self.assertFalse(release["execution_profile_changed"])

    def test_asset_tampering_rejected(self):
        (self.package/"synthetic-embeddings.json").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(core.CategoryError, "asset_mismatch"):
            core.verify(self.package)

    def test_metadata_tampering_rejected(self):
        meta = core.read(self.package/"package.json"); meta["state"] = "active"
        (self.package/"package.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(core.CategoryError, "fingerprint_mismatch"):
            core.verify(self.package)

    def test_training_flag_cannot_be_promoted_even_with_recomputed_hash(self):
        meta = core.read(self.package/"package.json"); meta["gradient_updates"] = True
        del meta["package_sha256"]; meta["package_sha256"] = core.fingerprint(meta)
        (self.package/"package.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(core.CategoryError, "no_gradient_updates"):
            core.verify(self.package)

    def test_path_traversal_rejected(self):
        outside = self.root/"outside"; outside.write_text("x")
        with self.assertRaisesRegex(core.CategoryError, "escaped_asset"):
            core.relative_asset(self.package, "../../outside")

    def test_evaluation_for_another_candidate_rejected(self):
        report = core.read(self.demo/"evaluation/report.json")
        report["package_sha256"] = "c"*64
        del report["evaluation_sha256"]; report["evaluation_sha256"] = core.fingerprint(report)
        (self.demo/"evaluation/report.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(core.CategoryError, "another_package"):
            core.export_candidate(self.package, self.demo/"evaluation", self.root/"export")

    def test_export_recomputes_metrics(self):
        report = core.read(self.demo/"evaluation/report.json")
        report["metrics"]["demo-part"]["tp"] = 999
        del report["evaluation_sha256"]; report["evaluation_sha256"] = core.fingerprint(report)
        (self.demo/"evaluation/report.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(core.CategoryError, "result_mismatch"):
            core.export_candidate(self.package, self.demo/"evaluation", self.root/"export")

    def test_export_cannot_write_inside_original_package(self):
        with self.assertRaisesRegex(core.CategoryError, "must_not_modify"):
            core.export_candidate(self.package, self.demo/"evaluation", self.package/"nested-export")

    def test_cli_never_adds_failure_file_to_existing_output(self):
        before = set(self.demo.iterdir())
        self.assertEqual(cli.main(["demo", "--output", str(self.demo)]), 1)
        self.assertEqual(set(self.demo.iterdir()), before)

    def test_cli_replay_and_export_do_not_require_ml_packages(self):
        output = self.root/"replay"
        self.assertEqual(cli.main(["replay", "--package", str(self.package), "--recording",
                                  str(self.demo/"recording.json"), "--output", str(output)]), 0)
        self.assertFalse(core.read(output/"report.json")["quality_gate_passed"])

    def test_runtime_network_guard_blocks_before_connection(self):
        script = Path(__file__).resolve().parents[1]/"scripts"
        source = f"import sys;sys.path.insert(0,{str(script)!r});from yoloe_backend import prohibit_network;prohibit_network();import socket\ntry:\n socket.create_connection(('127.0.0.1',1),timeout=.1)\n raise AssertionError('network allowed')\nexcept OSError as e:\n assert 'offline_network_access_blocked' in str(e)\n"
        result = subprocess.run([sys.executable, "-I", "-S", "-c", source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_entrypoint_works_in_isolated_python(self):
        script = Path(__file__).resolve().parents[1]/"scripts/add_sort_category.py"
        result = subprocess.run([sys.executable, "-I", "-S", str(script), "demo",
                                 "--output", str(self.root/"isolated-demo")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
