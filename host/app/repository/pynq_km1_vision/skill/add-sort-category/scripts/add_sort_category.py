#!/usr/bin/env python3
"""Create prompt-only category candidates, preview, evaluate and export. No motion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

# Keep sibling imports deterministic even under Python's isolated (-I -S) mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from category_core import (CategoryError, SCHEMA, digest, evaluate_rows, export_candidate,
    fingerprint, need, policy_defaults, prompt_map, read, save_evaluation, seal, slug,
    validate_spec, verify, write)


def make_spec(args):
    cat = dict(category_id=args.category_id, name=args.name, bin_id=args.bin_id,
               grasp_profile_id=args.grasp_profile)
    spec = dict(schema=SCHEMA, package_id=args.package_id, mode="text" if args.prompt else "visual",
                confidence=.3, nms_iou=.5, categories=[cat], references=[],
                routing_catalog=dict(bins=[args.bin_id], grasp_profiles=[args.grasp_profile]),
                evaluation_policy=policy_defaults())
    if args.prompt:
        cat["prompts"] = args.prompt
    else:
        need(args.reference and args.bbox and args.reference_group, "visual_reference_box_and_group_required")
        image = Path(args.reference).resolve()
        spec["references"] = [dict(reference_id="reference-one", category_id=args.category_id,
            image=str(image), image_sha256=digest(image), group_id=args.reference_group,
            bbox_xyxy=args.bbox, split="reference")]
    return validate_spec(spec)


def evaluate_live(args):
    from yoloe_backend import open_runtime
    meta, spec = verify(args.package)
    dataset = read(args.dataset)
    need(isinstance(dataset, dict) and dataset.get("schema") == "category-dataset-v1", "invalid_dataset_schema")
    purpose = dataset.get("purpose")
    need(purpose in ("development", "validation"), "test_scoring_not_supported")
    images = dataset.get("images")
    need(isinstance(images, list) and 1 <= len(images) <= 1000, "invalid_dataset_size")
    runtime = open_runtime(args.runtime)
    rows, seen_bytes, seen_pixels = [], set(), set()
    try:
        runtime.load(args.package)
        for item in images:
            need(isinstance(item, dict) and item.get("split") in ("development", "validation"),
                 "sealed_or_test_image_rejected")
            need(purpose != "validation" or item["split"] == "validation", "wrong_validation_split")
            image = (Path(args.dataset).resolve().parent/item["image"]).resolve()
            _, identity = runtime.image(image)
            need(identity["image_sha256"] == item.get("image_sha256"), "dataset_image_hash_mismatch")
            need(identity["image_sha256"] not in seen_bytes and
                 identity["decoded_bgr_sha256"] not in seen_pixels, "duplicate_evaluation_image")
            seen_bytes.add(identity["image_sha256"]); seen_pixels.add(identity["decoded_bgr_sha256"])
            if purpose == "validation":
                for ref in spec["references"]:
                    need(item.get("group_id") != ref["group_id"] and
                         identity["image_sha256"] != ref["image_sha256"] and
                         identity["decoded_bgr_sha256"] != ref["decoded_bgr_sha256"],
                         "validation_reference_or_split_leakage")
            _, observation = runtime.infer(image)
            row = dict(**identity, group_id=item["group_id"], split=item["split"], truth=item["truth"],
                       predictions=observation["instances"], inference_ms=observation["inference_ms"])
            rows.append(row)
        after = runtime.unchanged()
        report = evaluate_rows(rows, spec, "live-yoloe", purpose)
        report.update(package_sha256=meta["package_sha256"], dataset_manifest_sha256=digest(args.dataset),
                      base_weights_before=runtime.before, base_weights_after=after,
                      execution=runtime.execution, gradient_updates=False)
        save_evaluation(args.output, report, rows)
        return report
    finally:
        runtime.unchanged()
        if not (args.output/"observations.json").exists() and rows:
            write(args.output/"partial-observations.json", rows)


def demo(output):
    package = output/"candidate"
    package.mkdir()
    spec = dict(schema=SCHEMA, package_id="synthetic-demo", mode="text", confidence=.3, nms_iou=.5,
        categories=[dict(category_id="demo-part", name="演示物品", prompts=["small part"],
                         bin_id="demo-bin", grasp_profile_id="demo-grasp")], references=[],
        routing_catalog=dict(bins=["demo-bin"], grasp_profiles=["demo-grasp"]),
        evaluation_policy=dict(mask_iou=.5, min_precision=.9, min_recall=.8,
                               min_positive_images=2, min_negative_images=2))
    validate_spec(spec)
    write(package/"spec.json", spec)
    write(package/"synthetic-embeddings.json", dict(synthetic=True, not_a_model=True))
    fake_sha = hashlib.sha256(b"synthetic-not-a-model").hexdigest()
    meta = seal(package, dict(provenance="synthetic", spec_sha256=fingerprint(spec),
                prompt_map=prompt_map(spec), base_weights_before=fake_sha, base_weights_after=fake_sha,
                gradient_updates=False, execution=dict(implementation="synthetic-fixture")))
    rows = []
    instance = dict(category_id="demo-part", mask_rle=[0, 1, 2, 1])
    for i in range(4):
        rows.append(dict(image_sha256=hashlib.sha256(f"synthetic-file-{i}".encode()).hexdigest(),
            decoded_bgr_sha256=hashlib.sha256(f"synthetic-pixels-{i}".encode()).hexdigest(),
            group_id=f"synthetic-group-{i}", split="validation", image_size=[2, 2],
            truth=[instance] if i < 2 else [],
            predictions=[dict(instance, confidence=.9)] if i < 3 else [], inference_ms=1.0))
    report = evaluate_rows(rows, spec, "synthetic", "validation")
    report["package_sha256"] = meta["package_sha256"]
    evaluation = output/"evaluation"
    evaluation.mkdir()
    save_evaluation(evaluation, report, rows)
    write(output/"recording.json", dict(schema=SCHEMA, package_sha256=meta["package_sha256"],
                                       purpose="validation", rows=rows))
    export_candidate(package, evaluation, output/"portable-preview")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="Write an observation-only text or visual specification")
    for name in ("package-id", "category-id", "name", "bin-id", "grasp-profile"):
        new.add_argument("--"+name, required=True)
    choice = new.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prompt", action="append")
    choice.add_argument("--reference", type=Path)
    new.add_argument("--bbox", nargs=4, type=float)
    new.add_argument("--reference-group")
    new.add_argument("--output", type=Path, required=True)
    build = commands.add_parser("build", help="Cache text/visual embeddings using frozen local YOLOE")
    build.add_argument("--spec", type=Path, required=True)
    preview = commands.add_parser("preview", help="Predict a local image and draw masks, without any actuator")
    preview.add_argument("--image", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Score independent annotated positive/negative scenes")
    evaluate.add_argument("--dataset", type=Path, required=True)
    replay = commands.add_parser("replay", help="Evaluate recorded predictions; never opens the quality gate")
    replay.add_argument("--recording", type=Path, required=True)
    export = commands.add_parser("export", help="Export a portable candidate, never activates production")
    export.add_argument("--evaluation", type=Path, required=True)
    inspect = commands.add_parser("verify", help="Verify candidate integrity without ML dependencies")
    demo_parser = commands.add_parser("demo", help="Synthetic complete workflow, no third-party packages")
    for command in (build, preview, evaluate):
        command.add_argument("--runtime", type=Path, required=True)
    for command in (preview, evaluate, replay, export, inspect):
        command.add_argument("--package", type=Path, required=True)
    for command in (build, preview, evaluate, replay, export, demo_parser):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runtime = None
    created_output = False
    try:
        if args.command == "new":
            write(args.output, make_spec(args))
        elif args.command == "verify":
            meta, _ = verify(args.package)
            print(json.dumps(dict(package_sha256=meta["package_sha256"], provenance=meta["provenance"],
                                  state="preview", automatic_motion_allowed=False)))
        elif args.command == "export":
            release = export_candidate(args.package, args.evaluation, args.output)
            print("Exported " + release["recognition_state"] + "; production configuration unchanged.")
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            created_output = True
            if args.command == "build":
                from yoloe_backend import open_runtime
                runtime = open_runtime(args.runtime)
                runtime.build(args.spec, args.output)
            elif args.command == "preview":
                from yoloe_backend import open_runtime
                runtime = open_runtime(args.runtime)
                runtime.load(args.package)
                frame, observation = runtime.infer(args.image)
                runtime.unchanged()
                write(args.output/"observation.json", observation)
                runtime.overlay(frame, observation, args.output/"overlay.png")
            elif args.command == "evaluate":
                report = evaluate_live(args)
                print("Recognition quality gate: " + str(report["quality_gate_passed"]))
            elif args.command == "replay":
                meta, spec = verify(args.package)
                recording = read(args.recording)
                need(recording.get("schema") == SCHEMA and
                     recording.get("package_sha256") == meta["package_sha256"], "recording_package_mismatch")
                source = "synthetic" if meta["provenance"] == "synthetic" else "recorded"
                report = evaluate_rows(recording["rows"], spec, source, recording["purpose"])
                report["package_sha256"] = meta["package_sha256"]
                save_evaluation(args.output, report, recording["rows"])
            elif args.command == "demo":
                demo(args.output)
        if runtime is not None:
            runtime.unchanged()
        print("Completed; automatic_motion_allowed=false")
        # A failed quality gate is a valid evaluation result, not an execution error.
        return 0
    except (CategoryError, OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        reason = str(exc) if isinstance(exc, CategoryError) else "input_runtime_or_io_failure"
        if created_output and not (args.output/"failure.json").exists():
            write(args.output/"failure.json", dict(error=reason, automatic_motion_allowed=False))
        print("FAILED: "+reason, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
