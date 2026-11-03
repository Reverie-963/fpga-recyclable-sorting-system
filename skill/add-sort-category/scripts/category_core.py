"""Portable category contracts, mask metrics and immutable evidence packages.

Standard library only. No camera, board, network, training or actuator calls.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

SCHEMA = "add-sort-category-v1"


class CategoryError(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise CategoryError(reason)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in iter(lambda: stream.read(1024*1024), b""):
            value.update(raw)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    need(len(raw) <= 32*1024*1024, "json_too_large")
    return json.loads(raw.decode("utf-8-sig"), parse_constant=lambda _: None)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def slug(value):
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", value) is not None


def finite(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def sha_string(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def relative_asset(root, name):
    need(isinstance(name, str) and name and not Path(name).is_absolute()
         and "\\" not in name and ":" not in name, "invalid_relative_asset")
    target = (Path(root) / name).resolve()
    need(target.is_relative_to(Path(root).resolve()) and target.is_file(), "missing_or_escaped_asset")
    return target


def policy_defaults():
    # Team recommendations, not competition thresholds.
    return dict(mask_iou=.5, min_precision=.9, min_recall=.8,
                min_positive_images=30, min_negative_images=30)


def validate_spec(spec):
    need(isinstance(spec, dict) and spec.get("schema") == SCHEMA, "invalid_spec_schema")
    need(set(spec) == {"schema", "package_id", "mode", "confidence", "nms_iou", "categories",
                       "references", "routing_catalog", "evaluation_policy"}, "unsupported_spec_fields")
    need(slug(spec.get("package_id")), "invalid_package_id")
    need(spec.get("mode") in ("text", "visual"), "invalid_prompt_mode")
    need(finite(spec.get("confidence"), .01, .99) and finite(spec.get("nms_iou"), .01, .99),
         "invalid_inference_threshold")
    cats = spec.get("categories")
    need(isinstance(cats, list) and 1 <= len(cats) <= 20, "invalid_categories")
    catalog = spec.get("routing_catalog")
    need(isinstance(catalog, dict) and set(catalog) == {"bins", "grasp_profiles"}, "missing_routing_catalog")
    for key in ("bins", "grasp_profiles"):
        need(isinstance(catalog.get(key), list) and catalog[key]
             and all(slug(x) for x in catalog[key]) and len(set(catalog[key])) == len(catalog[key]),
             "invalid_routing_catalog")
    ids, prompts = set(), set()
    for cat in cats:
        need(isinstance(cat, dict) and slug(cat.get("category_id")), "invalid_category_id")
        need(set(cat) == ({"category_id", "name", "bin_id", "grasp_profile_id", "prompts"}
                         if spec["mode"] == "text" else {"category_id", "name", "bin_id", "grasp_profile_id"}),
             "unsupported_category_fields")
        need(cat["category_id"] not in ids, "duplicate_category_id")
        ids.add(cat["category_id"])
        need(isinstance(cat.get("name"), str) and 0 < len(cat["name"].strip()) <= 100,
             "invalid_category_name")
        need(cat.get("bin_id") in catalog["bins"] and
             cat.get("grasp_profile_id") in catalog["grasp_profiles"], "unknown_candidate_route")
        if spec["mode"] == "text":
            need(isinstance(cat.get("prompts"), list) and 1 <= len(cat["prompts"]) <= 5,
                 "missing_text_prompts")
            for text in cat["prompts"]:
                need(isinstance(text, str) and 0 < len(text.strip()) <= 120 and
                     text == text.strip() and text not in prompts, "invalid_or_duplicate_prompt")
                prompts.add(text)
    refs = spec.get("references")
    need(isinstance(refs, list), "invalid_references")
    need((spec["mode"] == "text" and not refs) or
         (spec["mode"] == "visual" and 1 <= len(refs) <= 30), "reference_mode_mismatch")
    ref_ids, represented = set(), set()
    for ref in refs:
        need(isinstance(ref, dict) and slug(ref.get("reference_id")) and
             ref["reference_id"] not in ref_ids, "invalid_reference_id")
        ref_ids.add(ref["reference_id"])
        need(set(ref).issubset({"reference_id", "category_id", "image", "image_sha256", "group_id",
                               "split", "bbox_xyxy", "decoded_bgr_sha256", "image_size"}),
             "unsupported_reference_fields")
        need(ref.get("category_id") in ids and sha_string(ref.get("image_sha256")), "invalid_reference_identity")
        need(isinstance(ref.get("group_id"), str) and ref["group_id"], "missing_reference_group")
        box = ref.get("bbox_xyxy")
        need(isinstance(box, list) and len(box) == 4 and all(finite(x, 0, 100000) for x in box)
             and box[0] < box[2] and box[1] < box[3], "invalid_reference_bbox")
        need(ref.get("split") in ("reference", "development"), "reference_must_not_use_validation_or_test")
        need(isinstance(ref.get("image"), str) and ref["image"], "missing_reference_image")
        represented.add(ref["category_id"])
    if spec["mode"] == "visual":
        need(represented == ids, "category_without_reference")
    need(len(prompts) <= 30, "too_many_prompts")
    policy = spec.get("evaluation_policy")
    need(isinstance(policy, dict) and set(policy) == set(policy_defaults()), "invalid_evaluation_policy")
    for key in ("mask_iou", "min_precision", "min_recall"):
        need(finite(policy[key], .01, 1), "invalid_evaluation_threshold")
    for key in ("min_positive_images", "min_negative_images"):
        need(type(policy[key]) is int and 1 <= policy[key] <= 100000, "invalid_sample_requirement")
    return spec


def prompt_map(spec):
    if spec["mode"] == "text":
        return [dict(prompt_id=i, text=text, category_id=cat["category_id"])
                for i, (cat, text) in enumerate((cat, text) for cat in spec["categories"]
                                               for text in cat["prompts"])]
    return [dict(prompt_id=i, text="example_" + ref["reference_id"],
                 category_id=ref["category_id"], reference_id=ref["reference_id"])
            for i, ref in enumerate(spec["references"])]


def seal(root, metadata):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*")
             if p.is_file() and p.name != "package.json"}
    need(files, "empty_package")
    metadata.update(schema=SCHEMA, assets=files, automatic_motion_allowed=False,
                    physical_acceptance_verified=False, state="preview")
    metadata["package_sha256"] = fingerprint(metadata)
    write(root / "package.json", metadata)
    return metadata


def verify(root):
    root = Path(root)
    meta = read(root / "package.json")
    need(isinstance(meta, dict) and meta.get("schema") == SCHEMA, "invalid_package_schema")
    frozen = dict(meta)
    identity = frozen.pop("package_sha256", None)
    need(sha_string(identity) and fingerprint(frozen) == identity, "package_fingerprint_mismatch")
    need(meta.get("automatic_motion_allowed") is False and
         meta.get("physical_acceptance_verified") is False and meta.get("state") == "preview",
         "package_must_remain_preview")
    need(meta.get("provenance") in ("real-yoloe", "synthetic"), "invalid_package_provenance")
    need(meta.get("gradient_updates") is False, "package_must_have_no_gradient_updates")
    need(isinstance(meta.get("assets"), dict) and meta["assets"], "invalid_package_assets")
    for name, expected in meta["assets"].items():
        need(sha_string(expected) and digest(relative_asset(root, name)) == expected, "package_asset_mismatch")
    spec = validate_spec(read(relative_asset(root, "spec.json")))
    need(meta.get("spec_sha256") == fingerprint(spec), "spec_fingerprint_mismatch")
    need(meta.get("prompt_map") == prompt_map(spec), "prompt_map_mismatch")
    need(meta.get("base_weights_before") == meta.get("base_weights_after") and
         sha_string(meta.get("base_weights_before")), "base_weight_identity_mismatch")
    for ref in spec["references"]:
        need(digest(relative_asset(root, ref["image"])) == ref["image_sha256"], "reference_hash_mismatch")
    return meta, spec


def image_count(size):
    need(isinstance(size, list) and len(size) == 2 and
         all(type(n) is int for n in size) and 1 <= size[0] <= 640 and 1 <= size[1] <= 480,
         "invalid_image_size")
    return size[0]*size[1]


def mask_intervals(runs, size):
    count = image_count(size)
    need(isinstance(runs, list) and 0 < len(runs) <= count*2+1 and
         all(type(n) is int and n >= 0 for n in runs) and sum(runs) == count, "invalid_mask_rle")
    intervals, start = [], 0
    for i, n in enumerate(runs):
        if i % 2 and n:
            intervals.append((start, start+n))
        start += n
    return intervals


def mask_iou(a, b, size):
    aa, bb = mask_intervals(a, size), mask_intervals(b, size)
    total_a, total_b = sum(y-x for x, y in aa), sum(y-x for x, y in bb)
    i, j, inter = 0, 0, 0
    while i < len(aa) and j < len(bb):
        inter += max(0, min(aa[i][1], bb[j][1])-max(aa[i][0], bb[j][0]))
        if aa[i][1] <= bb[j][1]:
            i += 1
        else:
            j += 1
    union = total_a+total_b-inter
    return inter/union if union else 0.0


def validate_instances(instances, size, category_ids, predictions=False):
    need(isinstance(instances, list) and len(instances) <= 100, "invalid_instances")
    for item in instances:
        need(isinstance(item, dict) and item.get("category_id") in category_ids, "unknown_instance_category")
        intervals = mask_intervals(item.get("mask_rle"), size)
        need(intervals, "empty_instance_mask")
        if predictions:
            need(finite(item.get("confidence"), 0, 1), "invalid_prediction_confidence")


def matching(predictions, truths, size, threshold):
    """Maximum cardinality matching, avoiding order-dependent greedy undercounts."""
    overlaps = [[mask_iou(p["mask_rle"], t["mask_rle"], size) for t in truths] for p in predictions]
    edges = [sorted((i for i, v in enumerate(row) if v >= threshold),
                    key=lambda i: row[i], reverse=True) for row in overlaps]
    assigned = {}
    def augment(p, seen):
        for t in edges[p]:
            if t in seen:
                continue
            seen.add(t)
            if t not in assigned or augment(assigned[t], seen):
                assigned[t] = p
                return True
        return False
    for p in range(len(predictions)):
        augment(p, set())
    return len(assigned), [overlaps[p][t] for t, p in assigned.items()]


def evaluate_rows(rows, spec, source, purpose):
    validate_spec(spec)
    need(source in ("live-yoloe", "recorded", "synthetic"), "invalid_evaluation_source")
    need(purpose in ("development", "validation"), "test_scoring_not_supported")
    need(isinstance(rows, list) and 1 <= len(rows) <= 1000, "invalid_evaluation_rows")
    ids = [c["category_id"] for c in spec["categories"]]
    metrics = {i: dict(tp=0, fp=0, fn=0, positive_images=0, negative_images=0, matched_ious=[]) for i in ids}
    hashes, decoded_hashes, groups, latency = set(), set(), set(), []
    reference_groups = {r["group_id"] for r in spec["references"]}
    reference_hashes = {r["image_sha256"] for r in spec["references"]}
    reference_decoded = {r.get("decoded_bgr_sha256") for r in spec["references"]}
    reference_overlap = False
    for row in rows:
        need(isinstance(row, dict) and sha_string(row.get("image_sha256")), "invalid_evaluation_image_identity")
        need(sha_string(row.get("decoded_bgr_sha256")), "missing_decoded_image_identity")
        need(row["image_sha256"] not in hashes and row["decoded_bgr_sha256"] not in decoded_hashes,
             "duplicate_evaluation_image")
        hashes.add(row["image_sha256"]); decoded_hashes.add(row["decoded_bgr_sha256"])
        need(isinstance(row.get("group_id"), str) and row["group_id"], "missing_evaluation_group")
        groups.add(row["group_id"])
        need(row.get("split") in ("development", "validation"), "sealed_or_test_image_rejected")
        overlap = row["image_sha256"] in reference_hashes or row["decoded_bgr_sha256"] in reference_decoded or row["group_id"] in reference_groups
        reference_overlap |= overlap
        if purpose == "validation":
            need(row["split"] == "validation" and not overlap, "validation_reference_or_split_leakage")
        size = row.get("image_size")
        truths, all_predictions = row.get("truth"), row.get("predictions")
        # Also validate size for negative scenes with no instances.
        image_count(size)
        validate_instances(truths, size, ids)
        validate_instances(all_predictions, size, ids, predictions=True)
        preds = [p for p in all_predictions if p["confidence"] >= spec["confidence"]]
        need(finite(row.get("inference_ms"), 0, 3600000), "invalid_inference_latency")
        latency.append(row["inference_ms"])
        for category_id in ids:
            pp = [p for p in preds if p["category_id"] == category_id]
            tt = [t for t in truths if t["category_id"] == category_id]
            tp, ious = matching(pp, tt, size, spec["evaluation_policy"]["mask_iou"])
            metric = metrics[category_id]
            metric["tp"] += tp; metric["fp"] += len(pp)-tp; metric["fn"] += len(tt)-tp
            metric["positive_images"] += bool(tt); metric["negative_images"] += not tt
            metric["matched_ious"].extend(ious)
    blockers, policy = [], spec["evaluation_policy"]
    for category_id, m in metrics.items():
        m["precision"] = m["tp"]/(m["tp"]+m["fp"]) if m["tp"]+m["fp"] else 0.0
        m["recall"] = m["tp"]/(m["tp"]+m["fn"]) if m["tp"]+m["fn"] else 0.0
        m["f1"] = 2*m["tp"]/(2*m["tp"]+m["fp"]+m["fn"]) if 2*m["tp"]+m["fp"]+m["fn"] else 0.0
        ious = m.pop("matched_ious")
        m["matched_mean_mask_iou"] = sum(ious)/len(ious) if ious else None
        for key, actual in (("min_positive_images", m["positive_images"]),
                            ("min_negative_images", m["negative_images"]),
                            ("min_precision", m["precision"]), ("min_recall", m["recall"])):
            if actual < policy[key]:
                blockers.append(category_id+":"+key)
    if source != "live-yoloe":
        blockers.append("no_live_inference_evidence")
    if purpose != "validation":
        blockers.append("development_only")
    if reference_overlap:
        blockers.append("reference_group_or_image_overlap")
    ordered = sorted(latency)
    def percentile(p):
        pos = (len(ordered)-1)*p
        low = int(pos); high = min(low+1, len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(pos-low)
    return dict(schema=SCHEMA, source=source, purpose=purpose, metrics=metrics, images=len(rows),
                unique_declared_groups=len(groups), reference_overlap=reference_overlap,
                policy=policy, quality_gate_passed=not blockers, blockers=blockers,
                inference_latency_ms=dict(samples=len(latency), median=percentile(.5), p95=percentile(.95),
                                          p99=percentile(.99) if len(latency) >= 100 else None, maximum=max(latency)),
                automatic_motion_allowed=False, physical_acceptance_verified=False,
                note="Declared data groups are not independently audited. Quality gate covers recognition only; no physical sorting acceptance.")


def save_evaluation(output, report, rows):
    output = Path(output)
    write(output / "observations.json", rows)
    report["observations_sha256"] = digest(output / "observations.json")
    report["evaluation_sha256"] = fingerprint(report)
    write(output / "report.json", report)
    lines = ["# 新物品接入评估", "", "- 来源："+report["source"], "- 用途："+report["purpose"],
             "- 识别质量门槛："+("通过" if report["quality_gate_passed"] else "未通过"),
             "- 自动机械执行：不允许；物理验收：未完成", "",
             "| 类别 | TP | FP | FN | Precision | Recall | F1 | 正/负场景 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, m in report["metrics"].items():
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['positive_images']}/{m['negative_images']} |")
    lines.extend(["", "未通过项目："+", ".join(report["blockers"]), "", report["note"], "",
                  "耗时为本机推理及结果转换，排除加载/提示编码；不代表整机分拣时延。样本不足100时不报告P99。"])
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines)+"\n")


def export_candidate(package, evaluation, output):
    need(not Path(output).resolve().is_relative_to(Path(package).resolve()), "export_must_not_modify_package")
    meta, spec = verify(package)
    report = read(Path(evaluation)/"report.json")
    frozen = dict(report); identity = frozen.pop("evaluation_sha256", None)
    need(fingerprint(frozen) == identity, "evaluation_fingerprint_mismatch")
    need(report.get("package_sha256") == meta["package_sha256"], "evaluation_for_another_package")
    rows = read(Path(evaluation)/"observations.json")
    need(digest(Path(evaluation)/"observations.json") == report["observations_sha256"], "observation_hash_mismatch")
    recalculated = evaluate_rows(rows, spec, report["source"], report["purpose"])
    for key in ("metrics", "quality_gate_passed", "blockers", "policy", "images"):
        need(report[key] == recalculated[key], "evaluation_result_mismatch")
    need(not report["quality_gate_passed"] or meta["provenance"] == "real-yoloe", "synthetic_cannot_pass_gate")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ["package.json", *meta["assets"]]:
        src = relative_asset(package, name)
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    (output/"evaluation").mkdir()
    for name in ("report.json", "observations.json", "report.md"):
        shutil.copyfile(Path(evaluation)/name, output/"evaluation"/name)
    release = dict(schema=SCHEMA, package_sha256=meta["package_sha256"],
                   recognition_state="evaluated_candidate" if report["quality_gate_passed"] else "preview_only",
                   evaluation_sha256=identity,
                   proposed_routes=[{k: c[k] for k in ("category_id", "bin_id", "grasp_profile_id")} for c in spec["categories"]],
                   automatic_motion_allowed=False, physical_acceptance_verified=False,
                   execution_profile_changed=False, created_at=datetime.now(timezone.utc).isoformat())
    write(output/"release.json", release)
    verify(output)
    return release
