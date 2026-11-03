---
name: add-sort-category
description: Add observation-only sorting categories using frozen YOLOE text prompts or boxed visual references, preview masks, evaluate independent positives and negatives, and export a versioned candidate package. Use when adding a new item without retraining or sharing a prompt configuration. Does not activate a robot or change production bindings.
---

# Add a sorting category

Translate the user's item definition and chosen destination into an immutable
candidate. The open-vocabulary model provides recognition; this skill supplies
the preparation, evaluation and portable configuration workflow. It does not
establish material composition or make an arbitrary object physically graspable.

Use `scripts/add_sort_category.py`. Resolve script/reference paths from this skill.
Read [README.md](README.md) for commands and [contracts](references/contracts.md)
for the runtime, category, dataset and state contracts.

## Prepare and build

1. Check existing local weights and the dedicated compatible environment before
   downloading or installing anything. This implementation is verified with
   YOLOE-11m, Ultralytics8.3.230 and CPU PyTorch2.6.0+cpu. Reuse that environment;
   do not upgrade the running workstation's inference stack.
2. Determine persistent category IDs, exact text prompts or reference boxes, the
   user-confirmed bin and an existing grasp-profile reference. Reuse already
   established routing choices. Ask only about genuine ambiguity; visual appearance
   alone cannot establish recycling policy. A routing catalog is a declaration,
   not evidence that a mechanical calibration exists or has passed.
3. Generate/edit a spec using `new`. Text prompts should use concrete supported
   names; do not promise automatic understanding of arbitrary Chinese descriptions.
   For visual mode, preserve reference file hash, object/session `group_id`, split
   and exclusive-right/bottom pixel box. All examples need explicit category IDs.
4. With the existing ML interpreter, use `build --spec ... --runtime ... --output ...`.
   Build caches prompt embeddings without gradient updates. It does not save or
   modify the base checkpoint. All network access is blocked in the process.
   If dependencies/assets are missing, report them instead of silently installing
   packages, downloading weights or swapping to another model.

## Preview, evaluate and export

Run `preview --package ... --runtime ... --image ... --output ...` on a different
local image. Read `observation.json` and inspect `overlay.png`. Prompt index is not
a category or bin number: the package's explicit prompt map supplies the identity.
Keep low scores, misses and background false positives visible in the assessment.

Freeze prompt/threshold choices before evaluating a declared validation set. The
dataset must include per-instance full-image RLE truth, explicit negative scenes,
file hashes, split and object/session group IDs. `evaluate` rejects validation
overlap with visual references by file hash, decoded pixels or declared group.
It also rejects duplicate decoded images and sealed/test splits. Development
images may be used for workflow smoke checks but can never open the quality gate.

Inspect per-category TP/FP/FN, precision/recall/F1, matched-mask IoU, positive and
negative scene counts, failure reasons and latency. Defaults of30 positive and30
negative images per category are team recommendations, not official thresholds.
Do not tune on a validation result and then call that same set untouched acceptance.
The tool cannot independently audit human-supplied groups or annotation truth.

Use `export --package ... --evaluation ... --output ...` to produce a portable
candidate with the evaluated configuration identity. Failed or insufficient
evidence exports as `preview_only`; live independent validation meeting the frozen
policy can export as `evaluated_candidate`. Both remain non-executable. Integrity
checks detect accidental changes; they are not signed attestation of provenance.

Use `verify --package ...` after copying/importing a candidate. Supply the same
trusted base checkpoint through a local runtime file on the other machine; it is
not bundled. Switching among candidate folders provides version selection without
overwriting a previous candidate. This is not production deployment or rollback.

## Report honestly

State whether a run is synthetic, recorded, real development inference or
independent validation. Recorded/synthetic results never open the gate. Existing
familiar objects demonstrate the workflow, not unseen-category generalization.
“No team retraining” does not mean the item was unseen during model pretraining.
The unchanged base hash and no-gradient path support the narrow no-retraining claim.

Keep `automatic_motion_allowed=false`, `physical_acceptance_verified=false` and
`sorting_bin=null`. Exported bin/grasp IDs are proposals. Actual integration must
follow the owning system's model/config identity, commissioning, ownership, STOP,
pending, freshness and scene checks. Reuse existing applicable authorization;
this skill itself neither grants motion authority nor writes production profiles.
No camera/serial/board/service calls are part of this tool.

For a complete offline demonstration without ML dependencies:

```text
python scripts/add_sort_category.py demo --output category-demo
python scripts/add_sort_category.py verify --package category-demo/portable-preview
```
