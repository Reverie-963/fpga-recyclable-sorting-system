"""Frozen local YOLOE prompt embeddings and observation-only inference.

Optional dependencies come from the user's existing, version-pinned runtime.
No downloads, installation, training, service management or device control.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from category_core import (CategoryError, SCHEMA, digest, fingerprint, need,
                           prompt_map, read, relative_asset, seal, sha_string,
                           validate_spec, verify, write)


def prohibit_network():
    def guard(event, args):
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo", "socket.sendto"):
            # Libraries probe connectivity during import and normally handle
            # OSError as offline. Block the operation without breaking that path.
            raise OSError("offline_network_access_blocked")
    sys.addaudithook(guard)


class Runtime:
    def __init__(self, config):
        config = Path(config).resolve()
        self.config = read(config)
        cfg = self.config
        need(isinstance(cfg, dict) and cfg.get("schema") == "yoloe-category-runtime-v1",
             "invalid_runtime_schema")
        need(cfg.get("device") == "cpu" and cfg.get("precision") == "fp32", "use_verified_cpu_fp32_runtime")
        need(type(cfg.get("threads")) is int and 1 <= cfg["threads"] <= 4, "invalid_cpu_threads")
        need(cfg.get("ultralytics_version") == "8.3.230" and cfg.get("torch_version") == "2.6.0+cpu",
             "use_existing_pinned_yoloe11_runtime")
        self.weights = (config.parent/cfg["weights"]).resolve()
        need(self.weights.is_file() and sha_string(cfg.get("weights_sha256")), "missing_frozen_weights")
        need(digest(self.weights) == cfg["weights_sha256"], "base_weights_hash_mismatch")
        need("-pf" not in self.weights.stem, "prompt_free_checkpoint_not_supported")
        self.before = cfg["weights_sha256"]
        prohibit_network()
        self.scratch = tempfile.TemporaryDirectory(prefix="category-yoloe-")
        os.environ.update(YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=self.scratch.name,
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          OPENBLAS_NUM_THREADS=str(cfg["threads"]), OMP_NUM_THREADS=str(cfg["threads"]))
        import torch
        import ultralytics
        import cv2
        import numpy as np
        from ultralytics import YOLOE
        need(str(torch.__version__) == cfg["torch_version"] and
             ultralytics.__version__ == cfg["ultralytics_version"], "installed_runtime_version_mismatch")
        torch.set_num_threads(cfg["threads"])
        cv2.setNumThreads(1)
        self.torch, self.cv2, self.np = torch, cv2, np
        self.model = YOLOE(str(self.weights)).to("cpu")
        self.execution = dict(device="cpu", precision="fp32", threads=cfg["threads"],
                              torch=str(torch.__version__), ultralytics=ultralytics.__version__,
                              numpy=np.__version__, opencv=cv2.__version__, network_allowed=False)

    def unchanged(self):
        after = digest(self.weights)
        need(after == self.before, "base_weights_changed_during_operation")
        return after

    def image(self, path):
        path = Path(path)
        need(path.is_file() and path.stat().st_size <= 20*1024*1024, "invalid_image_file")
        frame = self.cv2.imdecode(self.np.frombuffer(path.read_bytes(), self.np.uint8), self.cv2.IMREAD_COLOR)
        need(frame is not None and 1 <= frame.shape[1] <= 640 and 1 <= frame.shape[0] <= 480,
             "image_must_fit_640x480_without_implicit_resize")
        import hashlib
        return frame, dict(image_sha256=digest(path),
                           decoded_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                           image_size=[frame.shape[1], frame.shape[0]])

    def build(self, spec_path, output):
        spec_path, output = Path(spec_path).resolve(), Path(output)
        spec = validate_spec(read(spec_path))
        # JSON roundtrip detaches caller state; packaged paths are rewritten below.
        import json
        spec = json.loads(json.dumps(spec))
        prompts = prompt_map(spec)
        encoder_sha = None
        if spec["mode"] == "text":
            need(importlib.util.find_spec("clip") is not None, "existing_clip_tokenizer_missing")
            encoder = (Path(self.config_path).parent/self.config["text_encoder"]).resolve()
            need(encoder.is_file() and encoder.name == "mobileclip_blt.ts" and
                 digest(encoder) == self.config.get("text_encoder_sha256"), "text_encoder_hash_mismatch")
            encoder_sha = digest(encoder)
            old_cwd = Path.cwd()
            try:
                os.chdir(encoder.parent)
                with self.torch.inference_mode():
                    embeddings = self.model.get_text_pe([p["text"] for p in prompts]).detach().cpu().float()
            finally:
                os.chdir(old_cwd)
            need(digest(encoder) == encoder_sha, "text_encoder_changed")
        else:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            parts = []
            (output/"references").mkdir()
            for i, ref in enumerate(spec["references"]):
                image_path = (spec_path.parent/ref["image"]).resolve()
                frame, identity = self.image(image_path)
                need(identity["image_sha256"] == ref["image_sha256"], "reference_image_hash_mismatch")
                x0, y0, x1, y1 = ref["bbox_xyxy"]
                need(x1 <= frame.shape[1] and y1 <= frame.shape[0], "reference_bbox_out_of_image")
                suffix = image_path.suffix.lower()
                need(suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"), "unsupported_reference_extension")
                name = f"references/ref-{i:03d}"+suffix
                shutil.copyfile(image_path, output/name)
                ref.update(image=name, decoded_bgr_sha256=identity["decoded_bgr_sha256"],
                           image_size=identity["image_size"])
                predictor = YOLOEVPSegPredictor(overrides=dict(task="segment", mode="predict", save=False,
                    verbose=False, batch=1, device="cpu", half=False, imgsz=640), _callbacks=self.model.callbacks)
                self.model.model.model[-1].nc = 1
                self.model.model.names = ["object0"]
                predictor.set_prompts(dict(bboxes=[ref["bbox_xyxy"]], cls=[0]))
                predictor.setup_model(model=self.model.model, verbose=False)
                with self.torch.inference_mode():
                    parts.append(predictor.get_vpe(frame).detach().cpu().float())
            embeddings = self.torch.cat(parts, dim=1)
        values = embeddings.numpy()
        need(values.shape == (1, len(prompts), 512) and self.np.isfinite(values).all(), "invalid_prompt_embeddings")
        self.np.save(output/"embeddings.npy", values, allow_pickle=False)
        write(output/"spec.json", spec)
        after = self.unchanged()
        return seal(output, dict(provenance="real-yoloe", spec_sha256=fingerprint(spec),
                    prompt_map=prompt_map(spec), base_weights_before=self.before, base_weights_after=after,
                    base_model_family="yoloe-11-prompted", text_encoder_sha256=encoder_sha,
                    execution=self.execution, gradient_updates=False,
                    claim="Prompt embedding cache only; no team retraining. Pretraining exposure unknown."))

    def load(self, package):
        self.meta, self.spec = verify(package)
        need(self.meta["provenance"] == "real-yoloe", "synthetic_package_has_no_model")
        need(self.meta["base_weights_before"] == self.before, "package_uses_other_base_model")
        need(self.meta["execution"]["ultralytics"] == self.execution["ultralytics"] and
             self.meta["execution"]["torch"] == self.execution["torch"], "package_runtime_version_mismatch")
        values = self.np.load(relative_asset(package, "embeddings.npy"), allow_pickle=False)
        names = [p["text"] for p in self.meta["prompt_map"]]
        need(values.dtype == self.np.float32 and values.shape == (1, len(names), 512)
             and self.np.isfinite(values).all(), "invalid_cached_embeddings")
        with self.torch.inference_mode():
            self.model.set_classes(names, self.torch.from_numpy(values))
        self.model.predictor = None

    def infer(self, image):
        frame, identity = self.image(image)
        started = time.perf_counter()
        with self.torch.inference_mode():
            raw = self.model.predict(frame, device="cpu", half=False, imgsz=640, retina_masks=True,
                conf=self.spec["confidence"], iou=self.spec["nms_iou"], max_det=100,
                agnostic_nms=True, verbose=False, save=False)[0]
        instances = []
        boxes = raw.boxes
        if boxes is not None and len(boxes):
            need(raw.masks is not None and len(raw.masks.data) == len(boxes), "model_instance_masks_missing")
            masks = raw.masks.data.detach().cpu().numpy()
            need(masks.shape[1:] == frame.shape[:2], "mask_not_in_original_image")
            categories = {c["category_id"]: c for c in self.spec["categories"]}
            for i, (value, confidence) in enumerate(zip(boxes.cls.tolist(), boxes.conf.tolist())):
                prompt_id = int(value)
                need(value == prompt_id and 0 <= prompt_id < len(self.meta["prompt_map"]), "invalid_model_prompt_id")
                need(0 <= confidence <= 1 and self.np.isfinite(confidence), "invalid_model_confidence")
                prompt = self.meta["prompt_map"][prompt_id]
                mask = masks[i] > .5
                ys, xs = self.np.nonzero(mask)
                if not len(xs):
                    continue
                pixels = mask.astype(self.np.uint8).reshape(-1)
                edges = self.np.flatnonzero(pixels[1:] != pixels[:-1])+1
                runs = self.np.diff(self.np.r_[0, edges, len(pixels)]).tolist()
                if pixels[0]:
                    runs = [0]+runs
                cat = categories[prompt["category_id"]]
                instances.append(dict(instance_id=len(instances), category_id=cat["category_id"],
                    category_name=cat["name"], prompt_id=prompt_id, prompt_text=prompt["text"],
                    confidence=float(confidence), mask_rle=runs, area_px=int(mask.sum()),
                    bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                    proposed_bin_id=cat["bin_id"], sorting_bin=None))
        latency = (time.perf_counter()-started)*1000
        same_reference = any(identity["decoded_bgr_sha256"] == r["decoded_bgr_sha256"]
                             for r in self.spec["references"])
        return frame, dict(schema=SCHEMA, **identity, instances=instances, inference_ms=latency,
            package_sha256=self.meta["package_sha256"], base_model_sha256=self.before,
            prompt_mode=self.spec["mode"], execution=self.execution, reference_image_match=same_reference,
            automatic_motion_allowed=False, physical_acceptance_verified=False,
            material_identity_verified=False, mapping_status="operator_declared_candidate")

    def overlay(self, frame, observation, path):
        result = frame.copy()
        w, h = observation["image_size"]
        for instance in observation["instances"]:
            runs = instance["mask_rle"]
            mask = self.np.repeat(self.np.arange(len(runs)) % 2, runs).reshape(h, w).astype(self.np.uint8)
            contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
            self.cv2.drawContours(result, contours, -1, (30, 190, 240), 2)
            x0, y0, _, _ = instance["bbox_xyxy"]
            self.cv2.putText(result, instance["category_id"]+f" {instance['confidence']:.2f}",
                (x0, max(15, y0-5)), self.cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 190, 240), 1)
        ok, encoded = self.cv2.imencode(".png", result)
        need(ok, "overlay_encoding_failed")
        with Path(path).open("xb") as stream:
            stream.write(encoded.tobytes())


def open_runtime(config):
    runtime = Runtime(config)
    runtime.config_path = str(Path(config).resolve())
    return runtime
