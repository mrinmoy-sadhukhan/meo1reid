import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from models.conditional_detr_hybrid import ConditionalDETR
from models.mot17inference import HybridTemporalTracker


# ============================================================
# 🧠 Debug Hooks to Inspect Internal Flow
# ============================================================
def register_debug_hooks(model):
    """Attach hooks for ReID, memory projection, and temporal query updates."""
    hooks = []

    def hook_reid_head(module, input, output):
        print(f"  🔹 [HOOK] ReID Head Output: {tuple(output.shape)}")

    def hook_memory_proj(module, input, output):
        print(f"  🔸 [HOOK] Memory Projection: input {tuple(input[0].shape)} → output {tuple(output.shape)}")

    def hook_temporal_update(module, input, output):
        print(f"  🔹 [HOOK] Temporal Query Update: {tuple(output.shape)}")

    if hasattr(model, "reid_embed_head"):
        hooks.append(model.reid_embed_head.register_forward_hook(hook_reid_head))
    if hasattr(model, "memory_proj"):
        hooks.append(model.memory_proj.register_forward_hook(hook_memory_proj))
    if hasattr(model, "temporal_update"):
        hooks.append(model.temporal_update.register_forward_hook(hook_temporal_update))

    print("✅ Hooks registered: ReID head, memory projection, temporal update")
    return hooks


# ============================================================
# 🧰 Preprocess Function
# ============================================================
def preprocess(img, size=480):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0)


# ============================================================
# 🧪 MOT17 Debug Flow (on image sequence)
# ============================================================
def debug_mot17_flow(
    seq_dir,
    model_path="",
    conf_thresh=0.5,
    num_frames=6,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print("🚀 Starting Hybrid Tracker Debug (MOT17 Folder Mode)")
    print(f"Device: {device}")
    print(f"Sequence folder: {seq_dir}")

    # === Load Model ===
    model = ConditionalDETR(
        d_model=256,
        n_classes=2,
        n_layers=4,
        use_deformable=True,
        use_reid=True,
        use_temporal=True,
        use_reid_classifier=False
    ).to(device)

    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print(f"✅ Loaded model checkpoint: {model_path}")
    else:
        print("⚠️ Using randomly initialized model for debug test")

    model.eval()
    tracker = HybridTemporalTracker(model, device)
    hooks = register_debug_hooks(model)

    # === Collect frames ===
    img_dir = os.path.join(seq_dir, "img1")
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])[:num_frames]
    print(f"📸 Using first {len(frame_files)} frames: {frame_files}")

    for fid, fname in enumerate(frame_files, 1):
        fpath = os.path.join(img_dir, fname)
        frame = cv2.imread(fpath)
        if frame is None:
            print(f"⚠️ Could not read {fname}")
            continue

        print(f"\n=================== FRAME {fid} ({fname}) ===================")

        H, W = frame.shape[:2]
        inp = preprocess(frame).to(device)

        with torch.no_grad():
            cls_preds, box_preds, reid_feats = model(inp, return_reid=True, memory_cond=True)

        print(f"\n🔹 MODEL OUTPUT SHAPES:")
        print(f"  cls_preds : {tuple(cls_preds.shape)}  [B, layers, queries, classes]")
        print(f"  box_preds : {tuple(box_preds.shape)}  [B, layers, queries, 4]")
        print(f"  reid_feats: {tuple(reid_feats.shape)} [B, queries, 256]")

        # --- Postprocess detections ---
        cls = cls_preds[0, -1].softmax(-1)
        boxes = box_preds[0, -1]
        scores, labels = cls.max(-1)
        mask = (scores > conf_thresh) & (labels == 1)

        tracker.predict()

        if mask.sum() == 0:
            print("⚠️ No detections above confidence threshold")
            tracker._age_tracks()
            continue

        boxes = boxes[mask].cpu().numpy()
        if boxes.max() <= 1.01:
            boxes[:, [0, 2]] *= W
            boxes[:, [1, 3]] *= H

        boxes_xyxy = np.stack([
            boxes[:, 0] - boxes[:, 2] / 2,
            boxes[:, 1] - boxes[:, 3] / 2,
            boxes[:, 0] + boxes[:, 2] / 2,
            boxes[:, 1] + boxes[:, 3] / 2
        ], axis=1)
        feats = F.normalize(reid_feats[0, mask], dim=-1)

        print(f"\n🔹 TRACKER INPUTS:")
        print(f"  det_boxes: {boxes_xyxy.shape} | det_feats: {tuple(feats.shape)}")

        ids = tracker.update(boxes_xyxy, feats)
        print(f"  Assigned IDs: {ids}")

        # --- Inspect tracks and memory ---
        print(f"\n🔹 TRACK STATUS AFTER UPDATE:")
        for tid, tr in tracker.tracks.items():
            print(f"  🟢 Track {tid}: box={np.round(tr['box'], 1)}, age={tr['time_since_update']}")
            if tid in tracker.temporal_memory:
                mem = tracker.temporal_memory[tid]
                print(f"     Memory shape={tuple(mem.shape)}, mean={mem.mean():.4f}, std={mem.std():.4f}")

        # --- Inspect model internal memory ---
        if hasattr(model, "temporal_proj_memory") and hasattr(model, "memory_proj"):
            mem_count = len(getattr(model, "temporal_proj_memory", []))
            if mem_count > 0:
                last_mem = model.temporal_proj_memory[-1]
                print(f"  📦 Model Temporal Memory Bank: {mem_count} entries | Last shape={tuple(last_mem.shape)}")

    print("\n=================== DEBUG COMPLETE ===================")

    for h in hooks:
        h.remove()

