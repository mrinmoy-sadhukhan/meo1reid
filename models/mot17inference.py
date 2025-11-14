import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from datetime import datetime
from models.conditional_detr_hybrid import ConditionalDETR  # ✅ your hybrid model

# -------------------- Kalman Filter --------------------
class KalmanFilter:
    def __init__(self, dt=1.0):
        self.dt = dt
        self.dim_x = 6
        self.dim_z = 4
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)
        self.Q = np.diag([1, 1, 0.5, 0.5, 1, 1])
        self.R = np.diag([10, 10, 10, 10])
        self.P = np.eye(self.dim_x) * 100
        self.x = np.zeros(self.dim_x)

    def initiate(self, measurement):
        self.x[:4] = measurement
        self.x[4:] = 0
        self.P = np.eye(self.dim_x) * 100

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement):
        z = measurement
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def get_state_bbox(self):
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def mahalanobis(self, measurement):
        z = measurement
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return np.sqrt(y.T @ np.linalg.inv(S) @ y)


# -------------------- Hybrid MOTRv2-like Tracker --------------------
class HybridTemporalTracker:
    def __init__(self, model, device,
                 iou_weight=0.3, app_weight=0.5, motion_weight=0.2,
                 max_age=30, use_motion=True, use_fused_memory=True):
        self.model = model
        self.device = device
        self.iou_weight = iou_weight
        self.app_weight = app_weight
        self.motion_weight = motion_weight
        self.max_age = max_age
        self.use_motion = use_motion
        self.use_fused_memory = use_fused_memory

        self.tracks = {}
        self.temporal_memory = {}
        self.next_id = 1

    # ---------- Utility ----------
    def iou(self, a, b):
        xx1 = np.maximum(a[:, 0][:, None], b[:, 0])
        yy1 = np.maximum(a[:, 1][:, None], b[:, 1])
        xx2 = np.minimum(a[:, 2][:, None], b[:, 2])
        yy2 = np.minimum(a[:, 3][:, None], b[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_a = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1])
        area_b = (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
        return inter / (area_a[:, None] + area_b - inter + 1e-6)

    def compute_cosine_similarity(self, feats1, feats2):
        feats1 = F.normalize(feats1, dim=-1)
        feats2 = F.normalize(feats2, dim=-1)
        return torch.mm(feats1, feats2.T)

    def _bbox_to_cxcywh(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])

    # ---------- Prediction ----------
    def predict(self):
        for tid, tr in self.tracks.items():
            tr["prev_center"] = self._bbox_to_cxcywh(tr["box"])[:2]
            tr["kf"].predict()
            tr["time_since_update"] += 1

    # ---------- Temporal Memory Update ----------
    def update_memory(self, track_id, reid_feat, kf_state):
        """
        Push fused temporal features into model memory.
        reid_feat: [256]
        kf_state: (cx, cy, w, h)
        """
        reid_feat = reid_feat.detach().cpu()
        pos = torch.tensor(kf_state, dtype=torch.float32)

        # Normalize positions to [0,1] for consistency
        pos_norm = pos / 480.0

        # Optional motion velocity (approx)
        vel = torch.zeros(2)
        tr = self.tracks.get(track_id)
        if tr and "prev_center" in tr:
            vel[:2] = torch.tensor(pos[:2] - tr["prev_center"])

        if self.use_fused_memory:
            #print("i am here")
            #print(reid_feat.shape, pos_norm.shape, vel.shape)
            memory_feat = torch.cat([reid_feat, pos_norm, vel], dim=0)  # [256+4+2=262]
            #print(memory_feat.shape)
        else:
            #print("i am here else")
            memory_feat = reid_feat
            #print(memory_feat.shape)

        # Smooth memory (temporal EMA)
        if track_id not in self.temporal_memory:
            self.temporal_memory[track_id] = memory_feat.clone()
            print(self.temporal_memory[track_id].shape)
        else:
            print(track_id)
            #print(self.temporal_memory[track_id].shape, memory_feat.shape)
            prev = self.temporal_memory[track_id]
            #print(prev.shape, memory_feat.shape)
            self.temporal_memory[track_id] = 0.8 * prev + 0.2 * memory_feat

        # Update model internal memory
        if hasattr(self.model, "update_memory"):
            self.model.update_memory(self.temporal_memory[track_id].unsqueeze(0))

    # ---------- Main Update ----------
    def update(self, det_boxes, det_feats):
        if len(det_boxes) == 0:
            self._age_tracks()
            return []

        track_ids = list(self.tracks.keys())

        if len(track_ids) == 0:
            for i in range(len(det_boxes)):
                self._init_track(det_boxes[i], det_feats[i])
            return list(range(self.next_id - len(det_boxes), self.next_id))

        track_boxes = np.stack([tr["kf"].get_state_bbox() for tr in self.tracks.values()])
        track_feats = torch.stack([tr["feat"] for tr in self.tracks.values()]).to(self.device)
        det_feats = det_feats.to(self.device)

        sim_app = self.compute_cosine_similarity(det_feats, track_feats).cpu().numpy()
        iou_sim = self.iou(det_boxes, track_boxes)

        motion_cost = np.zeros_like(iou_sim)
        if self.use_motion:
            for i, box in enumerate(det_boxes):
                for j, (tid, tr) in enumerate(self.tracks.items()):
                    meas = self._bbox_to_cxcywh(box)
                    motion_cost[i, j] = np.exp(-tr["kf"].mahalanobis(meas))

        combined = (
            self.app_weight * sim_app +
            self.iou_weight * iou_sim +
            self.motion_weight * motion_cost
        )
        #########
        # === Diagnostic Logging ===
        print("\n[DEBUG] Association Diagnostics:")
        print(f"  IoU range: {iou_sim.min():.3f} → {iou_sim.max():.3f}")
        print(f"  AppSim range: {sim_app.min():.3f} → {sim_app.max():.3f}")
        print(f"  MotionCost range: {motion_cost.min():.3f} → {motion_cost.max():.3f}")
        print(f"  Combined range: {combined.min():.3f} → {combined.max():.3f}")
        print(f"  Weights: app={self.app_weight}, iou={self.iou_weight}, motion={self.motion_weight}")

        # If values are too small, it's why IDs keep changing
        if combined.max() < 0.2:
            print("  ⚠️ WARNING: Low association scores — likely no matches between frames!")

        # Continue with matching
        row_ind, col_ind = linear_sum_assignment(-combined)
        assigned = [-1] * len(det_boxes)
        matched, unmatched_dets, unmatched_trks = [], set(range(len(det_boxes))), set(range(len(track_boxes)))

        for r, c in zip(row_ind, col_ind):
            if combined[r, c] < 0.1:   # (Relaxed from 0.25)
                continue
            matched.append((r, c))
            unmatched_dets.discard(r)
            unmatched_trks.discard(c)

        print(f"  Matched pairs this frame: {len(matched)} / {len(det_boxes)} detections")
        #####
        row_ind, col_ind = linear_sum_assignment(-combined)
        assigned = [-1] * len(det_boxes)
        matched, unmatched_dets, unmatched_trks = [], set(range(len(det_boxes))), set(range(len(track_boxes)))

        for r, c in zip(row_ind, col_ind):
            if combined[r, c] < 0.25:
                continue
            matched.append((r, c))
            unmatched_dets.discard(r)
            unmatched_trks.discard(c)

        for r, c in matched:
            tid = track_ids[c]
            tr = self.tracks[tid]
            kf_state = self._bbox_to_cxcywh(det_boxes[r])
            tr["kf"].update(kf_state)
            tr["feat"] = 0.8 * tr["feat"] + 0.2 * det_feats[r].cpu()
            tr["box"] = det_boxes[r]
            tr["time_since_update"] = 0
            assigned[r] = tid
            #print(det_feats[r].shape, kf_state.shape)
            self.update_memory(tid, det_feats[r], kf_state)

        for i in unmatched_dets:
            self._init_track(det_boxes[i], det_feats[i])
            assigned[i] = self.next_id - 1

        self._cleanup_tracks()
        return assigned

    def _init_track(self, box, feat):
        kf = KalmanFilter()
        kf.initiate(self._bbox_to_cxcywh(box))
        self.tracks[self.next_id] = dict(kf=kf, feat=feat.cpu(), box=box, time_since_update=0)
        
        # Create consistent fused memory [256 + 4 + 2 = 262]
        pos = torch.tensor(self._bbox_to_cxcywh(box), dtype=torch.float32)
        pos_norm = pos / 480.0
        vel = torch.zeros(2)
        fused = torch.cat([feat.cpu(), pos_norm, vel], dim=0)
        
        #self.temporal_memory[self.next_id] = feat.clone()
        self.temporal_memory[self.next_id]=fused.clone()
        self.next_id += 1

    def _age_tracks(self):
        to_delete = []
        for tid, tr in self.tracks.items():
            tr["time_since_update"] += 1
            if tr["time_since_update"] > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)
            self.temporal_memory.pop(tid, None)

    def _cleanup_tracks(self):
        to_delete = [tid for tid, tr in self.tracks.items() if tr["time_since_update"] > self.max_age]
        for tid in to_delete:
            self.tracks.pop(tid, None)
            self.temporal_memory.pop(tid, None)


# -------------------- Preprocess --------------------
def preprocess(img, size=480):
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return t(img).unsqueeze(0)


# -------------------- Inference --------------------
def run_mot17_inference(
    video_dir,
    model_path="mot17_joint.pth",
    out_dir="results",
    device="cuda",
    conf_thresh=0.5,
    visualize=True
):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    model = ConditionalDETR(use_reid=True, use_temporal=True, use_reid_classifier=False).to(device)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print("Loaded pretrained DETR weights")
    else:
        print("Model path does not exist. Please check the path.")
        print("Using random weights")
        
    model.eval()

    tracker = HybridTemporalTracker(model, device)
    result_file = os.path.join(out_dir, "mot17_results.txt")
    results = []

    seq_dirs = sorted([p for p in os.listdir(video_dir)])
    for seq in seq_dirs:
        seq_path = os.path.join(video_dir, seq, "img1")
        frame_files = sorted([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f.endswith(".jpg")])
        print(f"🔹 Running on {seq} with {len(frame_files)} frames")

        out_video = os.path.join(out_dir, f"{seq}_tracked.avi")
        writer = None

        for fid, frame_path in enumerate(frame_files, 1):
            print(frame_path)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            H, W = frame.shape[:2]
            inp = preprocess(frame).to(device)

            with torch.no_grad():
                cls_preds, box_preds, reid_feats = model(inp, return_reid=True)

            cls = cls_preds[0, -1].softmax(-1)
            boxes = box_preds[0, -1]
            scores, labels = cls.max(-1)
            mask = (scores > conf_thresh) & (labels == 1)

            tracker.predict()
            if mask.sum() == 0:
                tracker._age_tracks()
                continue

            boxes = boxes[mask].cpu().numpy()
            if boxes.max() <= 1.01:
                boxes[:, [0, 2]] *= W
                boxes[:, [1, 3]] *= H
            boxes_xyxy = np.stack([
                boxes[:, 0] - boxes[:, 2]/2,
                boxes[:, 1] - boxes[:, 3]/2,
                boxes[:, 0] + boxes[:, 2]/2,
                boxes[:, 1] + boxes[:, 3]/2
            ], axis=1)

            feats = F.normalize(reid_feats[0, mask], dim=-1)
            ids = tracker.update(boxes_xyxy, feats)

            vis = frame.copy()
            for det_idx, tid in enumerate(ids):
                if tid < 0:
                    continue
                x1, y1, x2, y2 = boxes_xyxy[det_idx].astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"ID {tid}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                w, h = x2 - x1, y2 - y1
                results.append(f"{fid},{tid},{x1},{y1},{w},{h},{scores[mask][det_idx]:.3f},1,1\n")

            if writer is None:
                writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"XVID"), 20, (W, H))
            writer.write(vis)

            if visualize:
                cv2.imshow("Tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if writer:
            writer.release()
        print(f"✅ Saved tracked video: {out_video}")

    with open(result_file, "w") as f:
        f.writelines(results)
    print(f"✅ Results written to {result_file}")
    cv2.destroyAllWindows()
