import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Disable interactive mode for Kaggle / headless servers
import os, glob, re
from models.conditional_detr_hybrid import ConditionalDETR  # ✅ uncomment when model file is available


# ---------------- Dataset ----------------
class MarketFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_paths, self.pids, self.cams = [], [], []
        self.transform = transform
        pattern = re.compile(r'([-\d]+)_c(\d)')  # PID & cam from name

        # detect whether images are directly under root or inside subfolders
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if len(subdirs) > 0:
            # ── foldered structure: root/<pid>/*.jpg
            for pid_folder in subdirs:
                pid_path = os.path.join(root, pid_folder)
                imgs = sorted(glob.glob(os.path.join(pid_path, '*.jpg')))
                for img_path in imgs:
                    fname = os.path.basename(img_path)
                    match = pattern.search(fname)
                    if not match:
                        continue
                    pid, camid = map(int, match.groups())
                    if pid == -1:
                        continue
                    self.img_paths.append(img_path)
                    self.pids.append(pid)
                    self.cams.append(camid)
        else:
            # ── flat structure: root/*.jpg
            imgs = sorted(glob.glob(os.path.join(root, '*.jpg')))
            for img_path in imgs:
                fname = os.path.basename(img_path)
                match = pattern.search(fname)
                if not match:
                    continue
                pid, camid = map(int, match.groups())
                if pid == -1:
                    continue
                self.img_paths.append(img_path)
                self.pids.append(pid)
                self.cams.append(camid)

        print(f"📂 Loaded {len(self.img_paths)} images from {root}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.pids[idx], self.cams[idx], self.img_paths[idx]

# ---------------- Subset helper ----------------
def subset_dataset(dataset, fraction=0.1):
    subset_size = max(1, int(len(dataset) * fraction))
    indices = torch.randperm(len(dataset))[:subset_size]
    subset = torch.utils.data.Subset(dataset, indices)
    print(f"⚡ Using {subset_size}/{len(dataset)} samples (~{fraction * 100:.1f}%) from {dataset.__class__.__name__}")
    return subset


# ---------------- Feature extraction ----------------
def extract_features(model, dataloader, device):
    feats, pids, cams, paths = [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, pid, cam, path in tqdm(dataloader, desc="Extracting features", ncols=80):
            imgs = imgs.to(device)

            # Handle variable outputs from hybrid model
            out = model(imgs, return_reid=True)
            if len(out) == 5:
                _, _, reid_feats, _, _ = out
            else:
                _, _, reid_feats = out

            # mean-pool decoder embeddings
            feat = F.normalize(reid_feats.mean(1), dim=-1).cpu().numpy()
            feats.append(feat)
            pids += pid.tolist()
            cams += cam.tolist()
            paths += list(path)

    feats = np.vstack(feats)
    print(f"✅ Extracted features for {len(pids)} images.")
    return feats, np.array(pids), np.array(cams), paths


# ---------------- Visualization ----------------
def visualize_topk_retrieval(query_path, q_pid, gallery_paths, gallery_pids, sims, topk=10):
    os.makedirs("retrieval_vis", exist_ok=True)
    plt.figure(figsize=(16, 4))
    plt.suptitle(f"Query PID: {q_pid}", fontsize=14, fontweight="bold")

    # Query
    plt.subplot(1, topk + 1, 1)
    plt.imshow(Image.open(query_path))
    plt.title("Query", color="blue")
    plt.axis("off")

    # Gallery
    for i in range(topk):
        img = Image.open(gallery_paths[i])
        plt.subplot(1, topk + 1, i + 2)
        plt.imshow(img)
        color = "green" if gallery_pids[i] == q_pid else "red"
        plt.title(f"PID:{gallery_pids[i]}\n{(sims[i]):.2f}", color=color, fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join("retrieval_vis", f"retrieval_{q_pid}.png"))
    plt.close()


# ---------------- Evaluation ----------------
def evaluate_market(qf, qid, qcam, gf, gid, gcam, qpaths, gpaths, visualize_topk=True, topk=10):
    qf = qf / np.linalg.norm(qf, axis=1, keepdims=True)
    gf = gf / np.linalg.norm(gf, axis=1, keepdims=True)
    sim = np.dot(qf, gf.T)
    indices = np.argsort(-sim, axis=1)

    all_cmc, all_ap = [], []
    for i, q_pid in enumerate(qid):
        order = indices[i]

        # 🔹 Exclude same-camera same-person images
        valid = ~((gid[order] == q_pid) & (gcam[order] == qcam[i]))
        order = order[valid]

        matches = (gid[order] == q_pid).astype(np.int32)
        if matches.sum() == 0:
            continue

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:50])

        num_rel = matches.sum()
        prec = matches.cumsum() / (np.arange(len(matches)) + 1)
        ap = np.sum(prec * matches) / num_rel
        all_ap.append(ap)

        # 🔹 Visualization (only for first few queries)
        if visualize_topk and i < 5:
            visualize_topk_retrieval(
                qpaths[i], q_pid,
                [gpaths[k] for k in order[:topk]],
                [gid[k] for k in order[:topk]],
                [sim[i, k] for k in order[:topk]]
            )

    if len(all_ap) == 0:
        print("⚠️ No valid matches found across cameras.")
        return 0.0, np.zeros(50)

    cmc = np.mean(np.asarray(all_cmc), axis=0)
    mAP = np.mean(all_ap)

    print(f"\n✅ Evaluation Results (Cross-Camera Only):")
    print(f"mAP: {mAP * 100:.2f}%  Rank@1: {cmc[0] * 100:.2f}%  Rank@5: {cmc[4] * 100:.2f}%  Rank@10: {cmc[9] * 100:.2f}%")
    return mAP, cmc

# ---------------- Inference pipeline ----------------
def run_market_folder_inference(model_path, market_root, batch_size=32, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # === Load model ===
    model = ConditionalDETR(
        use_reid=True,
        use_temporal=False,
        use_reid_classifier=False
    ).to(device)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print("Loaded pretrained DETR weights")
    else:
        print("Model path does not exist. Please check the path.")
        print("Using random weights")
    model.eval()
    print("✅ Loaded pretrained ReID model.")

    # === Transforms ===
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # === Dataset setup ===
    query_ds = MarketFolderDataset(os.path.join(market_root, "query"), transform)
    gallery_ds = MarketFolderDataset(os.path.join(market_root, "gallery"), transform)

    # Use only 1% of dataset for quick test
    query_dl = torch.utils.data.DataLoader(subset_dataset(query_ds, 1.00), batch_size=batch_size, shuffle=False, num_workers=0)
    gallery_dl = torch.utils.data.DataLoader(subset_dataset(gallery_ds, 1.00), batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Query images: {len(query_ds)}, Gallery images: {len(gallery_ds)}")

    # === Extract embeddings ===
    q_feats, q_pids, q_cams, q_paths = extract_features(model, query_dl, device)
    g_feats, g_pids, g_cams, g_paths = extract_features(model, gallery_dl, device)

    # === Evaluate ===
    evaluate_market(q_feats, q_pids, q_cams, g_feats, g_pids, g_cams, q_paths, g_paths)
