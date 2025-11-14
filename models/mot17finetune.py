import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm
from models.custom_modules.cond_detr_decoder_layerV2_0 import ConditionalDecoderLayer
from models.conditional_detr_hybrid import ConditionalDETR  # 🔹 change path to your actual model file

# ================================================================
# COCO-style MOT Dataset for Detection + ReID Fine-Tuning
# ================================================================
class MOT17CocoReID(Dataset):
    def __init__(self, img_root, ann_path, transform=None):
        self.coco = COCO(ann_path)
        self.img_root = img_root
        self.transform = transform
        self.img_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, ids = [], []
        for a in anns:
            if a.get("category_id", 1) != 1:
                continue
            if "bbox" in a and a["bbox"][2] > 0 and a["bbox"][3] > 0:
                x, y, w, h = a["bbox"]
                boxes.append([x, y, x + w, y + h])
                ids.append(a.get("track_id", 0))

        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            ids = [0]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "ids": torch.tensor(ids, dtype=torch.long)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs, dim=0)
    return imgs, targets


# ================================================================
# Generalized IoU + Detection Loss
# ================================================================
def generalized_box_iou(boxes1, boxes2):
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    def box_area(box):
        return (box[:, 2] - box[:, 0]).clamp(min=0) * (box[:, 3] - box[:, 1]).clamp(min=0)

    area1, area2 = box_area(boxes1), box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter + 1e-7
    iou = inter / union
    enclosing_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = wh[:, :, 0] * wh[:, :, 1] + 1e-7
    giou = iou - (enclosing_area - union) / enclosing_area
    return giou


def detection_losses(pred_logits, pred_boxes, targets, device):
    B = len(targets)
    if B == 0:
        zero = torch.tensor(0., device=device)
        return zero, zero, zero

    loss_cls, loss_bbox, loss_giou = 0.0, 0.0, 0.0
    for b in range(B):
        gt_boxes = targets[b]["boxes"].to(device)
        if gt_boxes.numel() == 0:
            continue

        cls_pred = pred_logits[b]
        box_pred = pred_boxes[b]
        if cls_pred.dim() == 1:
            cls_pred = cls_pred.unsqueeze(0)
        if box_pred.dim() == 1:
            box_pred = box_pred.unsqueeze(0)

        Q, C = cls_pred.shape
        tgt = torch.ones(Q, dtype=torch.long, device=device)
        cls_loss = F.cross_entropy(cls_pred, tgt)

        pred = box_pred.sigmoid()
        gt = gt_boxes / 480.0
        pred_mean = pred.mean(0, keepdim=True)
        gt_mean = gt.mean(0, keepdim=True)
        loss_l1 = F.l1_loss(pred_mean, gt_mean)
        giou = generalized_box_iou(pred_mean, gt_mean)
        loss_giou_val = 1.0 - giou.mean()

        loss_cls += cls_loss
        loss_bbox += loss_l1
        loss_giou += loss_giou_val

    denom = max(1, B)
    return loss_cls / denom, loss_bbox / denom, loss_giou / denom


# ================================================================
# Joint Training for Detection + ReID
# ================================================================
def train_mot17_joint(
    mot17_root,
    ann_path,
    pretrained_reid="",
    save_path="mot17_joint.pth",
    lr=5e-5,
    batch_size=2,
    epochs=30,
    adaptive_mode="none",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}  |  Adaptive Mode: {adaptive_mode}")

    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = MOT17CocoReID(mot17_root, ann_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_fn)

    model = ConditionalDETR(
        use_reid=True,
        use_temporal=True,
        use_reid_classifier=True,
        n_classes=2,
        reid_dim=256,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    id_loss = nn.CrossEntropyLoss()
    # Adaptive params
    if adaptive_mode == "gradnorm":
        task_weights = nn.Parameter(torch.tensor([1.0, 1.0], device=device))
        optimizer.add_param_group({"params": [task_weights], "lr": lr})
    elif adaptive_mode == "uncertainty":
        model.log_sigma_det = nn.Parameter(torch.zeros(1, device=device))
        model.log_sigma_reid = nn.Parameter(torch.zeros(1, device=device))
        optimizer.add_param_group({"params": [model.log_sigma_det, model.log_sigma_reid], "lr": lr})
    for epoch in range(1, epochs + 1):
        model.train()
        total_cls = total_bbox = total_giou = total_reid = 0.0

        for imgs, targets in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()

            cls_preds, box_preds, reid_feats = model(imgs, return_reid=True)
            # Shapes:
            #   cls_preds -> [B, L, Q, C]
            #   box_preds -> [B, L, Q, 4]
            #   reid_feats -> [B, Q, D]
            cls_preds = cls_preds[:, -1]  # [B, Q, C]
            box_preds = box_preds[:, -1]  # [B, Q, 4]

            loss_cls, loss_bbox, loss_giou = detection_losses(cls_preds, box_preds, targets, device)
            loss_det = loss_cls + loss_bbox + 0.5 * loss_giou

            B, Q, D = reid_feats.shape
            reid_emb_flat = reid_feats.reshape(-1, D)
            logits = model.reid_classifier(reid_emb_flat)

            # === Build gt_ids_repeated matching [B*Q]
            gt_ids_repeated = torch.cat([
                t["ids"].repeat(Q // max(1, len(t["ids"])))[:Q]
                if len(t["ids"]) > 0 else torch.zeros(Q, dtype=torch.long, device=device)
                for t in targets
            ]).to(device)

            # Align
            min_len = min(len(gt_ids_repeated), logits.size(0))
            logits, gt_ids_repeated = logits[:min_len], gt_ids_repeated[:min_len]

            loss_id = id_loss(logits, gt_ids_repeated)

            # === Triplet Loss (aligned)
            reid_emb_flat = reid_emb_flat[:min_len]
            dist = torch.cdist(reid_emb_flat, reid_emb_flat)
            same_id = gt_ids_repeated.unsqueeze(1) == gt_ids_repeated.unsqueeze(0)

            if same_id.sum() > 1:
                pos_dist = (dist * same_id.float()).max(1)[0]
                neg_dist = (dist + 1e5 * same_id.float()).min(1)[0]
                loss_triplet = F.relu(pos_dist - neg_dist + 0.3).mean()
            else:
                loss_triplet = torch.tensor(0.0, device=device)

            loss_reid = 0.5 * loss_id + 0.25 * loss_triplet
            if adaptive_mode == "none":
                total_loss = loss_det + loss_reid
            elif adaptive_mode == "uncertainty":
                sigma_det = torch.exp(model.log_sigma_det)
                sigma_reid = torch.exp(model.log_sigma_reid)
                total_loss = (
                    (1.0 / (2 * sigma_det ** 2)) * loss_det +
                    (1.0 / (2 * sigma_reid ** 2)) * loss_reid +
                    model.log_sigma_det + model.log_sigma_reid
                )
            else:
                total_loss = loss_det + loss_reid

            total_loss.backward()
            optimizer.step()

            total_cls += loss_cls.item()
            total_bbox += loss_bbox.item()
            total_giou += loss_giou.item()
            total_reid += loss_reid.item()

        print(f"[Epoch {epoch}] cls={total_cls/len(dataloader):.4f} "
              f"bbox={total_bbox/len(dataloader):.4f} "
              f"giou={total_giou/len(dataloader):.4f} "
              f"reid={total_reid/len(dataloader):.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_ep{epoch}.pth")

    torch.save(model.state_dict(), save_path)
    print(f"✅ MOT17 joint fine-tuning completed → {save_path}")
