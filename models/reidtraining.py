import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models.conditional_detr_hybrid import ConditionalDETR  # hybrid model

def get_loaders(root, batch_size=32):
    tfm = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=tfm)
    val_path = os.path.join(root, "val")
    val_ds = datasets.ImageFolder(val_path, transform=tfm) if os.path.exists(val_path) else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0) if val_ds else None
    return train_dl, val_dl, len(train_ds.classes)

def train_reid_stage1(
    market_root, detr_ckpt="", save_path="reid_market.pth",
    lr=1e-4, epochs=40, batch_size=32, reid_dim=256, device="cuda"
):
    train_dl, val_dl, num_ids = get_loaders(market_root, batch_size)
    print(f"Loaded Market1501 with {num_ids} IDs")

    model = ConditionalDETR(
        use_reid=True, use_temporal=False, use_reid_classifier=True,
        reid_dim=reid_dim, n_reid_classes=num_ids, device=device
    ).to(device)

    if detr_ckpt and os.path.exists(detr_ckpt):
        ckpt = torch.load(detr_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)
        print("Loaded pretrained DETR weights")

    model.freeze_non_reid()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_id = 0.0

        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            out = model(imgs, return_reid=True)
            if len(out) == 5:
                _, _, reid_embeds, _, _ = out
            else:
                _, _, reid_embeds = out
            reid_embeds = reid_embeds.mean(1)
            logits = model.reid_classifier(reid_embeds)

            id_loss = ce_loss(logits, labels)
            loss = 0.5 * id_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_id += id_loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(train_dl):.4f}, id_loss={total_id/len(train_dl):.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{save_path}_ep{epoch}.pth")

    torch.save(model.state_dict(), save_path)
    print(f"✅ ReID fine-tuning completed → {save_path}")
