import torch
import torch.nn.functional as F
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def hungarian_match(cost_total):
    """Run Hungarian algorithm on CPU to save GPU RAM"""
    return linear_sum_assignment(cost_total.cpu().numpy())

def compute_sample_loss(
    o_bbox, t_bbox, o_cl, t_cl, t_mask,
    n_queries=100, empty_class_id=91, device="cuda"
):
    """
    Optimized DETR loss computation — reduced RAM & GPU copies
    """
    # --- Filter valid ground truths ---
    valid_gt = t_mask.nonzero(as_tuple=False).squeeze(-1)
    if valid_gt.numel() == 0:
        # No GT: only classification to "no object"
        loss_class = F.cross_entropy(o_cl, torch.full((n_queries,), empty_class_id, device=device))
        return loss_class, torch.tensor(0., device=device), torch.tensor(0., device=device)

    t_bbox = t_bbox[valid_gt]
    t_cl = t_cl[valid_gt]

    # --- Compute classification probabilities ---
    o_probs = o_cl.softmax(-1)

    # Compute per-pair costs without expanding all tensors permanently
    with torch.no_grad():
        # Only keep relevant columns for class cost (saves memory)
        C_classes = -o_probs[:, t_cl]          # [num_queries, num_gt]
        C_boxes = torch.cdist(o_bbox, t_bbox, p=1)  # [num_queries, num_gt]
        C_giou = -ops.generalized_box_iou(
            ops.box_convert(o_bbox, "cxcywh", "xyxy"),
            ops.box_convert(t_bbox, "cxcywh", "xyxy"),
        )

        # Weighted total cost (float32 precision)
        C_total = 1.0 * C_classes + 5.0 * C_boxes + 2.0 * C_giou

        # Hungarian matching on CPU (small 2D matrix)
        o_ixs, t_ixs = hungarian_match(C_total)

    # Convert back to torch tensors on GPU
    o_ixs = torch.as_tensor(o_ixs, device=device)
    t_ixs = torch.as_tensor(t_ixs, device=device)

    # --- Box regression losses ---
    num_boxes = len(t_bbox)
    matched_o_bbox = o_bbox[o_ixs]
    matched_t_bbox = t_bbox[t_ixs]

    loss_bbox = F.l1_loss(matched_o_bbox, matched_t_bbox, reduction="sum") / num_boxes

    giou = ops.generalized_box_iou(
        ops.box_convert(matched_o_bbox, "cxcywh", "xyxy"),
        ops.box_convert(matched_t_bbox, "cxcywh", "xyxy"),
    )
    loss_giou = (1 - torch.diag(giou)).mean()

    # --- Classification loss ---
    target_labels = torch.full((n_queries,), empty_class_id, device=device)
    target_labels[o_ixs] = t_cl[t_ixs]
    loss_class = F.cross_entropy(o_cl, target_labels)

    return loss_class, loss_bbox, loss_giou
