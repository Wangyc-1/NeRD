# compute_total_loss.py

import torch
import torch.nn.functional as F

def compute_total_loss(inputs_test,
                       score_test,
                       score_near,
                       score_far,
                       perturbations,
                       features_test,
                       args.alpha1,
                       args.alpha2,
                       pseudo_label,
                       args,
                       alpha):

    mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
    diag_num = torch.diag(mask)
    mask_diag = torch.diag_embed(diag_num)
    mask = mask - mask_diag
    copy = score_test.T
    dot_neg = score_test @ copy
    dot_neg = ((dot_neg ** 2) * mask.cuda()).sum(-1)
    neg_pred = torch.mean(dot_neg)
    loss_perturbations = 0.0
    for pert in perturbations:
        loss_perturbations += torch.mean(
            (F.kl_div(pert.unsqueeze(1).expand(-1, args.K, -1).cuda(),
                      score_near.cuda(), reduction="none").sum(-1)).sum(1)
        )
    # loss_perturbations = loss_perturbations / num_perturbations
    # === Feature Loss () ===
    loss = torch.mean(
        (F.kl_div(score_test.unsqueeze(1).expand(-1, args.K, -1).cuda(),
                  score_near.cuda(), reduction="none").sum(-1)).sum(1)
    ) - args.alpha1 * loss_perturbations \
        - args.alpha1 * torch.mean(
        (F.kl_div(score_near.cuda(),
                  score_test.unsqueeze(1).expand(-1, args.K, -1).cuda(), reduction="none").sum(-1)).sum(1)
    ) + args.alpha2 * torch.mean(
        (F.kl_div(score_test.unsqueeze(1).expand(-1, args.K, -1).cuda(),
                  score_far.cuda(), reduction="none").sum(-1)).sum(1)
    ) + neg_pred * alpha

    # === Pseudo-label Loss (pse) ===
    K = args.K
    B = features_test.size(0)
    features_norm = F.normalize(features_test, dim=1)  # [B, D]
    sim_matrix = torch.matmul(features_norm, features_norm.T)
    _, knn_indices = sim_matrix.topk(K + 1, dim=1)
    knn_indices = knn_indices[:, 1:]  # remove self
    loss_pse = 0.0
    ncl_count = 0
    for i in range(B):
        xi = score_test[i]
        label_i = pseudo_label[i]
        for j in knn_indices[i]:
            if pseudo_label[j] == label_i:
                xj = score_test[j]
                loss_pse += F.mse_loss(xi, xj, reduction='sum')
                ncl_count += 1
    if ncl_count > 0:
        loss_pse = loss_pse / ncl_count
    loss += args.dd * loss_pse

    return loss
