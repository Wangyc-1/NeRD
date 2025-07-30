# compute_total_loss.py

import torch
import torch.nn.functional as F
def compute_total_loss(inputs_test,
                       score_test,
                       score_near,
                       score_far,
                       perturbations,
                       features_test,
                       pseudo_label,
                       args,
                       alpha
                       ):
    # ===== conf =====
    batch_size = inputs_test.shape[0]
    self_mask = torch.eye(batch_size, batch_size).cuda()
    cross_mask = torch.ones_like(self_mask) - self_mask
    score_transpose = score_test.T
    cross_dot = score_test @ score_transpose
    contrastive_neg_dot = (cross_dot ** 2) * cross_mask
    contrastive_neg_pred = torch.mean(contrastive_neg_dot.sum(-1))
    def kl_div_with(p, q):
        e_constant = torch.exp(torch.tensor(1.0, device='cuda'))
        eps = 1e-10
        p_log_original = torch.log(p + eps)
        q_log_original = torch.log(q + eps)
        p_log_e = 1.0 + torch.log(p / e_constant + eps)
        q_log_e = 1.0 + torch.log(q / e_constant + eps)
        kl_original = p * (p_log_original - q_log_original)
        kl_with = p * (p_log_e - q_log_e)
        return kl_with
    loss_perturbations = 0.0
    for pert in perturbations:
        kl_div_pert = kl_div_with(
            pert.unsqueeze(1).expand(-1, args.K, -1).cuda(),
            score_near.cuda()
        ).sum(-1).sum(1)
        loss_perturbations += torch.mean(kl_div_pert)
    kl_div_positive = kl_div_with(
        score_test.unsqueeze(1).expand(-1, args.K, -1).cuda(),
        score_near.cuda()
    ).sum(-1).sum(1)
    loss_positive = torch.mean(kl_div_positive)
    kl_div_positive_rev = kl_div_with(
        score_near.cuda(),
        score_test.unsqueeze(1).expand(-1, args.K, -1).cuda()
    ).sum(-1).sum(1)
    loss_positive_rev = torch.mean(kl_div_positive_rev)
    kl_div_negative = kl_div_with(
        score_test.unsqueeze(1).expand(-1, args.K, -1).cuda(),
        score_far.cuda()
    ).sum(-1).sum(1)
    loss_negative = torch.mean(kl_div_negative)
    loss = (
            loss_positive
            - args.alpha1 * loss_perturbations
            - args.alpha1 * loss_positive_rev
            + args.alpha2 * loss_negative
            + contrastive_neg_pred * alpha
    )
    # === Feature distance Loss (pl) ===
    K = args.K
    B = features_test.size(0)
    features_norm = F.normalize(features_test, dim=1)  # [B, D]
    sim_matrix = torch.matmul(features_norm, features_norm.T)
    _, knn_indices = sim_matrix.topk(K + 1, dim=1)
    knn_indices = knn_indices[:, 1:]  # remove self
    loss_pl = 0.0
    ncl_count = 0
    for i in range(B):
        xi = score_test[i]
        label_i = pseudo_label[i]
        for j in knn_indices[i]:
            if pseudo_label[j] == label_i:
                xj = score_test[j]
                loss_pl += F.mse_loss(xi, xj, reduction='sum')
                ncl_count += 1
    if ncl_count > 0:
        loss_pl = loss_pl / ncl_count
    loss += args.Lambda * loss_pl

    return loss
