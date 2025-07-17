# neighbor_filter.py


import torch
import torch.nn.functional as F

def adjust_neighbors(idx_near, output_f_, fea_bank, pseudo_bank, score_bank, args):
    adjusted_idx_near = idx_near.clone()
    for i in range(output_f_.size(0)):
        a = i
        neighbors_a = idx_near[a]
        valid_neighbors = []
        for neighbor in neighbors_a:
            if neighbor >= fea_bank.size(0):
                continue
            neighbors_b = idx_near[neighbor % output_f_.size(0)]
            if a in neighbors_b:
                pseudo_a = pseudo_bank[a].float()
                pseudo_b = pseudo_bank[neighbor].float()
                pseudo_similarity = F.cosine_similarity(
                    pseudo_a.unsqueeze(0), pseudo_b.unsqueeze(0), dim=-1
                )
                if pseudo_similarity.item() > 0.5:
                    valid_neighbors.append(neighbor)

        if len(valid_neighbors) < args.K:
            nearest_idx = valid_neighbors[0] if valid_neighbors else neighbors_a[0]
            while len(valid_neighbors) < args.K:
                valid_neighbors.append(nearest_idx)

        adjusted_idx_near[a, :] = torch.tensor(valid_neighbors)

    score_nn = score_bank[adjusted_idx_near]
    return adjusted_idx_near, score_nn

