# kmeans_pseudo_update.py

import torch
import numpy as np
from sklearn.cluster import KMeans

def run_kmeans_update(iter_num, fea_bank, pseudo_bank, score_bank):

#Using KMeans clustering with high-confidence sample-based center initialization to update pseudo-labels.

    if iter_num % 500 == 0:
        initial_centers = []
        for label in range(12):
            label_indices = (pseudo_bank == label).nonzero(as_tuple=True)[0]
            if label_indices.numel() > 0:
                num_samples = label_indices.size(0)
                top_k_percentage = max(1, int(num_samples * 0.3))
                top_k_scores, top_k_indices = torch.topk(score_bank[label_indices], top_k_percentage, dim=0,
                                                         largest=True)
                top_k_features = fea_bank[label_indices[top_k_indices]]
                center_feature = top_k_features.mean(dim=0)
                # if label_indices.numel() > 0:
                #     max_score_indices = torch.argmax(score_bank[label_indices], dim=0)
                #     center_feature = fea_bank[label_indices[max_score_indices]]
                if center_feature.dim() > 1 and center_feature.shape[1] == 1:
                    center_feature = center_feature.squeeze(1)
                initial_centers.append(center_feature.cpu().detach().numpy())

        initial_centers = np.array(initial_centers)
        initial_centers = initial_centers[:, 0, :]

        all_features = fea_bank.cpu().numpy()
        kmeans = KMeans(n_clusters=12, init=initial_centers, n_init=1)
        kmeans.fit(all_features)
        first_cluster_centers = kmeans.cluster_centers_

        kmeans_second = KMeans(n_clusters=12, init=first_cluster_centers, n_init=1)
        kmeans_second.fit(all_features)
        second_cluster_centers = kmeans_second.cluster_centers_

        kmeans_third = KMeans(n_clusters=12, init=second_cluster_centers, n_init=1)
        kmeans_third.fit(all_features)

        cluster_labels = kmeans_third.labels_
        cluster_centers = kmeans_third.cluster_centers_

        all_features_tensor = torch.tensor(all_features).cuda()
        cluster_centers_tensor = torch.tensor(cluster_centers).cuda()
        distances = torch.cdist(all_features_tensor, cluster_centers_tensor)

        new_pseudo_bank = pseudo_bank.clone()
        for cluster_id in range(12):
            cluster_indices = torch.where(torch.tensor(cluster_labels).cuda() == cluster_id)[0]
            cluster_distances = distances[cluster_indices, cluster_id]
            num_to_update = max(1, int(len(cluster_distances) * 0.7))
            top_70_indices = torch.argsort(cluster_distances)[:num_to_update]
            indices_to_update = cluster_indices[top_70_indices]
            new_pseudo_bank[indices_to_update] = cluster_id

        pseudo_bank = new_pseudo_bank

    return pseudo_bank
