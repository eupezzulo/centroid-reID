from __future__ import print_function, division
from tqdm import tqdm
import torch.nn.functional as F
import torch

# https://github.com/VITA-Group/FAT
# See https://arxiv.org/pdf/1912.07863v1.pdf for more detail.

def get_centroids(cfg, model, loader):
    """
        Auxiliary method for calculating centroids.
    """
    model.train()
    centroids_ori = {}
    centroids_count = {}
    batch_size = 16

    # fixed 'CUDA OUT of memory' error with batch processing
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(loader)):
            images = sample[0].to(cfg.MODEL.DEVICE)
            labels = sample[1].to(cfg.MODEL.DEVICE)

            num_samples = len(images)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_images = images[start_idx : end_idx]
                batch_labels = labels[start_idx : end_idx]

                _, features = model(batch_images)

                for n, k in enumerate(batch_labels):                    
                    k_item = k.item()
                    if k_item in centroids_count:
                        centroids_ori[k_item] = centroids_ori[k_item] + features[n]
                        centroids_count[k_item] = centroids_count[k_item] + 1
                    else:
                        centroids_ori[k_item] = features[n]
                        centroids_count[k_item] = 1
            
            torch.cuda.empty_cache()

    # mean of not normalized feature vectors
    centroids_ori = {k : centroids_ori[k] / centroids_count[k] for k in centroids_ori}
    return centroids_ori


def get_gallery_set(centroids):
    """
       Auxiliary method to rearrange the centroids as a torch.tensor.
    """
    N = len(centroids)   
    g_vector = torch.t(torch.stack([centroids[k] for k in range(N)]))
     
    return g_vector


def get_pos_set(centroids, labels):
    """
        Auxiliary method to obtain the set of positive samples.
        Specifically, returned value is a torch.tensor that represent
        the not normalized positive samples.
    """
    pos = torch.stack([centroids[k.item()] for k in labels])
    return pos


def get_neg_set_batch(anchor, labels):
    """
        Auxliary method to obtain the set of negative samples.
        This set is computed from the current batch.
    """
    min_len = 5
    N = len(anchor)

    batch_distance = [F.pairwise_distance(anchor[k].expand(N, -1), anchor) for k in range(N)]
    sorted_key_ctrd = {k: torch.sort(batch_distance[k].cpu(), dim=0)[1] for k in range(N)}
    sorted_key_ctrd = {k: [int(n) for n in sorted_key_ctrd[k] if labels[int(n)] != labels[k]] for k in range(N)}

    min_len = min(min([len(sorted_key_ctrd[k]) for k in sorted_key_ctrd]), min_len)
    neg_ft = [torch.stack([anchor[sorted_key_ctrd[n][k]].data for n in range(N)]) for k in range(min_len)]
    
    return neg_ft