# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

class R1_mAP_centroids(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_centroids, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def get_centroids(self, gallery_ft, gallery_pid):
        # get centroids (optimized version)
        centroids = {}
        centroids_count = {}

        for n, k in enumerate(gallery_pid):
            k_item = k.item()
            if k_item in centroids_count: 
                centroids[k_item] = centroids[k_item] + gallery_ft[n]
                centroids_count[k_item] = centroids_count[k_item] + 1
            else:
                centroids[k_item] = gallery_ft[n]
                centroids_count[k_item] = 1
        
        return centroids, centroids_count
 
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # centroids gallery
        centroids, centroids_count = self.get_centroids(gf, g_pids)
        # 'centroids' contains the sum of all feature vectors of the same pid
        # 'centroids_count' contains for each pid how many feature vectors there are

        # in accordance with Market1501 evaluation guidelines, images
        # with the same camid and the same pid should not be considered in the
        # retrieval stage. So, the feature vector with the same camid is subtracted
        # from 'centroids[pid]' and one is substracted from 'centroids_count[pid]'.
        m, n = qf.shape[0], len(centroids)
        distmat = torch.zeros((m, n)) # shape of (# query, # pid)

        for k, query in enumerate(qf):
            # compute centroids for current query
            curr_pid = q_pids[k]
            curr_camid = q_camids[k]

            matching_indices = np.where((np.array(g_pids) == curr_pid) & (np.array(g_camids) == curr_camid))[0]

            if matching_indices.size > 0:
                index = matching_indices[0]
                centroids[curr_pid] = centroids[curr_pid] - gf[index]
                centroids_count[curr_pid] = centroids_count[curr_pid] - 1


            curr_centroids = {k: centroids[k] / centroids_count[k] for k in centroids}
            curr_centroids_tensor = torch.stack(list(curr_centroids.values()), dim=0)
            c_pids = np.array(list(curr_centroids.keys()))

            distances = torch.norm(query - curr_centroids_tensor, dim=1)
            distmat[k] = distances
        
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, c_pids, q_camids, g_camids, with_centroids=True)
        return cmc, mAP