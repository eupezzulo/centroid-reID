# encoding: utf-8
import torch.nn.functional as F
import torch 

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_centroids(cfg, num_classes):
    # creates a loss function that includes cross entropy 'xe',
    # trplet loss 'triMargin', and centroid-based loss 'ctrdMargin'
    # as terms
    if cfg.MODEL.METRIC_LOSS_TYPE == 'XE-tri-mg':
        loss_reg_weight = {'xe': 0.5,
                           'triplet': 1,
                           'centroid': 1
                          }
        
        margin = cfg.SOLVER.MARGIN

        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=cfg.MODEL.DEVICE!='cpu') 
            print("label smooth on, numclasses:", num_classes)
        
        def loss_func(outputs, labels, anchor, pos, neg_ft, loss_reg_weight=loss_reg_weight):
            """
                'centroids' is a dict like 'centroids[j] = feature[j]'.
                'labels' and 'pos' are torch.tensor objects.

                for example if labels is a tensor as tensor([0, 0, 1, 1])
                then 'pos' will be 'tensor([feature[0],
                                            feature[0],
                                            feature[1],
                                            feature[1]])

                'anchor' follows the same order as labels, so we can compute
                the distance between each extracted feature vector and its
                respective centroids as follows.
            """

            triplet_loss = 0
            for neg in neg_ft:
                dist_ap = F.pairwise_distance(anchor, pos)
                dist_an = F.pairwise_distance(anchor, neg)
                # clamp between 0 and (dist_ap - dist_an + margin)
                loss_triplet = torch.clamp(dist_ap - dist_an + margin, min=0.)
                triplet_loss += loss_triplet.mean()
            
            triplet_loss /= len(neg_ft)
            # triplet weight
            triplet_loss *= loss_reg_weight['triplet']

            # centroid weight
            centroid_loss = loss_reg_weight['centroid'] * torch.mean(F.pairwise_distance(pos, anchor))
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                xe = xent(outputs, labels)
                # xe weight
                return loss_reg_weight['xe'] * xe + triplet_loss + centroid_loss
            else:
                # xe weight
                xe = loss_reg_weight['xe'] * F.cross_entropy(outputs, labels)
                return loss_reg_weight['xe'] * xe + triplet_loss + centroid_loss
    else:
        print('expected METRIC_LOSS_TYPE should be XE-tri-mg '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    return loss_func
    

def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    elif cfg.MODE.NAME == 'efficientnet':
        feat_dim = 1280
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion


def make_loss_with_centroids_and_center(cfg, num_classes):
    # creates a loss function that includes cross entropy 'xe',
    # trplet loss 'triMargin', centroid-based loss 'ctrdMargin',
    # and center loss 'center_criterion' as terms
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    elif cfg.MODEL.NAME == 'efficientnet':
        feat_dim = 1280
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'XE-tri-mg-center':
        loss_reg_weight = {'xe': 1, 
                           'triplet': 1,
                           'centroid': 1
                          }
        margin = 1.

        if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
            print("center loss on")
            center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=cfg.MODEL.DEVICE!='cpu')

        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=cfg.MODEL.DEVICE!='cpu') 
            print("label smooth on, numclasses:", num_classes)
        
        def loss_func(outputs, labels, anchor, pos, neg_ft, loss_reg_weight=loss_reg_weight, use_margin=1.):

            triplet_loss = 0
            for neg in neg_ft:
                dist_ap = F.pairwise_distance(anchor, pos)
                dist_an = F.pairwise_distance(anchor, neg)

                loss_triplet = torch.clamp(dist_ap - dist_an + margin, min=0.)
                triplet_loss += loss_triplet.mean()
            
            triplet_loss /= len(neg_ft)
            triplet_loss *= loss_reg_weight['triplet']

            centroid_loss = loss_reg_weight['centroid'] * torch.mean(F.pairwise_distance(pos, anchor))
            
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                final_loss = loss_reg_weight['xe'] * xent(outputs, labels) + triplet_loss + centroid_loss
            else:
                xe = loss_reg_weight['xe'] * F.cross_entropy(outputs, labels)
                final_loss = xe + triplet_loss + centroid_loss

            return cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(anchor, labels) + final_loss

    else:
        print('expected METRIC_LOSS_TYPE should be XE-tri-mg '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    
    return loss_func, center_criterion