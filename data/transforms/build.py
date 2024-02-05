# encoding: utf-8
import torchvision.transforms as T

from .transforms import *

# In the config file, if you use 'imagenet' pre-trained backbone use the following values:
#   PIXEL_MEAN = [0.485, 0.456, 0.406]
#   PIXEL_STD = [0.229, 0.224, 0.225]
# Otherwise, if you use 'luperson' pre-trained backbone use the following values:
#   PIXEL_MEAN = [0.3525, 0.3106, 0.3140]
#   PIXEL_STD = [0.2260, 0.2522, 0.2505]
 
def build_transforms(cfg, is_train=True):
    # in according with original pre-processing of EfficientNetB1
    random_crop_size = cfg.INPUT.SIZE_TRAIN
    if cfg.MODEL.NAME == 'efficientnet':
        random_crop_size = [240, 240]
        
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
            RandomColorJitter(probability=cfg.INPUT.CJ_PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(random_crop_size),
            LGT(probability=cfg.INPUT.RCD_PROB),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
