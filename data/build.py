# encoding: utf-8
from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler
from .transforms import build_transforms

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    # data used to calculate centroids. Data augmentation is not applied
    centroids_transorms = build_transforms(cfg, is_train=False)
    val_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    
    if cfg.DATALOADER.SAMPLER == 'softmax':
        print('>> using RandomSampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn, pin_memory=True
        )
    else:
        print('>> using RandomIdentitySampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn, pin_memory=True
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn, pin_memory=True
    )

    # create centroid set and centroid loader
    # the latter will be useful for calculating centroids
    centroid_set = ImageDataset(dataset.train, centroids_transorms)
    centroid_loader = DataLoader(
        centroid_set, batch_size=cfg.SOLVER.IMS_PER_BATCH , shuffle=False, num_workers=num_workers,
        collate_fn=train_collate_fn, pin_memory=True
    )
    
    return train_loader, val_loader, len(dataset.query), num_classes, centroid_loader
