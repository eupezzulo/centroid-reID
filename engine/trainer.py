import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tqdm import tqdm


from utils.reid_metric import R1_mAP, R1_mAP_reranking, R1_mAP_centroids
from utils.centroids_utils import *


global centroids, ITER
ITER = 0

def create_supervised_trainer_with_centroids(model, optimizer, loss_fn, device=None):
    """
        Factory function for creating a trainer for supervised models.
        Specifically, the returned trainer uses the loss function given
        by cross-entropy, centroid-based loss, and triplet loss to guide training.
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        global centroids
        
        model.train()
        optimizer.zero_grad()

        img, labels = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels

        outputs, anchor = model(img)    
        # 'outputs' is cls_score calculated on features after BN layer  
        # 'anchor' is globa_feat, so features before BN layer

        # get set of positive samples
        pos = get_pos_set(centroids, labels)
        # get set of negative samples (from the batch)
        neg_ft = get_neg_set_batch(anchor, labels) 

        loss = loss_fn(outputs, labels, anchor, pos, neg_ft)
        loss.backward()
        optimizer.step()

        acc = (outputs.max(1)[1] == labels).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)

def create_supervised_trainer_with_centroids_and_center(model, center_criterion, optimizer, optimizer_center, loss_fn, center_loss_weight, device=None):
    """
        Factory function for creating a trainer for supervised models.
        Specifically, the returned trainer uses the loss function given
        by cross-entropy, centroid-based loss, triplet loss, and center loss
        to guide training.
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        global centroids
        
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()

        img, labels = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels

        outputs, anchor = model(img)      
        # 'outputs' is cls_score calculated on features after BN layer  
        # 'anchor' is globa_feat, so features before BN layer

        # get set of positive samples
        pos = get_pos_set(centroids, labels)
        # get set of negative samples (from the batch)
        neg_ft = get_neg_set_batch(anchor, labels) 

        loss = loss_fn(outputs, labels, anchor, pos, neg_ft)
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / center_loss_weight)
        optimizer_center.step()

        acc = (outputs.max(1)[1] == labels).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

# do train with loss composed by cross-entropy, triplet, and centroid-based terms.
def do_train_with_centroids(
        cfg,
        model,
        train_loader,
        centroid_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    re_ranking = cfg.TEST.RE_RANKING
    with_centroids = cfg.TEST.WITH_CENTROIDS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_centroids(model, optimizer, loss_fn, device=device)
    
    if with_centroids == 'yes':
        print("Create evaluator with centroids")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_centroids(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)    
    elif re_ranking == 'no':
        print("Create evaluator without re-ranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    elif re_ranking == 'yes':
        print("Create evaluator with re-ranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(re_ranking))

    # fixed EarlyStopping issue
    checkpoint_handler = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpoint_handler, {'model': model, 'optimizer': optimizer})

    timer = Timer(average=True)

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def check_do_initial_compute_val(engine):
        # before evaluation, calculate the centroids
        # with the updated features
        global centroids
        if engine.state.epoch % eval_period == 0:
            centroids = get_centroids(cfg, model, centroid_loader)


    @trainer.on(Events.STARTED)
    def check_do_initial_compute_train(engine):
        # at the start of the training, calculte the centroids
        global centroids
        engine.state.epoch = start_epoch
        centroids = get_centroids(cfg, model, centroid_loader)


    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    trainer.run(train_loader, max_epochs=epochs)

# do train with loss composed by cross-entropy, triplet, centroid-based, and center terms.
def do_train_with_centroids_and_center(            
            cfg,
            model,
            center_criterion,
            train_loader,
            centroid_loader,
            val_loader,
            optimizer,
            optimizer_center,
            scheduler,
            loss_fn,
            num_query,
            start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    re_ranking = cfg.TEST.RE_RANKING

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_centroids_and_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    
    if re_ranking == 'no':
        print("Create evaluator without re-ranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    elif re_ranking == 'yes':
        print("Create evaluator with re-ranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(re_ranking))

    # fixed EarlyStopping issue
    checkpoint_handler = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)    
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=checkpoint_period), checkpoint_handler, {'model': model, 'optimizer': optimizer})

    timer = Timer(average=True)

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def check_do_initial_compute_val(engine):
        # before evaluation, calculate the centroids
        # with the updated features
        global centroids
        if engine.state.epoch % eval_period == 0:
            centroids = get_centroids(cfg, model, centroid_loader)

    @trainer.on(Events.STARTED)
    def check_do_initial_compute_train(engine):
        # at the start of the training, calculte the centroids
        global centroids
        engine.state.epoch = start_epoch

        centroids = get_centroids(cfg, model, centroid_loader)


    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    trainer.run(train_loader, max_epochs=epochs)