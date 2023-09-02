import torch
import torch.nn as nn
from torchvision.transforms import Compose

from tools import builder
from utils import dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from tqdm import tqdm

import numpy as np
from datasets.data_transforms import *


class Acc_Metric:
    def __init__(self, oa = 0., macc = 0., ins_iou = 0., cls_iou = 0.):
        self.oa = oa
        self.macc = macc
        self.ins_iou = ins_iou
        self.cls_iou = cls_iou

    def better_than(self, other):
        if self.ins_iou > other.ins_iou:
            return True
        elif self.ins_iou == other.ins_iou and self.cls_iou > other.cls_iou:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['oa'] = self.oa
        _dict['macc'] = self.macc
        _dict['ins_iou'] = self.ins_iou
        _dict['cls_iou'] = self.cls_iou
        return _dict


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric()
    metrics = Acc_Metric()

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
    else:
        print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
        if config.model.NAME.startswith('DAM'):
            # DAM must generate prototypes first
            base_model.generate_prototypes(train_dataloader.dataset, logger)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger=logger)
    else:
        print_log('Using Data parallel ...' , logger=logger)
        base_model = nn.DataParallel(base_model).cuda()

    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    if hasattr(config, 'train_transforms'):
        train_transforms = []
        if config.train_transforms.scale:
            train_transforms.append(PointcloudScale())
        if config.train_transforms.rotate:
            train_transforms.append(PointcloudRotate())
        if config.train_transforms.translate:
            train_transforms.append(PointcloudTranslate())
        if config.train_transforms.jitter:
            train_transforms.append(PointcloudJitter())
        if config.train_transforms.flip:
            train_transforms.append(RandomHorizontalFlip())
        train_transforms = Compose(train_transforms)
    else:
        train_transforms = Compose([
            PointcloudScaleAndTranslate(),
        ])

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        for idx, (points, label, target) in enumerate(tqdm(train_dataloader)):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points, label, target = points.cuda(), label.squeeze(-1).long().cuda(), target.long().cuda()

            points = train_transforms(points)
            ret = base_model(points, label)
            loss, acc = base_model.module.get_loss_acc(ret, target)

            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)

        if not hasattr(config, 'save_only_best') or not config.save_only_best:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    num_part = 50
    npoints = config.npoints
    seg_classes = test_dataloader.dataset.seg_classes

    total_correct = 0
    total_seen = 0
    total_seen_class = [0] * num_part
    total_correct_class = [0] * num_part
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    base_model.eval()  # set model to eval mode
    with torch.no_grad():
        for _, (points, label, target) in enumerate(tqdm(test_dataloader)):
            cur_batch_size = points.size(0)
            points, label, target = points.cuda(), label.squeeze(-1).long().cuda(), target.long().cuda()
            ret = base_model(points, label)
            cur_pred_val_logits = ret.cpu().numpy()
            cur_pred_val = np.zeros((cur_batch_size, npoints)).astype(np.int32)
            target = target.cpu().numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * npoints)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0] *  len(seg_classes[cat])
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        oa = total_correct / float(total_seen)
        macc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        cls_iou = mean_shape_ious
        ins_iou = np.mean(all_shape_ious)
        for cat in sorted(shape_ious.keys()):
            print_log('eval mIoU of %s %.4f' % (cat + ' ' * (14 - len(cat)), 100 * shape_ious[cat]), logger=logger)

    if val_writer is not None:
        val_writer.add_scalar('Metric/oa', oa, epoch)
        val_writer.add_scalar('Metric/macc', macc, epoch)
        val_writer.add_scalar('Metric/cls_iou', cls_iou, epoch)
        val_writer.add_scalar('Metric/ins_iou', ins_iou, epoch)

    print_log(f'[Validation] EPOCH: {epoch}  ins_iou = {100 * ins_iou:.4f}  cls_iou = {100 * cls_iou:.4f}', logger=logger)

    return Acc_Metric(oa, macc, ins_iou, cls_iou)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    # builder.load_model(base_model, args.ckpts, logger=logger) # for finetuned transformer
    base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    test(base_model, test_dataloader, args, config, logger=logger)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    # builder.load_model(base_model, args.ckpts, logger=logger) # for finetuned transformer
    base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger = None):

    num_part = 50
    npoints = config.npoints
    seg_classes = test_dataloader.dataset.seg_classes

    total_correct = 0
    total_seen = 0
    total_seen_class = [0] * num_part
    total_correct_class = [0] * num_part
    shape_ious = {cat: [] for cat in seg_classes.keys()}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    base_model.eval()  # set model to eval mode
    with torch.no_grad():
        for _, (points, label, target) in enumerate(tqdm(test_dataloader)):
            cur_batch_size = points.size(0)
            points, label, target = points.cuda(), label.squeeze(-1).long().cuda(), target.long().cuda()
            ret = base_model(points, label)
            cur_pred_val_logits = ret.cpu().numpy()
            cur_pred_val = np.zeros((cur_batch_size, npoints)).astype(np.int32)
            target = target.cpu().numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * npoints)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0] *  len(seg_classes[cat])
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        oa = total_correct / float(total_seen)
        macc = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        cls_iou = mean_shape_ious
        ins_iou = np.mean(all_shape_ious)
        for cat in sorted(shape_ious.keys()):
            print_log('eval mIoU of %s %.4f' % (cat + ' ' * (14 - len(cat)), 100 * shape_ious[cat]), logger=logger)

    print_log(f'[Test] ins_iou = {100 * ins_iou:.4f}  cls_iou = {100 * cls_iou:.4f}', logger=logger)
