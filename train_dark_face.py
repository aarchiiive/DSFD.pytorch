import wandb

import pytz
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from data.config import cfg
from models.factory import build_net
from layers.modules import MultiBoxLoss
from layers.functions import Detect

KST = pytz.timezone('Asia/Seoul')

def calculate_map(detections, targets, num_classes=2):
    """
    Vectorized function for fast mAP calculation.
    Processes predictions and ground truths (GT) for each class at once,
    and computes the Average Precision (AP) for IoU thresholds ranging from 0.5 to 0.95.
    """
    # Define IoU thresholds: 0.5, 0.55, ..., 0.95 (total 10 thresholds)
    iou_thresholds = torch.arange(0.5, 1.0, 0.05).to(detections[0].device)
    aps_50 = []    # AP at IoU 0.5
    aps_50_95 = [] # Mean AP over IoU thresholds from 0.5 to 0.95

    # Calculate for each class (excluding background class 0)
    for cls in range(1, num_classes):
        pred_scores_list = []  # List to store prediction scores for each image
        tp_list = []           # List to store true positive flags (for each IoU threshold) for each image
        total_gts = 0          # Total number of GT boxes for the class

        for det, gt in zip(detections, targets):
            if det.shape[0] == 0:
                continue

            # Extract predicted boxes and confidence scores for the current class
            pred_boxes = det[cls, :, 1:]  # shape: (N, 4)
            scores = det[cls, :, 0]       # shape: (N,)

            # Select only the GT boxes corresponding to the current class
            gt_boxes = gt[:, :4]
            gt_labels = gt[:, 4].long()
            gt_boxes_cls = gt_boxes[gt_labels == cls]
            total_gts += len(gt_boxes_cls)

            if pred_boxes.numel() == 0:
                continue

            # Sort predictions by confidence score in descending order
            sorted_indices = torch.argsort(scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]
            scores = scores[sorted_indices]

            # If there are no GT boxes, mark all predictions as false positives
            if gt_boxes_cls.numel() == 0:
                tp_image = torch.zeros((pred_boxes.size(0), len(iou_thresholds)))
            else:
                # Compute IoU between predicted boxes and GT boxes (all combinations)
                ious = box_iou(pred_boxes, gt_boxes_cls)  # shape: (N, num_gt)
                max_ious, _ = ious.max(dim=1)              # Maximum IoU for each prediction (N,)
                tp_image = (max_ious.unsqueeze(1) >= iou_thresholds.unsqueeze(0)).float()

            pred_scores_list.append(scores)
            tp_list.append(tp_image)

        # If no GT boxes exist for the class, set AP to 0
        if total_gts == 0 or len(pred_scores_list) == 0:
            aps_per_thresh = [0.0 for _ in range(len(iou_thresholds))]
        else:
            # Concatenate predictions from all images
            all_scores = torch.cat(pred_scores_list)       # Shape: (total number of predictions,)
            all_tp = torch.cat(tp_list, dim=0)               # Shape: (total predictions, number of thresholds)
            sorted_indices = torch.argsort(all_scores, descending=True)
            all_scores = all_scores[sorted_indices]
            all_tp = all_tp[sorted_indices]

            aps_per_thresh = []
            # For each IoU threshold, compute precision-recall curve and calculate AP
            for t in range(len(iou_thresholds)):
                tp_thresh = all_tp[:, t]
                fp_thresh = 1 - tp_thresh

                tp_cumsum = torch.cumsum(tp_thresh, dim=0)
                fp_cumsum = torch.cumsum(fp_thresh, dim=0)

                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                recalls = tp_cumsum / (total_gts + 1e-6)

                sorted_recalls, indices = torch.sort(recalls)
                sorted_precisions = precisions[indices]

                ap = torch.trapz(sorted_precisions, sorted_recalls).item()
                aps_per_thresh.append(ap)

        aps_50.append(aps_per_thresh[0])
        aps_50_95.append(sum(aps_per_thresh) / len(aps_per_thresh))

    mAP50 = sum(aps_50) / len(aps_50) if aps_50 else 0.0
    mAP50_95 = sum(aps_50_95) / len(aps_50_95) if aps_50_95 else 0.0

    return mAP50, mAP50_95

def adjust_learning_rate(optimizer, lr, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    _lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = _lr

if __name__ == '__main__':
    ### Configurations
    method = 'default'
    num_classes = 2
    num_samples = None
    model_name = 'resnet50'
    dataset_name = 'dark_face'
    learning_rate = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    gamma = 0.1
    # lr_steps = [80000, 100000, 120000]
    lr_steps = [40000, 50000, 60000]
    negpos_ratio = 3
    variance = [0.1, 0.2]
    threshold = 0.5
    device = torch.device('cuda:0')

    batch_size = 32
    num_epochs = 300
    max_steps = 150000
    save_freq = 5
    num_workers = 8
    # pretrained_weights = 'weights/resnet50-19c8e357.pth'
    pretrained_weights = 'weights/wider_face/epoch_100.pth'
    resume = None
    use_wandb = True

    # data_dir = Path('../datasets/DarkFace_Train_2021')
    data_dir = Path('/home/ubuntu/data/DarkFace_Train_2021')
    image_dir = data_dir / 'image'
    label_dir = data_dir / 'label'
    train_meta = data_dir / 'mf_dsfd_dark_face_train_5500.txt'
    val_meta = data_dir / 'mf_dsfd_dark_face_val_500.txt'

    project_name = 'dark-face'
    exp_name = 'DSDF-Wider-Face-pretrained'

    if not resume:
        save_dir = Path('runs')
        timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
        save_dir = save_dir / f"{timestamp}_{exp_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(resume).parent

    if use_wandb:
        wandb.init(project=project_name, name=exp_name)
        wandb.config.update({
            'timestamp': timestamp,
            'method': method,
            'num_classes': num_classes,
            'num_samples': num_samples,
            'model_name': model_name,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'gamma': gamma,
            'lr_steps': lr_steps,
            'negpos_ratio': negpos_ratio,
            'variance': variance,
            'threshold': threshold,
            'device': device,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'max_steps': max_steps,
            'save_freq': save_freq,
            'num_workers': num_workers,
            'pretrained_weights': pretrained_weights,
            'data_dir': data_dir,
            'image_dir': image_dir,
            'label_dir': label_dir,
            'train_meta': train_meta,
            'val_meta': val_meta,
            'save_dir': save_dir
        })

    ### Dataset and DataLoader
    if dataset_name == 'dark_face':
        from data.dark_face import DarkFaceDataset, collate_fn
        train_dataset = DarkFaceDataset(
            data_dir,
            train_meta,
            num_samples=num_samples,
            method=method,
            phase='train'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )

        val_dataset = DarkFaceDataset(
            data_dir,
            val_meta,
            num_samples=num_samples,
            method=method,
            phase='val'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size*4,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    elif dataset_name == 'wider_face':
        from data.widerface import WiderFaceDataset, collate_fn
        train_dataset = WiderFaceDataset(
            cfg.FACE.TRAIN_FILE, mode='train'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        val_dataset = WiderFaceDataset(
            cfg.FACE.VAL_FILE, mode='val'
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size*4,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    ### Model, Loss, Optimizer
    model = build_net('train', num_classes, model_name)
    if not resume:
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            model.load_state_dict(state_dict['model'])

            ### ResNet50
            # new_state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            # model_dict = model.resnet.state_dict()
            # model_dict.update(new_state_dict)
            # model.resnet.load_state_dict(model_dict)

            # model.extras.apply(model.weights_init)
            # model.fpn_topdown.apply(model.weights_init)
            # model.fpn_latlayer.apply(model.weights_init)
            # model.fpn_fem.apply(model.weights_init)
            # model.loc_pal1.apply(model.weights_init)
            # model.conf_pal1.apply(model.weights_init)
            # model.loc_pal2.apply(model.weights_init)
            # model.conf_pal2.apply(model.weights_init)
    else:
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict['model'])

    model = model.to(device)
    detect = Detect(
        num_classes=num_classes,
        top_k=cfg.TOP_K,
        nms_thresh=cfg.NMS_THRESH,
        conf_thresh=cfg.CONF_THRESH,
        variance=cfg.VARIANCE,
        nms_top_k=cfg.NMS_TOP_K
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    criterion = MultiBoxLoss(
        num_classes=num_classes,
        negpos_ratio=negpos_ratio,
        variance=variance,
        threshold=threshold
    )

    if not resume:
        train_iter = 0
        step_index = 0
        best_mAP = 0.0
    else:
        train_iter = state_dict['train_iter']
        step_index = state_dict['lr_step']
        best_mAP = state_dict['best_mAP']
        adjust_learning_rate(optimizer, learning_rate, gamma, step_index)

    for epoch in range(num_epochs):
        epoch_loss = 0
        tbar = tqdm(train_loader)
        ### Training
        model.train()

        for i, (images, targets) in enumerate(tbar):
            if train_iter in lr_steps:
                step_index += 1
                adjust_learning_rate(optimizer, learning_rate, gamma, step_index)
            # print(images.shape, targets, image_paths)
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            outputs = model(images)

            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(outputs[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(outputs[3:], targets)
            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            train_iter += 1

            tbar.set_description(f'[Train] Epoch {epoch+1}/{num_epochs}, Iteration {train_iter}, Loss: {loss.item():.4f}')

        if (epoch+1) % save_freq == 0:
            # torch.save(model.state_dict(), save_dir / 'weights' / f'epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch+1,
                'train_iter': train_iter,
                'lr_step': step_index,
                'best_mAP': best_mAP,
                'model': model.state_dict(),
            }, save_dir / 'weights' / f'epoch_{epoch+1}.pt')

        if use_wandb:
            wandb.log({'train/loss': epoch_loss / len(train_loader)}, step=epoch)

        ## Validation
        model.eval()
        val_loss = 0
        all_detections = []
        all_targets = []

        with torch.no_grad():
            tbar = tqdm(val_loader)
            for images, targets in tbar:
                images, targets = images.to(device), [t.to(device) for t in targets]
                outputs = model(images)

                loss_l_pal1, loss_c_pal1 = criterion(outputs[:3], targets)
                loss_l_pal2, loss_c_pal2 = criterion(outputs[3:], targets)
                loss = loss_l_pal2 + loss_c_pal2
                val_loss += loss.item()

                detections = detect(outputs[3], F.softmax(outputs[4], dim=-1), outputs[5])
                all_detections.extend(detections)
                all_targets.extend(targets)

                tbar.set_description(f'[Val] Epoch {epoch+1}, Loss: {loss.item():.4f}')

            # mAP = calculate_map(all_detections, targets, num_classes=num_classes)
            mAP50, mAP50_95 = calculate_map(all_detections, all_targets, num_classes=num_classes)

            if use_wandb:
                wandb.log({'val/loss': val_loss / len(val_loader), 'val/mAP@50': mAP50, 'val/mAP@50-95': mAP50_95}, step=epoch)

        print(f"Epoch {epoch+1} - mAP@50: {mAP50:.4f}, mAP@50-95: {mAP50_95:.4f}")

        if mAP50_95 > best_mAP:
            best_mAP = mAP50_95
            torch.save(model.state_dict(), save_dir / 'weights' / 'best.pt')
            torch.save({
                'epoch': epoch+1,
                'train_iter': train_iter,
                'lr_step': step_index,
                'best_mAP': best_mAP,
                'model': model.state_dict(),
            }, save_dir / 'weights' / 'best.pt')
            print(f"New best model saved with mAP@50-95: {mAP50_95:.4f}")

        if train_iter >= max_steps:
            break