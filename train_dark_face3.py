import wandb

import pytz
import pickle
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.functional import F
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from data.config import cfg
from models.factory import build_net
from layers.modules import MultiBoxLoss
from layers.functions import Detect
from metrics import norm_score, eval_image, image_pr_info, save_pr_curve

KST = pytz.timezone('Asia/Seoul')

class Scheduler:
    def __init__(self, optimizer, gamma, lr_steps, default_lr=1e-4, warmup_lr=1e-6, start_iteration=0, warmup_iter=1000):
        self.iteration = start_iteration
        self.warmup_iter = warmup_iter
        self.default_lr = default_lr
        self.warmup_lr = warmup_lr
        self.lr = warmup_lr
        self.lr_steps = lr_steps
        self.step_index = 0
        self.gamma = gamma

        if start_iteration > warmup_iter:
            for step in self.lr_steps:
                if start_iteration > step:
                    self.step_index += 1
            self.lr = default_lr * (self.gamma ** self.step_index)
        else:
            self.lr = warmup_lr + (default_lr - warmup_lr) / warmup_iter * start_iteration

        self.update_param(optimizer)

    def update(self, optimizer):
        self.iteration += 1
        if self.iteration < self.warmup_iter:
            self.lr = self.warmup_lr + (self.default_lr - self.warmup_lr) / self.warmup_iter * self.iteration
            self.update_param(optimizer)
        elif self.iteration in self.lr_steps:
            self.step_index += 1
            self.lr = self.default_lr * (self.gamma ** self.step_index)
            self.update_param(optimizer)
        return self.lr

    def update_param(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

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
    # method = 'default'
    method = 'Ours'
    # method = 'SCI'
    num_classes = 2
    num_samples = None
    model_name = 'resnet50'
    dataset_name = 'dark_face'
    # dataset_name = 'wider_face'

    ### Loss, Optimizer
    learning_rate = 1e-3
    # learning_rate = 1e-4 # HLA-Face
    weight_decay = 5e-4
    momentum = 0.9
    gamma = 0.1
    lr_steps = [160000, 200000, 240000] # batch_size=8
    # lr_steps = [80000, 100000, 120000] # batch_size=16
    # lr_steps = [40000, 50000, 60000] # batch_size=32

    ## HLA-Face
    # lr_steps = [20000, 100000] # batch_size=8
    # lr_steps = [10000, 50000] # batch_size=16
    # lr_steps = [5000, 25000] # batch_size=32

    negpos_ratio = 3
    variance = [0.1, 0.2]
    threshold = 0.5
    device = torch.device('cuda:0')

    ### Training
    batch_size = 8
    num_epochs = 100
    max_steps = 300000 # batch_size=8
    # max_steps = 150000 # batch_size=16
    # max_steps = 75000 # batch_size=32

    ## HLA-Face
    # max_steps = 70000 # batch_size=8
    # max_steps = 35000 # batch_size=16
    # max_steps = 17500 # batch_size=32

    save_freq = 5
    num_workers = 8
    pretrained_weights = 'weights/resnet50-19c8e357.pth'
    # pretrained_weights = 'weights/wider_face/epoch_100.pth'
    resume = None
    use_wandb = True

    ### Evaluation
    iou_thresh = 0.5
    num_thresh = 1000

    ### Data
    # data_dir = Path('../datasets/DarkFace_Train_2021')
    data_dir = Path('datasets/DarkFace_Train_2021')
    image_dir = data_dir / 'image'
    label_dir = data_dir / 'label'
    train_meta = data_dir / 'mf_dsfd_dark_face_train_5500.txt'
    val_meta = data_dir / 'mf_dsfd_dark_face_val_500.txt'

    project_name = 'dark-face'
    # exp_name = 'DSDF-SCI-resnet-pretrained'
    # exp_name = 'DSDF-baseline-resnet-pretrained-epoch300'
    # exp_name = 'DSDF-Ours-resnet-pretrained-HLA-Face'
    # exp_name = 'DSDF-baseline-resnet-pretrained-HLA-Face'
    exp_name = 'DSFD-Ours-resnet-pretrained-cosine'

    if not resume:
        save_dir = Path('runs')
        timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
        save_dir = save_dir / f"{timestamp}_{exp_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
        (save_dir / 'pr_curve').mkdir(parents=True, exist_ok=True)
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
        from data.dark_face import DarkFaceDataset, collate_fn, collate_fn_fft
        if method == 'Ours':
            _collate_fn = collate_fn_fft
        else:
            _collate_fn = collate_fn

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
            collate_fn=_collate_fn,
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
            collate_fn=_collate_fn,
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
    if method == 'Ours':
        from models.DSFD_ours import build_net_resnet
        model = build_net_resnet('train', num_classes, model_name)
    else:
        from models.factory import build_net
        model = build_net('train', num_classes, model_name)

    if not resume:
        if pretrained_weights:
            state_dict = torch.load(pretrained_weights, map_location='cpu')
            # model.load_state_dict(state_dict['model'])

            ### ResNet50
            new_state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            model_dict = model.resnet.state_dict()
            model_dict.update(new_state_dict)
            model.resnet.load_state_dict(model_dict)

            model.extras.apply(model.weights_init)
            model.fpn_topdown.apply(model.weights_init)
            model.fpn_latlayer.apply(model.weights_init)
            model.fpn_fem.apply(model.weights_init)
            model.loc_pal1.apply(model.weights_init)
            model.conf_pal1.apply(model.weights_init)
            model.loc_pal2.apply(model.weights_init)
            model.conf_pal2.apply(model.weights_init)

            print(f"Pretrained weights loaded from {pretrained_weights}")
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
    # scheduler = Scheduler(optimizer, gamma, lr_steps, learning_rate)
    lr_init = 0.0001                   # initial learning rate (SGD=1E-2, Adam=1E-3)
    lr_min = 0.000001                   # minimum learning rate
    lr_max = 0.001
    warmup_ratio = 0.1
    num_steps_per_epoch = len(train_loader)
    total_steps = num_epochs * num_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_max,  # 최대 학습률
        total_steps=total_steps,
        pct_start=warmup_ratio,  # 전체 학습 단계 중 10%를 워밍업으로 사용
        anneal_strategy='cos',
        cycle_momentum=False  # AdamW에서는 모멘텀 사용 안 함
    )
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
        best_val_loss = float('inf')
    else:
        train_iter = state_dict['train_iter']
        step_index = state_dict['lr_step']
        best_mAP = state_dict['best_mAP']
        # adjust_learning_rate(optimizer, learning_rate, gamma, step_index)

    for epoch in range(num_epochs):
        epoch_loss = 0
        tbar = tqdm(train_loader)
        ### Training
        model.train()

        for i, data in enumerate(tbar):
            if method == 'Ours':
                images, fft_images, targets = data
                fft_images = fft_images.to(device)
            else:
                images, targets = data
            # if train_iter in lr_steps:
            #     step_index += 1
            #     adjust_learning_rate(optimizer, learning_rate, gamma, step_index)
            images = images.to(device)
            targets = [target.to(device) for target in targets]
            # outputs = model(images)
            if method == 'Ours':
                outputs, decoded_image = model(images)
            else:
                outputs = model(images)

            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(outputs[:3], targets)
            loss_l_pa12, loss_c_pal2 = criterion(outputs[3:], targets)
            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2
            if method == 'Ours':
                l1_loss = F.l1_loss(decoded_image, fft_images)
                loss += l1_loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            # lr = scheduler.update(optimizer)

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
            if method == 'Ours':
                wandb.log({
                    'train/low_img': [wandb.Image(images[0], caption=f"low_img_epoch{epoch}")],
                    'train/normal_img': [wandb.Image(fft_images[0], caption=f"normal_img_epoch{epoch}")],
                    'train/decoded_img': [wandb.Image(decoded_image[0], caption=f"decoded_img_epoch{epoch}")],
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/loss': epoch_loss / len(train_loader),
                    'train/l1_loss': l1_loss.item(),
                }, step=epoch)
            else:
                wandb.log({
                    'train/loss': epoch_loss / len(train_loader),
                }, step=epoch)


        ## Validation
        model.eval()
        val_loss = 0
        # all_preds = [] # [1, 2, 750, 5] X K
        # all_targets = [] # [[N, 5], [N, 5], ...] X K

        with torch.no_grad():
            tbar = tqdm(val_loader)
            for data in tbar:
                if method == 'Ours':
                    images, fft_images, targets = data
                    # fft_images = fft_images.to(device)
                else:
                    images, targets = data
                images, targets = images.to(device), [t.to(device) for t in targets]
                outputs = model(images)

                loss_l_pal1, loss_c_pal1 = criterion(outputs[:3], targets)
                loss_l_pal2, loss_c_pal2 = criterion(outputs[3:], targets)
                loss = loss_l_pal2 + loss_c_pal2
                val_loss += loss.item()

                # preds = detect(outputs[3], F.softmax(outputs[4], dim=-1), outputs[5])
                # preds = preds[..., [1, 2, 3, 4, 0]]
                # all_preds.append(preds) # [1, 2, 750, 5]
                # all_targets.append(targets) # [[N, 5], [N, 5], ...]

                tbar.set_description(f'[Val] Epoch {epoch+1}, Loss: {loss.item():.4f}')

            # mAP = calculate_map(all_detections, targets, num_classes=num_classes)
            # mAP50, mAP50_95 = calculate_map(all_detections, all_targets, num_classes=num_classes)

            if use_wandb:
                # wandb.log({'val/loss': val_loss / len(val_loader), 'val/mAP@50': mAP50, 'val/mAP@50-95': mAP50_95}, step=epoch)
                wandb.log({'val/loss': val_loss / len(val_loader)}, step=epoch)

            # num_faces = 0
            # img_pr_info_list = []
            # org_pr_curve = torch.zeros((num_thresh, 2)).to(device)
            # norm_preds = norm_score(all_preds)

            # for i, (preds, targets) in enumerate(zip(tqdm(norm_preds), all_targets)):
            #     for j in range(len(preds)):
            #         if len(targets[j]) > 0:
            #             num_faces += len(targets[j])
            #             pred_recall, proposal_list = eval_image(preds[j, 1], targets[j], iou_thresh)
            #             img_pr_info_list.append(image_pr_info(num_thresh, preds[j, 1], proposal_list, pred_recall))

            # for i in range(len(img_pr_info_list)):
            #     org_pr_curve[:, 0] += img_pr_info_list[i][:, 0]
            #     org_pr_curve[:, 1] += img_pr_info_list[i][:, 1]

            # pr_curve = torch.zeros((num_thresh, 2)).to(device)

            # for i in range(num_thresh):
            #     pr_curve[i, 0] = org_pr_curve[i, 1] / org_pr_curve[i, 0]
            #     pr_curve[i, 1] = org_pr_curve[i, 1] / num_faces

            # with open(save_dir / 'pr_curve' / f'epoch_{epoch+1}.pkl', 'wb') as f:
            #     pickle.dump(pr_curve, f)

            # ap = save_pr_curve(pr_curve, save_dir / 'pr_curve' / f'epoch_{epoch+1}.png')

        # if ap > best_mAP:
        if val_loss < best_val_loss:
            # best_mAP = ap
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'weights' / 'best.pt')
            torch.save({
                'epoch': epoch+1,
                'train_iter': train_iter,
                'lr_step': step_index,
                'best_mAP': best_mAP,
                'best_val_loss': best_val_loss,
                'model': model.state_dict(),
            }, save_dir / 'weights' / 'best.pt')
            print(f"New best model saved with mAP: {best_mAP:.4f}")

        if train_iter >= max_steps:
            break