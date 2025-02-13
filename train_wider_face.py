import wandb

import pytz
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# from data.dark_face import DarkFaceDataset, collate_fn
from data.config import cfg
from data.widerface import WiderFaceDataset, collate_fn
from models.factory import build_net
from layers.modules import MultiBoxLoss

KST = pytz.timezone('Asia/Seoul')

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
    learning_rate = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    gamma = 0.1
    lr_steps = [80000, 100000, 120000]
    negpos_ratio = 3
    variance = [0.1, 0.2]
    threshold = 0.5
    device = torch.device('cuda:1')

    batch_size = 16
    num_epochs = 100
    max_steps = 150000
    save_freq = 5
    num_workers = 8
    pretrained_weights = 'weights/resnet50-19c8e357.pth'
    # resume = 'runs/20250211_223733_DSDF-Wider-Face/weights/epoch_30.pth'
    resume = None

    # data_dir = Path('../datasets/DarkFace_Train_2021')
    # image_dir = data_dir / 'image'
    # label_dir = data_dir / 'label'
    # train_meta = data_dir / 'mf_dsfd_dark_face_train_5500.txt'
    # val_meta = data_dir / 'mf_dsfd_dark_face_val_500.txt'

    wandb_project = 'dark-face'
    wandb_name = 'DSDF-Wider-Face'

    if not resume:
        save_dir = Path('runs')
        timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
        save_dir = save_dir / f"{timestamp}_{wandb_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(resume).parent

    wandb.init(project=wandb_project, name=wandb_name)
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
        # 'data_dir': data_dir,
        # 'image_dir': image_dir,
        # 'label_dir': label_dir,
        # 'train_meta': train_meta,
        # 'val_meta': val_meta,
        'save_dir': save_dir
    })

    ### Dataset and DataLoader
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
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    ### Model, Loss, Optimizer
    model = build_net('train', num_classes, model_name)
    if not resume:
        state_dict = torch.load(pretrained_weights)
        new_state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        model_dict = model.resnet.state_dict()
        model_dict.update(new_state_dict)
        model.resnet.load_state_dict(model_dict)
    else:
        state_dict = torch.load(resume)
        model.load_state_dict(state_dict['model_state_dict'])

    model = model.to(device)

    if not resume:
        model.extras.apply(model.weights_init)
        model.fpn_topdown.apply(model.weights_init)
        model.fpn_latlayer.apply(model.weights_init)
        model.fpn_fem.apply(model.weights_init)
        model.loc_pal1.apply(model.weights_init)
        model.conf_pal1.apply(model.weights_init)
        model.loc_pal2.apply(model.weights_init)
        model.conf_pal2.apply(model.weights_init)

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
    else:
        train_iter = state_dict['train_iter']
        step_index = state_dict['lr_step']
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
            # torch.save(model.state_dict(), save_dir / 'weights' / f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch+1,
                'train_iter': train_iter,
                'lr_step': step_index,
                'model': model.state_dict(),
            }, save_dir / 'weights' / f'epoch_{epoch+1}.pth')

        wandb.log({
            'train/loss': epoch_loss / len(train_loader)
        }, step=epoch)

        ## Validation
        model.eval()

        with torch.no_grad():
            val_iter = 0
            val_loss = 0
            tbar = tqdm(val_loader)
            for i, (images, targets) in enumerate(tbar):
                images = images.to(device)
                targets = [target.to(device) for target in targets]
                outputs = model(images)

                loss_l_pa1l, loss_c_pal1 = criterion(outputs[:3], targets)
                loss_l_pa12, loss_c_pal2 = criterion(outputs[3:], targets)
                loss = loss_l_pa12 + loss_c_pal2

                val_loss += loss.item()
                val_iter += 1

                tbar.set_description(f'[Val] Epoch {epoch+1}/{num_epochs}, Iteration {val_iter}, Loss: {loss.item():.4f}')

            wandb.log({
                'val/loss': val_loss / len(val_loader)
            }, step=epoch)

        if train_iter >= max_steps:
            break