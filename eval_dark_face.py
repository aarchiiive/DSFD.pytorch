import pickle
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.config import cfg
from data.dark_face import DarkFaceDataset, collate_fn
from metrics import norm_score, eval_image, image_pr_info, save_pr_curve

if __name__ == "__main__":
    ### Configurations
    # method = 'Ours'
    # method = 'SCI'
    method = 'default'
    num_classes = 2
    num_samples = None
    model_name = 'resnet50'
    device = torch.device('cpu')

    batch_size = 4
    num_workers = 32
    # weights_path = 'runs/20250214_223548_DSDF-Ours-resnet-pretrained/weights/epoch_100.pt'
    # weights_path = 'runs/20250213_170459_DSDF-Wider-Face-pretrained/weights/epoch_100.pt'
    # weights_path = 'runs/DSFD-Wider-Face/weights/epoch_100.pth'
    # weights_path = 'runs/20250217_000550_DSDF-SCI-resnet-pretrained/weights/epoch_100.pt'
    weights_path = 'runs/20250217_161621_DSDF-baseline-resnet-pretrained-epoch300/weights/epoch_80.pt'
    # weights_path = 'runs/20250213_170459_DSDF-Wider-Face-pretrained/weights/epoch_100.pt'
    # weights_path = 'runs/20250216_085104_DSDF-baseline-resnet-pretrained/weights/epoch_80.pt'
    # weights_path = 'runs/DSFD-baseline/weights/epoch_95.pth'
    # weights_path = 'runs/DSFD-SCI/weights/epoch_95.pth'
    resume = None

    ## Eval
    iou_thresh = 0.5
    num_thresh = 1000

    data_dir = Path('/home/ubuntu/data/DarkFace_Train_2021')
    image_dir = data_dir / 'image'
    label_dir = data_dir / 'label'
    train_meta = data_dir / 'mf_dsfd_dark_face_train_5500.txt'
    val_meta = data_dir / 'mf_dsfd_dark_face_val_500.txt'

    val_dataset = DarkFaceDataset(
        data_dir,
        val_meta,
        num_samples=num_samples,
        method='default',
        phase='val'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    ### Ours
    if method == 'Ours':
        from models.DSFD_ours import build_net_resnet
        model = build_net_resnet('test', num_classes, model_name)
    else:
        from models.factory import build_net
        model = build_net('test', num_classes, model_name)

    state_dict = torch.load(weights_path, map_location='cpu')
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    ### Evaluation
    tbar = tqdm(val_loader, desc='Evaluating')
    all_preds = [] # [1, 2, 750, 5] X K
    all_targets = [] # [[N, 5], [N, 5], ...] X K

    max_images = 10000000
    for i, (images, targets) in enumerate(tbar):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            preds = model(images)

        # print(f"targets: {targets[0].shape}")
        if targets[0].shape[0] > 0:
            preds = preds[..., [1, 2, 3, 4, 0]] # confxyxy -> xyxyconf (swap)
            all_preds.append(preds) # [1, 2, 750, 5]
            all_targets.append(targets) # [[N, 5], [N, 5], ...]

        if i == max_images - 1:
            break

    ### Compute mAP & PR curve
    num_faces = 0
    img_pr_info_list = []
    org_pr_curve = torch.zeros((num_thresh, 2))
    norm_preds = norm_score(all_preds)

    for i, (preds, targets) in enumerate(zip(tqdm(norm_preds), all_targets)):
        for j in range(len(preds)):
            if len(targets[j]) > 0:
                num_faces += len(targets[j])
                pred_recall, proposal_list = eval_image(preds[j, 1], targets[j], iou_thresh)
                img_pr_info_list.append(image_pr_info(num_thresh, preds[j, 1], proposal_list, pred_recall))
        if i == max_images - 1:
            break

    for i in range(len(img_pr_info_list)):
        org_pr_curve[:, 0] += img_pr_info_list[i][:, 0]
        org_pr_curve[:, 1] += img_pr_info_list[i][:, 1]

    pr_curve = torch.zeros((num_thresh, 2))

    for i in range(num_thresh):
        pr_curve[i, 0] = org_pr_curve[i, 1] / org_pr_curve[i, 0]
        pr_curve[i, 1] = org_pr_curve[i, 1] / num_faces

    with open(Path(weights_path).parent.parent / 'pr_curve.pkl', 'wb') as f:
        pickle.dump(pr_curve, f)

    ap = save_pr_curve(pr_curve, Path(weights_path).parent.parent / 'pr_curve.png')