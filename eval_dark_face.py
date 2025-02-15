import pickle
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.config import cfg
from data.dark_face import DarkFaceDataset, collate_fn
from metrics import eval_image, image_pr_info, save_pr_curve

def norm_score(org_pred_list):
    norm_pred_list = []
    max_score = torch.finfo(torch.float32).min
    min_score = torch.finfo(torch.float32).max

    for i in range(len(org_pred_list)):
        pred = org_pred_list[i]
        scores = pred[..., 4]
        max_score = max(max_score, scores.max())
        min_score = min(min_score, scores.min())

    for i in range(len(org_pred_list)):
        pred = org_pred_list[i]
        scores = pred[..., 4]
        if max_score != min_score:
            norm_scores = (scores - min_score) / (max_score - min_score)
        else:
            norm_scores = (scores - (min_score - 1))
        org_pred_list[i][..., 4] = norm_scores

    norm_pred_list = org_pred_list
    return norm_pred_list

if __name__ == "__main__":
    ### Configurations
    method = 'Ours'
    num_classes = 2
    num_samples = None
    model_name = 'resnet50'
    device = torch.device('cuda:0')

    batch_size = 128
    num_workers = 8
    weights_path = 'runs/20250214_223548_DSDF-Ours-resnet-pretrained/weights/epoch_100.pt'
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
        method=method,
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
        model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])

    model.to(device)
    model.eval()

    ### Evaluation
    tbar = tqdm(val_loader, desc='Evaluating')
    all_preds = [] # [1, 2, 750, 5] X K
    all_targets = [] # [[N, 5], [N, 5], ...] X K

    max_images = 10000000
    for i, (images, fft_images, targets) in enumerate(tbar):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            preds = model(images)
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

    with open(Path(weights_path).parent / 'pr_curve.pkl', 'wb') as f:
        pickle.dump(pr_curve, f)

    save_pr_curve(pr_curve, 'pr_curve.png')