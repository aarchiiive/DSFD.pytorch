import torch
import pytz
import os
import time
from pathlib import Path
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from torch.cuda.amp import autocast
from models.factory import build_net
from utils.augmentations import to_chw_bgr
from data.dark_face import DarkFaceDataset

# Model inference function
def detect(model, img_path, thresh, device, img_mean, max_pixels, save_dir, vis=False):
    """Performs object detection on an image and optionally saves visualization."""
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape

    # Limit the maximum image size to max_pixels (only shrink if needed)
    max_im_shrink = np.sqrt(max_pixels / (height * width))
    img = cv2.resize(img, None, None, fx=max_im_shrink, fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    # Convert image to CHW format (PyTorch tensor format)
    x = to_chw_bgr(img).astype('float32')
    x -= img_mean  # Normalization using provided img_mean
    x = x[[2, 1, 0], :, :]  # RGB to BGR
    x = torch.from_numpy(x).unsqueeze(0).to(device)
    # x = x.half()  # Convert to half precision (FP16)
    # x = Variable(torch.from_numpy(x).unsqueeze(0)).to(device)

    # Perform model inference
    t1 = time.time()

    with torch.no_grad():
        detections = model(x) # ([x, y, x, y], [conf])

    # Scale for bbox conversion
    scale = torch.Tensor([height, width, height, width]).to(device)

    results = []
    img_vis = cv2.imread(img_path, cv2.IMREAD_COLOR) if vis else None  # Load image only if visualization is enabled

    # print(thresh)
    print(f"Number of objects above conf_thresh: {(detections[0, :, :, 0] > conf_thresh).sum().item()}")

    for i in range(detections.shape[1]):  # Iterate over classes
        j = 0
        # print(detections)
        while j < detections.shape[2] and detections[0, i, j, 0] > thresh:
            pt = detections[0, i, j, 1:] * scale
            score = detections[0, i, j, 0].item()

            # Clip bounding box coordinates to be within the image
            pt = pt.cpu().numpy()
            # score = score.cpu().numpy()
            x1, y1, x2, y2 = np.clip(pt, [0, 0, 0, 0], [img.shape[1]-1, img.shape[0]-1, img.shape[1]-1, img.shape[0]-1])

            results.append((x1, y1, x2, y2, score))

            if vis:  # Draw bounding box only if visualization is enabled
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Display confidence score
                conf_text = f"{score:.2f}"
                text_size, baseline = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                text_x, text_y = x1, y1 - text_size[1]

                cv2.rectangle(img_vis, (text_x, text_y - baseline - 2), (text_x + text_size[0], text_y + text_size[1]), (255, 0, 0), -1)
                cv2.putText(img_vis, conf_text, (text_x, text_y + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

            j += 1

    t2 = time.time()
    print(f"detect: {img_path} | time taken: {t2 - t1:.4f}s")

    # Save visualization only if enabled
    if vis:
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img_vis)

    return results  # Return bounding box results


# Main execution
if __name__ == '__main__':
    # Configuration settings
    vis = False  # Enable visualization
    num_classes = 2
    conf_thresh = 0.25
    model_name = 'resnet50'
    weights_path = Path('runs/20250211_055317/epoch_80.pth') # 개구린거 (low light image input)
    # weights_path = Path('runs/20250211_060036/epoch_80.pth') # SCI (enhanced input)

    img_dir = '../datasets/DarkFace_Train_2021/Track1.2_testing_samples'  # Directory containing input images
    # img_dir = '../datasets/DarkFace_Train_2021/DarkFace_Test_SCI'  # Directory containing input images
    save_root = 'submission'  # Root directory for saving results
    save_dir = weights_path.parent / save_root / Path(weights_path).stem  # Create a folder using weights name
    save_dir.mkdir(parents=True, exist_ok=True)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:2')

    # Image mean for normalization
    img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
    max_pixels = 1500 * 1000  # 1.5M pixel limit for resizing

    # Load model
    model = build_net('test', num_classes, model_name)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.to(device)
    # model.half()  # Convert model to half precision (FP16)
    model.eval()

    # dataset = DarkFaceDataset(
    #     data_dir='../datasets/DarkFace_Train_2021/Track1.2_testing_samples',
    #     meta_file=None,
    #     max_pixels=max_pixels,
    #     method='default',
    #     phase='test'
    # )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True
    # )

    # for i, (image, image_paths, img_width, img_height) in enumerate(tqdm(dataloader)):
    #     with torch.no_grad():
    #         detections = model(image.to(device))

    #     for i, img_path in enumerate(image_paths):
    #         results = []
    #         scale = torch.Tensor([img_width[i], img_height[i], img_width[i], img_height[i]]).to(device)

    #         for j in range(detections.size(1)):  # Iterate over classes
    #             k = 0
    #             while k < detections.shape[2] and detections[i, j, k, 0] > conf_thresh:
    #                 pt = detections[i, j, k, 1:] * scale
    #                 score = detections[i, j, k, 0].item()

    #                 x1, y1, x2, y2 = np.clip(pt, [0, 0, 0, 0], [img_width[i]-1, img_height[i]-1, img_width[i]-1, img_height[i]-1])
    #                 results.append((x1, y1, x2, y2, score))
    #                 k += 1

    #             # save results as .txt files
    #             save_path = save_dir / f"{Path(img_path).stem}.txt"
    #             with open(save_path, 'w') as f:
    #                 for x1, y1, x2, y2, conf in results:
    #                     f.write(f"{x1} {y1} {x2} {y2} {conf:.6f}\n")

    #             print(f"Saved results to {save_path}")


    # Get all image files (any format)
    img_list = [Path(f) for f in glob(f"{img_dir}/*") if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'))]

    for img_path in tqdm(img_list, desc="Processing images"):
        results = detect(
            model,
            img_path,
            thresh=conf_thresh,
            device=device,
            img_mean=img_mean,
            max_pixels=max_pixels,
            save_dir=save_dir,
            vis=vis
        )

        # Save results as .txt files
        save_path = save_dir / f"{img_path.stem}.txt"
        with open(save_path, 'w') as f:
            for x1, y1, x2, y2, conf in results:
                f.write(f"{x1} {y1} {x2} {y2} {conf:.6f}\n")  # Save confidence to 6 decimal places

        print(f"Saved results to {save_path}")
