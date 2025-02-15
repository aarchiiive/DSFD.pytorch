import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.augmentations import preprocess, preprocess_pair, to_chw_bgr


class DarkFaceDataset(Dataset):
    def __init__(self, data_dir, meta_file, max_pixels=1500*1000, num_samples=None, method='default', phase='train'):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.method = method
        self.phase = phase

        self.max_pixels = max_pixels
        self.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')

        self.filenames = []
        self.boxes = []
        self.labels = []

        '''
        [Label Format]
        12 -> Number of faces
        790 382 802 403 -> [x1, y1, x2, y2]
        783 391 791 408
        683 415 686 421
        657 421 661 427
        656 427 659 433
        636 423 643 432
        612 424 619 433
        554 423 561 433
        539 418 548 428
        992 63 1078 178
        620 429 625 434
        630 420 635 427
        '''

        if not self.phase == 'test':
            with open(meta_file, 'r') as f:
                meta_lines = f.readlines()

            for line in meta_lines:
                line = line.strip().split()
                if method in ['default', 'Ours']:
                    filename = self.data_dir / 'image' / Path(line[0]).name
                elif method == 'SCI':
                    filename = self.data_dir / 'SCI' / Path(line[0]).with_suffix('.jpg').name
                else:
                    raise ValueError(f'Invalid method: {method}')

                self.filenames.append(filename)

                label = self.data_dir / 'label' / filename.with_suffix('.txt').name
                with open(label, 'r') as f:
                    lines = f.readlines()
                    num_faces = int(lines[0])
                    box = []
                    label = []
                    for i in range(1, num_faces + 1):
                        x1, y1, x2, y2 = map(float, lines[i].split())
                        c = 1
                        if x2 <= x1 or y2 <= y1:
                            continue
                        box.append([x1, y1, x2, y2])
                        label.append(c)

                    if len(box) > 0:
                        self.boxes.append(box)
                        self.labels.append(label)
        else:
            self.filenames = sorted(self.data_dir.glob('*'))

        # print(sorted(self.data_dir.glob('*')))

        # Unzip the shuffled data
        if not self.phase == 'test':
            # Combine filenames, boxes, and labels into a single list for shuffling
            data = list(zip(self.filenames, self.boxes, self.labels))
            random.shuffle(data)  # Shuffle the dataset randomly
            self.filenames, self.boxes, self.labels = zip(*data)

        # Convert tuples back to lists
        self.filenames = list(self.filenames)
        if not self.phase == 'test':
            self.boxes = list(self.boxes)
            self.labels = list(self.labels)

        # Limit the dataset size to `num_samples` if specified
        if num_samples and num_samples < len(self.filenames):
            self.filenames = self.filenames[:num_samples]
            if not self.phase == 'test':
                self.boxes = self.boxes[:num_samples]
                self.labels = self.labels[:num_samples]

        self.num_samples = len(self.filenames)
        # print(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        if not self.phase == 'test':
            if self.method == 'Ours':
                img, fft_img, target, img_path = self.pull_item(index)
                return img, fft_img, target
            else:
                img, target, img_path = self.pull_item(index)
                return img, target, img_path
        else:
            img, img_path = self.pull_item(index)
            return img, img_path

    def pull_item(self, index):
        while True:
            image_path = self.filenames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            img_width, img_height = img.size
            if not self.phase == 'test':
                boxes = self.rescale(
                    np.array(self.boxes[index]), img_width, img_height)
                label = np.array(self.labels[index])
                bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
                if self.method == 'Ours':
                    fft_img_path = self.data_dir / 'FFT_fusion' / Path(image_path).with_suffix('.jpg').name
                    fft_img = Image.open(fft_img_path)
                    if fft_img.mode == 'L':
                        fft_img = fft_img.convert('RGB')
                    # fft_img = np.array(fft_img)
                    img, fft_img, sample_labels = preprocess_pair(img, fft_img, bbox_labels, self.phase, image_path)
                else:
                    img, sample_labels = preprocess(img, bbox_labels, self.phase, image_path)
                sample_labels = np.array(sample_labels)
                if len(sample_labels) > 0:
                    target = np.hstack(
                        (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                    assert (target[:, 2] > target[:, 0]).any()
                    assert (target[:, 3] > target[:, 1]).any()
                    break
                else:
                    index = random.randrange(0, self.num_samples)
            else:
                img = np.array(img)
                max_im_shrink = np.sqrt(self.max_pixels / (img_height * img_width))
                if max_im_shrink < 1:
                    img = cv2.resize(img, None, fx=max_im_shrink, fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

                img = to_chw_bgr(img).astype('float32')
                img -= self.img_mean  # Normalization using provided img_mean
                img = torch.from_numpy(img).unsqueeze(0)

        if not self.phase == 'test':
            if self.method == 'Ours':
                return torch.from_numpy(img), torch.from_numpy(fft_img), target, str(image_path)
            else:
                return torch.from_numpy(img), target, str(image_path)
        else:
            return torch.from_numpy(img), str(image_path)

    def rescale(self, boxes, img_width, img_height):
        boxes[:, 0] = boxes[:, 0] / img_width
        boxes[:, 1] = boxes[:, 1] / img_height
        boxes[:, 2] = boxes[:, 2] / img_width
        boxes[:, 3] = boxes[:, 3] / img_height
        return boxes

def collate_fn(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    # targets = []
    # imgs = []
    # paths = []
    # for sample in batch:
    #     imgs.append(sample[0])
    #     targets.append(torch.FloatTensor(sample[1]))
    #     paths.append(sample[2])
    # return torch.stack(imgs, 0), targets, paths
    imgs = []
    fft_imgs = []
    targets = []
    for sample in batch:
        imgs.append(sample[0])
        fft_imgs.append(sample[1])
        targets.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), torch.stack(fft_imgs, 0), targets