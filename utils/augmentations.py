#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from PIL import Image, ImageEnhance, ImageDraw
import math
import six
from data.config import cfg
import random


class sampler():

    def __init__(self,
                 max_sample,
                 max_trial,
                 min_scale,
                 max_scale,
                 min_aspect_ratio,
                 max_aspect_ratio,
                 min_jaccard_overlap,
                 max_jaccard_overlap,
                 min_object_coverage,
                 max_object_coverage,
                 use_square=False):
        self.max_sample = max_sample
        self.max_trial = max_trial
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_jaccard_overlap = min_jaccard_overlap
        self.max_jaccard_overlap = max_jaccard_overlap
        self.min_object_coverage = min_object_coverage
        self.max_object_coverage = max_object_coverage
        self.use_square = use_square


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class bbox():

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < cfg.brightness_prob:
        delta = np.random.uniform(-cfg.brightness_delta,
                                  cfg.brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < cfg.contrast_prob:
        delta = np.random.uniform(-cfg.contrast_delta,
                                  cfg.contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < cfg.saturation_prob:
        delta = np.random.uniform(-cfg.saturation_delta,
                                  cfg.saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    prob = np.random.uniform(0, 1)
    if prob < cfg.hue_prob:
        delta = np.random.uniform(-cfg.hue_delta, cfg.hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob > 0.5:
        img = random_brightness(img)
        img = random_contrast(img)
        img = random_saturation(img)
        img = random_hue(img)
    else:
        img = random_brightness(img)
        img = random_saturation(img)
        img = random_hue(img)
        img = random_contrast(img)
    return img


def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox.xmax + src_bbox.xmin) / 2
    center_y = (src_bbox.ymax + src_bbox.ymin) / 2
    if center_x >= sample_bbox.xmin and \
            center_x <= sample_bbox.xmax and \
            center_y >= sample_bbox.ymin and \
            center_y <= sample_bbox.ymax:
        return True
    return False


def project_bbox(object_bbox, sample_bbox):
    if object_bbox.xmin >= sample_bbox.xmax or \
       object_bbox.xmax <= sample_bbox.xmin or \
       object_bbox.ymin >= sample_bbox.ymax or \
       object_bbox.ymax <= sample_bbox.ymin:
        return False
    else:
        proj_bbox = bbox(0, 0, 0, 0)
        sample_width = sample_bbox.xmax - sample_bbox.xmin
        sample_height = sample_bbox.ymax - sample_bbox.ymin
        proj_bbox.xmin = (object_bbox.xmin - sample_bbox.xmin) / sample_width
        proj_bbox.ymin = (object_bbox.ymin - sample_bbox.ymin) / sample_height
        proj_bbox.xmax = (object_bbox.xmax - sample_bbox.xmin) / sample_width
        proj_bbox.ymax = (object_bbox.ymax - sample_bbox.ymin) / sample_height
        proj_bbox = clip_bbox(proj_bbox)
        if bbox_area(proj_bbox) > 0:
            return proj_bbox
        else:
            return False


def transform_labels(bbox_labels, sample_bbox):
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        proj_bbox = project_bbox(object_bbox, sample_bbox)
        if proj_bbox:
            sample_label.append(bbox_labels[i][0])
            sample_label.append(float(proj_bbox.xmin))
            sample_label.append(float(proj_bbox.ymin))
            sample_label.append(float(proj_bbox.xmax))
            sample_label.append(float(proj_bbox.ymax))
            sample_label = sample_label + bbox_labels[i][5:]
            sample_labels.append(sample_label)
    return sample_labels


def expand_image(img, bbox_labels, img_width, img_height):
    prob = np.random.uniform(0, 1)
    if prob < cfg.expand_prob:
        if cfg.expand_max_ratio - 1 >= 0.01:
            expand_ratio = np.random.uniform(1, cfg.expand_max_ratio)
            height = int(img_height * expand_ratio)
            width = int(img_width * expand_ratio)
            h_off = math.floor(np.random.uniform(0, height - img_height))
            w_off = math.floor(np.random.uniform(0, width - img_width))
            expand_bbox = bbox(-w_off / img_width, -h_off / img_height,
                               (width - w_off) / img_width,
                               (height - h_off) / img_height)
            expand_img = np.ones((height, width, 3))
            expand_img = np.uint8(expand_img * np.squeeze(cfg.img_mean))
            expand_img = Image.fromarray(expand_img)
            expand_img.paste(img, (int(w_off), int(h_off)))
            bbox_labels = transform_labels(bbox_labels, expand_bbox)
            return expand_img, bbox_labels, width, height
    return img, bbox_labels, img_width, img_height


def clip_bbox(src_bbox):
    src_bbox.xmin = max(min(src_bbox.xmin, 1.0), 0.0)
    src_bbox.ymin = max(min(src_bbox.ymin, 1.0), 0.0)
    src_bbox.xmax = max(min(src_bbox.xmax, 1.0), 0.0)
    src_bbox.ymax = max(min(src_bbox.ymax, 1.0), 0.0)
    return src_bbox


def bbox_area(src_bbox):
    if src_bbox.xmax < src_bbox.xmin or src_bbox.ymax < src_bbox.ymin:
        return 0.
    else:
        width = src_bbox.xmax - src_bbox.xmin
        height = src_bbox.ymax - src_bbox.ymin
        return width * height


def intersect_bbox(bbox1, bbox2):
    if bbox2.xmin > bbox1.xmax or bbox2.xmax < bbox1.xmin or \
            bbox2.ymin > bbox1.ymax or bbox2.ymax < bbox1.ymin:
        intersection_box = bbox(0.0, 0.0, 0.0, 0.0)
    else:
        intersection_box = bbox(
            max(bbox1.xmin, bbox2.xmin),
            max(bbox1.ymin, bbox2.ymin),
            min(bbox1.xmax, bbox2.xmax), min(bbox1.ymax, bbox2.ymax))
    return intersection_box


def bbox_coverage(bbox1, bbox2):
    inter_box = intersect_bbox(bbox1, bbox2)
    intersect_size = bbox_area(inter_box)

    if intersect_size > 0:
        bbox1_size = bbox_area(bbox1)
        return intersect_size / bbox1_size
    else:
        return 0.


def generate_batch_random_samples(batch_sampler, bbox_labels, image_width,
                                  image_height, scale_array, resize_width,
                                  resize_height):
    sampled_bbox = []
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = data_anchor_sampling(
                sampler, bbox_labels, image_width, image_height, scale_array,
                resize_width, resize_height)
            if sample_bbox == 0:
                break
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
    return sampled_bbox


def data_anchor_sampling(sampler, bbox_labels, image_width, image_height,
                         scale_array, resize_width, resize_height):
    num_gt = len(bbox_labels)
    # np.random.randint range: [low, high)
    rand_idx = np.random.randint(0, num_gt) if num_gt != 0 else 0

    if num_gt != 0:
        norm_xmin = bbox_labels[rand_idx][1]
        norm_ymin = bbox_labels[rand_idx][2]
        norm_xmax = bbox_labels[rand_idx][3]
        norm_ymax = bbox_labels[rand_idx][4]

        xmin = norm_xmin * image_width
        ymin = norm_ymin * image_height
        wid = image_width * (norm_xmax - norm_xmin)
        hei = image_height * (norm_ymax - norm_ymin)
        range_size = 0

        area = wid * hei
        for scale_ind in range(0, len(scale_array) - 1):
            if area > scale_array[scale_ind] ** 2 and area < \
                    scale_array[scale_ind + 1] ** 2:
                range_size = scale_ind + 1
                break

        if area > scale_array[len(scale_array) - 2]**2:
            range_size = len(scale_array) - 2
        scale_choose = 0.0
        if range_size == 0:
            rand_idx_size = 0
        else:
            # np.random.randint range: [low, high)
            rng_rand_size = np.random.randint(0, range_size + 1)
            rand_idx_size = rng_rand_size % (range_size + 1)

        if rand_idx_size == range_size:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = min(2.0 * scale_array[rand_idx_size],
                                 2 * math.sqrt(wid * hei))
            scale_choose = random.uniform(min_resize_val, max_resize_val)
        else:
            min_resize_val = scale_array[rand_idx_size] / 2.0
            max_resize_val = 2.0 * scale_array[rand_idx_size]
            scale_choose = random.uniform(min_resize_val, max_resize_val)

        sample_bbox_size = wid * resize_width / scale_choose

        w_off_orig = 0.0
        h_off_orig = 0.0
        if sample_bbox_size < max(image_height, image_width):
            if wid <= sample_bbox_size:
                w_off_orig = np.random.uniform(xmin + wid - sample_bbox_size,
                                               xmin)
            else:
                w_off_orig = np.random.uniform(xmin,
                                               xmin + wid - sample_bbox_size)

            if hei <= sample_bbox_size:
                h_off_orig = np.random.uniform(ymin + hei - sample_bbox_size,
                                               ymin)
            else:
                h_off_orig = np.random.uniform(ymin,
                                               ymin + hei - sample_bbox_size)

        else:
            w_off_orig = np.random.uniform(image_width - sample_bbox_size, 0.0)
            h_off_orig = np.random.uniform(
                image_height - sample_bbox_size, 0.0)

        w_off_orig = math.floor(w_off_orig)
        h_off_orig = math.floor(h_off_orig)

        # Figure out top left coordinates.
        w_off = 0.0
        h_off = 0.0
        w_off = float(w_off_orig / image_width)
        h_off = float(h_off_orig / image_height)

        sampled_bbox = bbox(w_off, h_off,
                            w_off + float(sample_bbox_size / image_width),
                            h_off + float(sample_bbox_size / image_height))

        return sampled_bbox
    else:
        return 0


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox.xmin >= object_bbox.xmax or \
            sample_bbox.xmax <= object_bbox.xmin or \
            sample_bbox.ymin >= object_bbox.ymax or \
            sample_bbox.ymax <= object_bbox.ymin:
        return 0
    intersect_xmin = max(sample_bbox.xmin, object_bbox.xmin)
    intersect_ymin = max(sample_bbox.ymin, object_bbox.ymin)
    intersect_xmax = min(sample_bbox.xmax, object_bbox.xmax)
    intersect_ymax = min(sample_bbox.ymax, object_bbox.ymax)
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
    if sampler.min_jaccard_overlap == 0 and sampler.max_jaccard_overlap == 0:
        has_jaccard_overlap = False
    else:
        has_jaccard_overlap = True
    if sampler.min_object_coverage == 0 and sampler.max_object_coverage == 0:
        has_object_coverage = False
    else:
        has_object_coverage = True

    if not has_jaccard_overlap and not has_object_coverage:
        return True
    found = False
    for i in range(len(bbox_labels)):
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if has_jaccard_overlap:
            overlap = jaccard_overlap(sample_bbox, object_bbox)
            if sampler.min_jaccard_overlap != 0 and \
                    overlap < sampler.min_jaccard_overlap:
                continue
            if sampler.max_jaccard_overlap != 0 and \
                    overlap > sampler.max_jaccard_overlap:
                continue
            found = True
        if has_object_coverage:
            object_coverage = bbox_coverage(object_bbox, sample_bbox)
            if sampler.min_object_coverage != 0 and \
                    object_coverage < sampler.min_object_coverage:
                continue
            if sampler.max_object_coverage != 0 and \
                    object_coverage > sampler.max_object_coverage:
                continue
            found = True
        if found:
            return True
    return found


def crop_image_sampling(img, bbox_labels, sample_bbox, image_width,
                        image_height, resize_width, resize_height,
                        min_face_size):
    # no clipping here
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)
    w_off = xmin
    h_off = ymin
    width = xmax - xmin
    height = ymax - ymin

    cross_xmin = max(0.0, float(w_off))
    cross_ymin = max(0.0, float(h_off))
    cross_xmax = min(float(w_off + width - 1.0), float(image_width))
    cross_ymax = min(float(h_off + height - 1.0), float(image_height))
    cross_width = cross_xmax - cross_xmin
    cross_height = cross_ymax - cross_ymin

    roi_xmin = 0 if w_off >= 0 else abs(w_off)
    roi_ymin = 0 if h_off >= 0 else abs(h_off)
    roi_width = cross_width
    roi_height = cross_height

    roi_y1 = int(roi_ymin)
    roi_y2 = int(roi_ymin + roi_height)
    roi_x1 = int(roi_xmin)
    roi_x2 = int(roi_xmin + roi_width)

    cross_y1 = int(cross_ymin)
    cross_y2 = int(cross_ymin + cross_height)
    cross_x1 = int(cross_xmin)
    cross_x2 = int(cross_xmin + cross_width)

    sample_img = np.zeros((height, width, 3))
    # print(sample_img.shape)
    sample_img[roi_y1 : roi_y2, roi_x1 : roi_x2] = \
        img[cross_y1: cross_y2, cross_x1: cross_x2]
    sample_img = cv2.resize(
        sample_img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    resize_val = resize_width
    sample_labels = transform_labels_sampling(bbox_labels, sample_bbox,
                                              resize_val, min_face_size)
    return sample_img, sample_labels


def transform_labels_sampling(bbox_labels, sample_bbox, resize_val,
                              min_face_size):
    sample_labels = []
    for i in range(len(bbox_labels)):
        sample_label = []
        object_bbox = bbox(bbox_labels[i][1], bbox_labels[i][2],
                           bbox_labels[i][3], bbox_labels[i][4])
        if not meet_emit_constraint(object_bbox, sample_bbox):
            continue
        proj_bbox = project_bbox(object_bbox, sample_bbox)
        if proj_bbox:
            real_width = float((proj_bbox.xmax - proj_bbox.xmin) * resize_val)
            real_height = float((proj_bbox.ymax - proj_bbox.ymin) * resize_val)
            if real_width * real_height < float(min_face_size * min_face_size):
                continue
            else:
                sample_label.append(bbox_labels[i][0])
                sample_label.append(float(proj_bbox.xmin))
                sample_label.append(float(proj_bbox.ymin))
                sample_label.append(float(proj_bbox.xmax))
                sample_label.append(float(proj_bbox.ymax))
                sample_label = sample_label + bbox_labels[i][5:]
                sample_labels.append(sample_label)

    return sample_labels


def generate_sample(sampler, image_width, image_height):
    scale = np.random.uniform(sampler.min_scale, sampler.max_scale)
    aspect_ratio = np.random.uniform(sampler.min_aspect_ratio,
                                     sampler.max_aspect_ratio)
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))

    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)

    # guarantee a squared image patch after cropping
    if sampler.use_square:
        if image_height < image_width:
            bbox_width = bbox_height * image_height / image_width
        else:
            bbox_height = bbox_width * image_width / image_height

    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = bbox(xmin, ymin, xmax, ymax)
    return sampled_bbox


def generate_batch_samples(batch_sampler, bbox_labels, image_width,
                           image_height):
    sampled_bbox = []
    for sampler in batch_sampler:
        found = 0
        for i in range(sampler.max_trial):
            if found >= sampler.max_sample:
                break
            sample_bbox = generate_sample(sampler, image_width, image_height)
            if satisfy_sample_constraint(sampler, sample_bbox, bbox_labels):
                sampled_bbox.append(sample_bbox)
                found = found + 1
    return sampled_bbox


def crop_image(img, bbox_labels, sample_bbox, image_width, image_height,
               resize_width, resize_height, min_face_size):
    sample_bbox = clip_bbox(sample_bbox)
    xmin = int(sample_bbox.xmin * image_width)
    xmax = int(sample_bbox.xmax * image_width)
    ymin = int(sample_bbox.ymin * image_height)
    ymax = int(sample_bbox.ymax * image_height)

    sample_img = img[ymin:ymax, xmin:xmax]
    resize_val = resize_width
    sample_labels = transform_labels_sampling(bbox_labels, sample_bbox,
                                              resize_val, min_face_size)
    return sample_img, sample_labels


def to_chw_bgr(image):
    """
    Transpose image from HWC to CHW and from RBG to BGR.
    Args:
        image (np.array): an image with HWC and RBG layout.
    """
    # HWC to CHW
    if len(image.shape) == 3:
        image = np.swapaxes(image, 1, 2)
        image = np.swapaxes(image, 1, 0)
    # RBG to BGR
    image = image[[2, 1, 0], :, :]
    return image


def anchor_crop_image_sampling(img,
                               bbox_labels,
                               scale_array,
                               img_width,
                               img_height):
    mean = np.array([104, 117, 123], dtype=np.float32)
    maxSize = 12000  # max size
    infDistance = 9999999
    bbox_labels = np.array(bbox_labels)
    scale = np.array([img_width, img_height, img_width, img_height])

    boxes = bbox_labels[:, 1:5] * scale
    labels = bbox_labels[:, 0]

    boxArea = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # argsort = np.argsort(boxArea)
    # rand_idx = random.randint(min(len(argsort),6))
    # print('rand idx',rand_idx)
    rand_idx = np.random.randint(len(boxArea))
    rand_Side = boxArea[rand_idx] ** 0.5
    # rand_Side = min(boxes[rand_idx,2] - boxes[rand_idx,0] + 1,
    # boxes[rand_idx,3] - boxes[rand_idx,1] + 1)

    distance = infDistance
    anchor_idx = 5
    for i, anchor in enumerate(scale_array):
        if abs(anchor - rand_Side) < distance:
            distance = abs(anchor - rand_Side)
            anchor_idx = i

    target_anchor = random.choice(scale_array[0:min(anchor_idx + 1, 5) + 1])
    ratio = float(target_anchor) / rand_Side
    ratio = ratio * (2**random.uniform(-1, 1))

    if int(img_height * ratio * img_width * ratio) > maxSize * maxSize:
        ratio = (maxSize * maxSize / (img_height * img_width))**0.5

    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC,
                      cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)
    image = cv2.resize(img, None, None, fx=ratio,
                       fy=ratio, interpolation=interp_method)

    boxes[:, 0] *= ratio
    boxes[:, 1] *= ratio
    boxes[:, 2] *= ratio
    boxes[:, 3] *= ratio

    height, width, _ = image.shape

    sample_boxes = []

    xmin = boxes[rand_idx, 0]
    ymin = boxes[rand_idx, 1]
    bw = (boxes[rand_idx, 2] - boxes[rand_idx, 0] + 1)
    bh = (boxes[rand_idx, 3] - boxes[rand_idx, 1] + 1)

    w = h = 640

    for _ in range(50):
        if w < max(height, width):
            if bw <= w:
                w_off = random.uniform(xmin + bw - w, xmin)
            else:
                w_off = random.uniform(xmin, xmin + bw - w)

            if bh <= h:
                h_off = random.uniform(ymin + bh - h, ymin)
            else:
                h_off = random.uniform(ymin, ymin + bh - h)
        else:
            w_off = random.uniform(width - w, 0)
            h_off = random.uniform(height - h, 0)

        w_off = math.floor(w_off)
        h_off = math.floor(h_off)

        # convert to integer rect x1,y1,x2,y2
        rect = np.array(
            [int(w_off), int(h_off), int(w_off + w), int(h_off + h)])

        # keep overlap with gt box IF center in sampled patch
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
        # mask in that both m1 and m2 are true
        mask = m1 * m2

        overlap = jaccard_numpy(boxes, rect)
        # have any valid boxes? try again if not
        if not mask.any() and not overlap.max() > 0.7:
            continue
        else:
            sample_boxes.append(rect)

    sampled_labels = []

    if len(sample_boxes) > 0:
        choice_idx = np.random.randint(len(sample_boxes))
        choice_box = sample_boxes[choice_idx]
        # print('crop the box :',choice_box)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        m1 = (choice_box[0] < centers[:, 0]) * \
            (choice_box[1] < centers[:, 1])
        m2 = (choice_box[2] > centers[:, 0]) * \
            (choice_box[3] > centers[:, 1])
        mask = m1 * m2
        current_boxes = boxes[mask, :].copy()
        current_labels = labels[mask]
        current_boxes[:, :2] -= choice_box[:2]
        current_boxes[:, 2:] -= choice_box[:2]

        if choice_box[0] < 0 or choice_box[1] < 0:
            new_img_width = width if choice_box[
                0] >= 0 else width - choice_box[0]
            new_img_height = height if choice_box[
                1] >= 0 else height - choice_box[1]
            image_pad = np.zeros(
                (new_img_height, new_img_width, 3), dtype=float)
            image_pad[:, :, :] = mean
            start_left = 0 if choice_box[0] >= 0 else -choice_box[0]
            start_top = 0 if choice_box[1] >= 0 else -choice_box[1]
            image_pad[start_top:, start_left:, :] = image

            choice_box_w = choice_box[2] - choice_box[0]
            choice_box_h = choice_box[3] - choice_box[1]

            start_left = choice_box[0] if choice_box[0] >= 0 else 0
            start_top = choice_box[1] if choice_box[1] >= 0 else 0
            end_right = start_left + choice_box_w
            end_bottom = start_top + choice_box_h
            current_image = image_pad[
                start_top:end_bottom, start_left:end_right, :].copy()
            image_height, image_width, _ = current_image.shape
            if cfg.filter_min_face:
                bbox_w = current_boxes[:, 2] - current_boxes[:, 0]
                bbox_h = current_boxes[:, 3] - current_boxes[:, 1]
                bbox_area = bbox_w * bbox_h
                mask = bbox_area > (cfg.min_face_size * cfg.min_face_size)
                current_boxes = current_boxes[mask]
                current_labels = current_labels[mask]
                for i in range(len(current_boxes)):
                    sample_label = []
                    sample_label.append(current_labels[i])
                    sample_label.append(current_boxes[i][0] / image_width)
                    sample_label.append(current_boxes[i][1] / image_height)
                    sample_label.append(current_boxes[i][2] / image_width)
                    sample_label.append(current_boxes[i][3] / image_height)
                    sampled_labels += [sample_label]
                sampled_labels = np.array(sampled_labels)
            else:
                current_boxes /= np.array([image_width,
                                           image_height, image_width, image_height])
                sampled_labels = np.hstack(
                    (current_labels[:, np.newaxis], current_boxes))

            return current_image, sampled_labels

        current_image = image[choice_box[1]:choice_box[
            3], choice_box[0]:choice_box[2], :].copy()
        image_height, image_width, _ = current_image.shape

        if cfg.filter_min_face:
            bbox_w = current_boxes[:, 2] - current_boxes[:, 0]
            bbox_h = current_boxes[:, 3] - current_boxes[:, 1]
            bbox_area = bbox_w * bbox_h
            mask = bbox_area > (cfg.min_face_size * cfg.min_face_size)
            current_boxes = current_boxes[mask]
            current_labels = current_labels[mask]
            for i in range(len(current_boxes)):
                sample_label = []
                sample_label.append(current_labels[i])
                sample_label.append(current_boxes[i][0] / image_width)
                sample_label.append(current_boxes[i][1] / image_height)
                sample_label.append(current_boxes[i][2] / image_width)
                sample_label.append(current_boxes[i][3] / image_height)
                sampled_labels += [sample_label]
            sampled_labels = np.array(sampled_labels)
        else:
            current_boxes /= np.array([image_width,
                                       image_height, image_width, image_height])
            sampled_labels = np.hstack(
                (current_labels[:, np.newaxis], current_boxes))

        return current_image, sampled_labels
    else:
        image_height, image_width, _ = image.shape
        if cfg.filter_min_face:
            bbox_w = boxes[:, 2] - boxes[:, 0]
            bbox_h = boxes[:, 3] - boxes[:, 1]
            bbox_area = bbox_w * bbox_h
            mask = bbox_area > (cfg.min_face_size * cfg.min_face_size)
            boxes = boxes[mask]
            labels = labels[mask]
            for i in range(len(boxes)):
                sample_label = []
                sample_label.append(labels[i])
                sample_label.append(boxes[i][0] / image_width)
                sample_label.append(boxes[i][1] / image_height)
                sample_label.append(boxes[i][2] / image_width)
                sample_label.append(boxes[i][3] / image_height)
                sampled_labels += [sample_label]
            sampled_labels = np.array(sampled_labels)
        else:
            boxes /= np.array([image_width, image_height,
                               image_width, image_height])
            sampled_labels = np.hstack(
                (labels[:, np.newaxis], boxes))

        return image, sampled_labels


def preprocess(img, bbox_labels, mode, image_path):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if cfg.apply_distort:
            img = distort_image(img)
        if cfg.apply_expand:
            img, bbox_labels, img_width, img_height = expand_image(
                img, bbox_labels, img_width, img_height)

        batch_sampler = []
        prob = np.random.uniform(0., 1.)
        if prob > cfg.data_anchor_sampling_prob and cfg.anchor_sampling:
            scale_array = np.array([16, 32, 64, 128, 256, 512])
            '''
            batch_sampler.append(
                sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.6, 0.0, True))
            sampled_bbox = generate_batch_random_samples(
                batch_sampler, bbox_labels, img_width, img_height, scale_array,
                cfg.resize_width, cfg.resize_height)
            '''
            img = np.array(img)
            img, sampled_labels = anchor_crop_image_sampling(
                img, bbox_labels, scale_array, img_width, img_height)
            '''
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                img, sampled_labels = crop_image_sampling(
                    img, bbox_labels, sampled_bbox[idx], img_width, img_height,
                    cfg.resize_width, cfg.resize_height, cfg.min_face_size)
            '''
            img = img.astype('uint8')
            img = Image.fromarray(img)
        else:
            batch_sampler.append(sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                         0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                         0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                         0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                         0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
                                         0.0, True))
            sampled_bbox = generate_batch_samples(
                batch_sampler, bbox_labels, img_width, img_height)

            img = np.array(img)
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                img, sampled_labels = crop_image(
                    img, bbox_labels, sampled_bbox[idx], img_width, img_height,
                    cfg.resize_width, cfg.resize_height, cfg.min_face_size)

            img = Image.fromarray(img)

    interp_mode = [
        Image.BILINEAR, Image.HAMMING, Image.NEAREST, Image.BICUBIC,
        Image.LANCZOS
    ]
    interp_indx = np.random.randint(0, 5)

    img = img.resize((cfg.resize_width, cfg.resize_height),
                     resample=interp_mode[interp_indx])

    img = np.array(img)

    if mode == 'train':
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp

    #img = Image.fromarray(img)
    img = to_chw_bgr(img)
    img = img.astype('float32')
    img -= cfg.img_mean
    img = img[[2, 1, 0], :, :]  # to RGB
    #img = img * cfg.scale

    return img, sampled_labels


# -----------------------------
# 아래 함수/클래스/변수들은 기존 코드대로 있다고 가정
# (sampler, expand_image, anchor_crop_image_sampling, generate_batch_samples,
#  crop_image, distort_image, to_chw_bgr, cfg 등)
# -----------------------------

def preprocess_pair(img, fft_img, bbox_labels, mode, image_path=None):
    """
    사용자님이 주신 기존 'preprocess' 로직을 그대로 따라가되,
    - 원본(img)에만 color distortion 적용
    - expand, anchor_crop, random crop, mirror, resize 등 기하학 변환은
      img와 fft_img에 동일 파라미터 적용
    - 최종 (img, fft_img, sampled_labels)를 반환

    img, fft_img : 둘 다 PIL.Image
    bbox_labels  : [[class, x1, y1, x2, y2], ...]  (0~1 정규화 좌표)
    mode         : 'train' or 'test'
    image_path   : (기존과 동일, 필요하면 사용)
    """
    img_width, img_height = img.size

    # sample_labels = bbox_labels 를 유지 (원본 코드와 동일)
    sampled_labels = bbox_labels

    # -----------------------
    # (1) mode=='train'일 때
    # -----------------------
    if mode == 'train':
        # 1-A. distortion: img만 적용, fft_img는 적용 X
        if cfg.apply_distort:
            img = distort_image(img)

        # 1-B. expand_image: (img, fft_img, bbox_labels) 모두에 동일 파라미터 적용
        if cfg.apply_expand:
            (img, fft_img,
             bbox_labels,  # expand_image 후 label 업데이트
             img_width, img_height) = expand_image_pair(img, fft_img, bbox_labels, img_width, img_height)

        # batch_sampler = []  # 원본 코드와 동일
        batch_sampler = []
        prob = np.random.uniform(0., 1.)

        # 1-C. anchor sampling vs. generate_batch_samples
        if prob > cfg.data_anchor_sampling_prob and cfg.anchor_sampling:
            scale_array = np.array([16, 32, 64, 128, 256, 512])
            '''
            batch_sampler.append(
                sampler(1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.6, 0.0, True))
            sampled_bbox = generate_batch_random_samples(
                batch_sampler, bbox_labels, img_width, img_height, scale_array,
                cfg.resize_width, cfg.resize_height)
            '''
            # PIL -> np.array
            img_np = np.array(img)
            fft_np = np.array(fft_img)

            # anchor_crop_image_sampling_pair 로직 (2장 처리)
            img_cropped, fft_cropped, sampled_labels = anchor_crop_image_sampling_pair(
                img_np, fft_np, bbox_labels, scale_array, img_width, img_height
            )

            # 최종 PIL로 복원
            img = Image.fromarray(img_cropped.astype('uint8'))
            fft_img = Image.fromarray(fft_cropped.astype('uint8'))

            '''
            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                img, sampled_labels = crop_image_sampling(
                    img, bbox_labels, sampled_bbox[idx], img_width, img_height,
                    cfg.resize_width, cfg.resize_height, cfg.min_face_size)
            '''
        else:
            # batch_sampler 5개 추가
            batch_sampler.append(sampler(1, 50, 1.0, 1.0, 1.0, 1.0,
                                         0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0,
                                         0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0,
                                         0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0,
                                         0.0, 0.0, 1.0, 0.0, True))
            batch_sampler.append(sampler(1, 50, 0.3, 1.0, 1.0, 1.0,
                                         0.0, 0.0, 1.0, 0.0, True))

            sampled_bbox = generate_batch_samples(
                batch_sampler, bbox_labels, img_width, img_height
            )

            # 원본 코드처럼 np.array 변환 후 crop
            img_np = np.array(img)
            fft_np = np.array(fft_img)

            if len(sampled_bbox) > 0:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                chosen_bbox = sampled_bbox[idx]

                # crop_image_pair
                img_cropped, fft_cropped, sampled_labels = crop_image_pair(
                    img_np, fft_np, bbox_labels,
                    chosen_bbox, img_width, img_height,
                    cfg.resize_width, cfg.resize_height, cfg.min_face_size
                )

                img = Image.fromarray(img_cropped.astype('uint8'))
                fft_img = Image.fromarray(fft_cropped.astype('uint8'))
            else:
                img = Image.fromarray(img_np.astype('uint8'))
                fft_img = Image.fromarray(fft_np.astype('uint8'))

        # (train) 분기 종료

    # -----------------------
    # (2) 이미지 리사이즈 (mode 무관)
    # -----------------------
    interp_mode = [
        Image.BILINEAR, Image.HAMMING, Image.NEAREST, Image.BICUBIC,
        Image.LANCZOS
    ]
    interp_indx = np.random.randint(0, 5)

    # print(f"Type of fft_img: {type(fft_img)}")
    img = img.resize((cfg.resize_width, cfg.resize_height),
                     resample=interp_mode[interp_indx])
    fft_img = fft_img.resize((cfg.resize_width, cfg.resize_height),
                             resample=interp_mode[interp_indx])

    # -----------------------
    # (3) mirror (mode=='train'일 때)
    # -----------------------
    if mode == 'train':
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            # 두 이미지 모두 좌우 반전
            img = np.array(img)[:, ::-1, :]
            fft_img_arr = np.array(fft_img)[:, ::-1, :]

            img = Image.fromarray(img.astype('uint8'))
            fft_img = Image.fromarray(fft_img_arr.astype('uint8'))

            # 라벨 뒤집기
            for i in six.moves.xrange(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp

    # -----------------------
    # (4) 최종: to_chw_bgr, float32, mean subtraction
    # -----------------------
    img = np.array(img)
    fft_img = np.array(fft_img)

    cv2.imwrite('img.jpg', img)
    cv2.imwrite('fft_img.jpg', fft_img)
    np.save('sampled_labels.npy', sampled_labels)

    img = to_chw_bgr(img)
    fft_img = to_chw_bgr(fft_img)

    img = img.astype('float32')
    fft_img = fft_img.astype('float32')

    img -= cfg.img_mean
    fft_img -= cfg.img_mean

    # 원본 코드처럼, 마지막에 RGB 순서로 바꾸려면 이미 to_chw_bgr에서 BGR 됐는데,
    # 다시 img[[2,1,0],:,:] 하는 부분이 있었음
    # => 주어진 코드에선 "img = img[[2,1,0],:,:]  # to RGB" 라고 되어 있는데
    #    이미 to_chw_bgr에서 (BGR) 됐으니, 그대로 맞춰줌
    img = img[[2, 1, 0], :, :]
    fft_img = fft_img[[2, 1, 0], :, :]

    return img, fft_img, sampled_labels


def expand_image_pair(img, fft_img, bbox_labels, img_width, img_height):
    """
    원본 코드의 expand_image와 동일 로직 + 두 이미지를 같이 적용.
    - 랜덤 expand_ratio, w_off, h_off 적용
    - expand된 새 (img, fft_img, bbox_labels, new_w, new_h) 리턴
    """
    prob = np.random.uniform(0, 1)
    if prob < cfg.expand_prob and (cfg.expand_max_ratio - 1.) >= 0.01:
        expand_ratio = np.random.uniform(1, cfg.expand_max_ratio)
        new_w = int(img_width * expand_ratio)
        new_h = int(img_height * expand_ratio)

        h_off = math.floor(np.random.uniform(0, new_h - img_height))
        w_off = math.floor(np.random.uniform(0, new_w - img_width))

        # (B, G, R) => cfg.img_mean
        expand_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        expand_fft = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        mean_val = np.uint8(cfg.img_mean)  # [104,117,123] 등
        expand_img[:] = mean_val
        expand_fft[:] = mean_val

        # PIL -> np
        img_np = np.array(img)
        fft_np = np.array(fft_img)

        expand_img[h_off:h_off+img_height, w_off:w_off+img_width] = img_np
        expand_fft[h_off:h_off+img_height, w_off:w_off+img_width] = fft_np

        # bbox 변환
        # expand_bbox = bbox(...) => transform_labels() 로직
        expand_bbox = bbox(
            -w_off / float(img_width),
            -h_off / float(img_height),
            (new_w - w_off) / float(img_width),
            (new_h - h_off) / float(img_height),
        )
        new_labels = transform_labels(bbox_labels, expand_bbox)

        # 결과를 PIL로
        img_expanded = Image.fromarray(expand_img.astype('uint8'))
        fft_expanded = Image.fromarray(expand_fft.astype('uint8'))

        return img_expanded, fft_expanded, new_labels, new_w, new_h
    else:
        # expand 안 함
        return img, fft_img, bbox_labels, img_width, img_height


def crop_image_pair(
    img_np, fft_np, bbox_labels,
    chosen_bbox, img_width, img_height,
    resize_w, resize_h, min_face_size
):
    """
    원본 코드의 crop_image 함수와 동일 로직을
    두 장(img_np, fft_np)에 대해 적용.
    chosen_bbox : [xmin, ymin, xmax, ymax] (정규화)
    - 픽셀 좌표로 변환 -> crop -> 라벨도 보정
    - (resize_w, resize_h) 로 최종 리사이즈
    """
    # 1) 픽셀 변환
    x1 = int(chosen_bbox.xmin * img_width)
    y1 = int(chosen_bbox.ymin * img_height)
    x2 = int(chosen_bbox.xmax * img_width)
    y2 = int(chosen_bbox.ymax * img_height)

    # 2) crop
    cropped_img = img_np[y1:y2, x1:x2]
    cropped_fft = fft_np[y1:y2, x1:x2]

    # 3) 라벨 보정
    new_labels = []
    cw = (x2-x1)
    ch = (y2-y1)
    for lb in bbox_labels:
        cls_, bx1, by1, bx2, by2 = lb
        # 픽셀 좌표
        px1 = bx1*img_width
        py1 = by1*img_height
        px2 = bx2*img_width
        py2 = by2*img_height

        # 박스가 crop 내부에 있는지 -> meet_emit_constraint 등
        # 간단히 중심이 crop 내부에 있는지
        cx = (px1+px2)/2.
        cy = (py1+py2)/2.
        if (cx >= x1 and cx <= x2) and (cy >= y1 and cy <= y2):
            # crop offset 보정
            px1 -= x1
            px2 -= x1
            py1 -= y1
            py2 -= y1
            # min_face_size 필터
            if (px2-px1)*(py2-py1) < (min_face_size*min_face_size):
                continue
            # 정규화
            nx1 = px1/cw
            ny1 = py1/ch
            nx2 = px2/cw
            ny2 = py2/ch
            # clip
            nx1 = max(0, min(nx1,1))
            nx2 = max(0, min(nx2,1))
            ny1 = max(0, min(ny1,1))
            ny2 = max(0, min(ny2,1))
            if nx2>nx1 and ny2>ny1:
                new_labels.append([cls_, nx1, ny1, nx2, ny2])

    # 4) 최종 리사이즈
    cropped_img = cv2.resize(cropped_img, (resize_w, resize_h),
                             interpolation=cv2.INTER_AREA)
    cropped_fft = cv2.resize(cropped_fft, (resize_w, resize_h),
                             interpolation=cv2.INTER_AREA)

    return cropped_img, cropped_fft, new_labels


def anchor_crop_image_sampling_pair(img_np, fft_np, bbox_labels, scale_array, img_width, img_height):
    """
    원본 anchor_crop_image_sampling을 'pair' 버전으로.
    - 동일 파라미터(배율, crop 영역 등)를 이용해 img_np와 fft_np를 함께 잘라낸다.
    - bbox_labels도 최종 crop 결과에 맞춰 업데이트
    - 반환: (img_cropped, fft_cropped, new_labels)
    """
    # 기존 anchor_crop_image_sampling 로직 복붙 후, 두 장 동시 처리
    # 아래는 간소화 예시이므로, 세부 로직(조건/확률)은 기존 코드와 동일하게 맞춰주세요.
    # -----------------------------------------------------------------

    # 1) img_np, fft_np를 anchor 방식으로 resize
    resized_img, resized_fft, boxes_labels = _anchor_resize_pair(
        img_np, fft_np, bbox_labels, scale_array, img_width, img_height
    )

    # 2) crop 영역 찾기 => 50번 시도
    #    원본 코드와 동일하게 center check 등
    cropped_img, cropped_fft, new_labels = _anchor_crop_after_resize(
        resized_img, resized_fft, boxes_labels
    )

    return cropped_img, cropped_fft, new_labels


def _anchor_resize_pair(img_np, fft_np, bbox_labels, scale_array, w, h):
    """
    anchor_crop_image_sampling 내부에 있던
    'ratio' 계산 + 두 이미지를 같이 resize + bbox도 스케일 조정
    """
    # 만약 bbox_labels가 0개면 그냥 반환
    if len(bbox_labels) == 0:
        return img_np, fft_np, bbox_labels

    # rand_idx 박스 골라 side 결정
    rand_idx = np.random.randint(len(bbox_labels))
    # pixel 좌표 => (x1 = bbox_labels[i][1]*w, ...)
    boxes_in_pixel = []
    for b in bbox_labels:
        cls_, x1, y1, x2, y2 = b
        boxes_in_pixel.append([cls_, x1*w, y1*h, x2*w, y2*h])
    boxes_in_pixel = np.array(boxes_in_pixel)

    bw = boxes_in_pixel[rand_idx, 3] - boxes_in_pixel[rand_idx, 1] + 1
    bh = boxes_in_pixel[rand_idx, 4] - boxes_in_pixel[rand_idx, 2] + 1
    area = bw * bh
    side = math.sqrt(area)

    # anchor_idx
    infDistance = 9999999
    anchor_idx = 5
    for i, anchor in enumerate(scale_array):
        if abs(anchor - side) < infDistance:
            infDistance = abs(anchor - side)
            anchor_idx = i

    # 예: target_anchor
    # 원본 코드처럼 ratio = (target_anchor / side) * 2^(random.uniform(-1,1))
    target_anchor = scale_array[anchor_idx] if anchor_idx < len(scale_array) else scale_array[-1]
    ratio = float(target_anchor) / side
    ratio *= (2 ** random.uniform(-1, 1))

    # maxSize 체크 등
    maxSize = 12000
    if (w*ratio)*(h*ratio) > maxSize*maxSize:
        ratio = math.sqrt((maxSize*maxSize) / (w*h))

    # interpolation
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
                      cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp = random.choice(interp_methods)

    newW = int(w*ratio)
    newH = int(h*ratio)

    # resize 두 장
    resized_img = cv2.resize(img_np, (newW, newH), interpolation=interp)
    resized_fft = cv2.resize(fft_np, (newW, newH), interpolation=interp)

    # bbox도 조정
    scaled_bboxes = []
    for b in boxes_in_pixel:
        cls_, px1, py1, px2, py2 = b
        # ratio 곱
        sx1 = px1*ratio
        sy1 = py1*ratio
        sx2 = px2*ratio
        sy2 = py2*ratio
        # 다시 0~1 정규화 (newW,newH)
        scaled_bboxes.append([
            cls_,
            sx1/newW, sy1/newH,
            sx2/newW, sy2/newH
        ])

    return resized_img, resized_fft, scaled_bboxes


def _anchor_crop_after_resize(img_np, fft_np, boxes_labels):
    """
    resize된 상태에서 crop 시도 (최대 50번), center-in-patch 등
    """
    h, w, _ = img_np.shape

    sample_boxes = []
    crop_w, crop_h = cfg.resize_width, cfg.resize_height  # 예시
    for _ in range(50):
        if crop_w < w:
            w_off = int(np.random.uniform(0, w-crop_w))
        else:
            w_off = 0
        if crop_h < h:
            h_off = int(np.random.uniform(0, h-crop_h))
        else:
            h_off = 0

        rect = np.array([w_off, h_off, w_off+crop_w, h_off+crop_h])
        # center check
        centers = []
        for b in boxes_labels:
            _, x1, y1, x2, y2 = b
            cx = (x1+x2)/2.0
            cy = (y1+y2)/2.0
            centers.append((cx, cy))
        centers = np.array(centers)
        # pixel 단위 center => centers[:,0]*w, ...
        px = centers[:,0]*w
        py = centers[:,1]*h

        m1 = (rect[0] <= px) * (rect[1] <= py)
        m2 = (rect[2] >= px) * (rect[3] >= py)
        mask = m1*m2
        if not mask.any():
            continue
        else:
            sample_boxes.append(rect)

    if len(sample_boxes)==0:
        # crop 실패 => 그냥 반환
        return img_np, fft_np, boxes_labels

    chosen_rect = random.choice(sample_boxes)
    cx1, cy1, cx2, cy2 = chosen_rect

    # crop
    cropped_img = img_np[cy1:cy2, cx1:cx2, :].copy()
    cropped_fft = fft_np[cy1:cy2, cx1:cx2, :].copy()
    cropH, cropW = cropped_img.shape[:2]

    # bbox 보정
    new_labels = []
    for b in boxes_labels:
        cls_, x1, y1, x2, y2 = b
        # 픽셀단위로
        px1 = x1*w
        py1 = y1*h
        px2 = x2*w
        py2 = y2*h
        cx = (px1+px2)/2.
        cy = (py1+py2)/2.
        if (cx1<=cx<=cx2) and (cy1<=cy<=cy2):
            # crop offset
            px1 -= cx1
            px2 -= cx1
            py1 -= cy1
            py2 -= cy1
            # 다시 정규화
            nx1 = px1/cropW
            ny1 = py1/cropH
            nx2 = px2/cropW
            ny2 = py2/cropH
            # clip
            nx1 = max(0, min(nx1,1))
            ny1 = max(0, min(ny1,1))
            nx2 = max(0, min(nx2,1))
            ny2 = max(0, min(ny2,1))
            if nx2>nx1 and ny2>ny1:
                new_labels.append([cls_, nx1, ny1, nx2, ny2])

    return cropped_img, cropped_fft, new_labels
