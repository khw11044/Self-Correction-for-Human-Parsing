#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   datasets.py
@Time    :   8/4/19 3:35 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
import json
import os
import numpy as np
import random
import torch
import cv2
from torch.utils import data
from utils.transforms import get_affine_transform

INDEX_DICT = {
    0: np.array([0, 0, 255], dtype=np.uint8),
    1: np.array([0, 255, 0], dtype=np.uint8),
    2: np.array([255, 0, 0], dtype=np.uint8),
    3: np.array([0, 255, 255], dtype=np.uint8),
    5: np.array([255, 255, 0], dtype=np.uint8),
    4: np.array([255, 0, 255], dtype=np.uint8),
    6: np.array([0, 0, 128], dtype=np.uint8),
    7: np.array([0, 128, 0], dtype=np.uint8),
    8: np.array([128, 0, 0], dtype=np.uint8),
    9: np.array([0, 128, 128], dtype=np.uint8),
    10: np.array([128, 128, 0], dtype=np.uint8),
    11: np.array([128, 0, 128], dtype=np.uint8),
    12: np.array([128, 128, 255], dtype=np.uint8),
    13: np.array([128, 255, 128], dtype=np.uint8)  # ,
    # 14: np.array([255, 128, 128], dtype=np.uint8),
}




class LIPDataSet(data.Dataset):
    def __init__(self, img_root, anno_root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        self.img_root = img_root
        self.anno_root = anno_root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.transform = transform
        self.dataset = dataset

        self.train_list = os.listdir(self.img_root)
        #self.train_list = [li for li in os.listdir(self.root) if li.split('_')[1] == 'front']
        self.number_samples = len(self.train_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        train_item = self.train_list[index]

        im_path = os.path.join(self.img_root, train_item)
        parsing_anno_path = os.path.join(self.anno_root, train_item.split('.')[0]+'.png')          # 라벨(seg된거) 불러오기

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            im_1 = cv2.imread(parsing_anno_path)

            # im, im_1 = corp(im,im_1,train_item)

            h, w, _ = im.shape
            # Get person center and scale
            person_center, s = self._box2cs([0, 0, w - 1, h - 1])
            r = 0
            a = 255
            b = 127

            try:
                dst0 = cv2.inRange(im_1, (255-10, 0, 0), (255, 0, 0))
                dst1 = cv2.inRange(im_1, (0, 255-10, 0), (0, 255, 0))
                dst2 = cv2.inRange(im_1, (0, 0, 255-10), (0, 0, 255))
                dst3 = cv2.inRange(im_1, (a-10, a-10, 0), (a, a, 0))
                dst4 = cv2.inRange(im_1, (0, a-10, a-10), (0, a, a))
                dst5 = cv2.inRange(im_1, (a-10, 0, a-10), (a, 0, a))
                dst6 = cv2.inRange(im_1, (a-10, 0, b - 10), (a, 0, b + 10))
                dst7 = cv2.inRange(im_1, (a-10, b - 10, 0), (a, b + 10, 0))
                dst8 = cv2.inRange(im_1, (0, b - 10, a-10), (0, b + 10, a))
                dst9 = cv2.inRange(
                    im_1, (b - 10, b - 10, b - 10), (b + 10, b + 10, b + 10))
                dst10 = cv2.inRange(
                    im_1, (b - 10, b - 10, a-10), (b + 10, b + 10, a))
                dst11 = cv2.inRange(
                    im_1, (b - 10, a-10, b - 10), (b + 10, a, b + 10))
                dst12 = cv2.inRange(im_1, (b - 10, a-10, a-10), (b + 10, a, a))
                dst13 = cv2.inRange(
                    im_1, (0, b - 10, b - 10), (0, b + 10, b + 10))
            except:
                print(im_path)
                print(parsing_anno_path)
            # dst14 = cv2.inRange(im_1, (a-10, a-10, a-10), (a, a, a))                # 골프클럽헤드
            a = np.dstack((dst0,
                           dst1,
                           dst2,
                           dst3,
                           dst4,
                           dst5,
                           dst6,
                           dst7,
                           dst8,
                           dst9,
                           dst10,
                           dst11,
                           dst12,
                           dst13  # ,
                           #   dst14
                           ))

            a[a > 0] = 1
            parsing_anno = a[:, :, 0]

            # 14개 라벨
            for i in range(13):
                parsing_anno += a[:, :, i + 1] * (i + 2)

            if self.dataset == 'train' or self.dataset == 'trainval':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf *
                            2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    person_center[0] = im.shape[1] - person_center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': train_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset == 'val' or self.dataset == 'test':
            return input, meta
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, meta


class LIPDataValSet(data.Dataset):
    def __init__(self, img_root, dataset='val', crop_size=[473, 473], transform=None, flip=False):
        self.img_root = img_root
        self.crop_size = crop_size
        self.transform = transform
        self.flip = flip
        self.dataset = dataset

        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)

        list_path = os.path.join(self.img_root, self.dataset + '_id.txt')
        val_list = [i_id.strip() for i_id in open(list_path)]

        self.val_list = val_list
        self.number_samples = len(self.val_list)

    def __len__(self):
        return len(self.val_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        val_item = self.val_list[index]
        # Load training image
        im_path = os.path.join(
            self.img_root, self.dataset + '_images', val_item + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
        input = self.transform(input)
        flip_input = input.flip(dims=[-1])
        if self.flip:
            batch_input_im = torch.stack([input, flip_input])
        else:
            batch_input_im = input

        meta = {
            'name': val_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return batch_input_im, meta
