#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
import cv2
import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
#from detectron2_master.aaa.model_123 import detect_model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'noonbody': {
        'input_size': [512, 512],
        'num_classes': 15,
        'label': ['Background', 'Torso', 'Right-hand', 'Left-hand', 'Left-foot', 'Right-foot', 'Right-thigh', 'Left-thigh', 'Right-calf', 'Left-calf', 'Left-arm', 'Right-arm', 'Left-forearm', 'Right-forearm', 'Head']
    },
    'lip': {
        'input_size': [473, 473],
        'num_classes': 16,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasse s', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip',
                        choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='checkpoints/schp_9_checkpoint.pth.tar',
                        help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    # parser.add_argument("--input-dir", type=str, default='./our_input', help="path of input image folder.")#이미지
    parser.add_argument("--input-dir", type=str, default='demo/demo_input',
                        help="path of input image folder.")  # 폴더
    # parser.add_argument("--output-dir", type=str, default='./our_output', help="path of output image folder.")#이미지
    parser.add_argument("--output-dir", type=str, default='demo/demo_output',
                        help="path of output image folder.")  # 폴
    parser.add_argument("--logits", action='store_true',
                        default=False, help="whether to save the logits.")

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = 'args.gpu'

    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = networks.init_model(
        'resnet101', num_classes=num_classes, pretrained=None)

    state_dict = torch.load(args.model_restore, map_location=torch.device('cpu'))[
        'state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
    ])

    folder_list = sorted(os.listdir(args.input_dir))
    f_count = 0
    s_count = 0
    for num in range(len(folder_list)):

        img_folder = os.path.join(args.input_dir, folder_list[num])
        dataset = SimpleFolderDataset(
            root=img_folder, input_size=input_size, transform=transform)
        dataloader = DataLoader(dataset)
        f_output = args.output_dir
        save_img_folder = os.path.join(f_output, folder_list[num])
        if not os.path.exists(save_img_folder):
            os.makedirs(save_img_folder)
        f_count += 1

        palette = get_palette(num_classes)
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader)):
                image, meta = batch
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                output = model(image.cuda())
                upsample = torch.nn.Upsample(
                    size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(
                    1, 2, 0)  # CHW -> HWC

                logits_result = transform_logits(
                    upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                parsing_result_path = os.path.join(
                    save_img_folder, img_name[:-4] + '.png')
                # 그레이색이야
                parsing_result = np.asarray(parsing_result, dtype=np.uint8)

                # 색으로 가요
                parsing_result = np.expand_dims(parsing_result, axis=2)
                parsing_result = np.concatenate(
                    (parsing_result, parsing_result, parsing_result), axis=2)
                a = 127
                b = 255
                index = [[0, 0, 0], [b, 0, 0], [0, b, 0], [0, 0, b], [b, b, 0], [0, b, b], [b, 0, b], [b, 0, a], [
                    b, a, 0], [0, a, b], [a, a, a], [a, a, b], [a, b, a], [a, b, b], [0, a, a], [b, b, b]]
                for i, j in enumerate(index):
                    parsing_result[:, :,
                                   0][parsing_result[:, :, 0] == i] = j[0]
                    parsing_result[:, :,
                                   1][parsing_result[:, :, 1] == i] = j[1]
                    parsing_result[:, :,
                                   2][parsing_result[:, :, 2] == i] = j[2]
                cv2.imwrite(parsing_result_path, parsing_result)
        print('Input_dir', img_folder, 'Output_dir',
              save_img_folder, '-----> Done')
    # 오류확인
    # folder_list = sorted(os.listdir(args.input_dir))
    # print(folder_list[352])
    # print(os.listdir(os.path.join(args.input_dir, folder_list[117])))
    # for num in range(len(folder_list[117])):
    #     img_folder = os.path.join(args.input_dir, folder_list[117])
    #     print(img_folder)
        print('front = ', f_count)
        print('side = ', s_count)
    return


if __name__ == '__main__':
    main()
