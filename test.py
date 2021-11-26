import argparse
import os
from collections import OrderedDict
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import networks
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import transform_logits


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='demo/demo_input')
    parser.add_argument('--save_path', type=str, default='demo/demo_output')
    parser.add_argument('--input_size', type=list, default=[473, 473])

    return parser.parse_args()


def main():
    args = get_arg()
    root = args.data_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = networks.init_model('resnet101', num_classes=16, pretrained=None)
    state_dict = torch.load('checkpoints/schp_16_checkpoint.pth.tar',
                            map_location=torch.device(device))['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[
                             0.225, 0.224, 0.229])
    ])

    dataset = SimpleFolderDataset(
        root=root, input_size=args.input_size, transform=transform)
    dataloader = DataLoader(dataset)
    # Path(args.save_path).mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(tqdm(dataloader)):
        image, meta = batch
        img_name = meta['name'][0]
        c = meta['center'].numpy()[0]
        s = meta['scale'].numpy()[0]
        w = meta['width'].numpy()[0]
        h = meta['height'].numpy()[0]

        output = model(image.cuda())
        upsample = torch.nn.Upsample(
            size=args.input_size, mode='bilinear', align_corners=True)
        upsample_output = upsample(output[0][-1][0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(
            1, 2, 0)  # CHW -> HWC

        logits_result = transform_logits(
            upsample_output.data.cpu().numpy(), c, s, w, h, input_size=args.input_size)
        parsing_result = np.argmax(logits_result, axis=2)
        parsing_result_path = os.path.join(
            args.save_path, img_name[:-4] + '.png')
  
        parsing_result = np.asarray(parsing_result, dtype=np.uint8)

        parsing_result = np.expand_dims(parsing_result, axis=2)
        parsing_result = np.concatenate(
            (parsing_result, parsing_result, parsing_result), axis=2)
        a = 127
        b = 255
        index = [[0, 0, 0], [b, 0, 0], [0, b, 0], [0, 0, b], [b, b, 0], [0, b, b], [b, 0, b], [b, 0, a], [
            b, a, 0], [0, a, b], [a, a, a], [a, a, b], [a, b, a], [a, b, b], [0, a, a]]
        for i, j in enumerate(index):
            parsing_result[:, :,
                           0][parsing_result[:, :, 0] == i] = j[0]
            parsing_result[:, :,
                           1][parsing_result[:, :, 1] == i] = j[1]
            parsing_result[:, :,
                           2][parsing_result[:, :, 2] == i] = j[2]
        # parsing_result = cv2.cvtColor(parsing_result, cv2.COLOR_BGR2RGB)
        # plt.imshow(parsing_result)  # 아웃풋 이미지
        # plt.show()
        cv2.imwrite(parsing_result_path,parsing_result)
        origin_img = cv2.imread(os.path.join(root,img_name))
        img_seg = cv2.addWeighted(origin_img, 0.5, parsing_result, 0.3, 0)     
        addh = cv2.hconcat([img_seg, parsing_result])

        addh = cv2.cvtColor(addh, cv2.COLOR_BGR2RGB)
        plt.imshow(addh)  # 아웃풋 이미지
        plt.show()
        # cv2.namedWindow('frame',flags=cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', width=800, height=1000)
    #     try:
    #         cv2.imshow('frame',addh)
    #         # frame_array.append(img_pose)
    #     except:
    #         pass

    #     if cv2.waitKey(0) & 0xFF == 27:
    #         break

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
