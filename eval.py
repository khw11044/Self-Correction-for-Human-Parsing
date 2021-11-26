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

def get_mIOU(seg_index,prid_area,seg_area):

    seg_unique, seg_counts = np.unique(seg_area, return_counts=True)
    print('seg_area',dict(zip(seg_unique, seg_counts)))

    prid_unique, prid_counts = np.unique(prid_area, return_counts=True)
    print('prid_area',dict(zip(prid_unique, prid_counts)))

    union = np.logical_or(prid_area == 1, seg_area == 1)
    inter = np.logical_and(prid_area == 1, seg_area == 1)

    union_unique, union_counts = np.unique(union, return_counts=True)
    union_TF = dict(zip(union_unique, union_counts))

    inter_unique, inter_counts = np.unique(inter, return_counts=True)
    inter_TF = dict(zip(inter_unique, inter_counts))
    try:
        iou = inter_TF[True] / union_TF[True]
    except:
        iou = 0
    return iou

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
    state_dict = torch.load('checkpoints/schp_6_checkpoint.pth.tar',
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

        seg_annot_name = './demo/demo_seg/' + img_name.split('.')[0] + '.png'
        seg_annot = cv2.imread(seg_annot_name)

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

        # index = [
        #     [0,0,0],                                                        # 배경
        #     [a,a,a], [0,0,b],                                               # 머리 몸통
        #     [0,b,b], [b,0,b], [b,b,0], [0,b,0], [a,b,a], [a,b,b],           # 왼팔위, 왼팔아래, 왼손, 왼다리위, 왼다리아래, 왼발
        #     [0,a,b], [b,0,a], [0,a,a], [b,0,0], [b,a,0], [a,a,b]            # 오른팔위, 오른팔아래, 오른다리위, 오른다리아래, 오른발
        # ]

        index = [[0, 0, 0],                      # 배경
        [b, 0, 0], [0, b, 0], [0, 0, b],         # 오른다리위 왼다리위 몸통
        [b, b, 0], [0, b, b], [b, 0, b],         # 왼팔손 왼팔위 왼팔아래
        [b, 0, a], [b, a, 0], [0, a, b],         # 오른팔아래 오른다리아래 오른팔위
        [a, a, a], [a, a, b], [a, b, a],         # 머리 오른발 왼다리아래
        [a, b, b], [0, a, a]]                    # 왼발 오른손 

        # 
        seg_area = []
        for b in range(len(index)):
            seg_area.append( np.full((parsing_result.shape[0],parsing_result.shape[1]), 0) )

        for x in range(parsing_result.shape[0]):
            for y in range(parsing_result.shape[1]):
                seg_rgb = list(seg_annot[x][y])
                if seg_rgb == index[0]:         # 배경
                    seg_area[0][x][y] = 1       
                elif seg_rgb == index[1]:       # 오른다리위
                    seg_area[1][x][y] = 1       
                elif seg_rgb == index[2]:       # 왼다리위
                    seg_area[2][x][y] = 1       
                elif seg_rgb == index[3]:       # 몸통
                    seg_area[3][x][y] = 1       
                elif seg_rgb == index[4]:       # 왼팔손
                    seg_area[4][x][y] = 1       
                elif seg_rgb == index[5]:       # 왼팔위
                    seg_area[5][x][y] = 1       
                elif seg_rgb == index[6]:       # 왼팔아래
                    seg_area[6][x][y] = 1       
                elif seg_rgb == index[7]:       # 오른팔아래
                    seg_area[7][x][y] = 1       
                elif seg_rgb == index[8]:       # 오른다리아래
                    seg_area[8][x][y] = 1       
                elif seg_rgb == index[9]:       # 오른팔위
                    seg_area[9][x][y] = 1       
                elif seg_rgb == index[10]:      # 머리
                    seg_area[10][x][y] = 1       
                elif seg_rgb == index[11]:      # 오른발
                    seg_area[11][x][y] = 1       
                elif seg_rgb == index[12]:      # 왼다리아래
                    seg_area[12][x][y] = 1       
                elif seg_rgb == index[13]:      # 왼발  
                    seg_area[13][x][y] = 1         
                elif seg_rgb == index[14]:      # 오른손
                    seg_area[14][x][y] = 1       
  
                
    
        
        for i, j in enumerate(index):
            prid_area = np.full((parsing_result.shape[0],parsing_result.shape[1]), 0)
            prid_area[parsing_result[:, :, 0] == i] = 1

            parsing_result[:, :,0][parsing_result[:, :, 0] == i] = j[0]
            parsing_result[:, :,1][parsing_result[:, :, 1] == i] = j[1]
            parsing_result[:, :,2][parsing_result[:, :, 2] == i] = j[2]
            # c1 = [parsing_result[:, :, 0] == i][0]
            

            unique, counts = np.unique(prid_area, return_counts=True)
            print('prid_area ',dict(zip(unique, counts)))

            iou = get_mIOU(j,prid_area,seg_area[i])
            print('class {} : {}%'.format(i,iou))
            
   
           
        # parsing_result = cv2.cvtColor(parsing_result, cv2.COLOR_BGR2RGB)
        # plt.imshow(parsing_result)  # 아웃풋 이미지
        # plt.show()
        cv2.imwrite(parsing_result_path,parsing_result)

        origin_img = cv2.imread(os.path.join(root,img_name))
        img_seg = cv2.addWeighted(origin_img, 0.5, parsing_result, 0.3, 0)     
        addh = cv2.hconcat([img_seg, parsing_result])
        # plt.imshow(addh)  # 아웃풋 이미지
        # plt.show()

        cv2.namedWindow('frame',flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', width=800, height=1000)
        addh = cv2.hconcat([img_seg, parsing_result])
        try:
            cv2.imshow('frame',addh)
        except:
            pass

        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
