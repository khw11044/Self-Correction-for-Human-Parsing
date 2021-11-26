import glob
from typing import Counter 
import cv2 
from PIL import Image
import matplotlib.pylab as plt
import time
import numpy as np
import time 

all_files = sorted(glob.glob('./seg/input_seg2' + '/*.png'))

save_true = False
count = 0 
print(len(all_files))
for i,folder in enumerate(all_files):

    start_time = time.time()
    print('Reading.......',folder)
    save_true = False
    #r = Image.open(folder)
    img = cv2.imread(folder)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            # rgb = r.getpixel((i,j))
            rgb = tuple(img[i][j])
            if rgb == (127,127,127): # 머리            # 127,127,127                  
                img[i][j] = np.array([14,14,14]) # 14,14,14
                save_true = True
            elif rgb == (0,0,255): # 몸통 255,0,0
                img[i][j] = np.array([1,1,1])     # bgr
                save_true = True
            elif rgb == (0,255,255): # 255,127,0
                img[i][j] = np.array([10,10,10])   # 왼팔 위           
                save_true = True
            elif rgb == (255,0,255): # 127,0,255
                img[i][j] = np.array([12,12,12])   # 왼팔 아래 [255,0,127]
                save_true = True
            elif rgb == (255,255,0):        # 127,127,0
                img[i][j] = np.array([3,3,3])   # 왼팔 손 [0,127,127]
                save_true = True
            elif rgb == (0,127,255): # 255,255,0
                img[i][j] = np.array([11,11,11])   # 오른팔 위 [0,255,255]
                save_true = True
            elif rgb == (255,0,127): # 2
                img[i][j] = np.array([13,13,13])   # 오른팔 아래 [255,0,255]
                save_true = True
            elif rgb == (0,127,127):    
                img[i][j] = np.array([2,2,2])   # 오른 손 [255,255,0]3
                save_true = True
            elif rgb == (0,255,0):
                img[i][j] = np.array([7,7,7])     # 왼다리 위 [255,0,0]
                save_true = True
            elif rgb == (127,255,127):
                img[i][j] = np.array([9,9,9]) # 왼다리 아래  [255,127,0]
                save_true = True
            elif rgb == (127,255,255):
                img[i][j] = np.array([4,4,4]) # 왼발 [127,127,255]
                save_true = True
            elif rgb == (255,0,0):
                img[i][j] = np.array([6,6,6])     # 오른다리 위 [0,255,0]
                save_true = True
            elif rgb == (255,127,0):
                img[i][j] = np.array([8,8,8])   # 오른다리 아래  [127,255,127]
                save_true = True
            elif rgb == (127,127,255):
                img[i][j] = np.array([5,5,5])     # 오른발 [127,255,255]
                save_true = True

                


    if save_true == True:
        cv2.imwrite(folder,img)
        # r.save(folder)
        print('change')
    count += 1
    print('time:',time.time() - start_time)
    print('folder {}/{}'.format(str(count),str(len(all_files))))