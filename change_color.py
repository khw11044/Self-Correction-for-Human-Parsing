import glob
from typing import Counter 
import cv2 
from PIL import Image
import matplotlib.pylab as plt
import time
import numpy as np
import time 

all_files = sorted(glob.glob('./seg' + '/*.png'))

save_true = False
count = 0 
for i,folder in enumerate(all_files):
    if i > 35:
        start_time = time.time()
        print('Reading.......',folder)
        save_true = False
        #r = Image.open(folder)
        img = cv2.imread(folder)
        for i in range(0,img.shape[0]):
            for j in range(0,img.shape[1]):
                # rgb = r.getpixel((i,j))
                rgb = tuple(img[i][j])
                # print('({},{})'.format(str(i),str(j)),rgb)
                if rgb == (14,14,14): # 머리                          
                    img[i][j] = np.array([127,127,127])
                    # print('({},{})'.format(str(i),str(j)),rgb)
                    #r.putpixel((i,j),(127,127,127))
                    save_true = True
                elif rgb == (1,1,1): # 몸통
                    img[i][j] = np.array([0,0,255])     # bgr
                    #r.putpixel((i,j),(255,0,0))
                    save_true = True
                elif rgb == (11,11,11):
                    img[i][j] = np.array([0,127,255])   # 왼팔 위           
                    #r.putpixel((i,j),(255,255,0))
                    save_true = True
                elif rgb == (13,13,13):
                    img[i][j] = np.array([255,0,127])   # 왼팔 아래 [255,0,127]
                    #r.putpixel((i,j),(255,0,255))
                    save_true = True
                elif rgb == (2,2,2):
                    img[i][j] = np.array([0,127,127])   # 왼팔 손 [0,127,127]
                    #r.putpixel((i,j),(0,255,255))
                    save_true = True
                elif rgb == (10,10,10):
                    img[i][j] = np.array([0,255,255])   # 오른팔 위 [0,255,255]
                    #r.putpixel((i,j),(255,127,0))
                    save_true = True
                elif rgb == (12,12,12):
                    img[i][j] = np.array([255,0,255])   # 오른팔 아래 [255,0,255]
                    #r.putpixel((i,j),(127,0,255))
                    save_true = True
                elif rgb == (3,3,3):
                    img[i][j] = np.array([255,255,0])   # 오른 손 [255,255,0]
                    #r.putpixel((i,j),(127,127,0))
                    save_true = True
                elif rgb == (6,6,6):
                    img[i][j] = np.array([255,0,0])     # 왼다리 위 [255,0,0]
                    #r.putpixel((i,j),(0,255,0))
                    save_true = True
                elif rgb == (8,8,8):
                    img[i][j] = np.array([255,127,0]) # 왼다리 아래  [255,127,0]
                    #r.putpixel((i,j),(127,255,127))
                    save_true = True
                elif rgb == (5,5,5):
                    img[i][j] = np.array([127,127,255]) # 왼발 [127,127,255]
                    #r.putpixel((i,j),(255,255,127))
                    save_true = True
                elif rgb == (7,7,7):
                    img[i][j] = np.array([0,255,0])     # 오른다리 위 [0,255,0]
                    #r.putpixel((i,j),(0,0,255))
                    save_true = True
                elif rgb == (9,9,9):
                    img[i][j] = np.array([127,255,127])   # 오른다리 아래  [127,255,127]
                    #r.putpixel((i,j),(0,127,255))
                    save_true = True
                elif rgb == (4,4,4):
                    img[i][j] = np.array([127,255,255])     # 오른발 [127,255,255]
                    #r.putpixel((i,j),(255,127,127))
                    save_true = True

                


        if save_true == True:
            cv2.imwrite(folder,img)
            # r.save(folder)
            print('change')
        count += 1
        print('time:',time.time() - start_time)
        print('folder {}/{}'.format(str(count),str(len(all_files))))