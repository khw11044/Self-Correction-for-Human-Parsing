import sys
import numpy as np
import cv2
import glob

oldx = oldy = -1 # 좌표 기본값 설정

# 내가 만든 마우스 함수를 실행시켜줘


def on_mouse(event, x, y, flags, param):
    # event는 마우스 동작 상수값, 클릭, 이동 등등
    # x, y는 내가 띄운 창을 기준으로 좌측 상단점이 0,0이 됌
    # flags는 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미, Shif+마우스 등 설정가능
    # param은 영상이룻도 있도 전달하고 싶은 데이타, 안쓰더라도 넣어줘야함

    global oldx, oldy # 밖에 있는 oldx, oldy 불러옴

    if event == cv2.EVENT_LBUTTONDOWN: # 왼쪽이 눌러지면 실행
        oldx, oldy = x, y # 마우스가 눌렀을 때 좌표 저장, 띄워진 영상에서의 좌측 상단 기준
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_LBUTTONUP: # 마우스 뗏을때 발생
        print('EVENT_LBUTTONUP: %d, %d' % (x, y)) # 좌표 출력

    elif event == cv2.EVENT_MOUSEMOVE: # 마우스가 움직일 때 발생
        if flags & cv2.EVENT_FLAG_LBUTTON: # ==를 쓰면 다른 키도 입력되었을 때 작동안하므로 &(and) 사용
            # cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # 단점이 빠르게 움직이면 끊김

            # circle은 끊기므로 line 이용
            # 마우스 클릭한 좌표에서 시작해서 마우스 좌표까지 그림
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 0), 10, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y # 그림을 그리고 또 좌표 저장


root = './seg'

all_files = sorted(glob.glob(root + '/*.*'))


# 단일
fail_list = ['007_05']

for img_file in fail_list:
    img_file = root + '/' + img_file +'.png'
    # img_file = './drawing_test/seg/inbody/001_01.png'
    img = cv2.imread(img_file, cv2.IMREAD_COLOR) # img에 이미지를 불러옴. 

    # 윈도우 창
    cv2.namedWindow('image',flags=cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', width=600, height=800)

    # 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
    # 마우스 이벤트가 발생하면 on_mouse 함수 실행
    cv2.setMouseCallback('image', on_mouse, img)


    # 영상 출력
    try:
        cv2.imshow('image', img)
        # cv2.waitKey()
        if cv2.waitKey() & 0xFF == 0x1B: # 무한 반복중, 키보드 ESC 아스키코드가 들어오면 빠져나감. 
            cv2.imwrite(img_file,img)
            print('save',img_file)
            cv2.destroyAllWindows()
    except:
        cv2.destroyAllWindows()


# 흰색 컬러 영상 생성
# img = np.ones((480, 640, 3), dtype=np.uint8) * 255

# for img_file in all_files:
    
#     # img_file = './drawing_test/seg/inbody/001_01.png'
#     img = cv2.imread(img_file, cv2.IMREAD_COLOR) # img에 이미지를 불러옴. 

#     # 윈도우 창
#     cv2.namedWindow('image',flags=cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('image', width=600, height=800)

#     # 마우스 입력, namedWIndow or imshow가 실행되어 창이 떠있는 상태에서만 사용가능
#     # 마우스 이벤트가 발생하면 on_mouse 함수 실행
#     cv2.setMouseCallback('image', on_mouse, img)


#     # 영상 출력
#     cv2.imshow('image', img)
#     # cv2.waitKey()
#     if cv2.waitKey() & 0xFF == 0x1B: # 무한 반복중, 키보드 ESC 아스키코드가 들어오면 빠져나감. 
#         cv2.imwrite(img_file,img)
#         print('save',img_file)
#         cv2.destroyAllWindows()
