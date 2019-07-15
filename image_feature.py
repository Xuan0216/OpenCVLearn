# import cv2

# # 打印版本号
# print(cv2.__version__)
# print(cv2.__file__)
# # 1、读取图片
# img=cv2.imread('./Julia.PNG')

# # 2、展示图片
# cv2.imshow('Julia',img)

# # 解决闪现
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # 英文输入法q退出

# # 3、保存图片
# cv2.imwrite('./Julia.png',img)

# # 键盘控制窗口
# k = cv2.waitKey(0)
# if k == ord('q'):
#     cv2.destroyAllWindows()
# elif k == ord('s'):
#     cv2.imwrite('Julia.PNG', img)
# cv2.destroyAllWindows()

# # 绘图库matplotlib的结合
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# img = cv2.imread('Julia.png', 0) # grayimage
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.axis('off')
# plt.show()

# # 图片的基本操作

# # 1、获取并修改像素值
# import numpy as np
# import cv2
# img = cv2.imread('Julia.png')
# px = img[200, 200]
# print(px)

# # accessing only blue pixel
# blue = img[200, 200, 0]
# print(blue)
# green = img[200, 200, 1]
# print(green)
# tmp=img[200:300,200:300]
# cv2.imwrite('Julia_px.png',tmp)
# tmp=img[200:300,200:300]
# cv2.imwrite('Julia_px.png',tmp)

import cv2
#加载图片
img=cv2.imread('woman.png')
#转换成灰度，提高计算速度
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#加载Haar特征分类器
face_cascade=cv2.CascadeClassifier('D:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
print(face_cascade,type(face_cascade))
#检测图片中的人脸
faces=face_cascade.detectMultiScale(
    gray,               	#要检测的图像
    scaleFactor=1.15,     	#图像尺寸每次缩小的比例
    minNeighbors=3,     	#一个目标至少要检测到3次才会被标记为人脸
    minSize=(5,5))    		#目标的最小尺寸

print(help(face_cascade.detectMultiScale))
#为每个人脸设置矩形值
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow('img',img)
cv2.waitKey(0)