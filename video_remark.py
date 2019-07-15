# 读取并且显示视频，框出检测到的人脸

import cv2
from time import sleep

cascPath = "D:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture('dance.mp4')

while True:
    if not cap.isOpened():
        print('Unable to load video.')
        sleep(5)
        pass
        # 读视频帧
    ret, frame = cap.read()
    # 转为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray)
    # 调用分类器进行检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )  # 画矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 显示视频
    cv2.resizeWindow('Video', 620, 350)
    cv2.imshow('Video', frame)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()