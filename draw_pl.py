import cv2 as cv
import numpy as np


# img_rgb = cv.imread("Traj_0_0_rgb.jpg")

img_rgb = np.zeros((800,800,3),np.uint8)
img_rgb[:] = [0,0,0]

pts = np.array([[241,   202],[230,   300],[203,  200],[185,   300]] , np.int32)     # 47 冰箱1
pts = pts.reshape((-1, 1, 2))

cv.polylines(img_rgb, [pts], True, (0, 255, 255),1,4)

cv.imwrite("draw_pl.jpg",img_rgb)
