import cv2 as cv
import numpy as np


img_rgb = cv.imread("./images/HOUSE2-2-Traj_0_0-instance.png",-1)
img_color = cv.imread("./images/HOUSE2-2-Traj_0_0-instance.png")
img_array = np.array(img_rgb)
# target_mapping_ins = [47]
img_ins_list = []
for l in range(0,480):
    for m in range(0,640):
        if img_rgb[l][m] in img_ins_list:
            pass
        else:
            img_ins_list.append(img_rgb[l][m])

print(img_ins_list)

for l in range(0,480):
    for m in range(0,640):
        # if img_rgb[l][m] in img_ins_list:
        if img_rgb[l][m] == 50:
            # print(img_rgb[l][m])
            img_color[l][m] = [img_rgb[l][m]*1.5,0,255]

cv.imwrite("draw_ins.png",img_color)