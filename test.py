# 此函数只是外部定义而已，大家可自行定义
# camera_matrix, rvec, tvec = camera_params()
import numpy as np
import cv2





camera_matrix = np.array([[201,0,320],[0,201,160],[0,0,1]],np.float)
tvec = np.array([1838.3618,-2244.7388,-45.0],np.float)
rvec = np.array([[-2036.7,793.5,464.9],[1,1,1],[1,1,1]], np.float)

print("相机内参:", camera_matrix)

print("平移向量:", tvec)

print("旋转矩阵:", rvec)

# # (R T, 0 1)矩阵
# Trans = np.hstack((rvec, [[tvec[0]], [tvec[1]], [tvec[2]]]))

# # 相机内参和相机外参 矩阵相乘
# temp = np.dot(camera_matrix, Trans)

# Pp = np.linalg.pinv(temp)

# # 点（u, v, 1) 对应代码里的 [605,341,1]
# p1 = np.array([605, 341, 1], np.float)

# print("像素坐标系的点:", p1)


# X = np.dot(Pp, p1)

# print("X:", X)

# # 与Zc相除 得到世界坐标系的某一个点
# X1 = np.array(X[:3], np.float)/X[3]

# print("X1:", X1)


## 3D 转成 2D
## cube为世界坐标系的8个点的三维坐标
cube = np.float64([[-3.102,-1.58400011, 9.29399872],[-3.102, -0.08400005, 9.29399872]
 ,[-1.27200007,-0.08400005 , 9.29399872]
 ,[-1.27200007, -1.58400011  ,9.29399872]
 ,[-3.102   ,   -1.58400011 ,13.8939991 ]
 ,[-3.102   ,   -0.08400005, 13.8939991 ]
 ,[-1.27200007 ,-0.08400005, 13.8939991 ]
 ,[-1.27200007, -1.58400011 ,13.8939991 ]])
result, _ = cv2.projectPoints(cube, rvec, tvec, camera_matrix, 0)
print("3D to 2D 的 8个点的坐标：", result)