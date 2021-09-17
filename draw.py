import cv2 as cv


img_rgb = cv.imread("Traj_0_0_rgb.jpg")


# bbox = [[255,   92],[230,   305],[265,   174],[251,   303],[346,   103],[355,   304],[332,   153],[334,   302]]        
bbox = [[300,   228],[299,   267],[300,   229],[299,   267],[319,   229],[319,   267],[318,   230],[318,   268]]  
for i in range(8):
        cv.circle(img_rgb,(bbox[i][0],bbox[i][1]),2,(0,0,255),2)
cv.imwrite("draw.jpg",img_rgb)