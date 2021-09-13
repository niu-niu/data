#!/usr/bin/env python3
import json
import os
import numpy as np
import geometry as geo  #pygeogmetry
import cv2 as cv

class CameraModel:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def cam_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def project(self, point):
        homo = self.cam_matrix() @ point
        return homo[0:2] / homo[2]

    def lift(self, pixel):
        return np.append((pixel - np.array([[self.cx], [self.cy]])) / np.array([[self.fx],[self.fy]]), 1.)

class Camera:
    def __init__(self, meta):
        self.id = meta['camera_id']
        self.room_id = meta['camera_room_id']
        self.width = meta['image_width']
        self.height = meta['image_height']

        self.vfov = meta['camera_vfov']
        rz = np.array([dst[1] - src[1] for src, dst in zip(meta['camera_position'].items(), meta['camera_look_at'].items())])
        rx = np.cross(rz, np.array([0, 0, 1]))
        ry = np.cross(rz, rx)
        rx = rx / np.linalg.norm(rx)
        ry = ry / np.linalg.norm(ry)
        rz = rz / np.linalg.norm(rz)
        self.R = np.array([rx, ry, rz]).transpose()
        self.position = np.array([v for k, v in meta['camera_position'].items()]) * 0.001
        self.fy = self.height / 2 / np.tan(self.vfov / 180 * np.pi / 2)
        self.model = CameraModel(self.fy, self.fy, self.width / 2, self.height / 2).cam_matrix()
        self.pose = geo.pose_from_rotation_translation(self.R, self.position)

    def summary(self):
        print('camera id {} in room {}'.format(self.id, self.room_id))
        print('camera matrix:')
        print(self.model)
        print('camera pose:')
        print(self.pose)
       
class Room:
    def __init__(self, meta):
        self.id = meta['id']
        self.type_id = meta['type_id']
        self.floor_polygon = np.array([[v for k,v in edge['start'].items()] for edge in meta['floor']]) * 0.001
        self.ceil_polygon = np.array([[v for k,v in edge['start'].items()] for edge in meta['ceil']]) * 0.001

def load_kjl(kjl_top):
    with open(os.path.join(kjl_top, 'data/user_output.json'), 'r') as f:
        structure = json.load(f)
        print("position:",structure[0]['camera_meta'][0]['camera_position'])
        print("look at :",structure[0]['camera_meta'][0]['camera_look_at'])
        cams = [Camera(c) for c in structure[0]['camera_meta']]
        print("cam position:",cams[0].position)
        rooms = [Room(r) for r in structure[1]['room_meta']]
        rgb_files = [os.path.abspath(os.path.join(kjl_top, 'render', cam.id, 'rgb.jpg')) for cam in cams]
        depth_files = [os.path.abspath(os.path.join(kjl_top, 'rasterize', cam.id, 'depth.png')) for cam in cams]
        instance_files = [os.path.abspath(os.path.join(kjl_top, 'rasterize', cam.id, 'instance.png')) for cam in cams]
        semantic_files = [os.path.abspath(os.path.join(kjl_top, 'rasterize', cam.id, 'semantic.png')) for cam in cams]
        return cams, rgb_files, depth_files, instance_files, semantic_files, rooms

if __name__ == '__main__':
    cams,_,_,_,_,rooms = load_kjl('.')

    print("vfov:", cams[0].vfov)
    print("fy:",cams[0].fy)
    fridge_c = np.array([-1865.811, 476.8019,902.35, 1.0])
    P_c = np.dot(cams[0].pose,fridge_c)
    print("Pc:", P_c)

    cams_model_pad = np.pad(cams[0].model,((0,0),(0,1)),'constant',constant_values=(0,0))
    print("cams_model_pad:",cams_model_pad)
    P_uvz = np.dot(cams_model_pad,P_c)
    Z = np.sqrt(np.sum(P_c)**2)
    print("Z:",Z)
    P_uv_tmp = P_uvz/Z
    P_uv = P_uv_tmp/P_uv_tmp[2]
    print("P_uv:",P_uv)
    print("------------------------------------------------")

    img_rgb = cv.imread("Traj_0_0_rgb.jpg")
    img_depth = cv.imread("Traj_0_0_depth.png")
    cv.circle(img_rgb,(int(P_uv[0]),int(P_uv[1])),5,(0,255,0),1)
    cv.imwrite("test.jpg", img_rgb)



    rvec = cams[0].R
    print("cam pose:", cams[0].pose)
    tvec = np.array([cams[0].pose[0][3],cams[0].pose[1][3],cams[0].pose[2][3]])
    print("tvec:",tvec)
    cube = np.float64([[-1865.811, 476.8019,902.35],])
    result, _ = cv.projectPoints(cube, rvec, tvec, cams[0].model, 0)
    print("P_uv opencv 3D to 2D：", result)