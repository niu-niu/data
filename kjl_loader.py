#!/usr/bin/env python3
import json
import os
import numpy as np
import geometry as geo
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

        vfov = meta['camera_vfov']
        rz = np.array([dst[1] - src[1] for src, dst in zip(meta['camera_position'].items(), meta['camera_look_at'].items())])
        rx = np.cross(rz, np.array([0, 0, 1]))
        ry = np.cross(rz, rx)
        rx = rx / np.linalg.norm(rx)
        ry = ry / np.linalg.norm(ry)
        rz = rz / np.linalg.norm(rz)
        self.R = np.array([rx, ry, rz]).transpose()
        self.position = np.array([v for k, v in meta['camera_position'].items()]) * 0.001
        fy = self.height / 2 / np.tan(vfov / 180 * np.pi / 2)
        self.tt = np.cross(rz, np.array([0, 0, 1]))

        self.model = CameraModel(fy, fy, self.width / 2, self.height / 2).cam_matrix()
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
        rooms = [Room(r) for r in structure[1]['room_meta']]
        rgb_files = [os.path.abspath(os.path.join(kjl_top, 'render', cam.id, 'rgb.jpg')) for cam in cams]
        depth_files = [os.path.abspath(os.path.join(kjl_top, 'rasterize', cam.id, 'depth.png')) for cam in cams]
        instance_files = [os.path.abspath(os.path.join(kjl_top, 'rasterize', cam.id, 'instance.png')) for cam in cams]
        semantic_files = [os.path.abspath(os.path.join(kjl_top, 'rasterize', cam.id, 'semantic.png')) for cam in cams]
        return cams, rgb_files, depth_files, instance_files, semantic_files, rooms

if __name__ == '__main__':
    cams,_,_,_,_,rooms = load_kjl('.')
    print(cams[0].summary())
    # print(rooms[0].floor_polygon())

    with open("HOUSE2.json", 'r') as f:
        data = json.load(f)
        ins_data = data[6:-1]
        fridge_data = ins_data[1]
        print(fridge_data)

    width = 640
    height = 480
    img_rgb = cv.imread("Traj_0_0_rgb.jpg")
    img_size = img_rgb
    img_depth = cv.imread("Traj_0_0_depth.png")
    print(cams[0].model)
    print(cams[0].pose)
    print(cams[0].position)
    cams_model_pad = np.pad(cams[0].model,((0,0),(0,1)),'constant',constant_values=(0,0))
    fridge_c = np.pad(fridge_data["ins_center"],(0,1),'constant',constant_values=(0,1))
    print(fridge_c.T)
    print(np.dot(cams[0].pose,fridge_c.T))
    # t = np.dot(cams_model_pad,cams[0].pose)
    # print(np.dot(t,fridge_c.T))
    rvec = cams[0].R
    tvec = cams[0].tt


    cube = np.float64([[-1865.811,476.8,902.35]])
    result, _ = cv.projectPoints(cube, rvec, tvec, cams[0].model, 0)
    print("3D to 2D 的 8个点的坐标：", result/209.56)
    cv.circle(img_rgb,(int(-6.3/209.56+320),int(6.3/209.56+160)),5,(0,255,0),1)
    cv.imwrite("test.jpg", img_rgb)





    