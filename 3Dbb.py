#!/usr/bin/env python3
import numpy as np
import json
import cv2
import os

def load_render_data(base_path,batch_name):
    render_root = os.path.join(base_path,batch_name)
    with open(os.path.join(render_root,"user_output.json"), 'r') as f:
        data = json.load(f)
        camera_meta = data[0]["camera_meta"]
        room_meta   = data[1]["room_meta"]
        for i in camera_meta:
            i["camera_id"] = batch_name[0:-1] + "_" + i["camera_id"]
            # print(i["camera_id"])
        print(camera_meta)
        # camera_meta = camera_meta[""]
        return camera_meta, room_meta

def load_house_data(kjl_root):
    with open("./house_json/" + batch_name[0:-3] + ".json", 'r') as f:
        data = json.load(f)
        room_num = len(data[0]["room_ids"])
        ins_num_ASSET = len(data[0]["ins_ids_ASSET"])
        print("room number:%s" %room_num)
        print("ins  number:%s" %ins_num_ASSET)
        room_info = data[1:room_num]
        ins_info = data[(room_num+1):-1]
        rooms_data = [Room(r) for r in room_info]
        intances_data = [Instance(i) for i in ins_info]
        return intances_data, rooms_data

class BoundingBoxBuilder:
    def __init__(self, ins)

class DataBuilder:
    def __init__(self, cam_meta, room_meta, instances_data, rooms_data):
        print(cam_info,room_info,id)

class Instance:
    def __init__(self, info):
        self.id = info["ins_id"]
        self.label = info["ins_label"]
        self.size = info["ins_size"]
        self.rotation_mat = info["ins_rotation"]
        self.center = info["ins_center"]
        self.tf = info["ins_tf"]

class Room:
    def __init__(self, info):
        self.id = info["room_ID"]
        self.name = info["room_name"]
        self.position = info["room_position"]
        self.boundary = info["room_boundary"]
        self.aream = info["room_area"]
    
class House:
    def __init__(self, instances, rooms):
        self.rooms = rooms
        self.instances = instances
        print(self.instances)

    def draw_house(self):
        img = np.zeros((10000,10000,3), np.uint8)
        point_size = 50
        point_color = (0, 0, 255) # BGR
        thickness = 50
        for r in self.rooms:
            # draw
            for b in r.boundary:
                # print(b[0]+5000, b[1]+5000)
                cv2.circle(img,(int(b[0]+5000),int(b[1]+5000)),point_size, point_color, thickness)

        for i in self.instances:
            print(i.center[0]+5000, i.center[1]+5000)
            cv2.circle(img,(int(i.center[0]+5000),int(i.center[1]+5000)),2, (0,255,2525), thickness)
            cv2.putText(img,'id:%s,lable:%s' % (i.id, i.label),(int(i.center[0]+5000),int(i.center[1]+5000)),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),15)
        while(True):
            cv2.namedWindow("image")
            cv2.imshow('image', img)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        cv2.destroyAllWindows()
    
if __name__ == '__main__':
    base_path = "/home/renping/Downloads/"
    batch_name = "HOUSE2-2/"
    ins_data, room_data = load_house_data(batch_name)
    H1 = House(ins_data, room_data)
    cam_meta, room_meta = load_render_data(base_path,batch_name)
    Builder = DataBuilder(cam_meta,room_meta, ins_data, room_data)
    
    # H1.draw_house()

# instance_center_p = [100,100,99]
# instance_size = [22,44,66]

# add = np.array([instance_center_p[i] + instance_size[i]/2 for i in [0,1,2]])
# mins = np.array([int(instance_center_p[i] - instance_size[i]/2) for i in [0,1,2]])

# output = np.array((add,mins)).T
# print(output)

# combinations = np.array([(x,y,z) for x in output[0] for y in output[1] for z in output[2]])
# print(combinations)



