#coding:utf-8

import os
import sys

rootdir0 = os.path.split(os.path.abspath(__file__))[0]
rootdir = os.path.split(rootdir0)[0]
package_path = os.path.join(rootdir,'mrtang-orgnized-py-yolo\yolov3')
package_path1 = os.path.join(rootdir,'KinectV2019')
sys.path.append(package_path)
sys.path.append(package_path1)

from darknet_yolov3 import *
from KinectClient import *

import multiprocessing
from multiprocessing import Queue,Event
import cv2
import time

import threading
import numpy.linalg as npla

def parse_obj(obj,ratio = 0.6):
    name = obj[0]
    rect = obj[2]
    x,y,w,h = obj[2]
    left_corner_x = int(x - w/2)
    left_corner_y = int(y - h/2)

    center = (rect[0],rect[1])
    size = (rect[2]*ratio,rect[3]*ratio)
    x_range = (int(rect[0] - size[0]/2),int(rect[0] + size[0]/2))
    y_range = (int(rect[1] - size[1]/2),int(rect[1] + size[1]/2))

    position = [0,0,0]
    return {'name':name,'size':size,'x_range':x_range,'y_range':y_range,
            'center':center,'position':position,'left_corner_x':left_corner_x,
            'left_corner_y':left_corner_y,'width':w,'height':h,'box':obj[2]}

# def parse_obj_info(th_que,th_ev,pro_que,info_que):
    # info_que.put('**[Yolo] sub thread started')
    # kk = KinectClientV2019(0)
    # target_object_list = ['person','cup','bottle','diningtable','sofa','chair']
    # while True:
        # if th_ev.is_set(): break
        # objs = th_que.get()
        # targets = []
        # for obj in objs:
            # tar = parse_obj(obj)
            # if tar['name'] in target_object_list:
                # if tar['name'] == 'diningtable':  tar['name'] = 'desk'
                
                # x,y,w,h = tar['box']
                # x += w/2
                # y += h/2
                # ps = kk.point_cloud[int(x-0.35*w):int(x+0.35*w),int(y-0.35*h):int(y+0.35*h),:]
                
                
                # ps = kk.point_cloud[tar['y_range'][0]:tar['y_range'][1],tar['x_range'][0]:tar['x_range'][1],:]   #切割该区域的点云
                # ii = np.where((ps[:,:,2]>700) & (ps[:,:,1]<-200))
                # ii = np.where(ps[:,:,2]>700)
                # ps = ps[ii[0],ii[1],:]  #有效点云
                
                # if ps.size > 0:    #切割到有效点云
                    # dis = npla.norm(ps,axis=1)
                    # total_num = dis.size
                    # avr_dis = np.mean(dis)
                    # std_dis = np.std(dis)
                    # er = np.fabs(dis - avr_dis) - 2.5 * std_dis #异常点
                    # ind1 = np.where(er > 0)[0]
                    # nu = total_num - ind1.size  #有效点数
                    
                    # if nu > 0:
                        # dis[ind1] = 99999.
                        # k = np.min((nu,10))            
                        # ind = np.argsort(dis)[:k]   #取前面K个
                        # pos = np.mean(ps[ind,:],axis=0)
                        # tar['position'] = pos                   
                        # targets.append(tar)
        # pro_que.put(targets)
        # info_que.put('**[Yolo] get %d targets'%(len(targets)))
    # info_que.put('**[Yolo] sub thread killed')
    
def parse_obj_info(th_que,th_ev,pro_que,info_que):
    info_que.put('**[Yolo] sub thread started')
    kk = KinectClientV2019(0)
    target_object_list = ['person','cup','bottle','diningtable','sofa','chair']
    while True:
        if th_ev.is_set(): break
        objs = th_que.get()
        targets = []
        for obj in objs:
            tar = parse_obj(obj)
            if tar['name'] in target_object_list:
                if tar['name'] == 'diningtable':  tar['name'] = 'desk'
                
                x,y,w,h = tar['box']
                x += w/2
                y += h/2
                rr = 0.6
                rrr = rr/2
                ps = kk.point_cloud[int(x-rrr*w):int(x+rrr*w),int(y-rrr*h):int(y+rrr*h),:]
                
                
                # ps = kk.point_cloud[tar['y_range'][0]:tar['y_range'][1],tar['x_range'][0]:tar['x_range'][1],:]   #切割该区域的点云
                #ii = np.where((ps[:,:,2]>700) & (ps[:,:,1]<-200))
                ii = np.where(ps[:,:,2]>700)
                ps = ps[ii[0],ii[1],:]  #有效点云
                p = np.mean(ps,axis=0)
                p[0] *= -1
                tar['position'] = p
                targets.append(tar)

        pro_que.put(targets)
        # info_que.put('**[Yolo] get %d targets'%(len(targets)))
    info_que.put('**[Yolo] sub thread killed')


def yolo_pro(pro_que,ev,info_que):
    info_que.put('**[Yolo] process started')
    kk = KinectClientV2019(0)
    #d = Detector(package_path + r'\cfg\yolov3-openimages.cfg',package_path + r'/weights/yolov3-openimages.weights',package_path + r'/data/openimages.data')
    #d = Detector(package_path + r'\cfg\yolov2.cfg',package_path + r'/weights/yolov2.weights',package_path + r'/data/coco.data')
    d = Detector()  #默认使用yolov3模型，大约可达到6帧速度
    rec_im = r'./tem.jpg'

    th_que = Queue()
    threading._start_new_thread(parse_obj_info,(th_que,ev,pro_que,info_que)) #启动子线程用于处理点云等

    while True:
        fr = kk.get_color_as_cvframe()
        cv2.imwrite(rec_im,fr)
        time.sleep(0.005)
        r = d.rec(rec_im)
        # info_que.put('**[Yolo] completed one detection')
        th_que.put(r)

        if ev.is_set():   break

    time.sleep(3)
    del d
    del kk
    info_que.put('**[Yolo] process killed')

if __name__ == '__main__':
    que = Queue()
    que1 = Queue()
    ev = Event()
    process = multiprocessing.Process(target = yolo_pro, args = (que,ev,que1))
    process.start()
    while True:
        st = que1.get().split('**')
        for txt in st:
            if len(txt)>0:  print txt