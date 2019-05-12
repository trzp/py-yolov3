#coding:utf-8

import os
import sys

from darknet_yolov3 import *
from pykinect import KinectClientV2019
from pykinect import locate_bottle,locate_obj
import multiprocessing
from multiprocessing import Queue,Event
import cv2
import time
import threading
from darknet_yolov3 import *
import numpy.linalg as nplg

class YoloPro(multiprocessing.Process):
    def __init__(self):
        self.dd = Detector()
        self._ev = Event()
        self._qu = Queue()
        self.targets = []
        self.kk = kk = KinectClientV2019()
        multiprocessing.Process.__init__(self)

    # def __del__(self):
        # self._ev.set()
        # while self._ev.is_set():
            # time.sleep(0.2)     #等待进程结束

    def new_rec(self):
        flg = False
        while not self._qu.empty():
            objs = self._qu.get()
            flg = True
        
        if flg:
            self.targets = []
            for obj in objs:
                r = self.parse_obj(obj)
                if r is not None:
                    self.targets.append(r)
            self.targets = sorted(self.targets,key = lambda tar:nplg.norm(tar['position'])) #将目标按照距离排序，前面的距离近
        return self.targets

    def parse_obj(self,obj):
        name,box = obj
        target_object_list = ['person','cup','bottle','diningtable','sofa','chair']
        if name in target_object_list:
            tar = {}
            x,y,w,h = box
            rect = ((x,y),(w,h))
            if name == 'diningtable':name = 'desk'
            tar['name'] = name
            tar['box'] = box
            tar['rect'] = rect
            if name == 'bottle':
                p = locate_bottle(rect,kk.point_cloud)
            else:
                p = locate_obj(rect,kk.point_cloud)
            tar['position'] = p
            return tar
        else:
            return None

    def run(self):
        kk = KinectClientV2019()    #不断将结果发送到队列
        rec_im = r'./tem.jpg'
        # while not self._ev.is_set():
        END = 0
        while not END:
            fr = kk.get_color_as_cvframe()
            cv2.imwrite(rec_im, fr)
            time.sleep(0.005)
            r = self.dd.rec(rec_im)
            if self._qu.qsize() > 50:   #限制队列长度
                while not self._qu.empty():
                    self._qu.get()
            self._qu.put(r)

        self._ev.clear()

def demo():
    import time
    yolo = YoloPro()
    yolo.start()
    while True:
        print yolo.new_rec()
        time.sleep(0.2)

if __name__ == '__main__':
    import time
    yolo = YoloPro()
    yolo.start()
    while True:
        print yolo.new_rec()
        time.sleep(0.2)