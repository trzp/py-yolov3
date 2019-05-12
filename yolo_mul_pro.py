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
import numpy.linalg as nplg
import pygame
from pygame.locals import *

def yolo_pro(args):
    ev = args['ev']
    qu = args['qu']
    kk = KinectClientV2019()
    dd = Detector()
    rec_im = r'./tem.jpg'

    while not ev.is_set():
        fr = kk.get_color_as_cvframe()
        cv2.imwrite(rec_im, fr)
        r = dd.rec(rec_im)
        if qu.qsize() > 200:   #限制队列长度
            while not qu.empty():
                qu.get()
        qu.put(r)

    ev.clear()


class Yolo_Pro():
    def __init__(self):
        self.ev = Event()
        self.qu = Queue()
        self.targets = []
        self.kk = KinectClientV2019()
        self.args = {'ev':self.ev,'qu':self.qu}

    def __del__(self):
        self.ev.set()
        while self.ev.is_set():
            time.sleep(0.2)     #等待进程结束

    def update(self):
        flg = False
        while not self.qu.empty():  #只取最新的结果
            objs = self.qu.get()
            flg = True
        
        if flg:
            self.targets = []
            for obj in objs:
                r = self.parse_obj(obj)
                if (r is not None) and (r['position'] is not None):
                    self.targets.append(r)
            if len(self.targets) > 0:
                self.targets = sorted(self.targets,key = lambda tar:nplg.norm(tar['position'])) #将目标按照距离排序，前面的距离近
        return self.targets

    def parse_obj(self,obj):
        name,_,box = obj
        target_object_list = ['person','cup','bottle','diningtable','sofa','chair']
        if name in target_object_list:
            tar = {}
            cx,cy,w,h = box
            x = int(cx - w/2)
            y = int(cy - h/2)
            box = (x,y,w,h)
            rect = ((x,y),(w,h))
            if name == 'diningtable':name = 'desk'
            tar['name'] = name
            tar['box'] = box
            tar['rect'] = rect
            if name == 'bottle':
                p = locate_bottle(rect,self.kk.point_cloud)
            else:
                p = locate_obj(rect,self.kk.point_cloud)
            tar['position'] = p
            return tar
        else:
            return None

def demo():
    kk = KinectClientV2019()
    yolo = Yolo_Pro()
    process = multiprocessing.Process(target = yolo_pro,args = (yolo.args,))
    process.start()
    
    pygame.init()
    pygame.font.init()
    scr = pygame.display.set_mode((640,480), 0, 32)  # 主界面初始化
    clk = pygame.time.Clock()

    END = False
    while not END:
        KINECT_SUR = kk.get_color_as_pgsurface()
        scr.blit(KINECT_SUR, (0, 0))
        
        targets = yolo.update()
        for tar in targets:
            pygame.draw.rect(scr, (255,0,0),tar['rect'],2)

        ev = pygame.event.get()
        for e in ev:
            if e.type == QUIT:    END = True
        pygame.display.update()
        clk.tick(60)
    kk.release()

if __name__ == '__main__':
    demo()