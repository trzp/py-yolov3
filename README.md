# py-yolov3
* 提供了python使用yolov3类的多进程程序，查看yolo_mul_pro
* 该类完全适用windows系统

* 部分文件未上传，查看readme.png
* yolo需要加载xx.data文件（比如coco.data）而xx.data文件又需要依赖于xx.names，这个xx.names的路径是在xx.data中指定的，是以相对路径给出。这极易造成无法找到xx.names文件，此时可以在xx.data中手动修改xx.names路径为绝对路径
