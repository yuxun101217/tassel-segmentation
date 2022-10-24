import os
import cv2
from collections import Counter


label_path = r"E:/数穗"
labs=os.listdir(label_path)
fname = open("E:/数穗/mingcheng.txt",'w')

for l in labs:
    print(l[0:-4])
    lab_path = os.path.join(label_path,l)
    fname.write("图像名称" + ";" + str(l[0:-4]) + "\n")
  