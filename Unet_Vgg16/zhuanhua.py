import cv2 
import os
import numpy as np 

path = "H:/unet-keras-master/img/UAV/val/"
out_path = "H:/unet-keras-master/img/UAV/val//"
filelist = os.listdir(path)
print("图片数目",len(filelist))
for i in filelist:
  print(i[0:-4])
  img = cv2.imread(path+str(i))
  img[img>0] =255
  iname=i[0:-4]+".jpg"
  cv2.imwrite(out_path+iname,img)
