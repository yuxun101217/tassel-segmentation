import os
import cv2
from collections import Counter


label_path = r"F:/yx/dataset/oridatapng/whole/quan"
labs=os.listdir(label_path)
pos=0
alls=0
fname = open("F:/yx/dataset/oridatapng/whole/quan/quan1.txt",'w')
for l in labs:
    print(l[0:-4])
    lab_path = os.path.join(label_path,l)
    lab = cv2.imread(lab_path)
    labband = lab[:,:,1]
    lab_list = list(labband.flatten())
    dt = Counter(lab_list)
    pos += dt[2]

    alls +=(dt[0]+dt[2]+0.0)
    posa = pos  
    fname.write("图像的像素2数目" + ";" + str(posa) + ";" +"图像的像素0数目" + ";" + str(alls)+ "\n")
    print("---"*40)
    print("图像的像素1数目",posa)
    pos=0
    alls=0
  



