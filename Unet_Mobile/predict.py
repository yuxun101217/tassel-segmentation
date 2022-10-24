from nets.unet import mobilenet_unet
from PIL import Image
import numpy as np
import random
import copy
import os
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))

class_colors = [(0,0,0),(1,1,1)]
NCLASSES = 2
HEIGHT = 416*2
WIDTH = 416*2


model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("F:/fenge/hiss_zhan-Semantic-Segmentation-master/Semantic-Segmentation/Unet_Mobile/logs/0logs/20201221_ep048-loss0.398-val_loss0.552.h5")

imgs = os.listdir("F:/yx/dataset/oridatajpg/whole/train")

for jpg in imgs:

    img = Image.open("F:/yx/dataset/oridatajpg/whole/train/"+jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT), int(WIDTH),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
    image = seg_img
    #image = Image.blend(old_img,seg_img,0.3)
    image.save("F:/yx/dataset/oridatajpg/whole/mobilenet/"+jpg)
    print(jpg," done!")

