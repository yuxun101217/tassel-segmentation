from unet import Unet
from PIL import Image
import numpy as np
import os
class miou_Unet(Unet):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        img, nw, nh = self.letterbox_image(image,(self.model_image_size[1],self.model_image_size[0]))
        img = [np.array(img)/255]
        img = np.asarray(img)
        
        pr = self.model.predict(img)[0]
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0],self.model_image_size[1]])
        pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h))

        return image

unet = miou_Unet()

image_ids = open(r"F:/unet-keras-master/VOCdevkit/VOC2007/ImageSets/Segmentation/yilou/test.txt",'r').read().splitlines()

if not os.path.exists("E:/yx/dataset/oridatajpg/whole/yilou/"):
    os.makedirs("E:/yx/dataset/oridatajpg/whole/yilou/")

for image_id in image_ids:
    image_path = "E:/yx/dataset/oridatajpg/whole/yilou/"+image_id+".jpg"
    image = Image.open(image_path)
    image = unet.detect_image(image)
    image.save("F:/unet-keras-master/img/whole/whole/" + image_id + ".jpg")
    print(image_id," done!")