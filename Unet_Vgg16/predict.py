#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from unet import Unet
from PIL import Image

unet = Unet()

while True:
    img = input('E:/yx/dataset/oridatajpg/whole/val/000002.JPG')
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        r_image = unet.detect_image(image)
        r_image.show()



