from nets.unet import mobilenet_unet
model = mobilenet_unet(2,4800,3200)
model.summary()