import time
import keras
import numpy as np
from nets.unet import Unet
from nets.unet_training import Generator, dice_loss_with_CE, CE
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.metrics import categorical_accuracy
from keras import backend as K
from PIL import Image
import os
#os.chdir("H:/fenge/unet-keras-master")
#from utils.metrics import Iou_score, f_score

from keras import backend
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))

def Iou_score(smooth = 1e-5, threhold = 0.5):
    def _Iou_score(y_true, y_pred):
        # score calculation
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())
        intersection = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        union = backend.sum(y_true[...,:-1] + y_pred, axis=[0,1,2]) - intersection

        score = (intersection + smooth) / (union + smooth)
        return score
    return _Iou_score

def f_score(beta=1, smooth = 1e-5, threhold = 0.5):
    def _f_score(y_true, y_pred):
        y_pred = backend.greater(y_pred, threhold)
        y_pred = backend.cast(y_pred, backend.floatx())

        tp = backend.sum(y_true[...,:-1] * y_pred, axis=[0,1,2])
        fp = backend.sum(y_pred         , axis=[0,1,2]) - tp
        fn = backend.sum(y_true[...,:-1], axis=[0,1,2]) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
        return score
    return _f_score

if __name__ == "__main__":    
    inputs_size = [512,512,3]
    log_dir = "H:/unet-keras-master/logs/UAVlogs/"
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    num_classes = 2
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = False

    # 获取model
    model = Unet(inputs_size,num_classes)
    # model.summary()

    model_path = "H:/unet-keras-master/model_data/unet_voc.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(r"H:/unet-keras-master/VOCdevkit/VOC2007/ImageSets/Segmentation/UAV/train.txt","r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"H:/unet-keras-master/VOCdevkit/VOC2007/ImageSets/Segmentation/UAV/val.txt","r") as f:
        val_lines = f.readlines()
        
    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(log_dir + '20210118_ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1)
    # tensorboard
    tensorboard = TensorBoard(log_dir=log_dir)

    freeze_layers = 17

    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    if True:
        lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        Batch_size = 2
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes).generate(False)
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//Batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
    
    
    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        Batch_size = 2
        # 交叉熵
        model.compile(loss = dice_loss_with_CE() if dice_loss else CE(),
                optimizer = Adam(lr=lr),
                metrics = [f_score()])
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(train_lines), len(val_lines), Batch_size))

        gen = Generator(Batch_size, train_lines, inputs_size, num_classes).generate()
        gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes).generate(False)
        # 开始训练
        model.fit_generator(gen,
                steps_per_epoch=max(1, len(train_lines)//Batch_size),
                validation_data=gen_val,
                validation_steps=max(1, len(val_lines)//Batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[checkpoint_period, reduce_lr,tensorboard])
