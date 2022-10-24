'''
#mask批量转化 三个通道改为255
'''
import os
import numpy as np
from PIL import Image

'''
def imgto_255(img_path):
	# 图片所在目录
	# img_path = r"H:\A剪雄整理数据1\RGB_label\zong\jpg/"
	img_list = os.listdir(img_path)
	for image in img_list:
		img_name = str(image)
		img = Image.open(os.path.join(img_path,image))
		img = np.array(img)
		img[img > 0] = 255	
		# rows, cols, dims = img.shape
		# print(img.shape)
		# for dim in range(dims):
		# 	for i in range(rows):
		# 		for j in range(cols):
		# 			if (img[i, j ,dim ] >= 1):
		# 				img[i, j, dim] = 1  #255
		# 			else:
		# 				img[i, j, dim] = 0

		img_new = os.path.join(img_path, img_name)
		new_img = Image.fromarray(np.uint8(img))
		print(np.max(new_img), np.min(new_img))
		new_img.save(os.path.join(img_path,image))
		print (image + " pixel values changed and saved to " + img_name)
'''

'''
计算miou

参考网址
https://blog.csdn.net/xijuezhu8128/article/details/104999872
https://blog.csdn.net/sinat_29047129/article/details/103642140
多分类 参考https://blog.csdn.net/jiongnima/article/details/84750819?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-4.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-4.nonecase


"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""


__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
	def __init__(self, numClass):
		self.numClass = numClass
		self.confusionMatrix = np.zeros((self.numClass,) * 2)

	def pixelAccuracy(self):
		# return all class overall pixel accuracy
		#  PA = acc = (TP + TN) / (TP + TN + FP + TN)
		acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
		return acc

	def classPixelAccuracy(self):
		# return each category pixel accuracy(A more accurate way to call it precision)
		# acc = (TP) / TP + FP
		classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
		return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

	def meanPixelAccuracy(self):
		classAcc = self.classPixelAccuracy()
		meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
		return meanAcc                  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

	def meanIntersectionOverUnion(self):
		# Intersection = TP Union = TP + FP + FN
		# IoU = TP / (TP + FP + FN)
		intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
		union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
			self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
		IoU = intersection / union  # 返回列表，其值为各个类别的IoU
		mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
		return mIoU , IoU

	def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
		# remove classes from unlabeled pixels in gt image and predict
		mask = (imgLabel >= 0) & (imgLabel < self.numClass)
		label = self.numClass * imgLabel[mask] + imgPredict[mask]
		count = np.bincount(label, minlength=self.numClass ** 2)
		confusionMatrix = count.reshape(self.numClass, self.numClass)
		return confusionMatrix

	def Frequency_Weighted_Intersection_over_Union(self):
		# FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
		freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
		iu = np.diag(self.confusion_matrix) / (
				np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
				np.diag(self.confusion_matrix))
		FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
		return FWIoU

	def addBatch(self, imgPredict, imgLabel):
		assert imgPredict.shape == imgLabel.shape
		self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

	def reset(self):
		self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':

	val_path = r"H:\mydatatemp\tempiou\testing-unet1\val/"
	val_list = os.listdir(val_path)
	with open("H:\mydatatemp/tempiou/testing-unet1/jingdumeters.txt", "w") as f:
		f.write("imgname " + " ; " +  " pa"+ " ; " +  " cpa"+" ; "+  "mpa" +" ; "+ " mIoU  "+" ; "+" IoU  "+ "\n")
		for val_imgname  in val_list:
			valid_imgpath = str(val_path + str(val_imgname))
			val_img = Image.open(valid_imgpath)
			imgLabel = np.array(val_img)[1]  # 可直接换成标注图片
			print("imgLabelmax", np.max(imgLabel))

			test_path =r"H:\mydatatemp\tempiou\testing-unet1\predict/"
			test_imgpath = test_path + str(val_imgname)[: -4] + "_predict.jpg"
			test_img = Image.open(test_imgpath)
			imgPredict = np.array(test_img)[1] # 可直接换成预测图片
			print("imgPredictmax",np.max(imgPredict))

			metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
			metric.addBatch(imgLabel,imgPredict )
			pa = metric.pixelAccuracy()
			cpa = metric.classPixelAccuracy()
			mpa = metric.meanPixelAccuracy()
			mIoU ,IoU = metric.meanIntersectionOverUnion()

			f.write(val_imgname + " ; " +str(pa)+" ; "+ "cpa" +" ; "  +  str(mpa)  +" ; " +  str(mIoU) + " ; "+str(IoU)+ "\n")
'''


import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import os


#p是精确度，r是召回率  F是F1系数也就是dice s是特异性

sumACC=0.0
sumP=0.0
sumR=0.0
sumF=0.0
sumS=0.0
sumI=0.0
img_GT_path =r'E:/yx/dataset/oridatapng/ZY8911/ZY8911val/'
img_R_path = r'E:/yx/dataset/oridatajpg/ZY8911/val/'

with open("E:/yx/dataset/oridatajpg/whole/miou/mobilenet/不同品种/UnetmobileZY8911valnew.txt", "w") as  f:
	f.write("imgname " + " ; " + " pa" + " ; " + "R召回率 " + " ; " + "F1系数DICE" + " ; " + " S特异性 " + " ; " + " IoU  " + "mIOU" + "\n")
	for i in os.listdir(img_GT_path):
		print(i[0:-4])
		img_GT = cv2.imread(img_GT_path+str(i),0)
		print(img_GT.shape)
		# b=img_R_path + str(i[0:-4]) + '_predict.jpg'
		img_R  = cv2.imread(img_R_path+str(i[0:-4])+'.jpg',0)   #'_predict_liantong.jpg'
		print(img_R.shape)

		ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		ret_R, binary_R   = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		img_GT = np.array(binary_GT)
		img_R = np.array(binary_R)
		img_GT=img_GT.reshape(-1)
		img_R=img_R.reshape(-1)
		print(img_GT.shape)
		print(img_R.shape)
		print(i)

		sumACC=sumACC+accuracy_score(img_GT, img_R)
		sumP=sumP+precision_score(img_GT, img_R,pos_label=255)
		sumR=sumR+recall_score(img_GT, img_R,pos_label=255)
		sumF=sumF+f1_score(img_GT, img_R,pos_label=255)

		acc = accuracy_score(img_GT, img_R)

		P = precision_score(img_GT, img_R,pos_label=255)

		R = recall_score(img_GT, img_R,pos_label=255)

		F = f1_score(img_GT, img_R,pos_label=255)

		a=confusion_matrix(img_GT, img_R)
		a=np.array(a,dtype='float64')

		sumS+=((a[0][0])/(a[0][0]+a[0][1]))
		S = (a[0][0])/(a[0][0]+a[0][1])

		sumI+=(a[1][1])/(a[0][1]+a[1][0]+a[1][1])
		IOU = (a[1][1])/(a[0][1]+a[1][0]+a[1][1])
		IOUF = (a[0][0]) / (a[0][1] + a[1][0] + a[0][0])
		MIOU = (IOU + IOUF) /2
		print('----------------------------------------')
		f.write(i + " ; " + str(P) + " ; " + str(R) + " ; " + str(F) + " ; " + str(S) + " ; " + str(IOU) + ";" + str(MIOU) +"\n")
	f.write("sum/219" + " ; " + str((sumP/219)) + " ; " + str(sumR/219) + " ; " + str(sumF/219) + " ; " + str(sumS/219) + " ; " + str(sumI/219) + "\n")
print("Acc:",)
print(sumACC/219)
print("Precision:",)
print(sumP/219)
print("Recall:",)
print(sumR/219)
print("F1:",)
print(sumF/219)
print("Specificity:",)
print(sumS/219)
print("mIoU:")
print(sumI/219)




