import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np   
import xlrd
from matplotlib.pyplot import MultipleLocator
import os
import seaborn as sns
# 保证图片在浏览器内正常显示
path = r'E:/yx/dataset/oridatajpg/whole/tongji/mobilenet/不同生育期/untfullyresult.xlsx'
rbook = xlrd.open_workbook(path)
rbook.sheets()
#rsheet0 = rbook.sheet_by_index(0)
rsheet1 = rbook.sheet_by_index(1)
rsheet2 = rbook.sheet_by_index(2)
#rsheet3 = rbook.sheet_by_index(18)
#rsheet4 = rbook.sheet_by_index(19)
#x0 = []
#y0 = []
fig,ax = plt.subplots(figsize=(10,12.5))

x1 =[]
y1 =[]

x2 =[]
y2 =[]

#x3 = []
#y3 = []

#x4 = []
#y4 = []

#for row in rsheet0.get_rows():
   # product_column1 = row[1]  # 品名所在的列
   # product_value1 = product_column1.value  # 项目名
   # product_column2 = row[2]  # 品名所在的列
    #product_value2 = product_column2.value  # 项目名

    #x0.append(float(product_value1))
    #y0.append(float(product_value2))
    
for row in rsheet1.get_rows():
    product_column1 = row[0]  # 品名所在的列
    product_value1 = product_column1.value  # 项目名
    product_column2 = row[1]  # 品名所在的列
    product_value2 = product_column2.value  # 项目名

    x1.append(product_value1)
    y1.append(product_value2)

for row in rsheet2.get_rows():
    product_column1 = row[0]  # 品名所在的列
    product_value1 = product_column1.value  # 项目名
    product_column2 = row[1]  # 品名所在的列
    product_value2 = product_column2.value  # 项目名

    x2.append(product_value1)
    y2.append(product_value2)
'''
#for row in rsheet3.get_rows():
 #   product_column1 = row[1]  # 品名所在的列
  #  product_value1 = product_column1.value  # 项目名
   # product_column2 = row[2]  # 品名所在的列
    #product_value2 = product_column2.value  # 项目名

   # x3.append(product_value1)
    #y3.append(product_value2)

#for row in rsheet4.get_rows():
 #   product_column1 = row[1]  # 品名所在的列
  #  product_value1 = product_column1.value  # 项目名
   # product_column2 = row[2]  # 品名所在的列
    #product_value2 = product_column2.value  # 项目名

   # x4.append(product_value1)
    #y4.append(product_value2)
'''
sns.regplot(np.array(x1), np.array(y1), fit_reg=True,scatter_kws= { 'color': 'white'},line_kws= { 'color': 'red'})
sns.regplot(np.array(x2), np.array(y2), fit_reg=True,scatter_kws= { 'color': 'white'},line_kws= { 'color': 'black'})
plt.scatter(x1, y1,  marker='o',label = 'Train',s = 25.0, alpha = 1)
plt.scatter(x2, y2,  marker='o', label = 'Val',s = 25.0, alpha = 1)
#plt.scatter(x3, y3, c='black', marker='^', label = 'Autumn',s = 25.0, alpha = 0.5 )
#plt.scatter(x4, y4, c='blue', marker='+', label = 'Winter',s = 25.0, alpha = 0.5 )

x5 = np.arange(0, 1300000, 1)
y5 = x5
plt.plot(x5, y5, c='blue', linewidth=2.0 )
font1 = {'family':'Times New Roman','size':20}
legend=plt.legend(prop=font1,loc='best')
font2 = {'family':'Times New Roman','size':20}
plt.xlabel('Tassel per image interactively counted',font2)
plt.ylabel('Tassel per image estimated from Faster R-CNN',font2)
#font3 = {'family':'Times New Roman','size':20}
#plt.title('H',font3)
x_major_locator=MultipleLocator(100000)
y_major_locator=MultipleLocator(100000)
plt.axis('scaled')
plt.tick_params(axis='both',which='major',labelsize=14)
ax=plt.gca()
plt.xticks(rotation = 90)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator) 
plt.xlim(0,1300000)#x和y坐标轴的范围
plt.ylim(0,1300000) #x和y坐标轴的范围
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#plt.tick_params(axis='both',which='major',labelsize=15)
#设置刻度的字号
#plt.text(50,0.5,fontsize=15)
plt.savefig("E:/yx/dataset/oridatajpg/whole/picture/untfully.png",dpi = 1000)
plt.show()