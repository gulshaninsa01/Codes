# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:28:27 2018

@author: Anshu Pandey
"""

import pandas
import numpy
import matplotlib.pyplot as plt

#load the data
path=r"D:\AI\IITD\semeion.data.txt"
data=pandas.read_csv(path,sep=' ',header=None)
data=data.iloc[0:1594,0:266]#dropping the last
x=numpy.array(data.iloc[0:1594,0:256])
y=numpy.array(data.iloc[0:1594,256:267])
plt.imshow(x[1098].reshape(16,16),cmap='gray')
plt.show()
y[1098]
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(x,y,test_size=0.2)
from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=(120,60),learning_rate_init=0.01,tol=1e-6,verbose=True)
#traint the algorithm
model.fit(xtr,ytr)
plt.imshow(xts[110].reshape(16,16),cmap='gray')
plt.show()
model.predict(xts[110].reshape(1,256))
#overall accuracy
model.score(xts,yts)
import cv2
img=cv2.imread(r"D:\AI\IITD\eight.png",0)
img.shape
model.predict(img.reshape(1,256))











