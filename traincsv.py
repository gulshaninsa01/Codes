#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd


# In[28]:


data=pd.read_csv('C:/Users/Insafo/Desktop/train.csv')


# In[29]:


data.tail()


# In[30]:


#data['value']=data.label.map({0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9})


# In[138]:


data1=data.values
type(data1)
data.shape
tdata=data.values


# In[132]:


X=data1[:,1:] #features !!! Its can be more
Y=data1[:,:1]  #label !!! It always one particular for each value 
Y.shape


# In[33]:


import matplotlib.pyplot as plt #for image generate
i=41000 #It means single row represent a single digit as show in label
d=X[i]
d.shape=(28,28)
plt.imshow(d)#,cmap='gray')#plt.cm.binary) #cmap print the black and white img
plt.show()
print('\nValue recognise is: ',data.label[i])


# In[34]:


plt.imshow(X[1098].reshape(28,28),cmap='gray')
plt.show()
#X.shape


# In[133]:


from sklearn.model_selection import train_test_split #Train and test the model
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)#,random_state=0) #Its gives the train test value to variables


# In[134]:


from sklearn.neighbors import KNeighborsClassifier


# In[135]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[136]:


knn.fit(xtrain,ytrain) #train the model


# In[174]:


ypred=knn.predict(xtest)
ypred 


# In[182]:


ypred.shape


# In[185]:


i=2300
plt.imshow(X[i].reshape(28,28), cmap='gray')
plt.show()
knn.predict([X[i]])


# In[48]:


knn.score(xtest,ytest) #Accuracy 


# ## Test data set without label 

# In[150]:


data2=pd.read_csv('C:/Users/Insafo/Desktop/test.csv')


# In[153]:


data2.head()
#type(data2)
#only predict the value and check the label by show a imgage


# In[110]:


type(tdata)


# In[91]:


#X1=datan[:,:] #xtest


# In[191]:


X1=tdata[:,1:] #features !!! Its can be more xtrain
Y1=tdata[:,:1]  #label !!! It always one particular for each value  ytrain
xts=data2.values #xtest
type(xts)


# In[192]:


from sklearn.model_selection import train_test_split #Train and test the model
xtrain,xtest,ytrain,ytest=train_test_split(X1,Y1,test_size=0.2)#,random_state=0) #Its gives the train test value to variables


# In[193]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)#neighbors show be even othewise groups makes equal


# In[194]:


knn.fit(X1,Y1)


# In[203]:


j=10
plt.imshow(xts[j].reshape(28,28))
plt.show()
knn.predict([xts[j]])


# In[204]:


knn.score(xtest,ytest) #Accuracy  


# In[ ]:




