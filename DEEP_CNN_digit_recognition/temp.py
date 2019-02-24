
# coding: utf-8

# In[1]:


#importing Keras, Library for deep learning 
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# preprocessing data
trainData = pd.read_csv('/home/deepak/Desktop/deepWork/machineLearning/dataset/mnistData/mnist_train.csv')
testData = pd.read_csv('/home/deepak/Desktop/deepWork/machineLearning/dataset/mnistData/mnist_test.csv')


# In[3]:


print(trainData.shape, testData.shape)


# In[4]:



data = trainData.append(testData)
data.shape


# In[5]:


# Reshapping the data because data is in rows and we need matrix for computation
# Convert into 28*28*1 using reshape fun (1 because it contains only blackandWhite)
data.iloc[1, 1:].values.reshape(28, 28).astype('uint8')


# In[6]:


#Storing Pixel array in form length width and channel in df_x
df_x = data.iloc[:,1:].values.reshape(len(data), 28, 28, 1)
# storing labels in y
y = data.iloc[:, 0].values


# In[7]:


# now y conatins 0...9 which may have relationship among them
# like our model may refer 2 = 2*1
# so we will convert it into categorical vectors
# like 0 will be [1 0 0 ...0], 1 = [0 1 0 .... 0]
#Converting labels to categorical features

df_y = keras.utils.to_categorical(y,num_classes=10)


# In[8]:


df_y


# In[9]:


df_x  =  np.array(df_x)
df_y  =  np.array(df_y)


# In[10]:


df_x.shape


# In[11]:


# test train split
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
# done with preprocessing


# In[12]:


#CNN model
model = Sequential()
# 32 filter 3*3 size
model.add(Convolution2D(32,3,data_format='channels_last',activation='relu',input_shape=(28,28,1)))
# reduce number of parameters by getting imporatant params
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten()) # converts all matrix to single vector
model.add(Dense(100))    #100 NN nodes 
model.add(Dropout(0.5))
model.add(Dense(10))     #output on NN will have 10 node as our output will be categorical nodes
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy']) # chose loss fun


# In[ ]:


model.summary()


# In[ ]:


#fitting it with just 100 images for testing 
model.fit(x_train, y_train, epochs=30, validation_data = (x_test, y_test) )


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


img = trainData.iloc[:, 1:]
img.shape


# In[ ]:


index = 8
testImg = trainData.iloc[index:(index+1), 1:]
img = testImg.values.reshape(1, 28, 28)
plt.imshow(img[0])
plt.show()
img = img.reshape(1, 28, 28, 1)


# In[ ]:


predicted = model.predict(img)
predicted = np.argmax(predicted, axis=None, out=None)
defined = trainData.iloc[index:(index+1), 0:1].values
np.squeeze(
print('-----predicted digit:', predicted,'digit in csv:', defined)

