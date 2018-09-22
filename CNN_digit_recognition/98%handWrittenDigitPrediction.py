
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
print('matplotlib', matplotlib.__version__+'\n','keras:', keras.__version__+'\n','sklearn:', sklearn.__version__+'\n', 'pandas:' + pandas.__version__+'\n','numpy:'+ numpy.__version__+'\n')


# In[2]:


# dataset https://www.kaggle.com/oddrationale/mnist-in-csv
# preprocessing data
trainData = pd.read_csv('/home/deepak/Desktop/deepWork/machineLearning/dataset/mnistData/mnist_train.csv')
testData = pd.read_csv('/home/deepak/Desktop/deepWork/machineLearning/dataset/mnistData/mnist_test.csv')


# In[3]:


print(trainData.shape, testData.shape)   


# In[9]:


data = trainData.append(testData)
data.shape


# In[10]:


# Reshapping the data because data is in rows and we need matrix for computation
# Convert into 28*28*1 using reshape fun (1 because it contains only blackandWhite)
# Unsigned integer (0 to 255)
data.iloc[1, 1:].values.reshape(28, 28)


# In[11]:


#Storing Pixel array in form length width and channel in df_x
df_x = data.iloc[:,1:].values.reshape(len(data), 28, 28, 1)
# storing labels in y
y = data.iloc[:, 0].values


# In[12]:


# now y conatins 0...9 which may have relationship among them
# like our model may refer 2 = 2*1
# so we will convert it into categorical vectors
# like 0 will be [1 0 0 ...0], 1 = [0 1 0 .... 0]
#Converting labels to categorical features

df_y = keras.utils.to_categorical(y,num_classes=10)


# In[13]:


df_y


# In[14]:


df_x  =  np.array(df_x)
df_y  =  np.array(df_y)


# In[15]:


df_x.shape


# In[16]:



# test train split# test t 
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)
# done with preprocessing


# In[17]:



#CNN model#CNN mod 
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


# In[18]:


model.summary()


# In[19]:


model.fit(x_train, y_train, epochs=30, validation_data = (x_test, y_test) )


# In[20]:


# evaluate model on test and train data
model.evaluate(x_test,y_test)


# In[21]:


# save model so that we can use it later.
model.save('model.h5')


# In[28]:


index = 8
testImg = trainData.iloc[index:(index+1), 1:]
img = testImg.values.reshape(28, 28)
print('----before reshape', img.shape, img)
plt.imshow(img)
plt.show()
img = img.reshape(1, 28, 28, 1).astype('uint8')


# In[27]:


predicted = model.predict(img)
predicted = np.argmax(predicted, axis=None, out=None)
defined = trainData.iloc[index:(index+1), 0:1].values
defined = np.squeeze(defined)
print('predicted digit:', predicted,'digit in csv:', defined)

