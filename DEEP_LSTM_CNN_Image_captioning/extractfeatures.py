
# coding: utf-8

# In[6]:


from keras.models import load_model;
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from os import listdir
from pickle import dump


# In[7]:


# VGG16 is a pretrained model on imagenet dataset for image classification 
# it is pretained model so we don't need to train it again.
# model can be downloaded from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
# and places under ~/.keras/models/

model = VGG16();
model.save('../../vgg16.h5');
vggModel = load_model('../../vgg16.h5');


# In[8]:


# will load each photo, prepare it for VGG, and collect the predicted features from the VGG model.
# The image features are a 1-dimensional 4,096 element vector.
vggModel.layers.pop()
vggModel = Model(inputs=vggModel.inputs, outputs=vggModel.layers[-1].output)

# summarize
print(vggModel.summary())


# In[9]:


# we will use VGG model to extract features
# extract features from each photo and save in file 
# directory is path of dataset
# featurePath is where you want to store fetaures.

def extract_features(directory, featurePath):
    features = dict()

    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        # convert image into 224*224*3
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model  (convert into 1*224, 224)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # get features
        feature = vggModel.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open(featurePath, 'wb'))
    return

