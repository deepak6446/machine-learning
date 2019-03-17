
# coding: utf-8

# In[1]:


import os;  
from ipynb.fs.full.extractfeatures import extract_features;
from ipynb.fs.full.textPreprocessor import processText;
from ipynb.fs.full.testData import load_clean_descriptions_photo_features;
from ipynb.fs.full.testData import create_tokenizer_max_length, create_sequences;
from ipynb.fs.full.model import define_model;
from keras.callbacks import ModelCheckpoint


# In[2]:


# create image feature file and cleaned description of image file
if (os.path.exists("./features.pkl") ==  False):
    extract_features("../../flickr8k_dataset/Flickr8k_Dataset/Flicker8k_Dataset", "./features.pkl")
if (os.path.exists("./descriptions.txt") ==  False):
    vocabulary = processText("../../flickr8k_dataset/Flickr8k_text/Flickr8k.token.txt", "./descriptions.txt")


# In[3]:


# prepare train dataset with startseq desc endseq.
train_descriptions, train_features  = load_clean_descriptions_photo_features("./descriptions.txt", "/home/gis-local/Desktop/built.io/deepakWork/flickr8k_dataset/Flickr8k_text/Flickr_8k.trainImages.txt", "./features.pkl")
# prepare tokenizer and determine max length of desc
tokenizer, max_length, vocab_size = create_tokenizer_max_length(train_descriptions)


# In[ ]:


# prepare sequences
# X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)


# In[ ]:


# X1train.shape, X2train.shape, ytrain.shape, ytrain


# In[ ]:


# prepare test dataset with startseq desc endseq.
# test_descriptions, test_features  = load_clean_descriptions_photo_features("./descriptions.txt", "/home/gis-local/Desktop/built.io/deepakWork/flickr8k_dataset/Flickr8k_text/Flickr_8k.testImages.txt", "./features.pkl")
# prepare sequences
# X1test, X2test, ytest, vocab_size = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)


# In[ ]:


# define the model
model = define_model(vocab_size, max_length)
# define checkpoint callback
# filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# # fit model
# model.fit([X1train, X2train], ytrain, epochs=1, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


# In[ ]:


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]
            
# run in progressive loading if you don't have enough memory
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    print("runnning epochs:", i)
    # create the data generator
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    # fit for one epoch
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # save model
    model.save('model_' + str(i) + '.h5')

