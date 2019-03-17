
# coding: utf-8

# In[1]:


from pickle import load;
from keras.preprocessing.sequence import pad_sequences;
from keras.utils import to_categorical;
from keras.preprocessing.text import Tokenizer;
from numpy import array;


# In[2]:


# calculate the length of the description with the most words

def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer_max_length(descriptions):
    
    words = to_lines(descriptions)    # get all words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    
    max_len = max_length(descriptions)
    print('Description Length: %d' % max_len)
    
    return tokenizer, max_len, vocab_size


# In[3]:


# load photo features
def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    from collections import defaultdict
    features = defaultdict(list)  
    features = {k: all_features[k] for k in dataset}
    return features
#     return {}


# In[4]:


# filename contains clean description 
# returns dictionary descriptions which has key:image_id and value as [startseq cleaned_desc1 endseq, startseq cleaned_desc2  endseq] 
def load_clean_descriptions_photo_features(filename, testDataFilePath, featureFile):
    dataset = load_set(testDataFilePath)
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    features = load_photo_features(featureFile, dataset)
    print('Descriptions: train=%d' % len(descriptions))
    print('Photos: train=%d' % len(features))
    return descriptions, features


# In[5]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# In[6]:


# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# In[7]:


# create sequences of images, input sequences and output words for an image
"""
photo   startseq,                                   little
photo   startseq, little,                           girl
photo   startseq, little, girl,                     running
photo   startseq, little, girl, running,            in
photo   startseq, little, girl, running, in,        field
photo   startseq, little, girl, running, in, field, endseq

"""

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

 

