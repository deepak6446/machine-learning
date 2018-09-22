
# coding: utf-8

# In[1]:


import cv2
import keras
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
print('cv2: '+cv2.__version__+'\n')

try:
    model = keras.models.load_model('model.h5')
except:
    print('error inloading model')


# In[3]:


# comman functions required for cv2
class VideoStream:
    
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()
            
    def read(self):
                # Return the latest frame
        return self.frame
    
    def stop(self):
        self.stream.release()
        cv2.destroyAllWindows()
        self.stopped = True


# In[4]:


def reshape(image):
    try:
        print('---before image', image.shape)
        # plt.imshow(image[0])
        # plt.show()
        image = image.reshape(1, 28, 28, 1).astype('uint8')
        return image
    except Exception as e: 
        print('Exception in predict:'+ e)


# In[5]:


# Init video stream
vs = VideoStream(src=0).start()


# In[6]:


frame_count=0
while True:
    frame = vs.read()
    if cv2.waitKey(1) == 27: 
            break  # esc to quit
    dim = (28, 28)
    frame = cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
    frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
    cv2.imshow('Video', frame)
    frame_count+=1
    try:
        # Only run every 15 framaes
        if frame_count%15==0:
            # Save the image as the fist layer of inception is a DecodeJpeg
            cv2.imwrite("current_frame.jpg",frame)
            frame_count = 0
            #_ = system('clear') 
            print('--------frame before reshape', frame.shape)
            frame = reshape(frame)
            print('--------frame after reshape', frame.shape)
            # print(exit)
            predicted = model.predict(frame)
            predicted = np.argmax(predicted, axis=None, out=None)
            print('predicted digit:', predicted)        
    except Exception as e: 
        print('Exception in predict:'+ e)
        # print(exit)
        vs.stop()
vs.stop()

