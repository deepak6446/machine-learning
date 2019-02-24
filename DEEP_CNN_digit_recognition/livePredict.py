
# coding: utf-8

# In[1]:


import cv2
import keras
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

print("cv2: " + cv2.__version__ + "\n" + "matplotlib: " + matplotlib.__version__)


# In[2]:


try:
    model = keras.models.load_model("model.h5")
except:
    print("error inloading model")


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
        image = image.reshape(1, 28, 28)
        print("---before image", image.shape)
        # plt.imshow(image[0])
        # plt.show()
        image = image.reshape(1, 28, 28, 1).astype("uint8")
        return image
    except Exception as e:
        print("Exception in predict:" + e)


# In[5]:


def changeImage(path):
    im = cv2.imread(path)
    morph = im.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # split the gradient image into channels
    image_channels = np.split(np.asarray(morph), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(
            image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY
        )
        image_channels[i] = np.reshape(
            image_channels[i], newshape=(channel_height, channel_width, 1)
        )

    # merge the channels
    image_channels = np.concatenate(
        (image_channels[0], image_channels[1], image_channels[2]), axis=2
    )
    image_channels = cv2.cvtColor(image_channels, cv2.COLOR_BGR2GRAY)
    image_channels[image_channels == 255] = 200
    image_channels[image_channels == 0] = 100
    image_channels[image_channels == 100] = 255
    image_channels[image_channels == 200] = 0
    # save the denoised image
    cv2.imwrite(path, image_channels)
    return image_channels


# In[6]:


def predict(path):
    img = changeImage(path)
    # img = cv2.imread(path,0)
    # print('----before reshape', img.shape)
    cv2.imshow("noise cancellation", img)
    img = img.reshape(1, 28, 28, 1).astype("uint8")
    predicted = model.predict(img)
    predicted = np.argmax(predicted, axis=None, out=None)
    print("predicted digit:", predicted)


# In[13]:


def chnagePixelAndPredict(frame):
    frame[frame < 100] = 0
    frame[frame > 99] = 255
    img = frame
    cv2.imshow("noise cancellation", img)
    img = img.reshape(1, 28, 28, 1).astype("uint8")
    predicted = model.predict(img)
    predicted = np.argmax(predicted, axis=None, out=None)
    print("predicted digit:", predicted)
    return frame


# In[16]:


# Init video stream
vs = VideoStream(src=0).start()


# In[17]:


"""
save image frame in black and white, then read its pixel value and predict the result.
"""
frame_count = 0
while True:
    frame = vs.read()
    if cv2.waitKey(1) == 27:
        break  # esc to quit
    dim = (28, 28)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video", frame)
    frame_count += 1
    try:
        # Only run every 15 framaes
        if frame_count % 15 == 0:
            # Save the image as the fist layer of inception is a DecodeJpeg
            frame = chnagePixelAndPredict(frame)
            cv2.imwrite("current_frame.jpg", frame)
            # predict("current_frame.jpg")
            frame_count = 0
            # _ = system('clear')
            # print('--------frame before reshape', frame.shape)
            # frame = reshape(frame)
            # print('--------frame before reshape', frame.shape)
            # print(exit)
            """predicted = model.predict(frame)
            predicted = np.argmax(predicted, axis=None, out=None)
            print('predicted digit:', predicted)         """
    except Exception as e:
        print("Exception in predict:", e)
        print(exit)
        vs.stop()
vs.stop()


# In[10]:


trainData = pd.read_csv(
    "/home/deepak/Desktop/deepWork/machineLearning/dataset/mnistData/mnist_train.csv"
)
index = 4
testImg = trainData.iloc[index : (index + 1), 1:]
cv2.imwrite("letter9.jpg", testImg.values.reshape(28, 28))


# In[11]:


img = testImg.values.reshape(28, 28)
# print('testset', img)
path = "letterOne.jpg"
predict(path)


# In[12]:


from PIL import Image

image_file = Image.open("one.jpg")  # open colour image
image_file = image_file.convert("1")  # convert image to black and white
image_file.save("result.png")

