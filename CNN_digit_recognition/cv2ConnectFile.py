
# coding: utf-8

# In[1]:


"""
display the contents of the webcam using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""
import cv2


# In[10]:


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


# In[9]:


show_webcam()

