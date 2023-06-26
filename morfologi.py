import cv2   
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt


#image = data.retina()
#image = data.astronaut()
image = imread(fname=r"C:\Users\ASUS\Documents\FILE SYAHDAN\semester 6\PCD\pertemuan 13\mirage.jpg")

print(image.shape)
plt.imshow(image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
# defining the range of masking 
blue1 = np.array([110, 50, 50]) 
blue2 = np.array([130, 255, 255]) 
      
# initializing the mask to be 
# convoluted over input image 
mask = cv2.inRange(hsv, blue1, blue2) 
  
# passing the bitwise_and over 
# each pixel convoluted 
res = cv2.bitwise_and(image, image, mask = mask) 
      
# defining the kernel i.e. Structuring element 
kernel = np.ones((5, 5), np.uint8) 
      
# defining the opening function  
# over the image and structuring element 
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(opening, cmap='gray')
ax[1].set_title('Citra Input 2')
# return video from the first webcam on your computer.   
screenRead = cv2.VideoCapture(0) 
  
# loop runs if capturing has been initialized. 
while(1): 
    # reads frames from a camera 
    _, image = screenRead.read() 
      
    # Converts to HSV color space, OCV reads colors as BGR  
    # frame is converted to hsv 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
    # defining the range of masking 
    blue1 = np.array([110, 50, 50]) 
    blue2 = np.array([130, 255, 255]) 
      
    # initializing the mask to be 
    # convoluted over input image 
    mask = cv2.inRange(hsv, blue1, blue2) 
  
    # passing the bitwise_and over 
    # each pixel convoluted 
    res = cv2.bitwise_and(image, image, mask = mask) 
      
    # defining the kernel i.e. Structuring element 
    kernel = np.ones((5, 5), np.uint8) 
      
    # defining the opening function  
    # over the image and structuring element 
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
     
    # The mask and opening operation 
    # is shown in the window  
    cv2.imshow('Mask', mask) 
    cv2.imshow('Opening', opening) 
      
    # Wait for 'a' key to stop the program  
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

# De-allocate any associated memory usage   
cv2.destroyAllWindows() 
  
# Close the window / Release webcam  
screenRead.release()

#Gradient
import cv2   
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt
#image = data.retina()
#image = data.astronaut()
image = imread(fname=r"C:\Users\ASUS\Documents\FILE SYAHDAN\semester 6\PCD\pertemuan 13\mirage.jpg")

print(image.shape)
plt.imshow(image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
# defining the range of masking 
blue1 = np.array([110, 50, 50]) 
blue2 = np.array([130, 255, 255]) 
      
# initializing the mask to be 
# convoluted over input image 
mask = cv2.inRange(hsv, blue1, blue2) 
  
# passing the bitwise_and over 
# each pixel convoluted 
res = cv2.bitwise_and(image, image, mask = mask) 
      
# defining the kernel i.e. Structuring element 
kernel = np.ones((5, 5), np.uint8) 
      
# defining the closing function  
# over the image and structuring element 
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(closing, cmap='gray')
ax[1].set_title('Citra Input 2')

# Python programe to illustrate 
# Gradient morphological operation 
# on input frames 
  
# organizing imports   
import cv2   
import numpy as np   
  
# return video from the first webcam on your computer.   
screenRead = cv2.VideoCapture(0) 
  
# loop runs if capturing has been initialized. 
while(1): 
    # reads frames from a camera 
    _, image = screenRead.read() 
      
    # Converts to HSV color space, OCV reads colors as BGR  
    # frame is converted to hsv 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
    # defining the range of masking 
    blue1 = np.array([110, 50, 50]) 
    blue2 = np.array([130, 255, 255]) 
      
    # initializing the mask to be 
    # convoluted over input image 
    mask = cv2.inRange(hsv, blue1, blue2) 
  
    # passing the bitwise_and over 
    # each pixel convoluted 
    res = cv2.bitwise_and(image, image, mask = mask) 
      
    # defining the kernel i.e. Structuring element 
    kernel = np.ones((5, 5), np.uint8) 
      
    # defining the gradient function  
    # over the image and structuring element 
    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) 
     
    # The mask and closing operation 
    # is shown in the window  
    cv2.imshow('Gradient', gradient) 
      
    # Wait for 'a' key to stop the program  
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

# De-allocate any associated memory usage   
cv2.destroyAllWindows() 
  
# Close the window / Release webcam  
screenRead.release() 
#Percobaan Dilasi dan erosi
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# Reading the input image 
img = cv2.imread(r"C:\Users\ASUS\Documents\FILE SYAHDAN\semester 6\PCD\pertemuan 13\mirage.jpg", 0) 
  
# Taking a matrix of size 5 as the kernel 
kernel = np.ones((5,5), np.uint8) 

img_erosion = cv2.erode(img, kernel, iterations=1) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 

fig, axes = plt.subplots(3, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(img, cmap = 'gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_erosion, cmap = 'gray')
ax[2].set_title("Citra Output Erosi")
ax[3].hist(img_erosion.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Output Erosi")

ax[4].imshow(img_dilation, cmap = 'gray')
ax[4].set_title("Citra Output Dilasi")
ax[5].hist(img_dilation.ravel(), bins = 256)
ax[5].set_title("Histogram Citra Output Erosi")

import cv2   
import numpy as np
from skimage import data
from skimage.io import imread

import matplotlib.pyplot as plt

#image = data.retina()
#image = data.astronaut()
image = imread(fname=r"C:\Users\ASUS\Documents\FILE SYAHDAN\semester 6\PCD\pertemuan 13\mirage.jpg")

print(image.shape)
plt.imshow(image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
# defining the range of masking 
blue1 = np.array([110, 50, 50]) 
blue2 = np.array([130, 255, 255]) 
      
# initializing the mask to be 
# convoluted over input image 
mask = cv2.inRange(hsv, blue1, blue2) 
  
# passing the bitwise_and over 
# each pixel convoluted 
res = cv2.bitwise_and(image, image, mask = mask) 
      
# defining the kernel i.e. Structuring element 
kernel = np.ones((5, 5), np.uint8) 
      
# defining the closing function  
# over the image and structuring element 
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 

fig, axes = plt.subplots(1, 2, figsize=(12, 12))
ax = axes.ravel()

ax[0].imshow(mask)
ax[0].set_title("Citra Input 1")

ax[1].imshow(closing, cmap='gray')
ax[1].set_title('Citra Input 2')

# Python programe to illustrate 
# Closing morphological operation 
# on an image 
  
# organizing imports   
import cv2   
import numpy as np   
  
# return video from the first webcam on your computer.   
screenRead = cv2.VideoCapture(0) 
  
# loop runs if capturing has been initialized. 
while(1): 
    # reads frames from a camera 
    _, image = screenRead.read() 
      
    # Converts to HSV color space, OCV reads colors as BGR  
    # frame is converted to hsv 
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
      
    # defining the range of masking 
    blue1 = np.array([110, 50, 50]) 
    blue2 = np.array([130, 255, 255]) 
      
    # initializing the mask to be 
    # convoluted over input image 
    mask = cv2.inRange(hsv, blue1, blue2) 
  
    # passing the bitwise_and over 
    # each pixel convoluted 
    res = cv2.bitwise_and(image, image, mask = mask) 
      
    # defining the kernel i.e. Structuring element 
    kernel = np.ones((5, 5), np.uint8) 
      
    # defining the closing function  
    # over the image and structuring element 
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
     
    # The mask and closing operation 
    # is shown in the window  
    cv2.imshow('Mask', mask) 
    cv2.imshow('Closing', closing) 
      
    # Wait for 'a' key to stop the program  
    if cv2.waitKey(1) & 0xFF == ord('a'): 
        break

# De-allocate any associated memory usage   
cv2.destroyAllWindows() 
  
# Close the window / Release webcam  
screenRead.release() 