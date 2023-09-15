import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from matplotlib import cm
import cv2
import math


def loss_and_accuracy(history):
  
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs , loss , label = 'training_loss')
  plt.plot(epochs , val_loss , label = 'validation_loss')
  plt.title('Loss Curve')
  plt.xlabel('epochs')
  plt.legend()
  
  plt.figure()

  plt.plot(epochs , accuracy , label = 'training_accuracy')
  plt.plot(epochs , val_accuracy , label = 'validation_accuracy')
  plt.title('Accuracy Curve')
  plt.xlabel('epochs')
  plt.legend()
    
    
def load_image(filename , img_shape=224):

  img = tf.io.read_file(filename)

  img = tf.image.decode_image(img , channels = 3)

  img = tf.image.resize(img , size = [img_shape , img_shape])

  img = img/255.

  return img


def make_predictions(model , filename , classnames):

  image = load_image(filename)

  pred = model.predict(tf.expand_dims(image , axis=0))

  pred_class = classnames[tf.argmax(tf.round(pred)[0])]

  plt.imshow(image)
  plt.title(f'Predicted class: {pred_class}')

  plt.axis(False)

def normalizeImages(img1, img2):
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2RGB)

    img1 = Image.fromarray(img1.astype('uint8'), 'RGB')
    img2 = Image.fromarray(img2.astype('uint8'), 'RGB')
    # first case: images already have the same size
    if(img1.size == img2.size):
        return img1, img2

    # second case: an image can be contained in another
    img1_is_container = img1.size[0] > img2.size[0] and img1.size[1] > img2.size[1]
    img2_is_container = img2.size[0] > img1.size[0] and img2.size[1] > img1.size[1]
    if(img1_is_container):
        return dragImg(img1, img2)
    elif(img2_is_container):
        img2, img1 = dragImg(img2, img1)
        return img1, img2

    # third case: one dimension has the same value, but not the other
    if(img1.size[0] == img2.size[0] and img1.size[1] > img2.size[1]):
        # same length, img2 less height
        new_diff = img1.size[1] - img2.size[1]
        diff = round(new_diff/2)
        diff2 = diff if new_diff % 2 == 0 else diff-1
        if diff2 < 0 : diff2 = 0
        return img1, addTop(addBottom(img2, diff), diff2)
    if(img1.size[0] == img2.size[0] and img1.size[1] < img2.size[1]):
        # same length, img1 less height
        new_diff = img2.size[1] - img1.size[1]
        diff = round(new_diff/2)
        diff2 = diff if new_diff % 2 == 0 else diff-1
        if diff2 < 0 : diff2 = 0
        return addTop(addBottom(img1, diff), diff2), img2
    if(img1.size[0] > img2.size[0] and img1.size[1] == img2.size[1]):
        # same height, img2 less length
        new_diff = img1.size[0] - img2.size[0]
        diff = round(new_diff/2)
        diff2 = diff if new_diff % 2 == 0 else diff-1
        if diff2 < 0 : diff2 = 0
        return img1, addStart(addEnd(img2, diff), diff2)
    if(img1.size[0] < img2.size[0] and img1.size[1] == img2.size[1]):
        # same height, img1 less length
        new_diff = img2.size[0] - img1.size[0]
        diff = round(new_diff/2)
        diff2 = diff if new_diff % 2 == 0 else diff-1
        if diff2 < 0 : diff2 = 0
        return addTop(addBottom(img1, diff), diff2), img2

    # fourth case: lenght1 is bigger but height1 is smaller or viceversa
    if(img1.size[0] > img2.size[0] and img1.size[1] < img2.size[1]):
        diff_len = round((img1.size[0] - img2.size[0])/2)
        diff_high = round((img2.size[1] - img1.size[1])/2)
        diff_len2 = diff_len if diff_len%2==0 else diff_len-1
        diff_high2 = diff_high if diff_high%2==0 else diff_high-1
        return addTop(addBottom(img1, diff_high), diff_high2), addStart(addEnd(img2, diff_len), diff_len2)
    
    if(img1.size[0] < img2.size[0] and img1.size[1] > img2.size[1]):
        diff_len = round((img2.size[0] - img1.size[0])/2)
        diff_high = round((img1.size[1] - img2.size[1])/2)
        diff_len2 = diff_len if diff_len%2==0 else diff_len-1
        if(diff_len2 == -1):
            diff_len2 = 1
        diff_high2 = diff_high if diff_high%2==0 else diff_high-1
        if(diff_high2 == -1):
            diff_high2 = 1
        return addStart(addEnd(img1, diff_len), diff_len2), addTop(addBottom(img2, diff_high), diff_high2)
