import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np      

import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers  

from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

def eval_model(Y_train_pred, Y_train, Y_test_pred, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf_matrix = confusion_matrix(Y_train, Y_train_pred)
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    ax.set_title('Training Set Performance: %s' % (sum(Y_train_pred == Y_train)/len(Y_train)))
    ax = fig.add_subplot(1, 2, 2)   
    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    ax.set_title('Test Set Performance: %s' % (sum(Y_test_pred == Y_test)/len(Y_test)));    
    print(classification_report(Y_test, Y_test_pred))

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

mobile_base = keras.applications.MobileNetV3Small(input_shape=(image_size, image_size, 3),
                                                  include_top=False,
                                                  include_preprocessing=False)