#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 14:47:57 2023

@author: Dario, inspired by Ewald

Dataset by Gonzalo Recio taken from kaggle:  https://www.kaggle.com/datasets/gonzalorecioc/futurama-frames-with-characteronscreen-data
"""

# imports
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import confusion_matrix
import seaborn as sns
import helper_functions

#%%
# flags
save_plot=True
save_model=True
load_model=not save_model

# Load the CSV files containing image file names and labels corresponding to each image
FILE_SHAPE = (135, 180, 3)
COLOR_MODE = 'rgb'
DATA_DIRECTORY = "../data/"
IMAGE_DIRECTORY = "../data/img/"
OUTPUT_DIR = "../output/" #Directory to store plots and csvs created
FAIL_DIR = OUTPUT_DIR+"erroneous_predictions" # directory to store erroneously predicted images


#Read dataframe
df = pd.read_csv(DATA_DIRECTORY+'/data.csv')
height, width = FILE_SHAPE[0], FILE_SHAPE[1]

# Bring images and labels together
image_paths = [os.path.join(IMAGE_DIRECTORY, filename) for filename in df['file'].values]

#Get the characteristics of the data
helper_functions.explore_data(df, output_dir=OUTPUT_DIR)

#Load the dataset with labels and create train, test, and validation datasets
train_dataset, test_dataset, val_dataset = helper_functions.load_dataset(image_paths, df)


#%%  Create the neural network
if load_model:
    print("Loading model")
    model= tf.keras.saving.load_model(OUTPUT_DIR+'futurama_model_dario.tf')
    #model= tf.keras.saving.load_model(OUTPUT_DIR+'futurama_model.tf')
else:
    print("Creating model")
    # Note that the layers are hardcoded in the function.
    model = helper_functions.create_nn()

    # compile the model
    LR=1e-3
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer,
                loss={'isLeela': 'binary_crossentropy',
                        'isFry': 'binary_crossentropy',
                        'isBender': 'binary_crossentropy'},
                metrics={'isLeela': 'accuracy',
                        'isFry': 'accuracy',
                        'isBender': 'accuracy'})


    #Fit the neural network
    print(2*'\n',10*'*', 'Fit Neural Network',10*'*')
    history, early_stopping_cb = helper_functions.fit(model, train_dataset, val_dataset, epochs = 250)
    helper_functions.plot_learning_curves(history, output_dir = OUTPUT_DIR, save_plot=save_plot)
    print('Beste Epoche: ', early_stopping_cb.best_epoch+1)

#Save the model
if save_model: model.save(OUTPUT_DIR+"futurama_model_dario.tf")

#Evaluate the model
print('Evaluate train data', model.evaluate(train_dataset))
print('Evaluate validation data', model.evaluate(val_dataset))
print('Evaluate test data', model.evaluate(test_dataset)) 

#Plot and save images of erroneously classified images/frames
model.trainable=False
helper_functions.plot_and_save_erroneous_images(df = df.head(500), model = model, image_folder=IMAGE_DIRECTORY, 
                                                fail_dir = FAIL_DIR, save_plots=True)
