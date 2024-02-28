#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 16 2024

@author: dario cantore

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
from PIL import Image


# flags
save_plot=True
save_model=True

# Load the CSV files containing image file names and labels corresponding to each image
FILE_SHAPE = (135, 180, 3)
COLOR_MODE = 'rgb'
DATA_DIRECTORY = "../data/"
IMAGE_DIRECTORY = "../data/img/"

df = pd.read_csv(DATA_DIRECTORY+'/data.csv')
height, width = FILE_SHAPE[0], FILE_SHAPE[1]
df.head()


def explore_data(df, output_dir, save_plot = False):
    """Function to display key characteristics of the data being read in"""
    #First, print an overview of the data frame
    # Overview
    print(10*'*', 'General information',10*'*')
    print('Dataframe shape: ', df.shape)
    print('Column names: ', df.columns)
    print()
    print('Data header\n', df.head())
    print()
    print('Column types\n', df.dtypes)
    print()
    print('unique values in character columns:\n',
        'Leela:  ', df['isLeela'].unique(), '\n',
        'Fry:    ', df['isFry'].unique(), '\n',
        'Bender: ', df['isBender'].unique()
        )
    
    #Then, print a count of how often each character appears
    num_img = df.shape[0]
    num_Fry = df['isFry'].sum()
    num_Leela = df['isLeela'].sum()
    num_Bender = df['isBender'].sum()
    #Note that the "isXXX" columns are the only integer columns. therefore the below works and is slightly easier
    num_no_char = sum(df.select_dtypes("int").sum(axis = 1) == 0)
    print(2*'\n',10*'*', 'Verteilung der Charaktere',10*'*')
    print('Number images: ', num_img)
    print()
    print('Images per character:\n',
        'w. Leela:  ', num_Leela, '\t w/o Leela:  ', num_img - num_Leela , '\n',
        'w. Fry:    ', num_Fry, '\t w/o Fry:    ', num_img - num_Fry, '\n',
        'w. Bender: ', num_Bender, '\t w/o Bender: ', num_img - num_Bender
        )
    print()
    print('Images w/o any: ', num_no_char)

    # Histogram of the distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[:, ['isLeela', 'isFry', 'isBender']],
            bins=[-.5,.5,1.5],
            edgecolor='black',  
            color=['purple','orange','grey'] # colors
            )

    plt.legend(['Leela', 'Fry', 'Bender'])
    plt.xticks((0, 1), ('in frame', 'not in frame'))  #x-label
    plt.title('Number of character appearances in frames')


    if save_plot: plt.savefig(output_dir+'Appearances.png', dpi=300)
    plt.show()

    #Note that the dataframe is not being modified in this function
    return(df)


# Helper function to import pictures
def load_and_preprocess(image_path, label):
    """This function is used to read in individual images of the dataset"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0  # Min-Max Normalize [0, 1]

    return image, label


def load_dataset(image_paths, df, train_size = 0.6, test_size = 0.2, validation_size = 0.2 ,\
                 leela_col = "isLeela", fry_col = "isFry", bender_col = "isBender"):
    """Function to load the dataset and extract train, test, and validation splits.
    Note that the label columns are hardcoded.
    Returns a train dataset, test dataset, and validation dataset"""
    #Load the dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, {"isLeela": df[leela_col] ,
                                                                "isFry": df[fry_col],
                                                                "isBender":df[bender_col] }))

    dataset = dataset.map(load_and_preprocess)

    #Make a sense check:
    if (train_size + test_size + validation_size) == 1:
        pass
    else:
        validation_size = 1-train_size-test_size
        print("Train, test, and validation size do not add up to one. Setting validation size to {}".format(validation_size))
    
    # Determine size of dataset
    total_samples = len(image_paths)
    train_size = int(train_size * total_samples)
    val_size = int(validation_size * total_samples)
    test_size = total_samples - train_size - val_size 

    # randomize
    dataset = dataset.shuffle(buffer_size=total_samples)

    # split in train, test, validation
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)
    test_dataset=test_dataset.shuffle(test_size, reshuffle_each_iteration=False)

    # batches anlegen
    batch_size = 16
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset, val_dataset


def plot_images(images, labels, output_dir, filename=None, eval_labels=None):
    """Function to plot images with labels"""
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    axes = axes.flatten() # einfacheres iterieren

    for i in range(12):
        image = images[i].numpy() # numpy format wird gebraucht
        label = labels["isLeela"][i].numpy(), labels["isFry"][i].numpy(), labels["isBender"][i].numpy()

        axes[i].imshow(image)
        axes[i].set_title(f'Leela: {label[0]} Fry: {label[1]} Bender: {label[2]}')
        axes[i].axis('off')

    plt.tight_layout()
    if filename: plt.savefig(output_dir, filename,dpi=300)
    plt.show()


def create_nn():
    """Function to create a convoluted neural network with 3 branches"""
    
    print(2*'\n',10*'*', 'Netzwerk anlegen',10*'*')

    # Input
    input_layer = layers.Input(shape=(height, width, 3), name='input_image')

    # preprocessing netzwerk elemente
    x = layers.RandomRotation(0.05,
                            fill_mode='nearest',
                            interpolation='bilinear',
                            seed=None,
                            fill_value=0.5)(input_layer)
    x = layers.RandomZoom(0.1,
                        width_factor=0.1,
                        fill_mode="nearest",
                        interpolation="bilinear",
                        seed=None,
                        fill_value=0.0)(x)
    x = layers.RandomFlip(mode="horizontal", seed=None)(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomBrightness([-0.3, 0.6], value_range=(0, 1))(x)


    # Convolution
    x = layers.Conv2D(16, (3, 3), activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)


    # Branch 1 - isLeela
    output1 = layers.Dense(32, activation='relu')(x)
    output1 = layers.Dense(16, activation='relu')(output1)
    output1 = layers.Dense(1, activation='sigmoid', name='isLeela')(output1)

    # Branch 2 - isFry
    output2 = layers.Dense(32, activation='relu')(x)
    output2 = layers.Dense(16, activation='relu')(output2)
    output2 = layers.Dense(1, activation='sigmoid', name='isFry')(output2)

    # Branch 3 - isBender
    output3 = layers.Dense(32, activation='relu')(x)
    output3 = layers.Dense(16, activation='relu')(output3)
    output3 = layers.Dense(1, activation='sigmoid', name='isBender')(output3)

    # Create model
    model = keras.models.Model(inputs=input_layer, outputs=[output1, output2, output3])
    return(model)

def fit(model, train_dataset, val_dataset, epochs = 200, ):
    """Function to fit the neural network, using early stopping based on average accuracy across the three classes.
    Note again that things like column names are hardcoded.
    Returns the history and early stopping instance"""
    
    # callback to get average accuracy. Also used for early stopping
    class MeanAcc(Callback):
        def __init__(self):
            super(MeanAcc, self).__init__()

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            logs['mean_accuracy'] = (logs["isLeela_accuracy"] + logs["isFry_accuracy"] + logs["isBender_accuracy"]) / 3
            logs['mean_val_accuracy'] = (logs["val_isLeela_accuracy"] + logs["val_isFry_accuracy"] + logs["val_isBender_accuracy"]) / 3


    mean_acc_cb=MeanAcc() # instantiate

    early_stopping_cb = EarlyStopping(monitor='mean_val_accuracy',  # use insance for early stopping
                                    patience=25,
                                    restore_best_weights=True,
                                    verbose=1,
                                    start_from_epoch=20
                                    )

    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=epochs,
                        verbose=True,
                        callbacks=[mean_acc_cb,early_stopping_cb]
    )
    return(history, early_stopping_cb)


def create_confusion_matrices(model, test_dataset, save_plot = True):
    """Function to evaluate a model on the test dataset.
    Note again that the column names are hardcoded.
    Does not return anything - creates plots and saves them as files where required"""
    # Create lists for true labels
    y_true_isLeela = []
    y_true_isFry = []
    y_true_isBender = []

    # Create lists for true predictions
    correct_predictions_isLeela = []
    correct_predictions_isFry = []
    correct_predictions_isBender = []

    # Create lists for wrong predictions
    mismatched_predictions_isLeela = []
    mismatched_predictions_isFry = []
    mismatched_predictions_isBender = []

    y_pred_all = []

    for images, labels in test_dataset:
        for i in range(images.shape[0]):
            # ein bild und label extrahieren
            single_image = np.expand_dims(images[i], axis=0)
            single_label_isLeela = labels['isLeela'][i]
            single_label_isFry = labels['isFry'][i]
            single_label_isBender = labels['isBender'][i]

            # Prediction
            y_pred = model.predict(single_image, verbose=1)

            # Runden
            y_pred_rounded = np.round(y_pred).astype(int)

            # 'Wahre' label an die Listen anhängen
            y_true_isLeela.append(single_label_isLeela)
            y_true_isFry.append(single_label_isFry)
            y_true_isBender.append(single_label_isBender)

            # Predictions an eigene Listen anhängen
            correct_predictions_isLeela.append(y_true_isLeela[-1] == y_pred_rounded[0])
            correct_predictions_isFry.append(y_true_isFry[-1] == y_pred_rounded[1])
            correct_predictions_isBender.append(y_true_isBender[-1] == y_pred_rounded[2])

            # Mismatches anhängen an die jeweiligen Listen
            if not correct_predictions_isLeela[-1]:
                mismatched_predictions_isLeela.append((int(y_true_isLeela[-1]), int(y_pred_rounded[0])))
            if not correct_predictions_isFry[-1]:
                mismatched_predictions_isFry.append((int(y_true_isFry[-1]), int(y_pred_rounded[1])))
            if not correct_predictions_isBender[-1]:
                mismatched_predictions_isBender.append((int(y_true_isBender[-1]), int(y_pred_rounded[2])))

            # Gesamtliste
            y_pred_all.append(y_pred_rounded)

    # Accuracy berechnen je label
    accuracy_isLeela = np.mean(correct_predictions_isLeela)
    accuracy_isFry = np.mean(correct_predictions_isFry)
    accuracy_isBender = np.mean(correct_predictions_isBender)

    print("Accuracy (isLeela):", accuracy_isLeela)
    print("Accuracy (isFry):", accuracy_isFry)
    print("Accuracy (isBender):", accuracy_isBender)

    # Datentype umwandeln
    y_true_isLeela_flat = [int(label) for label in y_true_isLeela]
    y_pred_isLeela_flat = [int(p[0]) for p in y_pred_all]
    y_true_isFry_flat = [int(label) for label in y_true_isFry]
    y_pred_isFry_flat = [int(p[1]) for p in y_pred_all]
    y_true_isBender_flat = [int(label) for label in y_true_isBender]
    y_pred_isBender_flat = [int(p[2]) for p in y_pred_all]

    # Confusion matrix für jedes label erstellen
    cm_isLeela = confusion_matrix(y_true_isLeela_flat,y_pred_isLeela_flat)
    cm_isFry = confusion_matrix(y_true_isFry_flat,y_pred_isFry_flat)
    cm_isBender = confusion_matrix(y_true_isBender_flat,y_pred_isBender_flat)

    # Plotten
    labels = ['absent', 'present']  # Assuming your labels are 0 and 1

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(cm_isLeela, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Leela)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 3, 2)
    sns.heatmap(cm_isFry, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Fry)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 3, 3)
    sns.heatmap(cm_isBender, annot=True, fmt="d", cmap="Greys", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Bender)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    if save_plot: plt.savefig('ConfusionMatrix.png', dpi=150)
    plt.show()


def create_confusion_matrices(model, test_dataset, save_plot = True):
    """Function to create predictions for each individual image, and create confusion matrices out of them"""
    y_pred_all = []
    # 'true' label
    y_true_isLeela = []
    y_true_isFry = []
    y_true_isBender = []

    # Lists for predictions
    correct_predictions_isLeela = []
    correct_predictions_isFry = []
    correct_predictions_isBender = []

    #lists for erroneous predictions
    mismatched_predictions_isLeela = []
    mismatched_predictions_isFry = []
    mismatched_predictions_isBender = []

    for images, labels in test_dataset:
        for i in range(images.shape[0]):
            # extract image and label
            single_image = np.expand_dims(images[i], axis=0)
            single_label_isLeela = labels['isLeela'][i]
            single_label_isFry = labels['isFry'][i]
            single_label_isBender = labels['isBender'][i]

            # Prediction
            y_pred = model.predict(single_image, verbose=1)

            # round to closest integer
            y_pred_rounded = np.round(y_pred).astype(int)

            # Append true labels
            y_true_isLeela.append(single_label_isLeela)
            y_true_isFry.append(single_label_isFry)
            y_true_isBender.append(single_label_isBender)

            # Append predictions to lists
            correct_predictions_isLeela.append(y_true_isLeela[-1] == y_pred_rounded[0])
            correct_predictions_isFry.append(y_true_isFry[-1] == y_pred_rounded[1])
            correct_predictions_isBender.append(y_true_isBender[-1] == y_pred_rounded[2])

            # append mistmatches
            if not correct_predictions_isLeela[-1]:
                mismatched_predictions_isLeela.append((int(y_true_isLeela[-1]), int(y_pred_rounded[0])))
            if not correct_predictions_isFry[-1]:
                mismatched_predictions_isFry.append((int(y_true_isFry[-1]), int(y_pred_rounded[1])))
            if not correct_predictions_isBender[-1]:
                mismatched_predictions_isBender.append((int(y_true_isBender[-1]), int(y_pred_rounded[2])))

            # Create total lists
            y_pred_all.append(y_pred_rounded)
    
    # Get accuracy per label
    accuracy_isLeela = np.mean(correct_predictions_isLeela)
    accuracy_isFry = np.mean(correct_predictions_isFry)
    accuracy_isBender = np.mean(correct_predictions_isBender)

    print("Accuracy (isLeela):", accuracy_isLeela)
    print("Accuracy (isFry):", accuracy_isFry)
    print("Accuracy (isBender):", accuracy_isBender)

    # Change datatypes
    y_true_isLeela_flat = [int(label) for label in y_true_isLeela]
    y_pred_isLeela_flat = [int(p[0]) for p in y_pred_all]
    y_true_isFry_flat = [int(label) for label in y_true_isFry]
    y_pred_isFry_flat = [int(p[1]) for p in y_pred_all]
    y_true_isBender_flat = [int(label) for label in y_true_isBender]
    y_pred_isBender_flat = [int(p[2]) for p in y_pred_all]

    # Confusion matrix per label
    cm_isLeela = confusion_matrix(y_true_isLeela_flat,y_pred_isLeela_flat)
    cm_isFry = confusion_matrix(y_true_isFry_flat,y_pred_isFry_flat)
    cm_isBender = confusion_matrix(y_true_isBender_flat,y_pred_isBender_flat)

    # Create plot
    labels = ['absent', 'present']  # Assuming your labels are 0 and 1

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(cm_isLeela, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Leela)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 3, 2)
    sns.heatmap(cm_isFry, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Fry)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.subplot(1, 3, 3)
    sns.heatmap(cm_isBender, annot=True, fmt="d", cmap="Greys", xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Bender)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.tight_layout()
    if save_plot: plt.savefig('ConfusionMatrix.png', dpi=150)
    plt.show()
    return None

def plot_learning_curves(history, output_dir,save_plot = True):
    """Function to plot the learning curves"""
    plt.plot(history.history['isLeela_accuracy'], label='Leela Train')
    plt.plot(history.history['val_isLeela_accuracy'], label='Leela Val.')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_plot: plt.savefig(output_dir+'LeelaPlot.png', dpi=150)
    plt.show()

    plt.plot(history.history['isFry_accuracy'], label='Fry Train')
    plt.plot(history.history['val_isFry_accuracy'], label='Fry Val.')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_plot: plt.savefig(output_dir+'FryPlot.png', dpi=150)
    plt.show()

    plt.plot(history.history['isBender_accuracy'], label='Bender Train')
    plt.plot(history.history['val_isBender_accuracy'], label='Bender Val.')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    if save_plot: plt.savefig(output_dir+'BenderPlot.png', dpi=150)
    plt.show()
    
    return None

def plot_and_save_erroneous_images(model, df, save_plots, image_folder, fail_dir,
                                   target_width = 180, target_height = 135, threshold = 0.5):
    """Function to create predictions for images and plot and save erronous images.
    Does not return anything"""
    
    target_width = target_width  # Target image width
    target_height = target_height  # Target image height
    threshold = threshold  # Threshold for prediction True or false
    file_no = len(df)
    fail_dir = fail_dir #Folder where plots will be stored

    print('Prediction running, this might take a moment.')
    
    #Loop over the dataframe, where each row is a path to an image as well as labels
    for index, row in df.iterrows():
        
        #Get paths to the image
        image_name = row['file']
        image_path = f'{image_folder}/{image_name}'
        try:
            # load image
            img = Image.open(image_path)

            # scale
            scaled_img = img.resize((target_width, target_height))

            # convert to numpy and set dimensions to model dimensions
            img_array = np.array(scaled_img)
            img_array = img_array/255.0 # min-max!
            img_array = np.expand_dims(img_array, axis=0)

            # close files
            img.close()
            scaled_img.close()
            
            # predict probabilities
            predictions = model.predict(img_array, verbose=0)
                    
            # insert probabilities into dataframe
            df.at[index, 'predLeela'] = predictions[0]
            df.at[index, 'predFry'] = predictions[1]
            df.at[index, 'predBender'] = predictions[2]
            
            file_no = file_no-1
            if file_no % 100 == 0: print(f'{file_no} images to go.')
            
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
    
    #Calculate correct predictions 
    # Add new columns that indicates whether the entire prediction is correct  

    if set(['isLeela','isFry','isBender']).issubset(df.columns):
        
        df['pred_correct'] = ((abs(df['isLeela'] - df['predLeela'])<threshold) &
                            (abs(df['isFry'] - df['predFry'])<threshold)  &
                            (abs(df['isBender'] - df['predBender'])<threshold))
        
        print(df['pred_correct'].value_counts())
    

        df_mistakes=df.loc[df.loc[:,'pred_correct']==False] # new dt with only erroneous images
    
    if save_plot == True:
        if len(df_mistakes)>1:
            df_mistakes.to_csv('false_predictions.csv', sep=',') # save the csv if the dataframe is not empty
        df.to_csv('all_predictions.csv', sep=',') # Save a dataframe with all predictions


    # Load images in false_precitions.csv
    fig, axes = None, None
    plot_counter = 0

    for index, row in df_mistakes.iterrows():
        #Once 12 erroneous images have been found, plot them
        if plot_counter % 12 == 0:
            if axes is not None:
                plt.tight_layout()
                plt.savefig(os.path.join(fail_dir, f'combined_plot_{plot_counter // 12}.png'))
                plt.close()


            remaining_images = len(df_mistakes) - plot_counter
            num_images_to_display = min(remaining_images, 12)

            fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12))
        
        image_name = row['file']
        image_path = f'{image_folder}/{image_name}'

        try:
            img = Image.open(image_path)

            ax = axes[plot_counter % 3, plot_counter % 4]

            ax.imshow(img)
            ax.axis('off')

            ax.set_title(f"True Leela:{row['isLeela']} Fry:{row['isFry']} Bender:{row['isBender']}\n"
                        f"Pred Leela:{row['predLeela']:.0f} Fry:{row['predFry']:.0f} Bender:{row['predBender']:.0f}",
                        size=14)

            img.close()

            plot_counter += 1

        except FileNotFoundError:
            print(f"Image file not found: {image_path}")

    # Save if last plot not complete empty, save it anyways
    if axes is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(fail_dir, f'combined_plot_{plot_counter // 12}.png'))
        plt.close()
        
    print('Plotting done!')