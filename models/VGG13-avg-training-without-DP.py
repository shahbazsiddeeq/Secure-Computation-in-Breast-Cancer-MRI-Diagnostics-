import os
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import array
from PIL import Image, ImageFile
from keras.callbacks import ModelCheckpoint

PATH = os.getcwd()
data_dir_list = os.listdir(PATH)
print(PATH)

target_data = pd.read_csv(PATH + '/INbreast_r.csv')
dataset = 'aug_dataset'
train_dataset = 'aug_dataset/training'
test_dataset = 'aug_dataset/testing'
img_list=os.listdir(PATH+'/'+dataset)
train_img_list=os.listdir(PATH+'/'+train_dataset)
test_img_list=os.listdir(PATH+'/'+test_dataset)

img_data_list=[]
train_img_data_list=[]
test_img_data_list=[]

def label_img(img):
    word_label = img.split('_')[0]
    find = target_data.loc[target_data['File Name'].isin([int(word_label)])]
    for cell in find['Bi-Rads']:
        findf = cell
    if findf == '4a' or findf == '4b' or findf == '4c' :
        findf = '4'
    findf = int(findf)
    return findf-1

for img in train_img_list:
    if img == '.DS_Store':
        continue
    img_path = PATH + '/'+ train_dataset + '/'+ img
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = load_img(img_path, target_size=(64, 64), color_mode='rgb')
    x = img_to_array(img)
    train_img_data_list.append(x)

x_train = np.array(train_img_data_list)
x_train.shape[0]
x_train = x_train.reshape(x_train.shape[0], 64, 64, 3)
x_train = x_train.astype('float32')
x_train /= 255

train_labels_arr = array.array('i', [])
for img in train_img_list:
    if img == '.DS_Store':
        continue
    train_label = label_img(img)
    train_labels_arr.append(train_label)

y_train = np.array(train_labels_arr, dtype='uint8')
y_train

for img in test_img_list:
    if img == '.DS_Store':
        continue
    img_path = PATH + '/'+ test_dataset + '/'+ img
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = load_img(img_path, target_size=(64, 64), color_mode='rgb')
    x = img_to_array(img)
    test_img_data_list.append(x)

x_test = np.array(test_img_data_list)
x_test.shape
x_test = x_test.reshape(x_test.shape[0], 64, 64, 3)
x_test = x_test.astype('float32')
x_test /= 255

test_labels_arr = array.array('i', [])
for img in test_img_list:
    if img == '.DS_Store':
        continue
    test_label = label_img(img)
    test_labels_arr.append(test_label)

y_test = np.array(test_labels_arr, dtype='uint8')

num_classes = 6
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

input_shape = (64, 64, 3)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 16
epochs = 50

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(64, (3, 3), input_shape=(64, 64, 3), padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
  tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',),
  tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(6, activation='softmax')
])

mcp = ModelCheckpoint(filepath=PATH+'/model_vgg13.h5',monitor="val_acc", save_best_only=True, save_weights_only=False)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[mcp],
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





