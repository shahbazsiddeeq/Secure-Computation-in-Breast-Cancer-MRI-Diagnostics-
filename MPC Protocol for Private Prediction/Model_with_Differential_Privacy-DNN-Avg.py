from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import array
from PIL import Image, ImageFile
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer



num_classes = 6

# input image dimensions
img_rows, img_cols = 64, 64

PATH = 'D:/School/shahbaz/ppml/Archive'
data_dir_list = os.listdir(PATH)

target_data = pd.read_csv(PATH + '/INbreast_r.csv')
num_classes = 6
dataset = ''
train_dataset = '/training'
test_dataset = '/testing'
img_list=os.listdir(PATH+'/'+dataset)
train_img_list=os.listdir(PATH+'/'+train_dataset)
test_img_list=os.listdir(PATH+'/'+test_dataset)

img_data_list=[]
train_img_data_list=[]
test_img_data_list=[]


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
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
def label_img(img):
    word_label = img.split('_')[0]
#     print(word_label)
    find = target_data.loc[target_data['File Name'].isin([int(word_label)])]
#     print(find)
    for cell in find['Bi-Rads']:
        findf = cell
    #findf = find['Bi-Rads']
    if findf == '4a' or findf == '4b' or findf == '4c' :
        findf = '4'
    findf = int(findf)
    return findf-1
train_labels_arr = array.array('i', [])

for img in train_img_list:
    if img == '.DS_Store':
        continue
    train_label = label_img(img)
    #print(label)
    train_labels_arr.append(train_label)
#     train_labels.append(train_label)
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
test_labels_arr
for img in test_img_list:
    if img == '.DS_Store':
        continue
    test_label = label_img(img)
    test_labels_arr.append(test_label)

y_test = np.array(test_labels_arr, dtype='uint8')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
input_shape = (64, 64, 3)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 16
num_classes = 6
epochs = 100

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 input_shape=(64, 64, 3)),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(6, activation='softmax')
  ])
l2_norm_clip = 1.0  # Adjust based on your specific requirements
noise_multiplier = 0.1  # Can vary between 0.1 and 1.0
num_microbatches = 1  # Can be the same as your batch size
learning_rate = 0.001  # Adjust as needed

optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate)

loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # Assuming logits as the output
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.save('model_DNN_avg_16_w1.h5')

