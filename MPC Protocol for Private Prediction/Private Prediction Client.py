import numpy as np
import tensorflow as tf
import tf_encrypted as tfe
import os
import pandas as pd
from keras.utils import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import array
from PIL import Image, ImageFile

PATH = os.getcwd()
data_dir_list = os.listdir(PATH)
print(PATH)

target_data = pd.read_csv(PATH + '/INbreast_r.csv')
num_classes = 6

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
test_labels_arr = array.array('i', [])
for img in test_img_list:
    if img == '.DS_Store':
        continue
    test_label = label_img(img)
    #print(label)
    test_labels_arr.append(test_label)

y_test = np.array(test_labels_arr, dtype='uint8')

# input image dimensions
img_rows, img_cols = 64, 64
input_shape = (img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
x_test = x_test.astype('float32')
x_test /= 255


config = tfe.RemoteConfig.load("./tfe.config")
config.connect_servers()
tfe.set_config(config)
tfe.set_protocol(tfe.protocol.SecureNN())

input_shape = (1, 64, 64, 3)
output_shape = (1, 6)
client = tfe.serving.QueueClient(input_shape=input_shape, output_shape=output_shape)

# User inputs
num_tests = 10
images, expected_labels = x_test[:num_tests], y_test[:num_tests]
expected_labels

for image, expected_label in zip(images, expected_labels):
    
    res = client.run(image.reshape((1, 64, 64, 3)))
    print(res)

    predicted_label = np.argmax(res)

    print("The image had label {} and was {} classified as {}".format(
        expected_label,
        "correctly" if expected_label == predicted_label else "incorrectly",
        predicted_label))

