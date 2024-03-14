import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split


n_classes = 2
path_pos = 'data/Face' 
path_neg = 'data/NotFace' 

def read_data(path, label):
    images = []
    labels = []
    dirs = os.listdir(path)
    for files in dirs:
        file_name = os.path.join(path, files)
        image = cv2.imread(file_name, 0)
        image = cv2.resize(image, (100, 40))  
        image = np.reshape(image, 40 * 100)
        images.append(image)
        labels.append(label)
    return images, labels

images_pos, labels_pos = read_data(path_pos, 1)
images_neg, labels_neg = read_data(path_neg, 0)

images = images_pos + images_neg
labels = labels_pos + labels_neg

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=41)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# In thông tin
print("Kích thước dữ liệu huấn luyện:", x_train.shape)
print("Kích thước nhãn huấn luyện:", y_train.shape)