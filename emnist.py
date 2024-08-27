import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

num_classes = 10
f, ax = plt.subplots(1, num_classes, figsize=(20,20))

for i in range(0, num_classes):
  sample = x_train[y_train == i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Label: {}".format(i), fontsize=16)
     
# import sys
# import sklearn
# import numpy as np
# from sklearn.datasets import fetch_openml
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
# from sklearn import metrics
# mnist = fetch_openml('mnist_784', version=1)
# mnist.keys()
# data = mnist["data"]
# target = mnist["target"]
# data_train, data_test, target_train, target_test = data[:60000], data[60000:], target[:60000], target[60000:]
# data_train = data_train[:1000]
# target_train = target_train[:1000]
# data_train = data_train/255.0
# data_test = data_test/255.0
# print("done1")

# non_linear_model = SVC(kernel='rbf',decision_function_shape='ovr')
# # linear_model = LinearSVC()
# print("done2")
# non_linear_model.fit(data_train, target_train)
# print("done3")
# target_pred = non_linear_model.predict(data_test)
# print("accuracy:", metrics.accuracy_score(target_test, target_pred))

# //--------------------------------
# from tensorflow import keras
# from keras.datasets import mnist
# from keras import layers, models
# import matplotlib.pyplot as plt
# import numpy as np
# f = open("weights.txt", "w")
# (train_x, train_y), (test_x, test_y) = mnist.load_data()

# # Flatten the training and testing data

# new_train_x = []
# for sample in train_x:
#     to_add = []

#     for row in sample:
#         for pixel in row:
#             to_add.append(pixel)

#     new_train_x.append(np.array(to_add))

# train_x = np.array(new_train_x)

# new_test_x = []
# for sample in test_x:
#     to_add = []

#     for row in sample:
#         for pixel in row:
#             to_add.append(pixel)

#     new_test_x.append(np.array(to_add))

# test_x = np.array(new_test_x)

# # Create model
# model = keras.Sequential()
# model.add(keras.layers.Input(shape=train_x.shape))
# model.add(keras.layers.Dense(512, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))
# model.summary()

# model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# history = model.fit(train_x, train_y, epochs=30, validation_data=(test_x, test_y))

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

# test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
# model.save('mnist')
# print('Testing accuracy:', test_acc)

# for lay in model.layers:
#     for i in lay.get_weights():
#         f.write(str(i.tolist())[1:-1])
#         f.write("\n")