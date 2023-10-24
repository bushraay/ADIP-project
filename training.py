import cv2 as cv
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# Function to load images and labels from a folder
def load_images_and_labels(folder_path, class_label):
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            image = cv.imread(os.path.join(folder_path, filename))
            # You can optionally resize the image if needed
            if image is not None:
                # You can optionally resize the image if needed
                image = cv.resize(image, (32, 32))
                images.append(image)
                labels.append(class_label)

    return images, labels

# # Define class names
class_names = ['plastic', 'metal']

# Initialize empty lists for training and testing data
training_images = []
training_labels = []
testing_images = []
testing_labels = []

# Set the split ratio (70% for training, 30% for testing)
split_ratio = 0.7

# Load data from the specified directories
for class_label, class_name in enumerate(class_names):
    data_folder = os.path.join('trainingDataP', class_name)
    image_files = [f for f in os.listdir(data_folder) if f.endswith('.jpeg') or f.endswith('.png')]
    # Shuffle the image files for randomness
    random.shuffle(image_files)

    # Determine the split point based on the ratio
    split_point = int(len(image_files) * split_ratio)
    # Load training data
    images, labels = load_images_and_labels(data_folder, class_label)
    training_images.extend(images[:split_point])
    training_labels.extend(labels[:split_point])

    # Load testing data
    testing_images.extend(images[split_point:])
    testing_labels.extend(labels[split_point:])

# Convert the lists to numpy arrays
training_images = np.array(training_images)
training_labels = np.array(training_labels)
testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)

# Normalize the image data
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# model definition and training
# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])

# model.fit(training_images,training_labels,epochs=10, validation_data=(testing_images,testing_labels))

# model.save('image_classifier.model')


model = models.load_model('image_classifier.model')
loss, accuracy = model.evaluate(testing_images,testing_labels)
print("Loss =",loss)
print("Accuracy =", accuracy)

a = cv.imread("ds/1.jpeg")
b = cv.imread("ds/2.jpeg")
c = cv.imread("ds/3.jpeg")
d = cv.imread("ds/4.jpeg")
e = cv.imread("ds/5.jpeg")
f = cv.imread("ds/6.jpeg")
a = cv.cvtColor(a, cv.COLOR_BGR2RGB);Ia=cv.resize(a, (32,32))
outa = model.predict(np.array([Ia])/255);outa = np.argmax(outa)
b = cv.cvtColor(b, cv.COLOR_BGR2RGB);Ib=cv.resize(b, (32,32))
outb = model.predict(np.array([Ib])/255);outb = np.argmax(outb)
c = cv.cvtColor(c, cv.COLOR_BGR2RGB);Ic=cv.resize(c, (32,32))
outc = model.predict(np.array([Ic])/255);outc = np.argmax(outc)
d = cv.cvtColor(d, cv.COLOR_BGR2RGB);Id=cv.resize(d, (32,32))
outd = model.predict(np.array([Id])/255);outd = np.argmax(outd)
e = cv.cvtColor(e, cv.COLOR_BGR2RGB);Ie=cv.resize(e, (32,32))
oute = model.predict(np.array([Ie])/255);oute = np.argmax(oute)
f = cv.cvtColor(f, cv.COLOR_BGR2RGB);If=cv.resize(f, (32,32))
outf = model.predict(np.array([If])/255);outf = np.argmax(outf)
plt.subplot(2,3,1);plt.imshow(a);plt.xlabel(class_names[outa])
plt.subplot(2,3,2);plt.imshow(b);plt.xlabel(class_names[outb])
plt.subplot(2,3,3);plt.imshow(c);plt.xlabel(class_names[outc])
plt.subplot(2,3,4);plt.imshow(d);plt.xlabel(class_names[outd])
plt.subplot(2,3,5);plt.imshow(e);plt.xlabel(class_names[oute])
plt.subplot(2,3,6);plt.imshow(f);plt.xlabel(class_names[outf])

plt.show()

# a = cv.imread("ds/can.jpg")
# b = cv.imread("ds/can2.jpg")
# c = cv.imread("ds/can3.jpg")
# d = cv.imread("ds/can4.jpg")