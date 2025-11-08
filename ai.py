import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

## Avoid OOM errors by limit GPU Memory Consumption Growth------------------------
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

## Remove Dodgy Images------------------------------------------------------------
data_dir = "data"
image_exts = ["jpeg", "jpg", "bmp", "png"]

# print(os.listdir(data_dir))

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)

        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            #print("GOOD")
            if tip not in image_exts:
                print(f"Image not in ext list {image_path}")
                os.remove(image_path)
        except Exception as e:
            print(f"Issue with image {image_path}")

## Load Data---------------------------------------------------

data = tf.keras.utils.image_dataset_from_directory("data")
#data_iterator = data.as_numpy_iterator()
#batch = data_iterator.next()

#fig, ax = plt.subplots(ncols = 4, figsize=(20,20))
#for idx, img in enumerate(batch[0][:4]):
#    ax[idx].imshow(img)
#    ax[idx].title.set_text(batch[1][idx])
#
#plt.show()

## Preprocess Data---------------------------------------------

data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()[0]

train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2)+1
test_size = int(len(data)*0.1)+1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

## Deep Model -> 1.) Build Deep Learning Model 2.) Train 3.) Plot Performance

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])
model.summary()

logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=25, validation_data=val, callbacks=[tensorboard_callback])

#fig = plt.figure()
#plt.plot(hist.history["loss"], color="teal", label="loss")
#plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
#plt.legend(loc="upper left")
#plt.show()

#fig = plt.figure()
#plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
#plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
#plt.legend(loc="upper left")
#plt.show()

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    yhat = model.predict(x)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}")

## Test
img = cv2.imread("tests/happy/happy_test.jpg")
resize = tf.image.resize(img, (256,256))

yhat = model.predict(np.expand_dims(resize/255, 0))
if yhat > 0.5:
    print("SAD")
else:
    print("HAPPY")

## Save Model
model.save(os.path.join("models", "happysadmodel.h5"))
