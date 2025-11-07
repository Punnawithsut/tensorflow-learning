import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

## Avoid OOM errors by limit GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)

## Remove Dodgy Images
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

## Load Data
data = tf.keras.utils.image_dataset_from_directory("data")
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

#fig, ax = plt.subplots(ncols = 4, figsize=(20,20))
#for idx, img in enumerate(batch[0][:4]):
#    ax[idx].imshow(img.astype(int))
#    ax[idx].title.set_text(batch[1][idx])
#
#plt.show()