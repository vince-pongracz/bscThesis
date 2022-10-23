
import re
from PIL import Image
from rawkit.options import WhiteBalance
from rawkit.raw import Raw
import imageio
import rawpy
import scipy.misc as misc
from skimage import exposure
import skimage
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras import datasets, layers, models
import pandas as pd
import os
import datetime

# image processing
import cv2

from exiftool import ExifToolHelper

# TensorFlow
import tensorflow as tf
ResizeMethod = tf.image.ResizeMethod


# TODO good to check:
# scikit-image - for image resize, compress
# ImageDataGenerator - tensorflow/keras: for data augmentation (https://www.youtube.com/watch?v=nU_T2PPigUQ&ab_channel=JeffHeaton)


def process_meta(exifMeta: list):
    # this transformation is required to get the exif fields
    exifMeta = exifMeta[0]
    processed = [
        # "colorTemp":
        int(exifMeta["XMP:ColorTemperature"]),
        # "tint":
        int(exifMeta["XMP:Tint"]),
        # "brightness":
        float(exifMeta["XMP:Exposure2012"]),
        # "contrast":
        int(exifMeta["XMP:Contrast2012"]),
        # "vibrance":
        int(exifMeta["XMP:Vibrance"])
        # occasionally add other fields, as start it will be goo like this
    ]
    return processed


def load_images_from_folder(folder: str):
    """
    loads images
    :param folder: filename in str
    :return: list of images in dim. [x * y * z]
    """
    imagesWithMeta = []
    for filename in os.listdir(folder):
        fullName = os.path.join(folder, filename)
        img = cv2.imread(fullName)

        with ExifToolHelper() as et:
            meta = et.execute_json(fullName)
            processedExif = process_meta(meta)
            if img is not None:
                normed_img = img / 255
                compressed = tf.image.adjust_jpeg_quality(normed_img, 90)
                new_size = [200, 300]
                bilin = tf.image.resize(
                    compressed, new_size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=True).numpy()
                # imagesWithMeta.append({
                #     'picture': bilin,
                #     'exif': processedExif})
                imagesWithMeta.append({
                    'pic': bilin,
                    'exif': processedExif
                })

    return imagesWithMeta


def get_dataset_partitions(df: pd.DataFrame, train_split=0.8, val_split=0.1, test_split=0.1, shuffleData=True):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(df)
    if shuffleData:
        # Specify seed to always have the same split distribution between runs
        df = shuffle(df)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = df.take([0, train_size])
    val_ds = df.drop([0, train_size]).take([0, val_size])
    test_ds = df.drop([0, train_size+val_size])

    return train_ds, val_ds, test_ds


def convert(images):
    df = pd.DataFrame(images)
    df.to_pickle('dataframe.pkl')
    # print(df)
    # print(df.describe())
    return get_dataset_partitions(df)


def readFromPkl(filename: str):
    df = pd.read_pickle(filename)
    return get_dataset_partitions(df)


def buildModel(in_shape, out_shape):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=in_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(out_shape, activation='linear'))

    # model.summary()

    return model


def compileAndTrainModel(model: models.Sequential, trainData: pd.DataFrame, testData: pd.DataFrame, epochs: int = 20):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.mean_squared_error(from_logits=True),
                  metrics='mae')

    # split trainData, testData into images-label
    train_images, train_labels = trainData['pic'], trainData['exif']
    test_images, test_labels = testData['pic'], testData['exif']

    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels))


def readData(directory: str = 'resources\\raws'):
    imageData = []
    xmpData = []
    
    for filename in os.listdir(directory):
        fullName = os.path.join(directory, filename)
        if fullName.endswith('.xmp'):
            with ExifToolHelper() as et:
                meta = et.execute_json(fullName)
                processedExif = process_meta(meta)
                xmpData.append(processedExif)
        else:
            rawPic = readRawSimple(fullName)
            imageData.append(rawPic)

    imagesWithMeta = zip(imageData, xmpData)
    return imagesWithMeta


def readRawSimple(fullNameWithPath: str = 'resources\\raws\\temp.ARW'):
    with rawpy.imread(fullNameWithPath) as rawImg:
        rgbImg = rawImg.postprocess(rawpy.Params(use_camera_wb=True))
        rgbNormed = rgbImg / 255.0
        new_size = [200, 300]
        bilinear = tf.image.resize(
            rgbNormed, new_size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=True).numpy()
        return bilinear


def readRaw(path: str = 'resources\\raws', picName: str = 'temp.ARW'):
    """
    Read raw file from path with name picName

    Args:
        path (str, optional): Path to dir. Loads raws from this dir. Defaults to 'resources\raws'.
        picName (str, optional): Name of the picture to load. Defaults to 'temp.ARW'.

    # TODO extend
    Returns:
        _type_: _description_
    """
    with rawpy.imread(os.path.join(path, picName)) as rawImg:
        # rawImg = rawImg.raw_image # gets raw image in ndarray, not RGB format
        # convert raw image ndarray into sRGB
        rgbImg = rawImg.postprocess(rawpy.Params(use_camera_wb=True))
        rgbNormed = rgbImg / 255.0

        # show image
        # size = 4
        # plt.figure(figsize=(size,size))
        # plt.imshow(rgbImg)
        # plt.show()
        return rgbNormed


class Correction:
    def __init__(self, colorTemp, tint, brightness, contrast, vibrance):
        self.colorTemp = colorTemp
        self.tint = tint
        self.brightness = brightness
        self.contrast = contrast
        self.vibrance = vibrance


def generateXmpResult(path: str = 'resources\\raws', pictureName: str = 'sample4.NEF', correction: Correction = Correction(7000, -4, 1.3, 10, 4)):
    fullFileNameRaw = os.path.join(path, pictureName)
    fullFileNameXmp = fullFileNameRaw.split(".")[0] + ".xmp"

    date = datetime.datetime.now()

    xmpString = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, {date}        ">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about=""
                xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
                crs:Temperature="{correction.colorTemp}"
                crs:Tint="{correction.tint}"
                crs:Exposure2012="{correction.brightness}"
                crs:Contrast2012="{correction.contrast}"
                crs:Vibrance="{correction.vibrance}">
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>"""

    with open(fullFileNameXmp, "w") as xmpFile:
        xmpFile.write(xmpString)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # path = "resources\\testJpgs"
    # images = load_images_from_folder(path)
    # tr, val, tst = convert(images)
    # tr, val, tst = readFromPkl('dataframe.pkl')

    # print(val)
    # rgbNormed = readRaw()
    # xmpTest = generateXmpResult()
    # rawTest = readRawSimple()
    
    imagesWithMeta = list(readData())
    # item = imagesWithMeta[0]
    
    
    print('done')
