
import threading
from tensorflow.keras.models import load_model
from datasets import Dataset, Features
import datasets
import shutil
from copyreg import pickle
import re
# from PIL import Image
# from rawkit.options import WhiteBalance
# from rawkit.raw import Raw
# import imageio
import rawpy
# import scipy.misc as misc
# from skimage import exposure
# import skimage
# import math
# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
from keras import datasets, layers, models
import pandas as pd
import os
import datetime

# Python program to demonstrate
# HDF5 file
import numpy as np
import h5py

# image processing
import cv2

from exiftool import ExifToolHelper
import tensorboard

import subprocess

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


def get_dataset_partitions(df: pd.DataFrame, train_split=0.7, test_val_split=0.5):

    train, testAndValid = train_test_split(df, test_size=(
        1-train_split), random_state=42, shuffle=True)

    test, valid = train_test_split(
        testAndValid, test_size=test_val_split, random_state=43, shuffle=True)

    return train, test, valid


def convert(images, shuffleData=True, serialize=False, pklName: str = 'dataframe'):
    if shuffleData:
        # Specify seed to always have the same split distribution between runs
        images = shuffle(images)

    df = pd.DataFrame(images, columns=('raw', 'exif'))

    train, test, valid = get_dataset_partitions(df)
    if serialize:
        df.to_pickle(f'{pklName}_all.pkl', protocol=4)
        train.to_pickle(f'{pklName}_train.pkl', protocol=4)
        test.to_pickle(f'{pklName}_test.pkl', protocol=4)
        valid.to_pickle(f'{pklName}_valid.pkl', protocol=4)

    # print(df)
    # print(df.describe())
    return train, test, valid


def readFromPkl(filename: str):
    df = pd.read_pickle(filename)
    return get_dataset_partitions(df)


def buildModel(in_shape, out_shape):

    model = nn_models.Sequential()

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


def readRawSimple(fullNameWithPath: str = 'resources\\raws\\temp.ARW', new_size: list[int] = [133, 200], return_as_tensor: bool = False):
    with rawpy.imread(fullNameWithPath) as rawImg:
        rgbImg = rawImg.postprocess(rawpy.Params(use_camera_wb=True))
        rgbNormed = rgbImg / 255.0
        bilinear = tf.image.resize(
            rgbNormed, new_size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=True)
        if not return_as_tensor:
            bilinear = bilinear.numpy
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
    num_pred: int = 5
    # def __init__(self, colorTemp, tint, brightness, contrast, vibrance):
    #     self.colorTemp = colorTemp
    #     self.tint = tint
    #     self.brightness = brightness
    #     self.contrast = contrast
    #     self.vibrance = vibrance
    def __init__(self, pred_list: list):
        list_len = len(pred_list)
        for i in range(Correction.num_pred - list_len):
            pred_list.append(0)
            
        [self.colorTemp, self.tint, self.brightness, self.contrast, self.vibrance] = pred_list


def generateXmpResult(path: str = 'resources\\raws', pictureName: str = 'sample4.NEF', correction: Correction = Correction([7000, -4, 1.3, 10, 4])):
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


def cleanData(directory: str = 'resources\\rawTest') -> None:
    files = os.listdir(directory)

    # TODO unique constraint for the samples

    acceptedExtensions = ['xmp', 'ARW', 'arw', 'NEF', 'nef', 'cr2', 'CR2']

    for file in files:
        fullName = os.path.join(directory, file)
        name, extension = file.split('.')
        if (extension not in acceptedExtensions):
            os.remove(fullName)
            files.remove(file)
        else:
            regex = re.compile(f'{name}.*')
            count = len(list(filter(regex.match, files)))
            if (count != 2):
                # print(f'remove: {fullName}')
                os.remove(fullName)
                files.remove(file)


def data_load_and_preprocess(directory: str = f'resources{os.path.sep}raws', pathToJPGs: str = f'resources{os.path.sep}genJPGs'):

    rawImages: np.array = []
    xmpData: np.array = []
    files = os.listdir(directory)
    if not os.path.exists(pathToJPGs):
        os.mkdir(pathToJPGs)
    convertedImageSize = [133, 200]

    for file in files:
        name, ext = file.split('.')
        if ext != 'xmp':
            with rawpy.imread(f'{directory}{os.path.sep}{file}') as rawImg:
                rgbImg = rawImg.postprocess(rawpy.Params(use_camera_wb=True))
                rgbNormed = rgbImg / 255.0
                bilinear = tf.image.resize(
                    rgbNormed, convertedImageSize, method=ResizeMethod.BILINEAR)
                jpgName = f'{pathToJPGs}{os.path.sep}{name}.jpg'
                tf.keras.utils.save_img(jpgName, bilinear)
                rawImages.append(bilinear)
        else:
            with ExifToolHelper() as et:
                fileName = f'{directory}{os.path.sep}{file}'
                shutil.copyfile(fileName, f'{pathToJPGs}{os.path.sep}{file}')
                meta = et.execute_json(fileName)
                processedExif = tf.convert_to_tensor(process_meta(meta))
                xmpData.append(processedExif)

    # images = tf.keras.utils.image_dataset_from_directory(directory=directory, image_size=tuple(convertedImageSize))
    print('test tensorflow load')

    return rawImages, xmpData


def saveDatasets(imgs, xmps, path: str = f'datasets{os.path.sep}ds') -> None:
    ds = Dataset.from_dict({"img": imgs, "exif": xmps})
    ds.save_to_disk(path)
    # TODO push to google drive :D
    print('dataset saved!')


def loadDataset(path: str = f'datasets{os.path.sep}ds') -> Dataset:
    ds = datasets.load_from_disk(path)
    print('load done')
    return ds


def prepareToPrediction(raw_dir: str = f'resources{os.path.sep}raws', jpg_dir: str = f'resources{os.path.sep}predictOnThem', save_jpg: bool = False) -> list:

    images: np.array = []
    image_names = []
    files = os.listdir(raw_dir)

    if save_jpg and not os.path.exists(jpg_dir):
        os.mkdir(jpg_dir)

    convertedImageSize = [133, 200]

    for file in files:
        name, ext = file.split('.')
        if ext != 'xmp':
            with rawpy.imread(f'{raw_dir}{os.path.sep}{file}') as rawImg:
                rgbImg = rawImg.postprocess(rawpy.Params(use_camera_wb=True))
                rgbNormed = rgbImg / 255.0
                bilinear_img = tf.image.resize(
                    rgbNormed, convertedImageSize, method=ResizeMethod.BILINEAR)
                if save_jpg:
                    jpgName = f'{jpg_dir}{os.path.sep}{name}.jpg'
                    tf.keras.utils.save_img(jpgName, bilinear_img)
                images.append(bilinear_img)
                image_names.append(name)

    print('preparePredict')

    return images, image_names


def load_models(path: str = 'models', file_format: str = 'h5'):
    print(f'load models from: {path}')
    model_files = os.listdir(path)
    num_models = len(model_files)
    models = []
    for i in range(num_models):
        model = load_model(f'{path}{os.path.sep}model_{i}.{file_format}')
        models.append(model)
    return models


def predict_on_models(images, models: list):
    all_predictions = []
    for img in images:
        img_predictions = []
        img = tf.expand_dims(img, axis=0)
        for m in models:
            # print(img.shape)
            m_res = m.predict(img)
            m_res = np.ndarray.flatten(m_res)[0]
            img_predictions.append(m_res)
        # print(img_predictions)
        all_predictions.append(img_predictions)

    return all_predictions


def gen_batch_xmps(target_dir:str, img_names_and_predictions):
    for pic_name, pred in img_names_and_predictions:
        corr = Correction(pred)
        generateXmpResult(target_dir, pictureName=pic_name, correction=corr)
    

def move_results(source: str, target: str) -> None:
    allfiles = os.listdir(source)
    
    for file in allfiles:
        target_name = target + os.path.sep + file
        
        if os.path.exists(target_name):
            print(target_name,'exists in the destination path - overwrite!')
            os.remove(target_name)
            
        shutil.move(os.path.join(source, file), target)

    print('move done')
    

def open_lr(lr_path:str):
    subprocess.run([lr_path])


if __name__ == '__main__':
    assert (pd.__version__ == '1.3.5')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('Tensorflow version: ' + tf.__version__)

    raw_path = "resources\\raws"
    lr_target_dir = 'resources\\auto_imp_dir'
    lr_executable_path = 'C:\\Program Files\\Adobe\\Adobe Lightroom Classic\\Lightroom.exe'
    rawBigger_path = f'resources{os.path.sep}rawTest'

    nn_models = load_models()
    # imgs, xmps = data_load_and_preprocess(directory=rawBigger_path)
    # saveDatasets(imgs, xmps)

    images, image_names = prepareToPrediction(raw_path)
    preds = predict_on_models(images, nn_models)
    pred_result = list(zip(image_names, preds))
    gen_batch_xmps(raw_path, pred_result)
    
    move_results(raw_path, lr_target_dir)
    
    open_lr(lr_executable_path)
    
    # print(pred_result)

    # cleanData(path)
    # imagesWithMeta = list(readData(path))
    # tr, val, tst = convert(imagesWithMeta)

    # imgs, xmps = data_load_and_preprocess(directory=rawBigger_path)
    # saveDatasets(imgs, xmps)

    # print(val)
    # rgbNormed = readRaw()
    # xmpTest = generateXmpResult()
    # rawTest = readRawSimple()

    # item = imagesWithMeta[0]
    print('run ended')
