
import pandas as pd
import os
import string

# image processing
import cv2

from exiftool import ExifToolHelper

# TensorFlow
import tensorflow as tf
import tensorflow_datasets as tfds
ResizeMethod = tf.image.ResizeMethod

from sklearn.utils import shuffle

# TODO good to check:
# scikit-image - for image resize, compress
#


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


def load_images_from_folder(folder: string):
    """
    loads images
    :param folder: filename in string
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
                new_size = [600, 900]
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
    
    train_ds = df.take([0,train_size])
    val_ds = df.drop([0, train_size]).take([0, val_size])
    test_ds = df.drop([0,train_size+val_size])

    return train_ds, val_ds, test_ds


def convert(images):
    df = pd.DataFrame(images)
    df.to_pickle('dataframe.pkl')
    # print(df)
    # print(df.describe())
    return get_dataset_partitions(df)

def readFromPkl(filename: string):
    df = pd.read_pickle(filename)
    return get_dataset_partitions(df)




if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # path = "resources\\testJpgs"
    # images = load_images_from_folder(path)
    # tr, val, tst = convert(images)
    tr, val, tst = readFromPkl('dataframe.pkl')
    
    print('done')
