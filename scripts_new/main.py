
import cv2
import os

import tensorflow as tf
ResizeMethod = tf.image.ResizeMethod


def load_images_from_folder(folder):
    """
    loads images
    :param folder: filename in string
    :return: list of images in dim. [x * y * z]
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        withoutExtTag = filename.replace('.jpg', '')
        if img is not None:
            img = img / 255
            compressed = tf.image.adjust_jpeg_quality(img, 85)
            new_size = [900, 600]
            
            bilin = tf.image.resize(compressed, new_size, method=ResizeMethod.BILINEAR, preserve_aspect_ratio=True).numpy()
            lanczos1 = tf.image.resize(compressed, new_size, method=ResizeMethod.LANCZOS3, preserve_aspect_ratio=True).numpy()
            gaussian = tf.image.resize(compressed, new_size, method=ResizeMethod.GAUSSIAN, preserve_aspect_ratio=True).numpy()
            
            
            
            cv2.imwrite(filename=withoutExtTag + '_bilin' + '.jpg', img=bilin*255)
            cv2.imwrite(filename=withoutExtTag + '_lanczos' + '.jpg', img=lanczos1*255)
            cv2.imwrite(filename=withoutExtTag + '_gaussian' + '.jpg', img=gaussian*255)
            

            images.append(img)
    return images


if __name__ == '__main__':
    path = "resources\\testJpgs"
    images = load_images_from_folder(path)
