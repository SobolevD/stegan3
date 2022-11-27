import numpy as np
import cv2
from PIL import Image
from skimage.io import imsave, imread

from lab2.utils.in_out import write_image, read_image


def cut(image, container, v=0.25):
    new_shape = (np.array(image.shape) * np.sqrt(v)).astype('int')
    image_part = image[0:new_shape[0], 0:new_shape[1]]

    result_image = container.copy()

    result_image[0:new_shape[0], 0:new_shape[1]] = image_part
    return result_image


def cut_bulk(image, container, p_min, p_max, p_delta):
    cut_images = []
    items_count = int(np.round((p_max-p_min) / p_delta)) + 1
    p_current = p_min
    for i in range(0, items_count):
        cut_images.append(cut(image, container, p_current))
        p_current += p_delta

    return np.array(cut_images)


def scale(image, k=0.25):

    new_shape = (np.array(image.shape) * k).astype('int')

    write_image(image, 'resources/tmp.png')
    img = Image.open('resources/tmp.png')

    # изменяем размер
    new_image = img.resize((new_shape[0], new_shape[1]))

    # сохранение картинки
    new_image.save('resources/tmp.png')

    scaled_image = read_image('resources/tmp.png')

    result_image = ''
    if scaled_image.shape[0] > image.shape[0]:
        result_image = scaled_image[0:image.shape[0], 0:image.shape[1]]
        return result_image.copy()

    zeros = np.zeros(image.shape)
    zeros[0:new_shape[0], 0:new_shape[1]] = scaled_image
    result_image = zeros

    return result_image


def scale_bulk(image, p_min, p_max, p_delta):
    scale_images = []
    items_count = int(np.round((p_max-p_min) / p_delta)) + 1
    p_current = p_min
    for i in range(0, items_count):
        scale_images.append(scale(image, p_current))
        p_current += p_delta

    return np.array(scale_images).astype('int')


def smooth(image, M=3):
    return cv2.blur(image, (M, M))


def smooth_bulk(image, p_min, p_max, p_delta):
    smooth_images = []
    items_count = int(np.round((p_max-p_min) / p_delta)) + 1

    p_current = p_min
    for i in range(0, items_count):
        smooth_images.append(smooth(image, p_current))
        p_current += p_delta

    return np.array(smooth_images)


def jpeg(image, quality=85):
    imsave('resources/barb.jpeg', image, quality=quality)
    jpeg_image = imread('resources/barb.jpeg')
    return jpeg_image


def jpeg_bulk(image, p_min, p_max, p_delta):
    jpeg_images = []
    items_count = int(np.round((p_max-p_min) / p_delta)) + 1

    p_current = p_min
    for i in range(0, items_count):
        jpeg_images.append(jpeg(image, p_current))
        p_current += p_delta

    return np.array(jpeg_images)


def cut_and_jpeg_bulk(image, container, p1_min, p1_max, p1_delta, p2_min, p2_max, p2_delta):
    cut_and_jpeg_images = []
    items_count_cut = int(np.round((p1_max-p1_min) / p1_delta)) + 1
    items_count_jpeg = int(np.round((p2_max-p2_min) / p2_delta)) + 1

    p1_current = p1_min
    p2_current = p2_min
    for i in range(0, items_count_jpeg):
        for j in range(0, items_count_cut):
            cut_image = cut(image, container, p1_current)
            cut_and_jpeg_image = jpeg(cut_image, p2_current)
            cut_and_jpeg_images.append(cut_and_jpeg_image)
            p1_current += p1_delta
        p2_current += p2_delta
    return np.array(cut_and_jpeg_images)
