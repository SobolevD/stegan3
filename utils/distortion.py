import numpy as np
import cv2
from PIL import Image
from skimage.io import imsave, imread
import pandas as pd

from lab2.task import get_rho_for_image
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

    images = np.array(cut_images)
    for i in range(0, images.shape[0]):
        write_image(images[i], f'resources/cut/cut_{i}.png')
    return images


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

    images = np.array(scale_images).astype('int')
    for i in range(0, images.shape[0]):
        write_image(images[i], f'resources/scale/scale_{i}.png')
    return images

def smooth(image, M=3):
    return cv2.blur(image, (M, M))


def smooth_bulk(image, p_min, p_max, p_delta):
    smooth_images = []
    items_count = int(np.round((p_max-p_min) / p_delta)) + 1

    p_current = p_min
    for i in range(0, items_count):
        smooth_images.append(smooth(image, p_current))
        p_current += p_delta

    images = np.array(smooth_images)
    for i in range(0, images.shape[0]):
        write_image(images[i], f'resources/smooth/smooth_{i}.png')
    return images

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

    images = np.array(jpeg_images)
    for i in range(0, images.shape[0]):
        write_image(images[i], f'resources/jpeg/jpeg_{i}.png')
    return images

def cut_and_jpeg_bulk(image, container, p1_min, p1_max, p1_delta, p2_min, p2_max, p2_delta, H_zone, watermark):
    cut_and_jpeg_images = []
    items_count_cut = int(np.round((p1_max - p1_min) / p1_delta)) + 1
    items_count_jpeg = int(np.round((p2_max - p2_min) / p2_delta)) + 1

    p1_current = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p1_ind = p1_current.copy()
    p2_current = p2_min
    p2 = []
    rho = np.zeros(shape=(7, 8))
    for i in range(0, items_count_jpeg):
        for j in range(0, items_count_cut):

            P1 = p1_ind[j]
            cut_image = cut(image, container, p1_ind[j])
            cut_and_jpeg_image = jpeg(cut_image, p2_current)
            cut_and_jpeg_images.append(cut_and_jpeg_image)
            # rho.append(get_rho_for_image(H_zone, watermark, cut_and_jpeg_image))
            rho[i][j] = get_rho_for_image(H_zone, watermark, cut_and_jpeg_image)
        p2.append(p2_current)
        p2_current += p2_delta

    df = pd.DataFrame(rho, columns=p1_current, index = p2)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): # more options can be specified also
        print(df)

    return np.array(cut_and_jpeg_images)
