import numpy as np
from PIL import Image
from skimage.io import imsave, imread

from lab2.utils.in_out import write_image, read_image


def cut(image, container, v=0.25):
    new_shape = image.shape * np.sqrt(v)
    image_part = image[0:new_shape[0], 0:new_shape[1]]

    result_image = container.copy()

    result_image[0:new_shape[0], 0:new_shape[1]] = image_part
    return result_image


def scale(image, k=0.25):
    zeros = np.zeros(image.shape)

    new_shape = image.shape * np.sqrt(k)

    write_image(image, 'resources/tmp.png')
    img = Image.open('resources/tmp.png')

    # изменяем размер
    new_image = img.resize((new_shape.shape[0], new_shape.shape[1]))
    new_image.show()

    # сохранение картинки
    new_image.save('resources/tmp.png')

    scaled_image = read_image('resources/tmp.png')

    result_image = ''
    if scaled_image.shape[0] > image.shape[0]:
        result_image = scaled_image[0:image.shape[0], 0:image.shape[1]]
        return result_image.copy()

    zeros[0:new_shape[0], 0:new_shape[1]] = scaled_image
    result_image = zeros

    return result_image


def smooth(image, M=3):
    N = image.shape
    g = (1. / (M ** 2)) * np.ones((M, M))

    image_copy = image.copy()
    for i in range(0, N[0] - M):
        for j in range(0, N[1] - M):
            image_copy[i:i+M, j:j+M] = g * image_copy[i:i+M, j:j+M]

    return image_copy


def jpeg(image, quality=0.95):
    imsave('resources/bridge.jpeg', image, quality=quality)
    jpeg_image = imread('resources/bridge.jpeg')
    return jpeg_image