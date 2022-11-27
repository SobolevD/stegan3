import numpy as np

from lab2.task import do_embedding, get_rho_for_image
from lab2.utils.in_out import read_image, write_image
from utils.distortion import cut, scale, smooth, jpeg, cut_bulk, scale_bulk, smooth_bulk, jpeg_bulk, cut_and_jpeg_bulk
from matplotlib import pyplot as plt

if __name__ == '__main__':

    container = read_image('resources/barb.tif')
    H_zone, watermark, image = do_embedding(container)#read_image('resources/black.png')#

    rho = get_rho_for_image(H_zone, watermark, image)
    print(f'Original rho: {rho}')

    cut_images = cut_bulk(image, container, 0.2, 0.9, 0.1)
    scale_images = scale_bulk(image, 0.55, 1.45, 0.15)
    smooth_images = smooth_bulk(image, 3, 15, 2)
    jpeg_images = jpeg_bulk(image, 30, 90, 10)

    cut_rhos = []
    for i in range(0, cut_images.shape[0]):
        cut_rhos.append(get_rho_for_image(H_zone, watermark, cut_images[i]))

    plt.title('Rhos (cut)')
    x = np.arange(0.2, 1.0, 0.1)
    plt.plot(x, cut_rhos)
    plt.show()

    scale_rhos = []
    for i in range(0, scale_images.shape[0]):
        scale_rhos.append(get_rho_for_image(H_zone, watermark, scale_images[i]))

    plt.title('Rhos (scale)')
    x = np.arange(0.55, 1.5, 0.15)
    plt.plot(x, scale_rhos)
    plt.show()

    smooth_rhos = []
    for i in range(0, smooth_images.shape[0]):
        smooth_rhos.append(get_rho_for_image(H_zone, watermark, smooth_images[i]))

    plt.title('Rhos (smooth)')
    x = np.arange(3, 17, 2)
    plt.plot(x, smooth_rhos)
    plt.show()

    jpeg_rhos = []
    for i in range(0, jpeg_images.shape[0]):
        jpeg_rhos.append(get_rho_for_image(H_zone, watermark, jpeg_images[i]))

    plt.title('Rhos (jpeg)')
    x = np.arange(30, 91, 10)
    plt.plot(x, jpeg_rhos)
    plt.show()

    cut_and_jpeg = cut_and_jpeg_bulk(image, container, 0.2, 0.9, 0.1, 30, 90, 10)
    a = 3




