from lab2.task import do_embedding
from lab2.utils.in_out import read_image, write_image
from utils.distortion import cut, scale, smooth

if __name__ == '__main__':

    container = read_image('resources/barb.tif')
    image = do_embedding(container)#read_image('resources/black.png')#

    #result1 = cut(image, container)
    #write_image(result1, 'resources/tmp1.png')

    # result2 = scale(image, 2)
    # write_image(result2, 'resources/tmp2.png')

    result3 = smooth(image)
    write_image(result3, 'resources/tmp3.png')
    a = 3

