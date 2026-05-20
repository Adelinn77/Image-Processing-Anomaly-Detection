import numpy as np
import cv2

def convert_to_grayscale(image):
    # image is a 3D array in a [height, width, channels(Blue Green Red)] format 

    b = image[:, :, 0]  # all the values of the blue channel
    g = image[:, :, 1]  # all the values of the green channel
    r = image[:, :, 2]  # all the values of the red channel

    grayscale_image =  b/3 + g/3 + r/3 

    return grayscale_image.astype(np.ubyte)


def convert_to_grayscale_using_luminosity(image):
    # image is a 3D array in a [height, width, channels(Blue Green Red)] format 

    b = image[:, :, 0]  # all the values of the blue channel
    g = image[:, :, 1]  # all the values of the green channel
    r = image[:, :, 2]  # all the values of the red channel

    grayscale_image = 0.299 * r + 0.587 * g + 0.114 * b

    return grayscale_image.astype(np.ubyte)