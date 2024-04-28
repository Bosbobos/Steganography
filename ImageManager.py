from PIL import Image
import numpy as np


def image_to_array(path):
    with Image.open(path) as img:
        return np.array(img)


def array_to_image(arr, path):
    img = Image.fromarray(arr)
    img.save(path)


def calibrate_image(imgPath, calibr_const=3):
    matrix = image_to_array(imgPath)

    calibr_value = 10 * calibr_const
    for line in range(len(matrix)):
        for row in range(len(matrix[line])):
            for channel in range(3):
                matrix[line][row][channel] = max(calibr_value, min(255 - calibr_value, matrix[line][row][channel]))

    Image.fromarray(matrix).save(imgPath[:-4] + '_calibrated' + imgPath[-4:])
