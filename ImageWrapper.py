from PIL import Image
import numpy as np


def string_to_binary(text):
    print(' '.join(list(str(ord(i)) for i in text)))
    return ''.join(format(ord(i), '08b') for i in text)


def binary_to_string(binary):
    print(binary)
    return ''.join(chr(int(binary[i:i+8],2)) for i in range(0, len(binary), 8))


def image_to_array(path):
    with Image.open(path) as img:
        return np.array(img)


def array_to_image(arr):
    img = Image.fromarray(arr)
    img.save('result.bmp')
