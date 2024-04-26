import ImageWrapper as Img
from math import floor, log2
from numpy import int16

def find_max_size(imgPath):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    cnt = 0
    for i in range(height - 1):
        for j in range(0, width - 2, 2):
            diff = int16(img[i][j][0]) - int16(img[i][j+1][0])
            n = floor(log2(abs(diff))) if diff != 0 else 0
            if n < 3: n = 3
            cnt += n

    return cnt


print(find_max_size('input/lasttry.png'))


def find_bin_text_length(text):
    return len(Img.string_to_binary(text))
while 1:
    print(find_bin_text_length(input()))