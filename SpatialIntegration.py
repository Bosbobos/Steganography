import ImageManager as Img
import TextManager as Txt
from math import ceil, floor, log2
from numpy import int16


def calibrate_image(imgPath, calibr_const):
    Img.calibrate_image(imgPath, calibr_const)


def pvd_encode(imgPath, msg):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    binMsg = Txt.string_to_binary(msg) + '0' * 16
    cnt = 0
    for i in range(height - 1):
        for j in range(0, width - 2, 2):
            diff = int16(img[i][j][0]) - int16(img[i][j+1][0])
            n = floor(log2(abs(diff))) if diff != 0 else 0
            if n < 3: l, n = 0, 3
            else: l = 2**n
            binM = binMsg[cnt:min(cnt+n, len(binMsg))]
            m = int(binM, 2) if binM != '' else 0
            cnt += n
            d = l + m if diff >= 0 else -(l + m)

            if diff % 2 != 0:
                img[i][j][0] += ceil((d - diff)/2)
                img[i][j+1][0] -= floor((d - diff)/2)
            else:
                img[i][j][0] += floor((d - diff)/2)
                img[i][j+1][0] -= ceil((d - diff)/2)

            if cnt + 1 >= len(binMsg):
                i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
                path = 'output/' + imgPath[i:]
                Img.array_to_image(img, path)
                return

    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = 'output/' + imgPath[i:]
    Img.array_to_image(img, path)


def pvd_decode(imgPath):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    binMsg = ''
    for i in range(height - 1):
        for j in range(0, width - 2, 2):
            diff = int16(img[i][j][0]) - int16(img[i][j+1][0])
            n = floor(log2(abs(diff))) if diff != 0 else 0
            if n < 3: l, n = 0, 3
            else: l = 2**n

            m = abs(diff) - l
            binMsg += f'{m:0{n}b}'
            if binMsg[-16:] == '0' * 16:
                return Txt.binary_to_string(binMsg[:-16])

    return Txt.binary_to_string(binMsg)
