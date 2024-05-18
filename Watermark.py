import ImageManager as Img
import numpy as np
from scipy.fft import dct


def arnold(matrix):
    m, n = matrix.shape
    new_matrix = matrix.copy()
    k = np.array([[1, 1], [1, 2]])
    for i in range(m):
        for j in range(m):
            ij = np.array([[i],[j]])
            new_ij = np.matmul(k, ij)
            new_i, new_j = new_ij[0][0], new_ij[1][0]
            new_matrix[new_i % m][new_j % m] = matrix[i][j]

    return new_matrix


def reverse_arnold(matrix):
    m, n = matrix.shape
    new_matrix = matrix.copy()
    k = np.array([[2, -1], [-1, 1]])
    for i in range(m):
        for j in range(m):
            ij = np.array([[i],[j]])
            new_ij = np.matmul(k, ij)
            new_i, new_j = new_ij[0][0], new_ij[1][0]
            new_matrix[new_i % m][new_j % m] = matrix[i][j]

    return new_matrix


def embed_into_block_dct(block_dct, next_dct, wm_bit):
    # Find the median of first 9 coefficients starting from 0 and going zigzag
    # For visual representation find picture 1 in corresponding report
    ac = []
    for k in range(4):
        for l in range(4-k):
            if not k == l == 0: ac.append(block_dct[k][4-l])
    med = np.median(ac)
    dc = block_dct[0, 0]
    diff = abs(dc - med)
    if (abs(dc) > 1000 or abs(dc) < 1) and med != 0.0 or diff > 0.0001:
        mod_power = abs(2 * med)
    else:
        mod_power = abs(2 * diff / dc)

    next = next_dct[3, 3]
    delta = block_dct[3, 3] - next

    t, k = 80, 12
    if wm_bit == 1:
        if delta > t - k:
            while delta > t - k:
                block_dct[3, 3] -= mod_power
                delta = block_dct[3, 3] - next
        elif k > delta > -t/2:
            while delta < k:
                block_dct[3, 3] += mod_power
                delta = block_dct[3, 3] - next
        elif delta < -t/2:
            while delta > -t - k:
                block_dct[3, 3] -= mod_power
                delta = block_dct[3, 3] - next
    else:
        if delta > t/2:
            while delta <= t+k:
                block_dct[3, 3] += mod_power
                delta = block_dct[3, 3] - next
        elif -k < delta < t/2:
            while delta >= -k:
                block_dct[3, 3] -= mod_power
                delta = block_dct[3, 3] - next
        elif delta < k-t:
            while delta <= k-t:
                block_dct[3, 3] += mod_power
                delta = block_dct[3, 3] - next

def DKP_difference(imgPath, watermarkPath):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    wm = Img.binarize_image(watermarkPath)
    #wm = arnold(wm).flatten()
    wm = wm.flatten()
    z = 0
    for i in range(0, height, 8):
        for j in range(0, width-8, 8):
            block = np.int16(img[i:i+8, j:j+8, 0]) - 128
            next = np.int16(img[i:i+8, j+8:j+16, 0]) - 128
            block_dct = dct(block, norm="ortho")
            next_dct = dct(next, norm="ortho")

            embed_into_block_dct(block_dct, next_dct, wm[z])

            #unedited_image = dct(block_dct, norm="ortho", type=3)



            img[i:i+8, j:j+8, 0] = np.uint8(dct(block_dct, norm="ortho", type=3) + 128)
            z += 1

    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = 'output/' + imgPath[i:]
    Img.array_to_image(img, path)


def extract_from_block_dct(block_dct, next_dct):
    delta = block_dct[3, 3] - next_dct[3, 3]
    t, k = 80, 12
    if delta < -t or 0 < delta < t:
        return 1

    return 0


def DKP_difference_extract(imgPath):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    wm = np.zeros(width*height//64, dtype=np.uint8)
    z = 0
    for i in range(0, height, 8):
        for j in range(8, width-8, 8):
            block = np.int16(img[i:i+8, j:j+8, 0]) - 128
            next = np.int16(img[i:i+8, j+8:j+16, 0]) - 128
            block_dct = dct(block, norm="ortho")
            next_dct = dct(next, norm="ortho")

            wm[z] = extract_from_block_dct(block_dct, next_dct)
            z += 1

    #watermark = reverse_arnold(wm.reshape(width//8, height//8)) * 255
    watermark = wm.reshape(width//8, height//8) * 255
    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = ''.join(['output/', 'WM', imgPath[i:]])
    Img.array_to_image(watermark, path, True)

from SpatialIntegration import calibrate_image as ci
ci('input/newest.png', 8)
DKP_difference('output/newest.png', 'input/wm.png')
DKP_difference_extract('output/newest.png')
