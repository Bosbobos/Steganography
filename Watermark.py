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
    if abs(dc) > 1000 or abs(dc) < 1:
        mod_power = abs(2 * med)
    else:
        mod_power = abs(2 * (dc - med) / dc)
    delta = block_dct[3, 3] - next_dct[3, 3]

    t, k = 80, 12
    if wm_bit == 1:
        if delta > t - k:
            while delta > t - k:
                block_dct -= mod_power
                delta = block_dct[3, 3] - next_dct[3, 3]
        elif k > delta > -t/2:
            while delta < k:
                block_dct += mod_power
                delta = block_dct[3, 3] - next_dct[3, 3]
        else:
            while delta > -t - k:
                block_dct -= mod_power
                delta = block_dct[3, 3] - next_dct[3, 3]
    else:
        if delta > t/2:
            while delta <= t+k:
                block_dct += mod_power
                delta = block_dct[3, 3] - next_dct[3, 3]
        elif -k < delta < t/2:
            while delta >= -k:
                block_dct -= mod_power
                delta = block_dct[3, 3] - next_dct[3, 3]
        else:
            while delta <= k-t:
                block_dct += mod_power
                delta = block_dct[3, 3] - next_dct[3, 3]

def DKP_difference(imgPath, watermarkPath):
    img = np.int16(Img.image_to_array(imgPath)[:, :, 0]) - 128
    width, height = img.shape
    wm = Img.binarize_image(watermarkPath)
    wm = arnold(wm).flatten()
    z = 0
    for i in range(0, height, 8):
        for j in range(8, width-8, 8):
            block = img[i:i+8, j:j+8]
            next = img[i:i+8, j+8:j+16]
            block_dct = dct(block, norm="ortho")
            next_dct = dct(next, norm="ortho")

            embed_into_block_dct(block_dct, next_dct, wm[z])

            img[i:i+8, j:j+8] = dct(next_dct, norm="ortho", type=3)
            z += 1

    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = 'output/' + imgPath[i:]
    Img.array_to_image(img, path)

DKP_difference('input/newest.png', 'input/64x64.png')
