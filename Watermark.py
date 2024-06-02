import ImageManager as Img
import numpy as np
from scipy.fft import dct, idct


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
    if abs(dc) > 1000 or abs(dc) < 1:
        mod_power = abs(2 * med)
    else:
        mod_power = abs(2 * diff / dc)

    M = max(0.01, mod_power)
    T, K = 80, 12
    delta = block_dct[4][4] - next_dct[4][4]
    if wm_bit == 1:
        if delta > T - K:
            while delta > T - K:
                block_dct[4][4] -= M
                delta = block_dct[4][4] - next_dct[4][4]
        elif K > delta > -T / 2:
            while delta < K:
                block_dct[4][4] += M
                delta = block_dct[4][4] - next_dct[4][4]
        elif delta < -T / 2:
            while delta > -T - K:
                block_dct[4][4] -= M
                delta = block_dct[4][4] - next_dct[4][4]
    else:
        if delta > T / 2:
            while delta <= T + K:
                block_dct[4][4] += M
                delta = block_dct[4][4] - next_dct[4][4]
        elif -K < delta < T / 2:
            while delta >= -K:
                block_dct[4][4] -= M
                delta = block_dct[4][4] - next_dct[4][4]
        elif delta < K - T:
            while delta <= K - T:
                block_dct[4][4] += M
                delta = block_dct[4][4] - next_dct[4][4]

    return block_dct


def DKP_difference(imgPath, watermarkPath):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    wm = Img.binarize_image(watermarkPath)
    wm = arnold(wm).flatten()
    wm = wm.flatten()
    z = 0
    for block_i in range(height // 8):
        for block_j in range(width // 8 - 1):
            block = [[img[block_i * 8 + i][block_j * 8 + j][1] - 128 for j in range(8)] for i in range(8)]
            next = [[img[block_i * 8 + i][block_j * 8 + j + 8][1] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            next_dct = dct(dct(next, axis=0, norm='ortho'), axis=1, norm='ortho')

            block_dct = embed_into_block_dct(block_dct, next_dct, wm[z])
            block_idct = idct(idct(block_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
            for i in range(8):
                for j in range(8):
                    g = max(min(int(block_idct[i][j]) + 128, 255), 0)
                    img[block_i * 8 + i][block_j * 8 + j][1] = g
            z += 1

    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = 'output/' + imgPath[i:]
    Img.array_to_image(img, path)


def DKP_difference_extract(imgPath):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    wm = np.zeros(width*height//64, dtype=np.uint8)
    z = 0
    T = 80
    for block_i in range(height // 8):
        for block_j in range(width // 8 - 1):
            block = [[img[block_i * 8 + i][block_j * 8 + j][1] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            delta = block_dct[4][4]
            wm[z] = 1 if delta < -T or (0 < delta < T) else 0
            z += 1

    watermark = reverse_arnold(wm.reshape(width//8, height//8)) * 255
    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = ''.join(['output/', 'WM', imgPath[i:]])
    Img.array_to_image(watermark, path, True)
