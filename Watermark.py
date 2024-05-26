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
    #wm = arnold(wm).flatten()
    wm = wm.flatten()
    z = 0
    for block_i in range(height // 8):
        for block_j in range(width // 8 - 1):
            block = [[img[block_i * 8 + i][block_j * 8 + j][0] - 128 for j in range(8)] for i in range(8)]
            next = [[img[block_i * 8 + i][block_j * 8 + j + 8][0] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            next_dct = dct(dct(next, axis=0, norm='ortho'), axis=1, norm='ortho')

            block_dct = embed_into_block_dct(block_dct, next_dct, wm[z])

            #unedited_image = dct(block_dct, norm="ortho", type=3)

            block_idct = idct(idct(block_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
            for i in range(8):
                for j in range(8):
                    g = max(min(int(block_idct[i][j]) + 128, 255), 0)
                    img[block_i * 8 + i][block_j * 8 + j][1] = g
            z += 1

    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = 'output/' + imgPath[i:]
    Img.array_to_image(img, path)

def embed_watermark(matrix: np.array, wm):
    width, height = matrix.shape[1], matrix.shape[0]
    wm_bits = Img.binarize_image(wm).flatten()
    num_blocks_x = width // 8
    num_blocks_y = height // 8
    index_of_px_in_wm = 0
    for block_i in range(num_blocks_y):
        for block_j in range(num_blocks_x):
            block = [[matrix[block_i * 8 + i][block_j * 8 + j][1] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
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
            if block_j < num_blocks_x - 1:
                neighbor_block = [[matrix[block_i * 8 + i][block_j * 8 + j + 8][1] for j in range(8)] for i in range(8)]
                neighbor_block_dct = dct(dct(neighbor_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                T, K = 80, 12
                if index_of_px_in_wm < len(wm_bits):
                    watermark_bit = wm_bits[index_of_px_in_wm]
                    delta = block_dct[4][4] - neighbor_block_dct[4][4]
                    if watermark_bit == 1:
                        if delta > T - K:
                            while delta > T - K:
                                block_dct[4][4] -= M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif K > delta > -T / 2:
                            while delta < K:
                                block_dct[4][4] += M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif delta < -T / 2:
                            while delta > -T - K:
                                block_dct[4][4] -= M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                    else:
                        if delta > T / 2:
                            while delta <= T + K:
                                block_dct[4][4] += M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif -K < delta < T / 2:
                            while delta >= -K:
                                block_dct[4][4] -= M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif delta < K - T:
                            while delta <= K - T:
                                block_dct[4][4] += M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                    index_of_px_in_wm += 1
                block_idct = idct(idct(block_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
                for i in range(8):
                    for j in range(8):
                        g = max(min(int(block_idct[i][j]) + 128, 255), 0)
                        matrix[block_i * 8 + i][block_j * 8 + j][1] = g
    return matrix



def extract_from_block_dct(block_dct, next_dct):
    delta = block_dct[4, 4] - next_dct[4, 4]
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
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            next_dct = dct(dct(next, axis=0, norm='ortho'), axis=1, norm='ortho')

            wm[z] = extract_from_block_dct(block_dct, next_dct)
            z += 1

    #watermark = reverse_arnold(wm.reshape(width//8, height//8)) * 255
    watermark = wm.reshape(width//8, height//8) * 255
    i = len(imgPath) - 1 - imgPath[::-1].index('/') if '/' in imgPath else 0  #Finds file name without directories
    path = ''.join(['output/', 'WM', imgPath[i:]])
    Img.array_to_image(watermark, path, True)

def extract(matrix):
    width, height = matrix.shape[1], matrix.shape[0]
    num_blocks_x = width // 8
    num_blocks_y = height // 8
    wm_bits = []
    T = 80
    for block_i in range(num_blocks_y):
        for block_j in range(num_blocks_x - 1):
            block = [[matrix[block_i * 8 + i][block_j * 8 + j][1] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            delta = block_dct[4][4]
            wm_bits.append(1 if delta < -T or (0 < delta < T) else 0)
    wm_matrix = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_bits)):
        wm_matrix[i // 64][i % 64] = wm_bits[i]
    wm_matrix_img = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            if wm_matrix[i][j] == 1:
                wm_matrix_img[i][j] = [255, 255, 255, 255]
            else:
                wm_matrix_img[i][j] = [0, 0, 0, 255]
    out = np.array(wm_matrix_img).astype("uint8")
    return out

#DKP_difference('input/Dima.png', 'input/watermark.png')
matrix = embed_watermark(Img.image_to_array('input/newest_calibrated.png'), 'input/wm.png')
Img.array_to_image(matrix, 'output/DimaAlg.png')
wm = extract(Img.image_to_array('output/DimaAlg.png'))
Img.array_to_image(wm, 'output/WM/GodHelpMe.png')
