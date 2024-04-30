import ImageManager as Img
from math import floor, log2, log10
from numpy import int16, histogram
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt


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


def mse(path1, path2):
    img1, img2 = Img.image_to_array(path1), Img.image_to_array(path2)
    width, height, _ = img1.shape
    summ = 0
    for i in range(height - 1):
        for j in range(width - 1):
            diff = int16(img1[i][j][0]) - int16(img2[i][j][0])
            summ += diff**2

    summ /= width * height
    return summ


def rmse(MSE):
    return MSE ** 0.5


def psnr(MSE):
    return 10 * log10(255**2 / MSE)

def ssim(path1, path2):
    img1, img2 = Img.image_to_array(path1), Img.image_to_array(path2)
    return structural_similarity(img1, img2, channel_axis=-1, data_range=255)


def plot_red_histogram(path):
    img = Img.image_to_array(path)
    red_channel = img[:, :, 0]

    # вычисление гистограммы
    hstm, bin_edges = histogram(red_channel, bins=256, range=(0,256))

    # построение гистограммы
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hstm, width=1, color='red')
    plt.xlim([0, 256])
    plt.title('Red pixels histogram ' + path)
    plt.xlabel('Pixel value (red channel)')
    plt.ylabel('Number of pixels')
    plt.show()


def plot_mean_histogram(path):
    img = Img.image_to_array(path)
    mean_channel = img.mean(axis=2)

    hstm, bin_edges = histogram(mean_channel, bins=256, range=(0,256))

    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hstm, width=1, color='gray')
    plt.xlim([0, 256])
    plt.title('Average value of pixels histogram ' + path)
    plt.xlabel('Average pixel value')
    plt.ylabel('Number of pixels')
    plt.show()

if __name__ == '__main__':
    path1, path2 = 'input/newest_calibrated.png', 'output/newest_calibrated.png'
    MSE,  SSIM = mse(path1, path2), ssim(path1, path2)
    RMSE, PSNR = rmse(MSE), psnr(MSE)
    print(f'MSE: {MSE}')
    print(f'RMSE: {RMSE}')
    print(f'PSNR: {PSNR}')
    print(f'SSIM: {SSIM}')

    plot_mean_histogram(path1)
    plot_mean_histogram(path2)
