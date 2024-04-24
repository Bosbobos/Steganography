import ImageWrapper as Img
from math import ceil, floor, log2
from numpy import int16


def pvd_encode(imgPath, msg):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    binMsg = Img.string_to_binary(msg+'a') # Extra letter because it's always wrong
    print(binMsg)
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
            print(l, end=' ')
            print(int16(img[i][j][0]) - int16(img[i][j+1][0]), end=' ')
            print(m)
            if cnt + 1 >= len(binMsg):
                Img.array_to_image(img)
                return


def pvd_decode(imgPath, size=200):
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
            if len(binMsg) >= size:
                return Img.binary_to_string(binMsg[:size])


pvd_encode('4.bmp', '150')
print('--------------------')
print(pvd_decode('result.bmp', 150))