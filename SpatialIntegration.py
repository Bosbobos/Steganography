import ImageWrapper as Img
from math import ceil, floor
from numpy import int16


def pvd_encode(imgPath, msg):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    binMsg = Img.string_to_binary(msg)
    print(binMsg)
    cnt = 0
    for i in range(height - 1):
        for j in range(0, width - 2, 2):
            n = 3
            diff = int16(img[i][j][0]) - int16(img[i][j+1][0])
            while abs(diff) >= 2**n: n += 1
            l = 0 if n == 3 else 2**(n-1)
            binM = binMsg[cnt:min(cnt+n, len(binMsg))]
            m = int(binM, 2) if binM != '' else 0
            cnt += n
            d = l + m if diff >= 0 else -(l + m)

            if d%2 != 0:
                img[i][j][0] += ceil((d - diff)/2)
                img[i][j+1][0] -= floor((d - diff)/2)
            else:
                img[i][j][0] += floor((d - diff)/2)
                img[i][j+1][0] -= ceil((d - diff)/2)

            print(img[i][j][0] - img[i][j+1][0])
            if cnt + 1 >= len(binMsg):
                Img.array_to_image(img)
                return


def pvd_decode(imgPath, size=20):
    img = Img.image_to_array(imgPath)
    width, height, _ = img.shape
    binMsg = ''
    for i in range(height - 1):
        for j in range(0, width - 2, 2):
            n = 3
            diff = int16(img[i][j][0]) - int16(img[i][j+1][0])
            print(diff, end = ' ')
            while abs(diff) >= 2**n: n += 1
            n -= 1
            l = 0 if n == 3 else 2**n

            m = (diff % 256) - l
            print(n, m, bin(m))
            binMsg += f'{m:0{n}b}'
            if len(binMsg) >= size:
                return Img.binary_to_string(binMsg[:size])


pvd_encode('4.bmp', 'hello world!')
print(pvd_decode('result.bmp', 96))
