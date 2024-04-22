from PIL import Image
import numpy as np
from math import ceil, floor

# Conversion functions
def string_to_binary(text):
    """Convert a string to binary representation."""
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_string(binary):
    """Convert binary representation to a string."""
    text = ''.join(chr(int(binary[i:i + 8], 2)) for i in range(0, len(binary), 8))
    return text

# Image functions
def image_to_array(path):
    """Load an image from the file path and convert it to a numpy array."""
    with Image.open(path) as img:
        return np.array(img)

def array_to_image(arr, save_path):
    """Convert a numpy array to an image and save it to the specified path."""
    img = Image.fromarray(arr)
    img.save(save_path)

# PVD encoding function
def pvd_encode(img_path, msg, result_path='result.jpg'):
    """Encode a message in the image using Pixel Value Differencing (PVD)."""
    img = image_to_array(img_path)
    height, width, _ = img.shape
    bin_msg = string_to_binary(msg)
    msg_length = len(bin_msg)
    cnt = 0

    for i in range(height - 1):
        for j in range(0, width - 1, 2):
            # Calculate the difference between pixels
            diff = int(img[i][j][0]) - int(img[i][j + 1][0])
            n = 3

            # Determine the required bit length for the current difference
            while abs(diff) >= 2 ** n:
                n += 1

            # Calculate the lower bound for the difference
            l = 0 if n == 3 else 2 ** (n - 1)

            # Extract binary message bits to encode
            if cnt + n <= msg_length:
                bin_m = bin_msg[cnt:cnt + n]
                m = int(bin_m, 2)
                cnt += n
            else:
                m = 0  # No more bits to encode, use zero

            # Calculate new difference (d) based on l and m
            d = l + m if diff >= 0 else -(l + m)

            # Adjust pixel values based on the new difference
            if d % 2 != 0:
                img[i][j][0] += ceil((d - diff) / 2)
                img[i][j + 1][0] -= floor((d - diff) / 2)
            else:
                img[i][j][0] += floor((d - diff) / 2)
                img[i][j + 1][0] -= ceil((d - diff) / 2)

            # If the message has been fully encoded, save the image and return
            if cnt >= msg_length:
                array_to_image(img, result_path)
                return

    # Save the encoded image
    array_to_image(img, result_path)

# PVD decoding function
def pvd_decode(img_path, msg_length):
    """Decode a message from the image using Pixel Value Differencing (PVD)."""
    img = image_to_array(img_path)
    height, width, _ = img.shape
    bin_msg = ''
    total_bits = msg_length * 8  # Convert message length from characters to bits

    for i in range(height - 1):
        for j in range(0, width - 1, 2):
            # Calculate the difference between pixels
            diff = int(img[i][j][0]) - int(img[i][j + 1][0])
            n = 3

            # Determine the required bit length for the current difference
            while abs(diff) >= 2 ** n:
                n += 1

            # Calculate the lower bound for the difference
            l = 0 if n == 3 else 2 ** (n - 1)

            # Extract message bits from the image
            m = abs(diff) - l
            bin_msg += f'{m:0{n}b}'

            # Stop when the desired message length is reached
            if len(bin_msg) >= total_bits:
                # Trim the binary message to the desired length and convert to string
                return binary_to_string(bin_msg[:total_bits])

    # Return the decoded message
    return binary_to_string(bin_msg[:total_bits])

# Example usage
pvd_encode('1.jpg', 'hello world!', 'result.jpg')
decoded_message = pvd_decode('result.jpg', 12)  # 'hello world!' is 12 characters long
print(decoded_message)