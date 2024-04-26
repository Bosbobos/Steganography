import SpatialIntegration as SI

msg = '''Please choose the method:
    0: exit
    1: PVD
    '''


def get_image_path():
    return input('Write the path to the image: ')


def get_message():
    return input('Write the message to encode: ')


def get_bits():
    return int(input('Write the number of bits to decode (you can start with 200 and increase if needed): '))


def PVD():
    msg = '''Please choose the operation:
    1: Encode
    2: Decode
    '''
    op = int(input(msg))
    match op:
        case 1:
            name = get_image_path()
            SI.pvd_encode(name, get_message())
            print('The picture with encoded message is in the output folder.')
        case 2:
            name = get_image_path()
            bits = get_bits()
            message = SI.pvd_decode(name, bits)
            print(message)
        case _:
            raise Exception("Unknown command")


if __name__=='__main__':
    while True:
        message = get_message()
        SI.pvd_encode('input/lasttry.png', message)
        print(SI.pvd_decode('output/lasttry.png', 6480))
    #operation = int(input(msg))
    #while (operation != 0):
    #    match operation:
    #        case 1:
    #            PVD()
    #        case _:
    #            raise Exception("Unknown command")
