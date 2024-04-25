import SpatialIntegration as SI

msg = '''Please choose the method:
0: exit
1: PVD
'''

def get_image_name():
    return input('Put the image into "input" folder and write its name: ')

def get_message():
    return input('Write the message to encode: ')

def PVD():
    msg = '''Please choose the operation:
    1: Encode
    2: Decode
    '''
    op = int(input(msg))
    match op:
        case 1:
            name = get_image_name()
            SI.pvd_encode('input/' + name, get_message())
            print('The picture with encoded message is in folder "output".')

if __name__=='__main__':
    operation = int(input(msg))
    while (operation != 0):
        match operation:
            case 1:
                PVD()
            case _:
                raise Exception("Unknown command")
