import SpatialIntegration as SI
import TextManager as TM

msg = '''Please choose the method:
    0: exit
    1: PVD
    '''


def get_image_path():
    return input('Write the path to the image: ')


def get_message():
    with open('input_text.txt', 'r') as f:
        text = ''.join(i for i in f.readlines())
    return TM.to_ascii(text)


def get_calibr_const():
    return int(input('Write how strong you want the calibration to be (from 1 to 10). '
                     'Better start from the lower and increase if needed: '))

def PVD():
    msg = '''Please choose the operation:
    0: Calibration
    1: Encode
    2: Decode
    
    Note that this method might not work for high contrast image without prior calibration.
    If you notice out-of-place pixels and the message doesn't decode, perform calibration before encoding.
    '''
    op = int(input(msg))
    match op:
        case 0:
            name = get_image_path()
            SI.calibrate_image(name, get_calibr_const())
            name = name[:-4] + '_calibrated' + name[-4:]
            SI.pvd_encode(name, get_message())
            i = len(name) - 1 - name[::-1].index('/') if '/' in name else 0  #Finds file name without directories
            path = 'output/' + name[i:]
            print(SI.pvd_decode(path))
        case 1:
            print('Please put the text to encode in input_text.txt.')
            name = get_image_path()
            SI.pvd_encode(name, get_message())
            i = len(name) - 1 - name[::-1].index('/') if '/' in name else 0  #Finds file name without directories
            path = 'output/' + name[i:]
            print(SI.pvd_decode(path))
        case 2:
            name = get_image_path()
            message = SI.pvd_decode(name)
            print(message)
        case _:
            raise Exception("Unknown command")


if __name__=='__main__':
    operation = int(input(msg))
    while operation != 0:
       match operation:
           case 1:
               PVD()
           case _:
               raise Exception("Unknown command")
