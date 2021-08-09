import os
import sys
import cv2
import numpy as np
from utils import image_resize, add_frame_tags
from processors import FrameTagger

if __name__ == '__main__':
    # qrcode test
    #frame = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'testimages', 'qrcodes_2.png'))
    # placa test

    # TEST detecting plates
    # frame = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'testimages', 'carros_varios.jpg'))
    # frame = image_resize(frame, width=800)
    # tagger = FrameTagger('plate,face,qrcode')
    # tags = tagger.detect_frame_tags(frame)
    # tagged_frame = add_frame_tags(frame, tags)
    # cv2.imshow('Capture - Plate detection', tagged_frame)
    # cv2.waitKey(0)

    frame = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'testimages', 'carros_varios.jpg'))
    coords = [[81, 39], [146, 39], [146, 104], [81, 104]]
    x, y, w, h = cv2.boundingRect(np.array(coords))
    print(x, y, w, h)