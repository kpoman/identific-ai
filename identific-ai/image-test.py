import os
import sys
import cv2
from utils import image_resize, add_frame_tags
from processors import FrameTagger

if __name__ == '__main__':
    # qrcode test
    #frame = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'testimages', 'qrcodes_2.png'))
    # placa test
    frame = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'testimages', 'carros_varios.jpg'))
    frame = image_resize(frame, width=800)
    tagger = FrameTagger('plate,face,qrcode')
    tags = tagger.detect_frame_tags(frame)
    tagged_frame = add_frame_tags(frame, tags)
    cv2.imshow('Capture - Plate detection', tagged_frame)
    cv2.waitKey(0)