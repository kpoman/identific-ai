import os
import sys
import time
import socket
import datetime
import traceback
import json
import click
import cv2
import numpy as np
from click.types import DateTime
import imagezmq
from imutils.video import VideoStream
from utils import apply_transform, totuple, add_frame_tags
from processors import VideoStreamSubscriber, FrameTagger
from qr_extractor import extract


@click.command()
@click.option('--src-type', default='v4l2', help='type of the input stream (v4l2, picamera, hubstream, netstream)')
@click.option('--src-index', default='0', help='index or url of the device')
@click.option('--dst-ip', default='127.0.0.1')
@click.option('--dst-port', default=5555)
@click.option('--transform', default=None, help='a json array of operation and params')
@click.option('--tagging', default='', help='available taggings: qrcode,plate,face')
@click.option('--xdebug', default=False, help='debug locally with an X server')
def create_input_stream(src_type, src_index, dst_ip, dst_port, transform, tagging, xdebug):
    
    hostname = socket.gethostname()

    # create a capture object from supported source
    if src_type =='v4l2':
        capture = VideoStream(src=int(src_index), usePiCamera=False)
    elif src_type =='picamera':
        #capture = VideoStream(usePiCamera=True, resolution=(1280,720), framerate=10)
        capture = VideoStream(usePiCamera=True, resolution=(1920,1080), framerate=10)
    elif src_type == 'youtube':
        import pafy
        url = src_index
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        capture = VideoStream(best.url)
    elif src_type == 'hubstream':
        capture = VideoStreamSubscriber(hostname=src_index.split(':')[0], port=src_index.split(':')[1])
    
    # create a sender object where to send captured data
    sender = imagezmq.ImageSender(f'tcp://{dst_ip}:{dst_port}', REQ_REP=False)

    # create a tagger to do f= object detection
    tagger = FrameTagger(tagging)

    # start capturing
    capture.start()
    time.sleep(2.0)          

    try:
        while True:
            frame = capture.read()
            print('original shape', frame.shape)
            frame = apply_transform(frame, transform)
            print('new shape', frame.shape)
            tags = tagger.detect_frame_tags(frame)
            
            if xdebug:
                debug_frame = add_frame_tags(frame, tags)
                cv2.imshow('Capture detection', debug_frame)
                cv2.waitKey(1)

            metadata = {
                'hostname': hostname, 
                'datetime': datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f'),
                'tags': tags,
            }
            try:
                sender.send_image(json.dumps(metadata, cls=npEncoder), frame)
            except Exception as e:
                print(e, ':')
                print(metadata)
            time.sleep(0.1)
    except Exception as ex:
        print('exception caught:', ex)
        capture.stop()
        traceback.print_exc()

class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    create_input_stream()
