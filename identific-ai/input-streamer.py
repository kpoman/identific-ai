import os
import sys
import socket
import time
import datetime
import json
import click
from click.types import DateTime
import imagezmq
from imutils.video import VideoStream

@click.command()
@click.option('--src-type', default='v4l2', help='type of the input stream (v4l2, picamera, netstream)')
@click.option('--src-index', default=0, help='index or url of the device')
@click.option('--dst-ip', default='127.0.0.1')
@click.option('--dst-port', default=5555)
@click.option('--transform', default=None)
@click.option('--tagging', default=None)
def create_input_stream(src_type, src_index, dst_ip, dst_port, transform, tagging):
    sender = imagezmq.ImageSender(f'tcp://{dst_ip}:{dst_port}', REQ_REP=False)
    if src_type=='v4l2':
        capture = VideoStream(src_index)
    elif src_type=='picamera':
        capture = VideoStream(usePiCamera=True)
    
    hostname = socket.gethostname()

    capture.start()
    time.sleep(2.0)

    while True:
        frame = capture.read()
        metadata = {'hostname': hostname, 'datetime':datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f')}
        sender.send_image(json.dumps(metadata), frame)
        time.sleep(0.5)
        print(metadata)



if __name__ == '__main__':
    create_input_stream()