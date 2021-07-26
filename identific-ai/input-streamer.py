import os
import sys
import socket
import time
import datetime
import json
import traceback
import click
import cv2
import numpy as np
from click.types import DateTime
import imagezmq
from imutils.video import VideoStream

@click.command()
@click.option('--src-type', default='v4l2', help='type of the input stream (v4l2, picamera, netstream)')
@click.option('--src-index', default='0', help='index or url of the device')
@click.option('--dst-ip', default='127.0.0.1')
@click.option('--dst-port', default=5555)
@click.option('--transform', default=None, help='a json array of operation and params')
@click.option('--tagging', default='qrcode,plates,faces')
@click.option('--xdebug', default=False, help='debug locally with an X server')
def create_input_stream(src_type, src_index, dst_ip, dst_port, transform, tagging, xdebug):
    sender = imagezmq.ImageSender(f'tcp://{dst_ip}:{dst_port}', REQ_REP=False)
    if src_type =='v4l2':
        capture = VideoStream(src=int(src_index), usePiCamera=False)
    elif src_type =='picamera':
        capture = VideoStream(usePiCamera=True)
    elif src_type == 'youtube':
        import pafy
        url = src_index
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        capture = VideoStream(best.url)

    capture.start()
    time.sleep(2.0)

    if tagging:
        if 'qrcode' in tagging:
            qrDecoder = cv2.QRCodeDetector()
        if 'plates' in tagging:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'br.xml')
            plate_classifier = cv2.CascadeClassifier(path)
        if 'faces' in tagging:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'haarcascade_frontalface_alt.xml')
            face_classifier = cv2.CascadeClassifier(path)            

    hostname = socket.gethostname()
    try:
        while True:
            frame = capture.read()
            if transform:
                frame = apply_transform(frame, transform)
            #frame = image_resize(frame, width=400)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
            faces = plates = qrcodes = []
            if tagging:
                if 'qrcode' in tagging:
                    timer_s = datetime.datetime.now()
                    qrcodes,bbox,rectifiedImage = qrDecoder.detectAndDecode(frame_gray)
                    print(datetime.datetime.now()-timer_s, 'qrcode detect', qrcodes)
                if 'plates' in tagging:
                    timer_s = datetime.datetime.now()
                    plates = plate_classifier.detectMultiScale(frame_gray)
                    if len(plates) > 0:
                        plates = plates[0].tolist()
                    print(datetime.datetime.now()-timer_s, 'plate detect', plates)
                if 'faces' in tagging:
                    timer_s = datetime.datetime.now()
                    faces = face_classifier.detectMultiScale(frame_gray)
                    if len(faces) > 0:
                        faces = faces[0].tolist()                
                    print(datetime.datetime.now()-timer_s, 'face detect', faces)

            if xdebug:
                cv2.imshow('Capture - Plate detection', frame)
                cv2.waitKey(1)

            metadata = {
                'hostname': hostname, 
                'datetime':datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f'),
                'faces': faces,
                'plates': plates,
                'qrcodes': qrcodes
            }
            sender.send_image(json.dumps(metadata), frame)
    except Exception as ex:
        print('exception caught:', ex)
        capture.stop()
        traceback.print_exc()


def apply_transform(image, transform):
    """
    {
        "transformation": "Crop",
        "width": 100,
        "height": 100,
        "xPosition": 0,
        "yPosition": 0,
        "gravity": "NorthWest"
    },
    {
        "transformation": "Rotate",
        "degrees": 45.3
    }
    """
    operations = json.loads(transform)
    for o in operations:
        if o['transformation'] == 'Rotate':
            image = rotate_image(image, int(o['Degrees']))
        if o['transformation'] == 'Crop':
            image = image[o['yPosition']:o['yPosition']+o['height'], o['xPosition']:+o['xPosition']+o['width']]
    return image

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

if __name__ == '__main__':
    create_input_stream()
