import os
import sys
import socket
import time
import datetime
import json
import click
import cv2
from click.types import DateTime
import imagezmq
from imutils.video import VideoStream

@click.command()
@click.option('--src-type', default='v4l2', help='type of the input stream (v4l2, picamera, netstream)')
@click.option('--src-index', default='0', help='index or url of the device')
@click.option('--dst-ip', default='127.0.0.1')
@click.option('--dst-port', default=5555)
@click.option('--transform', default=None)
@click.option('--tagging', default='qrcode,plates')
def create_input_stream(src_type, src_index, dst_ip, dst_port, transform, tagging):
    sender = imagezmq.ImageSender(f'tcp://{dst_ip}:{dst_port}', REQ_REP=False)
    if src_type =='v4l2':
        capture = VideoStream(int(src_index))
    elif src_type =='picamera':
        capture = VideoStream(usePiCamera=True)
    elif src_type == 'youtube':
        import pafy
        url = src_index
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        #capture = cv2.VideoCapture(best.url)
        capture = VideoStream(best.url)

    capture.start()
    time.sleep(2.0)

    if tagging:
        if 'qrcode' in tagging:
            qrDecoder = cv2.QRCodeDetector()
        if 'plates' in tagging:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'br.xml')
            classifier = cv2.CascadeClassifier(path)

    hostname = socket.gethostname()

    while True:
        frame = capture.read()
        frame = image_resize(frame, width=800)
        if tagging:
            if 'qrcode' in tagging:
                data,bbox,rectifiedImage = qrDecoder.detectAndDecode(frame)
                # if data:
                #     print(data)
            if 'plates' in tagging:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_gray = cv2.equalizeHist(frame_gray)
                plates = classifier.detectMultiScale(frame_gray)
                for (x,y,w,h) in plates:
                    center = (x + w//2, y + h//2)
                    frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
                    faceROI = frame_gray[y:y+h,x:x+w]

        cv2.imshow('Capture - Plate detection', frame)
        cv2.waitKey(1)

        metadata = {'hostname': hostname, 'datetime':datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f')}
        sender.send_image(json.dumps(metadata), frame)
        #time.sleep(0.1)
        #print(metadata)

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
