import os
import sys
import socket
import time
import datetime
import json
import traceback
from threading import Thread, Event
import click
import cv2
import numpy as np
from click.types import DateTime
import imagezmq
from imutils.video import VideoStream
from utils import apply_transform, totuple
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
    elif src_type == 'hubstream':
        capture = VideoStreamSubscriber(hostname=src_index.split(':')[0], port=src_index.split(':')[1])

    capture.start()
    time.sleep(2.0)

    print('tagging:')
    print(tagging)

    if tagging:
        if 'qrcode' in tagging:
            qrDecoder = cv2.QRCodeDetector()
        if 'plate' in tagging:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'br.xml')
            plate_classifier = cv2.CascadeClassifier(path)
        if 'face' in tagging:
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
                    # codes, output = extract(frame, debug=True)
                    # cv2.imshow('qrcode', output)
                    # cv2.waitKey(1)
                    #print(res)
                    timer_s = datetime.datetime.now()
                    qrcodes = qrDecoder.detectMulti(frame_gray)
                    if qrcodes[0]:  # True
                        qrcodes = qrcodes[1][0].astype(np.int32).tolist() #list(totuple(qrcodes[1][0].astype(np.int32)))
                    else:
                        qrcodes = []
                    print(datetime.datetime.now()-timer_s, 'qrcode detect', qrcodes)
                if 'plate' in tagging:
                    timer_s = datetime.datetime.now()
                    plates = plate_classifier.detectMultiScale(frame_gray)
                    if len(plates) > 0:
                        plates = plates[0].tolist()
                    print(datetime.datetime.now()-timer_s, 'plate detect', plates)
                if 'face' in tagging:
                    timer_s = datetime.datetime.now()
                    faces = face_classifier.detectMultiScale(frame_gray)
                    if len(faces) > 0:
                        faces = faces[0].tolist()                
                    print(datetime.datetime.now()-timer_s, 'face detect', faces)

            if xdebug:
                if len(qrcodes) > 0:
                    print(np.array(qrcodes))
                    frame = cv2.drawContours(frame, [np.array(qrcodes)], 0, (255,0,0,0))
                cv2.imshow('Capture - Plate detection', frame)
                cv2.waitKey(1)

            metadata = {
                'hostname': hostname, 
                'datetime':datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f'),
                'faces': faces,
                'plates': plates,
                'qrcodes': qrcodes
            }
            try:
                sender.send_image(json.dumps(metadata), frame)
            except Exception as e:
                print(e, ':')
                print(type(metadata['qrcodes'][0][0]))
                print(metadata)
            #time.sleep(0.5)
    except Exception as ex:
        print('exception caught:', ex)
        capture.stop()
        traceback.print_exc()

# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = Event()
        self._thread = Thread(target=self._run, args=())
        self._thread.daemon = True


    def start(self):
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def read(self):
        msg, frame = self.receive()
        return frame #cv2.imdecode(np.frombuffer(frame, dtype='uint8'), -1)

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        while not self._stop:
            self._data = receiver.recv_image() #.recv_jpg()
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True

    def stop(self):
        self.close()

if __name__ == '__main__':
    create_input_stream()
