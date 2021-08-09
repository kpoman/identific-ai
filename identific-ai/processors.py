import datetime
import os
import time
import json
import cv2
import imagezmq
from utils import rect_as_points
from threading import Thread, Event
from time import perf_counter
from codetiming import Timer

import numpy as np

# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = Event()
        self._thread = Thread(target=self._run, args=())
        self._thread.daemon = True
        self.timer = Timer()

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

class FrameTagger():
    
    def __init__(self, tagging):
        self.tagging = tagging
        self.detectors = {}
        self.timer = Timer()
        self.build_detectors()
    
    def build_detectors(self):
        if 'qrcode' in self.tagging:
            self.detectors['qrcode'] = cv2.QRCodeDetector()
        if 'plate' in self.tagging:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'br.xml')
            self.detectors['plate'] = cv2.CascadeClassifier(path)
        if 'face' in self.tagging:
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', 'haarcascade_frontalface_alt.xml')
            self.detectors['face'] = cv2.CascadeClassifier(path)
    
    def detect_frame_tags(self, frame):
        tags = {}
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        if 'qrcode' in self.tagging:
            with Timer(name='qrcode-dtc', text='{name} {milliseconds:.0f} ms'):
                qrcodes = self.detectors['qrcode'].detectMulti(frame)
                tags['qrcode'] = []
                if qrcodes[0]:  # True
                    for q in qrcodes[1]:
                        tags['qrcode'].append(q.astype(np.int32).tolist()) #list(totuple(qrcodes[1][0].astype(np.int32)))
        if 'plate' in self.tagging:
            with Timer(name='plate-dtc', text='{name} {milliseconds:.0f} ms'):
                plates = self.detectors['plate'].detectMultiScale(frame_gray)
                tags['plate'] = []
                for p in plates:
                    tags['plate'].append(rect_as_points(p))
        if 'face' in self.tagging:
            with Timer(name='face-dtc', text='{name} {milliseconds:.0f} ms'):
                faces = self.detectors['face'].detectMultiScale(frame_gray)
                tags['face'] = []
                for f in faces:
                    tags['face'].append(rect_as_points(f)) #faces[0].tolist()                
        return tags

def tag_dump(input_stack, output_dir='tag_dump/'):
    print('created tag_dump')
    while True:
        try:
            res = input_stack.pop()
            if res:
                obj = json.loads(res[0])
                if 'tags' in obj:
                    for tag in obj['tags']:
                        cnt = 0
                        for item in obj['tags'][tag]:
                            #print(f'item-{tag}-{cnt} at {obj["tags"][tag][cnt]}')
                            filename = f'{obj["hostname"]}-{obj["datetime"]}-{tag}-{cnt}.jpg'
                            coords = obj['tags'][tag][cnt]
                            print(filename, 'at', coords)
                            x, y, w, h = cv2.boundingRect(np.array(coords))
                            cropped = res[1][y:y+h,x:x+w]
                            cv2.imwrite(output_dir+filename, cropped)
                            cnt += 1
                    #if len(obj['tags']['face']) > 0:
                    #    for face in obj['tags']['face']:
                    #        print(face)
        except Exception as e:
            pass #print(e)
        time.sleep(0.1)
