import sys
import cv2
from imutils.video import VideoStream
import imagezmq
import threading
import numpy as np
from time import sleep
import click
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

frame_stack = []

@click.command()
@click.option('--src-ip', default='127.0.0.1')
@click.option('--src-port', default=5555)
@click.option('--processor', default='web')
def create_stream_processor(src_ip, src_port, processor):
    receiver = VideoStreamSubscriber(src_ip, src_port)
    if processor == 'web':
        x = threading.Thread(target=run_simple, args=('0.0.0.0', 4000, application,), daemon=True)
        x.start()
    try:
        while True:
            msg, frame = receiver.receive()
            frame_stack.append((msg, frame))
    except (KeyboardInterrupt, SystemExit):
        print('Exit due to keyboard interrupt')
    except TimeoutError as ex:
        print('Timeout error, streamer is gone ...')
    except Exception as ex:
        print('Python error with no Exception handler:')
        print('Traceback error:', ex)
        traceback.print_exc()
    finally:
        receiver.close()
        sys.exit()

def sendImagesToWeb():
    while True:
        try:
            camName, frame = frame_stack.pop()#receiver.recv_image()
            jpg = cv2.imencode('.jpg', frame)[1]
            yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n'+jpg.tostring()+b'\r\n'    
        except Exception as e:
            pass

@Request.application
def application(request):
    return Response(sendImagesToWeb(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Helper class implementing an IO deamon thread
class VideoStreamSubscriber:

    def __init__(self, hostname, port):
        self.hostname = hostname
        self.port = port
        self._stop = False
        self._data_ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=())
        self._thread.daemon = True
        self._thread.start()

    def receive(self, timeout=15.0):
        flag = self._data_ready.wait(timeout=timeout)
        if not flag:
            raise TimeoutError(
                "Timeout while reading from subscriber tcp://{}:{}".format(self.hostname, self.port))
        self._data_ready.clear()
        return self._data

    def _run(self):
        receiver = imagezmq.ImageHub("tcp://{}:{}".format(self.hostname, self.port), REQ_REP=False)
        while not self._stop:
            self._data = receiver.recv_image()
            self._data_ready.set()
        receiver.close()

    def close(self):
        self._stop = True



if __name__ == "__main__":
    create_stream_processor()