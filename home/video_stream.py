import cv2
from threading import Thread
from queue import Queue
q = Queue(maxsize=1)

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FPS, 5)
        # self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False
        self.img=None

    def start(self):
        # start the thread to read frames from the video stream
        t= Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            # self.img = letterbox(self.frame, (640,640), stride=0, auto='')[0]
            # if q.empty():
            #     # time.sleep(0.2)
            #     # q.get()
            #     q.put(self.frame)
            # print("H")

    def read(self):
        # return the frame most recently read
        return self.grabbed,self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stream.release()
        self.stopped = True

    def camera(self):
        return self.stream
