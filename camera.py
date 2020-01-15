import cv2
from imutils.video.pivideostream import PiVideoStream
from imutils.video import VideoStream
import imutils
import time
import numpy as np
import datetime
from imutils.object_detection import non_max_suppression

class VideoCamera(object):

    def __init__(self, flip = False):
        # self.vs = VideoStream() # Uncomment this line and comment the next line to use usb camera or built in camera's (Laptops)
        self.vs = PiVideoStream(resolution=(720, 720)).start() # Change resolution as per need
        self.flip = flip
        time.sleep(2.0)
        self.avg = None

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_object(self, hog):
        found_objects = False
        frame = self.flip_if_needed(self.vs.read()).copy() 
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        timestamp = datetime.datetime.now()
        text = "Unoccupied"

        # objects = classifier.detectMultiScale(
        #     gray,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 30),
        #     flags=cv2.CASCADE_SCALE_IMAGE
        # )

        # if len(objects) > 0:
        #     found_objects = True

        # # Draw a rectangle around the objects
        # for (x, y, w, h) in objects:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ret, jpeg = cv2.imencode('.jpg', frame)

        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
        padding=(8, 8), scale=1.05)

        # convert image to grayscale and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the average frame is None, initialize it
        if self.avg is None:
            print("[INFO] starting background model...")
            self.avg = gray.copy().astype("float")

        # accumulate the weighted average between the current frame and
        # previous frames, then compute the difference between the current
        # frame and running average
        cv2.accumulateWeighted(gray, self.avg, 0.5)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        # threshold the delta image, dilate the thresholded image to fill
        # in holes, then find contours on thresholded image
        thresh = cv2.threshold(frameDelta, 5, 255,
            cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 5000:
                continue

            # draw the original bounding boxes
            # for (x, y, w, h) in rects:
            #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            text = "Occupied"

        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

        if text == "Occupied":
            found_objects = True

        ret, jpeg = cv2.imencode('.jpg', frame)

        return (jpeg.tobytes(), found_objects)


