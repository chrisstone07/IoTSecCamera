import cv2
import sys
from mail import sendEmail
from flask import Flask, render_template, Response
from camera import VideoCamera
from flask_basicauth import BasicAuth
import time
#import threading
import multiprocessing
import credentials as creds
import lcd
import RPi.GPIO as GPIO

# LCD Pins
D4=6
D5=13
D6=19
D7=26
RS=20
EN=21
mylcd=lcd.lcd()
mylcd.begin(D4,D5,D6,D7,RS,EN)
mylcd.Print("Safe: No intruder detected")

email_update_interval = 600 # sends an email only once in this time interval
video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier

# App Globals (do not edit)
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = creds.BASIC_AUTH_USERNAME
app.config['BASIC_AUTH_PASSWORD'] = creds.BASIC_AUTH_PASSWORD
app.config['BASIC_AUTH_FORCE'] = creds.BASIC_AUTH_FORCE

basic_auth = BasicAuth(app)
last_epoch = 0

def check_for_objects():
	global last_epoch
	while True:
		try:
			frame, found_obj = video_camera.get_object(object_classifier)
			if found_obj and (time.time() - last_epoch) > email_update_interval:
				last_epoch = time.time()
				print("Sending email...")
				sendEmail(frame)
				print("done!")
                #trying to fix indent
                mylcd.clear()
                mylcd.Print("Alert! Intruder Detected")

            else:
                    mylcd.clear()
                    mylcd.Print("Safe: No intruder detected")
		except:
			print("Error sending email: ", sys.exc_info()[0])

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # t = threading.Thread(target=check_for_objects, args=())
    # t.daemon = True
    # t.start()

    d = multiprocessing.Process(name='daemon', target=check_for_objects)
    d.daemon = True
    d.start()
    app.run(host='0.0.0.0', debug=False)

