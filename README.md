# Smart-Security-Camera
IoT Raspberry Pi security camera running open-cv for object detection. The camera will send an email with an image of any objects it detects. It also runs a server that provides a live video stream over the internet.

[Watch the original video here](https://youtu.be/Y2QFu-tTvTI)

## Setup

This project uses a Raspberry Pi Camera to stream video. Before running the code, make sure to configure the raspberry pi camera on your device.

Open the terminal and run

```
sudo raspi-config
```

Select `Interface Options`, then `Pi Camera` and toggle on. Press `Finish` and exit.

You can verify that the camera works by running

```
raspistill -o image.jpg
```
which will save a image from the camera in your current directory. You can open up the file inspector and view the image.

## Installing Dependencies

This project uses openCV to detect objects in the video feed. You can install openCV by using the following [tutorial](https://www.pyimagesearch.com/2019/09/16/install-opencv-4-on-raspberry-pi-4-and-raspbian-buster/). I used the Python 3 version of the tutorial. Also i recommend installing it using pip rather than compiling from source as it takes a heck lot of time.

The installation took almost 8 hours (!!) on my Raspberry Pi Zero, but it would be considerably faster on a more powerful board like the Raspberry Pi 3.

The tutorial will prompt you to create a virtual environment. Make sure you are using the virtual environment by typing the following commands

```bash
source ~/.profile
workon cv
```

Next, navigate to the repository directory

```
cd Smart-Security-Camera
```

and install the dependencies for the project

```
pip3 install -r requirements.txt
```


## Customization

To get emails when objects are detected, you'll need to make a couple modifications to the `credentials.py` file.

Open `credentials.py` with nano `nano credentials.py`, then edit it. 

```
# Basic auth credentials
BASIC_AUTH_USERNAME = 'CHANGE_ME_USERNAME'
BASIC_AUTH_PASSWORD = 'CHANGE_ME_PLEASE'
BASIC_AUTH_FORCE = True

# Email you want to send the update from (only works with gmail)
fromEmail = 'myemail@gmail.com'
fromEmailPassword = 'password1234'

# Email you want to send the update to
toEmail = 'anotheremail@gmail.com'
```
and replace with your own email/credentials. Save with `ctrl + x` then press `y` to save. The `mail.py` file logs into a gmail SMTP server and sends an email with an image of the object detected by the security camera. 



You can also modify the `main.py` file to change some other properties.

```
email_update_interval = 600 # sends an email only once in this time interval
video_camera = VideoCamera(flip=True) # creates a camera object, flip vertically
object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml") # an opencv classifier
```
Notably, you can use a different object detector by changing the path `"models/fullbody_recognition_model.xml"` in `object_classifier = cv2.CascadeClassifier("models/fullbody_recognition_model.xml")`.

to a new model in the models directory.

```
facial_recognition_model.xml
fullbody_recognition_model.xml
upperbody_recognition_model.xml
```

## Running the Program

Run the program

```
python main.py
```

You can view a live stream by visiting the ip address of your pi in a browser on the same network. You can find the ip address of your Raspberry Pi by typing `ifconfig` in the terminal and looking for the `inet` address. 

Visit `<raspberrypi_ip>:5000` in your browser to view the stream.

Note: To view the live stream on a different network than your Raspberry Pi, you can use [ngrok](https://ngrok.com/) to expose a local tunnel. Once downloaded, run ngrok with `./ngrok http 5000` and visit one of the generated links in your browser.

Note: The video stream will not start automatically on startup. To start the video stream automatically, you will need to run the program  from your `/etc/rc.local` file see this [video](https://youtu.be/51dg2MsYHns?t=7m4s) for more information about how to configure that.

## Credits

Credits to [HackerShackOfficial](https://github.com/HackerShackOfficial) for the original code base at [Smart-Security-Camera](https://github.com/HackerShackOfficial/Smart-Security-Camera)
