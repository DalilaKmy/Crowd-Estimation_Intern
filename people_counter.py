from flask import Flask, request, jsonify, render_template, redirect
import pusher
from database import db_session
from models import Flight
import datetime as Hari
from datetime import datetime
import os

import numpy as np
import cv2
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
import threading
import argparse
import imutils
import time
import sqlite3

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS
import dlib

app = Flask(__name__)

#using Pusher credentials
pusher_client = pusher.Pusher(
    'app_id',
    'key',
    'secret',
    ssl=True,
    cluster='ap1')

@app.teardown_appcontext
def shutdown_session(exception=None):
    db_session.remove()

@app.route('/')
def index():
		conn = sqlite3.connect('database.db')
		conn.row_factory = dict_factory
		curr = conn.cursor()
		curr.execute("DELETE FROM names;")
		conn.commit()

		conn = sqlite3.connect('database.db')
		conn.row_factory = dict_factory
		curr = conn.cursor()
		display = curr.execute("SELECT * from names;").fetchall()
		return render_template('facergns.html', names=display)


@app.route('/resetAll')
def resetAll():
        conn = sqlite3.connect('database.db')
        conn.row_factory = dict_factory
        curr = conn.cursor()
        curr.execute("DELETE FROM names;")
        conn.commit()
		
        conn = sqlite3.connect('database.db')
        conn.row_factory = dict_factory
        curr = conn.cursor()
        all_faces = curr.execute("SELECT * from names").fetchall()        
        return render_template('facergns.html', names=all_faces)

	
outputFrame = None
lock = threading.Lock()

def dict_factory(cursor, row):
    d ={}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def Enquiry(lis1): 
    if len(lis1) == 0: 
        return 0
    else: 
        return 1

def detect_motion(frameCount):
	frameCount=0
	global outputFrame, lock
	
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

	# if a video path was not supplied, grab a reference to the webcam
	if not args.get("input", False):
		print("[INFO] starting video stream...")
		#vs = VideoStream(src='rtsp://192.168.0.104:8554/live.sdp').start()
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(args["input"])

	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	writer = None

	W = None
	H = None

	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	totalFrames = 0
	totalDown = 0
	totalUp = 0
	fps = FPS().start()

	while True:

		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		if args["input"] is not None and frame is None:
			break

		frame = imutils.resize(frame, width=500)

		rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		rgb = cv2.GaussianBlur(rgb, (7, 7), 0)

		if W is None or H is None:
			(H, W) = frame.shape[:2]


		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)


		status = "Waiting for People"
		rects = []

		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting Person"
			trackers = []

			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]

				if confidence > args["confidence"]:
					idx = int(detections[0, 0, i, 1])

					if CLASSES[idx] != "person":
						continue

					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
					tracker.start_track(rgb, rect)

					trackers.append(tracker)

		else:
			for tracker in trackers:
				status = "Tracking Person"
				
				tracker.update(rgb)
				pos = tracker.get_position()

				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				rects.append((startX, startY, endX, endY))


		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

		objects = ct.update(rects)

		for (objectID, centroid) in objects.items():
			to = trackableObjects.get(objectID, None)

			if to is None:
				to = TrackableObject(objectID, centroid)

			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				if not to.counted:

					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						to.counted = True

					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						to.counted = True

			trackableObjects[objectID] = to

			text = "ID {}".format(objectID)

			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		info = [
				("People In", totalUp),
				("People Out", totalDown),
				("Status", status),
				]

		totalPeople = totalUp + totalDown
		current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

		#Write output data to frame
		cv2.putText(frame, current_time, (10, H - ((i * 20) + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
		
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		#Pusher part to add real-time data
		data = {
				"id":1,
				"peopleIn":totalUp,
				"peopleOut":totalDown,
				"totalPeople":totalPeople,
				"dateTime":current_time
				}

		pusher_client.trigger('table', 'update-record', {'data': data })
		
		#Database part to create table and insert data
		conn = sqlite3.connect('database.db')
		conn.row_factory = dict_factory
		curr = conn.cursor()
		
		curr.execute("""
		CREATE TABLE IF NOT EXISTS names
		(id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, status TEXT, peopleIn INT, peopleOut INT, totalPeople INT, dateTime DATETIME)""")

		curr.execute("""
		INSERT INTO names
		(status, peopleIn, peopleOut, totalPeople, dateTime) VALUES (?,?,?,?,?)""", (status,totalUp,totalDown,totalPeople,current_time))
		
		conn.commit()
		curr.close()
		conn.close()

		if writer is not None:
			writer.write(frame)

		# show the output frame
		totalFrames += 1
		fps.update()

		
    # Normal
        
		if total > frameCount:
			motion = md.detect(rgb)
			if motion is not None:
				(thresh, (minX, minY, maxX, maxY)) = motion
				
		md.update(rgb)
		total += 1

		with lock:
			outputFrame = frame.copy()
			

def generate():
    
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue

            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
	
@app.route("/video_feed")
def video_feed():
    
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':

	ap = argparse.ArgumentParser()

	ap.add_argument("-p", "--prototxt", default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
        help="# of frames used to construct the background model")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    
	t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
	t.daemon = True
	t.start()
	app.run(debug=True, threaded=True, use_reloader=True)

