from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
import argparse
import time

prototxtPath = os.path.sep.join([settings.BASE_DIR, "face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join([settings.BASE_DIR,"face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(os.path.join(settings.BASE_DIR,'face_detector/mask_detector.model'))




class FaceDetect(object):
	def __init__(self):
		# initialize the video stream, then allow the camera sensor to warm up
		self.vs = VideoStream(src=0).start()
		# start the FPS throughput estimator
		self.fps = FPS().start()

	def __del__(self):
		cv2.destroyAllWindows()

	def detect_and_predict_mask(self, frame, faceNet, maskNet):

		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
									 (104.0, 177.0, 123.0))

		faceNet.setInput(blob)
		detections = faceNet.forward()


		faces = []
		locs = []
		preds = []


		for i in range(0, detections.shape[2]):

			confidence = detections[0, 0, i, 2]
			conf = 0.5

			if confidence > conf:

				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")


				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				face = frame[startY:endY, startX:endX]
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis=0)

				faces.append(face)
				locs.append((startX, startY, endX, endY))


		if len(faces) > 0:
			preds = maskNet.predict(faces)
		return (locs, preds)

	def get_current_frame(self):

		# load our serialized face detector model from disk
		print("[INFO] loading face detector model...")
		
		# load the face mask detector model from disk
		print("[INFO] loading face mask detector model...")
		
		# initialize the video stream and allow the camera sensor to warm up
		print("[INFO] starting video stream...")

		frame = self.vs.read()
		frame = imutils.resize(frame, width=650)
		frame = cv2.flip(frame, 1)

		(locs, preds) = self.detect_and_predict_mask( frame, faceNet, maskNet)

		for (box, pred) in zip(locs, preds):

			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		# show the output frame
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()
	

	


	

	
