
from django.shortcuts import render
import pandas as pd
import json
import subprocess
import time
from django.shortcuts import redirect,HttpResponse
from django.shortcuts import render
import cv2
import threading
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from rjvs_port_app.camera import FaceDetect
from rjvs_port_app.train_mask_detection import FaceMaskTrain
from rjvs_port_app.testclass import sampleDetect



# Create your views here.


# create a function
def index(request):
	return render(request, "index.html")
	
#################################Jupyter Notebook##########################################
def open_jupiter_notbook(request):
    b= subprocess.check_output("jupyter-lab list".split()).decode('utf-8')
    if "9999" not in b:
        a=subprocess.Popen("jupyter-lab  --no-browser --port 9999".split())
    start_time = time.time()
    unreachable_time = 10
    while "9999" not in b:
        timer = time.time()
        elapsed_time = timer-start_time
        b= subprocess.check_output("jupyter-lab list".split()).decode('utf-8')
        if "9999" in b:
            break
        if elapsed_time > unreachable_time:
            return HttpResponse("Unreachable")
    path = b.split('\n')[1].split('::',1)[0]

    return render(request, 'Project1.html')


#################################Facemask Detection##########################################
def Face_Mask(request):
	return render(request, 'Face_Mask.html')

def gen(camera):
	while True:
		frame = camera.get_current_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		
def facecam_feed(request):
	return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')



#################################Facemask Training Detection##########################################


def Face_Mask_train(request ):
    reader = FaceMaskTrain()
    rets = reader.training()
    return render(request, 'Facemask_train.html')

def train_mask(train_mask_detection):
    train_mask_detection.FaceMaskTrain(training())
    return redirect('Face_Mask_train')









