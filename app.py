from flask import Flask, render_template , request , jsonify, send_file
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from yolo_detection_images import runModel
from datetime import datetime

app = Flask(__name__)

############################################## THE REAL DEAL ###############################################
@app.route('/detectObject' , methods=['POST'])
def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################

	img = runModel(img)

	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})

##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

@app.route('/test' , methods=['GET','POST'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})

@app.route('/get_doc_content')
def get_doc_content():
    # Đường dẫn đến tệp doc.txt, hãy đảm bảo rằng đường dẫn là chính xác
    doc_path = 'D:\Learning\webapp\detect_result.txt'
    with open(doc_path, 'r', encoding='utf-8') as file:
        content = file.read()
    parts = content.split()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return {'img_name': parts[0], 'ID': parts[1], 'time': current_time}

@app.route('/')
def home():
	return render_template('./index.html')

	
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run(debug = True)
