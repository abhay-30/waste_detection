# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
# from BankNotes import BankNote

import yolov5

# load model
model = yolov5.load('keremberke/yolov5m-garbage')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image



# 2. Create the app object
app = FastAPI()
# pickle_in = open("classifier.pkl","rb")
# classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
 
# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def waste_detect(img_url):

    # set image
    img =  img_url

    # perform inference
    results = model(img, size=640)

    # inference with test time augmentation
    results = model(img, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4]  # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    # show detection bounding boxes on image
    results.show()

    print(results)  # results output image 1/1: 720x1280 2 biodegradables, 1 paper

    sample = str(results).split()

    # print(type(results))

    # regex to match the particular things

    ct = 0
    last_word = ''
    for words in sample:
        if(words == 'paper' or words == 'plastic' or words == 'rubber'):
            ct += int(last_word)
        else:
            last_word = words

    if(ct > 5):
         return {'message':'true'}
    else:
          return {'message':'true'}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload