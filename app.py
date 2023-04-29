import uvicorn
from fastapi import FastAPI, HTTPException
import requests
import io
from PIL import Image
import yolov5


app = FastAPI()

# load model
model = yolov5.load('keremberke/yolov5m-garbage')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image


@app.get('/predict')
async def process_url(url: str):
    # make a GET request to the URL
    print(url)
    print("\n\n\n\n")
    
    img = url
    results = model(img, size=640)

    # show detection bounding boxes on image
    results.show()

    print(results)  # results output image 1/1: 720x1280 2 biodegradables, 1 paper

    # count the number of detected objects
    count = 0
    for label in results.names:
        if label in ['paper', 'plastic', 'rubber']:
            count += results.labels.count(label)

    # return the result
    if count > 5:
        return {'message': 'true'}
    else:
        return {'message': 'false'}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
