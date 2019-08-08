import cv2
import numpy as np
from Boxes_Drawing import Operations
from Object_Tracking import CentroidTracker
import tensorflow as tf
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




# read pre-trained model and config file



# function to get the output layer names
# in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

operations = Operations()
ct = CentroidTracker()

necessary_process = Operations()


# for each detetion from each output layer
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)


model = tf.keras.models.load_model(r"models\CNN-4")

net = cv2.dnn.readNet(r"models\yolov3.weights", r"models\yolov3.cfg")

# read class names from text file
classes = None
with open(r"models\yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]


# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def detection_people(controlling_frame, frame):

    (H, W) = (None, None)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(frame, scale, (90, 90), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))


    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:

        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8 and class_id == 0:

                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([round(x), round(y), round(w+x), round(h+y)])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        boxes2 = [round(x), round(y), round(x + w), round(y + h)]
        rectangles = necessary_process.selecting_boxes(frame, boxes2, W, H)
#        print(boxes2)
        objects = ct.update(rectangles)
        bounding_boxes = ct.bounding_box2
#        print(len(bounding_boxes))
#        print(bounding_boxes)
        necessary_process.adjusting_boxes(controlling_frame, bounding_boxes, frame, model)
        
