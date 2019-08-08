import cv2
from threading import Thread
import os
import numpy as np


class Operations():

    def __init__(self):
        pass

    def selecting_boxes(self,image_np, boxes, W, H):

        '''

        selecting_boxes() takes boxes and distinguish them according to scores in order to send to tracker.

        :param image_np: this image from webcamera in order to determine the rectangles to send to tracker.
        :param boxes: These belongs to coco model and they will select if the score is bigger than 0.7
        :param scores: Scores determine which boxes either show a people or not
        :param W: W indicate the Weight of the image
        :param H: H indicate the Weight of the image
        :return: rectangle which was manipulated according to scores and these rectangle goes to tracker
        as a parameter.

        '''

        self.rectangle = []


        # print("Number of scores : ", len(scores), "Number of boxes : ", len(boxes))
        # print(scores[0])




            # if value > 0.8:

        # (left, right, top, bottom) = (int(boxes[order][1] * W), int(boxes[order][3] * W),
        #                         int(boxes[order][0] * H), int(boxes[order][2] * H))
        (left, right, top, bottom) = (int(boxes[0]), int(boxes[2]), int(boxes[1]), int(boxes[3]))
        self.rectangle.append([left, right, top, bottom])

        cv2.rectangle(image_np, (left, top), (right, bottom), (0, 255, 0), 2)
        # cv2.rectangle(image_np, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)

        return self.rectangle

    def adjusting_boxes(self, control_frame, bounding_box, image, model):

        '''

            It takes bounding boxes and object that come from tracker and visualize the information through the
            method of opencv libraries.

        :param objects: are coordinates of boxes that is coming from ssd mobilnet model.
        :param bounding_box: are boxes of each object in the frame from tracker.
        :param image: was caputred from VideoCapture via cv2
        :param model: refers to prediction model, in this case it uses our model that trained with collected images from
        non-copyright image websites.
        :return: does not return back anything but it returns the images as written categories and drawn circle
        '''

        path_of_data = 'Increased6(60x60)\\'
        categories = os.listdir(path_of_data)

        # value = list(objects.values())
        # if not list(objects.values()):
        #     pass

        for index, value in enumerate(bounding_box):
            for index2, value2 in enumerate(value):
                if value2 < 0:
                    index_of_negative = value.index(value2)
                    bounding_box[index][index_of_negative] = 0


        # print("Deneme", bounding_box)

        for sep2 in bounding_box:

            # value = list(objects.values())[bounding_box.index(sep2)]
            if sep2 != [] and control_frame == True:
                cropped_image = image[sep2[2]:sep2[3], sep2[0]:sep2[1]]

                # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                cropped_image = np.array(cropped_image, dtype=np.uint8)
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                gray = gray.astype(np.float32)

                gray = cv2.resize(gray, (60, 60))
                gray = gray.reshape(-1, 60, 60, 1)

                prediction = model.predict(gray)
                prediction = prediction.tolist()

                max_value_of_prediction = max(prediction[0])
                our_prediction = prediction[0].index(max_value_of_prediction)

                text = categories[our_prediction]
                print(text)
                # if value[0] or value[1]:
                #     pass


                cv2.putText(image, text, (int(sep2[1]/2) -10, sep2[2] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # cv2.circle(image, (sep2[1]-70, sep2[2] + 20), 4, (0, 255, 0), -1)
            else:
                pass

    def scale(self, data, min_value, max_value):

        '''

            This method getting some data in and then scale with respect to given
            parameter of value of min and max.

        :param data: it contains the data that will be normalized
        :param min_value: minimum value of scaling
        :param max_value: maximum value of scaling
        scaling process value will be between min_value and max_value.

        :return: it will return the data that was scaled

        '''

        nom = (data - data.min(axis=0)) * (max_value - min_value)
        denom = data.max(axis=0) - data.min(axis=0)
        denom[denom == 0] = 1
        return min_value + nom / denom

