from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:

    def __init__(self, maxDisappearred = 3):

        '''

        :param maxDisappearred: bir nesne yakalandıktan sonra yeni bir ID için kaç kare geçmesi gerektiğini
        belirtmektedir.
        '''

        self.bounding_box2 = []
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappearred

    def register(self, centroid):

        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):

        del self.objects[objectID]
        del self.disappeared[objectID]


    def update(self, rects):

        self.bounding_box2 =[]

        # If there is no bounding box after 3 frame it will be deleted in the list

        if len(rects) == 0:

            for objectID in self.disappeared.keys():
                self.disappeared[objectID] +=1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        # If list of rects is not empty, added the bounding boxes center coordinate in the inputCentroids
        # numpy list

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for(i, (top, left, bottom, right)) in enumerate(rects):


            cX = top
            cY = right

            inputCentroids[i] = (cX,cY)

        if len(self.objects) == 0:

            for i in range(0,len(inputCentroids)):
                self.register(inputCentroids[i])

        # Getting existed objects of center coordinate in order to compare
        else:

            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids),inputCentroids)

            # Looking for nearest row and columns to measure Euclidean distances.

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]

                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # Finding which coordinate belongs to which bounding box

                itemindex = np.where(inputCentroids == inputCentroids[col])

                row_1 , col_1 =itemindex

                sıra = row_1[0]

                self.bounding_box2.append(rects[sıra])
#                print("rect sayısı", len(rects))


                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0,D.shape[0])).difference(usedRows)
            unusedCols = set(range(0,D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])


        return self.objects

