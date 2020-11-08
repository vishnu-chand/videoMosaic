from .pixCommon import *


def frameFit(img, bbox):
    '''
    ensure the bbox will not go away from the image boundary
    '''
    imHeight, imWidht = img.shape[:2]
    x0, y0, width, height = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = x0 + int(width), y0 + int(height)
    x1, y1 = min(x1, imWidht), min(y1, imHeight)
    return np.array((x0, y0, max(0, x1 - x0), max(0, y1 - y0)))


def bbox2cent(bbox):
    '''
    calculate the centroid of bbox
    '''
    x, y, w, h = bbox
    return np.array((x + w / 2, y + h / 2))


def img2bbox(img):
    return np.array([0, 0, img.shape[1], img.shape[0]])


def bboxScale(img, bbox, scaleWH):
    try:
        sw, sh = scaleWH
    except:
        sw, sh = scaleWH, scaleWH
    x, y, w, h = bbox
    xc, yc = (x + w / 2, y + h / 2)
    w *= sw
    h *= sh
    x, y = xc - w / 2, yc - h / 2
    return frameFit(img, (x, y, w, h))


def bbox2dbbox(img, bbox):
    '''
    convert cv bbox to dlib bbox
    '''
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    return dlib.rectangle(x, y, x + w, y + h)


def dbbox2bbox(img, dbbox):
    '''
    convert dlib bbox to cv bbox
    '''
    x = dbbox.left()
    y = dbbox.top()
    w = dbbox.right() - x
    h = dbbox.bottom() - y
    return frameFit(img, (x, y, w, h))


def getTrajectory(p1, p2, returnSemiCircle=False):
    '''
    when returnSemiCircle is true  it will return in range -180 to 180
    when returnSemiCircle is false it will return in range 0 to 360
            self.matrix, self.chessImgW, self.chessImgH = None, None, None

    '''
    delta = np.array(p1) - np.array(p2)
    theta = cv2.fastAtan2(delta[1], delta[0])
    magnitude = cv2.norm(np.array(delta))
    if returnSemiCircle:
        if theta > 180:
            theta -= 360
    return np.array((magnitude, theta))


def skeletonIt(img):
    skel = np.zeros_like(img)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = img.size - cv2.countNonZero(img)
        if zeros == img.size:
            done = True
    return skel


def getOverlap(bbox1, bbox2, reference='iou'):
    x1min, y1min, x1max, y1max = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x2min, y2min, x2max, y2max = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    width_of_overlap_area = min(x1max, x2max) - max(x1min, x2min)
    height_of_overlap_area = min(y1max, y2max) - max(y1min, y2min)
    score = 0.0
    if width_of_overlap_area > 0 and height_of_overlap_area > 0:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
        if reference == 'iou':
            box_1_area = (y1max - y1min) * (x1max - x1min)
            box_2_area = (y2max - y2min) * (x2max - x2min)
            denominator = box_1_area + box_2_area - area_of_overlap
        elif reference == 'minBbox':
            bbox1Area = bbox1[2] * bbox1[2]
            bbox2Area = bbox2[2] * bbox2[2]
            denominator = bbox1Area if bbox1Area < bbox2Area else bbox2Area
        elif reference == 'maxBbox':
            bbox1Area = bbox1[2] * bbox1[2]
            bbox2Area = bbox2[2] * bbox2[2]
            denominator = bbox1Area if bbox1Area > bbox2Area else bbox2Area
        else:  # a reference bbox has been passed
            denominator = reference[2] * reference[3]
        score = area_of_overlap / denominator
    return score

# class ImLog:
#     def __init__(self):
#         self.__imgs = []
#         self.__rcsize = None
#         self.size = 0
#
#     def log(self, name, img, loc=(30, 30), color=(255, 255, 255), txtSize=1, txtFont=cv2.FONT_HERSHEY_SIMPLEX, txtThickness=3, txtColor=None):
#         self.size += 1
#         img = bboxLabel(img.copy(), name, loc, color, 3, txtSize, txtFont, txtThickness, txtColor)
#         self.__imgs.append(img)
#         return self
#
#     def reset(self):
#         self.__imgs = []
#         self.__rcsize = None
#         self.size = 0
#
#     def setSize(self, rcsize):
#         self.__rcsize = rcsize[:2]
#
#     def getImg(self, rcsize=None, nCol=None, resize_method=cv2.INTER_LINEAR, fit=False, asgray=False):
#         if rcsize is not None:
#             self.__rcsize = rcsize
#         if self.__rcsize is None:  # take the first image size
#             self.__rcsize = self.__imgs[0].shape[:2]
#         img = photoframe(self.__imgs, self.__rcsize, nCol, resize_method, fit, asgray)
#         self.reset()
#         return img
#
#
# def simpleTrackBar(readOnlyInputImg, trackBarsLengths, winname='trackBar'):
#     cv2.namedWindow(winname, 0)
#     for ix, maxValue in enumerate(trackBarsLengths):
#         cv2.createTrackbar(str(ix), winname, 0, maxValue, lambda x: None)
#     outputImg = readOnlyInputImg.copy()
#     while True:
#         cv2.imshow(winname, outputImg)
#         k = cv2.waitKey(100) & 0xFF
#         if k == 27:
#             break
#         vals = [readOnlyInputImg, outputImg]
#         for ix, _ in enumerate(trackBarsLengths):
#             vals.append(cv2.getTrackbarPos(str(ix), winname))
#         yield vals
#
#
# class __MarkRoi:
#     def __init__(self, oimg, color):
#         self.contours = []
#         self.oimg = oimg.copy()
#         self.dispImg = self.oimg.copy()
#         self.color = color
#
#     def reset(self):
#         self.contours = []
#         self.dispImg = self.oimg.copy()
#
#     def drawRoi(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDBLCLK:
#             self.dispImg = self.oimg.copy()
#             self.contours.append((x, y))
#             contour = np.array(self.contours).astype(np.int32)
#             cv2.drawContours(self.dispImg, [contour], -1, self.color, 10)
#
#     def markRoi(self, key=ord('m'), winname='mark roi by double click'):
#         res = None
#         if key == ord('m'):
#             cv2.namedWindow(winname, 0)
#             cv2.setMouseCallback(winname, self.drawRoi)
#             while 1:
#                 cv2.imshow(winname, self.dispImg)
#                 k = cv2.waitKey(1) & 0xFF
#                 if k == 13:
#                     break
#                 if k == 27:
#                     self.reset()
#             val = np.array([np.array(self.contours).astype(np.int32)])
#             if val.any():
#                 res = val
#         cv2.destroyAllWindows()
#         return res
#
#
# def markRoi(oimg, key=ord('m'), winname='mark roi by double click', color=(255, 0, 0)):
#     return __MarkRoi(oimg, color).markRoi(key, winname)
