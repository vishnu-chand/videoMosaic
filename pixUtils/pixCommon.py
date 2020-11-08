import os
import re
import cv2
import ast
import sys
import json
import time
import shutil
import pickle
import argparse
import traceback
import numpy as np
from glob import glob
from tqdm import tqdm
import subprocess as sp
from os.path import join
from pathlib import Path
from os.path import exists
from os.path import dirname
from os.path import basename
from itertools import groupby
from itertools import zip_longest
from itertools import permutations
from itertools import combinations
from collections import OrderedDict
from collections import defaultdict
from datetime import datetime as dt

try:
    import yaml
except:
    pass

try:
    from matplotlib import pyplot as plt
except:
    pass

try:
    import dlib
except:
    pass

try:
    from PIL import Image
except:
    pass

try:
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
except:
    pass

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, formatter={'float': lambda x: "{0:8.3f}".format(x)})


class DotDict(dict):
    def __init__(self, datas=None):
        super().__init__()
        if isinstance(datas, argparse.Namespace):
            datas = vars(datas)
        datas = dict() if datas is None else datas
        for k, v in datas.items():
            self[k] = v

    def __getattr__(self, key):
        if key not in self:
            print("56 __getattr__ pixCommon key: ", key)

            raise AttributeError(key)
        else:
            return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        keys = list(self.keys())
        nSpace = len(max(keys, key=lambda x: len(x))) + 2
        keys = sorted(keys)
        data = [f'{key:{nSpace}}: {self[key]},' for key in keys]
        data = '{\n%s\n}' % '\n'.join(data)
        return data

    def copy(self):
        return DotDict(super().copy())

    def toJson(self):
        res = OrderedDict()
        for k, v in self.items():
            try:
                json.dumps({k: v})
                res[k] = v
            except:
                res[k] = str(v)
        return json.dumps(res)

    def toDict(self):
        res = OrderedDict()
        for k, v in self.items():
            try:
                json.dumps({k: v})
                res[k] = v
            except:
                res[k] = str(v)
        return res


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def readYaml(src, defaultDict=None):
    if os.path.exists(src) or defaultDict is None:
        with open(src, 'r') as book:
            data = yaml.safe_load(book)
    else:
        data = defaultDict
    return DotDict(data)


def writeYaml(yamlPath, jObjs):
    with open(yamlPath, 'w') as book:
        yaml.dump(yaml.safe_load(jObjs), book, default_flow_style=False, sort_keys=False)


def readPkl(pklPath, defaultData=None):
    if not os.path.exists(pklPath):
        print("loading pklPath: ", pklPath)
        return defaultData
    return pickle.load(open(pklPath, 'rb'))


def writePkl(pklPath, objs):
    pickle.dump(objs, open(dirop(pklPath), 'wb'))


def dir2(var):
    '''
    list all the methods and attributes present in object
    '''
    for v in dir(var):
        print(v)
    print("34 dir2 common : ", );
    quit()


# def checkAttr(obj, b, getAttr=False):
#     a = set(vars(obj).keys())
#     if getAttr:
#         print(a)
#     extra = a - a.intersection(b)
#     if len(extra):
#         raise Exception(extra)


def bboxLabel(img, txt="", loc=(30, 30), color=(255, 255, 255), thickness=3, txtSize=1, txtFont=cv2.QT_FONT_NORMAL, txtThickness=3, txtColor=None):
    if len(loc) == 4:
        x0, y0, w, h = loc
        x0, y0, rw, rh = int(x0), int(y0), int(w), int(h)
        cv2.rectangle(img, (x0, y0), (x0 + rw, y0 + rh), list(color), thickness)
    else:
        x0, y0, rw, rh = int(loc[0]), int(loc[1]), 0, 0
    txt = str(txt)
    if txt != "":
        if txtColor is None:
            txtColor = (0, 0, 0)
        (w, h), baseLine = cv2.getTextSize(txt, txtFont, txtSize, txtThickness)
        # baseLine -> to fit char like p,y in box
        cv2.rectangle(img, (x0, y0 + rh), (x0 + w, y0 + rh - h - baseLine), color, -1)
        cv2.putText(img, txt, (x0, y0 + rh - baseLine), txtFont, txtSize, txtColor, txtThickness, cv2.LINE_AA)
    return img


def drawText(img, txt, loc, color=(255, 255, 255), txtSize=1, txtFont=cv2.FONT_HERSHEY_SIMPLEX, txtThickness=3, txtColor=None):
    (w, h), baseLine = cv2.getTextSize(txt, txtFont, txtSize, txtThickness)
    x0, y0 = int(loc[0]), int(loc[1])
    if txtColor is None:
        txtColor = (0, 0, 0)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 - h - baseLine), color, -1)
    cv2.putText(img, txt, (x0, y0 - baseLine), txtFont, txtSize, txtColor, txtThickness)
    return img


def putSubImg(mainImg, subImg, loc, interpolation=cv2.INTER_CUBIC):
    '''
    place the sub image inside the genFrame image
    '''
    if len(loc) == 2:
        x, y = int(loc[0]), int(loc[1])
        h, w = subImg.shape[:2]
    else:
        x, y, w, h = int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
        subImg = cv2.resize(subImg, (w, h), interpolation=interpolation)
    x, y, w, h = frameFit(mainImg, (x, y, w, h))
    mainImg[y:y + h, x:x + w] = getSubImg(subImg, (0, 0, w, h))
    return mainImg


def getSubImg(im1, bbox):
    '''
    crop sub image from the given input image and bbox
    '''
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    img = im1[y:y + h, x:x + w]
    if img.shape[0] and img.shape[1]:
        return img


def maskIt(roi, roiMask):
    '''
    apply mask on the image. It can accept both gray and colors image
    '''
    if len(roi.shape) == 3 and len(roiMask.shape) == 2:
        roiMask = cv2.cvtColor(roiMask, cv2.COLOR_GRAY2BGR)
    elif len(roi.shape) == 2 and len(roiMask.shape) == 3:
        roiMask = cv2.cvtColor(roiMask, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(roi, roiMask)


def imResize(img, sizeRC=None, scaleRC=None, interpolation=cv2.INTER_LINEAR):
    if sizeRC is not None:
        r, c = sizeRC[:2]
    else:
        try:
            dr, dc = scaleRC
        except:
            dr, dc = scaleRC, scaleRC
        r, c = img.shape[:2]
        r, c = r * dr, c * dc
    if interpolation == 'aa':
        img = np.array(Image.fromarray(img).resize((int(c), int(r)), Image.ANTIALIAS))
    else:
        img = cv2.resize(img, (int(c), int(r)), interpolation=interpolation)
    return img


def imHconcat(imgs, sizeRC, interpolation=cv2.INTER_LINEAR):
    rh, rw = sizeRC[:2]
    res = []
    for queryImg in imgs:
        qh, qw = queryImg.shape[:2]
        queryImg = cv2.resize(queryImg, (int(rw * qw / qh), int(rh)), interpolation=interpolation)
        res.append(queryImg)
    return cv2.hconcat(res)


def imVconcat(imgs, sizeRC, interpolation=cv2.INTER_LINEAR):
    rh, rw = sizeRC[:2]
    res = []
    for queryImg in imgs:
        qh, qw = queryImg.shape[:2]
        queryImg = cv2.resize(queryImg, (int(rw), int(rh * qh / qw)), interpolation=interpolation)
        res.append(queryImg)
    return cv2.vconcat(res)


class VideoWrtier:
    """mjpg xvid mp4v"""

    def __init__(self, path, camFps, size=None, codec='mp4v'):
        self.path = path
        try:
            self.fps = camFps.get(cv2.CAP_PROP_FPS)
        except:
            self.fps = camFps
        self.__vWriter = None
        self.__size = size
        self.__codec = cv2.VideoWriter_fourcc(*(codec.upper()))
        print("writing :", path, '@', self.fps, 'fps')

    def write(self, img):
        if self.__vWriter is None:
            if self.__size is None:
                self.__size = tuple(img.shape[:2])
            self.__vWriter = cv2.VideoWriter(self.path, self.__codec, self.fps, self.__size[::-1])
        if tuple(img.shape[:2]) != self.__size:
            img = cv2.resize(img, self.__size)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.__vWriter.write(img)

    def close(self):
        self.__vWriter.release()


class clk:
    def __init__(self):
        self.tik = dt.now()
        self.__lapse = 0

    def tok(self, reset=True):
        lapse = dt.now() - self.tik
        self.__lapse = lapse.seconds + (lapse.microseconds / 1000000)
        if reset:
            self.tik = dt.now()
        return self

    def fps(self, nFrames, roundBy=4):
        lapse = nFrames / self.__lapse
        # self.__lapse = 0
        return round(lapse, roundBy) if roundBy else int(lapse)

    def __repr__(self):
        lapse = self.__lapse
        # self.__lapse = 0
        return str(round(lapse, 4))

    def sec(self, roundBy=4):
        lapse = self.__lapse
        # self.__lapse = 0
        return round(lapse, roundBy) if roundBy else int(lapse)


# class Clock:
#     def __init__(self):
#         self.__lapse = 0
#         self.__tik = dt.now()
#         self.__cycle = 0
#
#     def tik(self):
#         self.__tik = dt.now()
#
#     def tok(self, reset=True):
#         lapse = dt.now() - self.__tik
#         lapse = lapse.seconds + (lapse.microseconds / 1000000)
#         self.__lapse += lapse
#         self.__cycle += 1
#         return f'{round(lapse, 4)} {self.__cycle}'
#
#     def __repr__(self):
#         lapse = self.__lapse / self.__cycle
#         return f'{round(lapse, 4)} {round(1 / lapse, 4)}'


class Wait:
    def __init__(self):
        self.pause = False

    def __call__(self, delay=1):
        if self.pause:
            delay = 0
        key = cv2.waitKey(delay)
        if key == 32:
            self.pause = True
        if key == 13:
            self.pause = False
        return key


__wait = Wait()


def showImg(winname='output', imC=None, delay=None, windowConfig=0, nRow=None, chFirst=False):
    winname = str(winname)
    if imC is not None:
        if type(imC) is not list:
            imC = [imC]
        imC = photoframe(imC, nRow=nRow, chFirst=chFirst)
        cv2.namedWindow(winname, windowConfig)
        cv2.imshow(winname, imC)

    if delay is not None:
        key = __wait(delay)
        return key
    return imC


def pshowImg(winname=None, imC=None, delay=0):
    winname = str(winname)
    if imC is not None:
        if type(imC) is list:
            pass
        plt.imshow(imC)
    if delay is not None:
        if delay == 0:
            plt.show()
        # else:
        #     plt.pause(delay / 1000)
        return 1
    return imC


def str2path(*dirpath):
    dirpath = list(map(str, dirpath))
    path = join(*dirpath)
    if path.startswith('home/ec2-user'):
        path = join('/', path)
    return path


def moveCopy(src, des, op, isFile, remove):
    des = str2path(des)
    if isFile and not os.path.splitext(des)[-1]:
        raise Exception(f'''Fail des: {des}
                                    should be file''')
    if not remove and exists(des):
        raise Exception(f'''Fail des: {des}
                                    already exists delete it before operation''')
    if isFile:
        if remove and exists(des):
            os.remove(des)
        mkpath = dirname(des)
        if not exists(mkpath):
            os.makedirs(mkpath)
    else:
        if remove and exists(des):
            shutil.rmtree(des, ignore_errors=True)
    return op(src, des)


def dirop(*dirpath, **kw):
    mkdir, remove, mode = kw.get('mkdir', True), kw.get('remove'), kw.get('mode', 0o777)
    copyTo, moveTo = kw.get('copyTo'), kw.get('moveTo')
    path = str2path(*dirpath)
    isFile = os.path.splitext(path)[-1]
    if copyTo or moveTo:
        if not exists(path):
            raise Exception(f'''Fail src: {path} 
                                            not found''')
    elif remove is True and exists(path):
        if isFile:
            os.remove(path)
        else:
            shutil.rmtree(path, ignore_errors=True)
    mkpath = dirname(path) if isFile else path
    if mkdir and not exists(mkpath) and mkpath:
        os.makedirs(mkpath)
    if copyTo:
        copy = shutil.copy if isFile else shutil.copytree
        path = moveCopy(path, copyTo, copy, isFile, remove=remove)
    elif moveTo:
        path = moveCopy(path, moveTo, shutil.move, isFile, remove=remove)
    return path

def downloadDB(link, des, remove=False):
    dirop(des)
    os.system(f'cd {des};wget -nd -c "{link}"')
    unzipIt(join(des, basename(link)), des, remove=remove)


def zipIt(src, desZip, remove=False):
    if not exists(src):
        raise Exception(f'''Fail src: {src} \n\tnot found''')
    if exists(desZip):
        if remove:
            os.remove(desZip)
        else:
            raise Exception(f'''Fail des: {desZip} \n\talready exists delete it before operation''')
    desZip, zipExt = os.path.splitext(desZip)
    if os.path.isfile(src):
        tempDir = join(dirname(src), getTimeStamp())
        if os.path.exists(tempDir):
            raise Exception(f'''Fail tempDir: {tempDir} \n\talready exists delete it before operation''')
        os.makedirs(tempDir)
        shutil.copy(src, tempDir)
        desZip = shutil.make_archive(desZip, zipExt[1:], tempDir)
        shutil.rmtree(tempDir, ignore_errors=True)
    else:
        desZip = shutil.make_archive(desZip, zipExt[1:], src)
    return desZip


def unzipIt(src, desDir, remove=False):
    if not exists(src):
        raise Exception(f'''Fail src: {src} \n\tnot found''')
    if os.path.splitext(desDir)[-1]:
        raise Exception(f'''Fail desDir: {desDir} \n\tshould be folder''')
    tempDir = join(dirname(desDir), getTimeStamp())
    shutil.unpack_archive(src, tempDir)
    if not exists(desDir):
        os.makedirs(desDir)
    for mvSrc in os.listdir(tempDir):
        mvSrc = join(tempDir, mvSrc)
        mvDes = join(desDir, basename(mvSrc))
        if remove is True and exists(mvDes):
            if os.path.isfile(mvDes):
                os.remove(mvDes)
            else:
                shutil.rmtree(mvDes, ignore_errors=True)
        try:
            shutil.move(mvSrc, desDir)
        except Exception as exp:
            shutil.rmtree(tempDir, ignore_errors=True)
            raise Exception(exp)
    shutil.rmtree(tempDir, ignore_errors=True)
    return desDir


# def float2img(img, pixmin=0, pixmax=255, dtype=0):
#     '''
#     convert oldFeature to (0 to 255) range
#     '''
#     return cv2.normalize(img, None, pixmin, pixmax, 32, dtype)


def float2img(img, min=None, max=None):
    min = img.min() if min is None else min
    max = img.max() if max is None else max
    img = img.astype('f4')
    img -= min
    img /= max
    return (255 * img).astype('u1')


def photoframe(imgs, rcsize=None, nRow=None, resize_method=cv2.INTER_LINEAR, fit=False, asgray=False, chFirst=False):
    '''
    # This method pack the array of images in a visually pleasing manner.
    # If the nCol is not specified then the nRow and nCol are equally divided
    # This method can automatically pack images of different size. Default stitch size is 128,128
    # when fit is True final photo frame size will be rcsize
    #          is False individual image size will be rcsize
    # Examples
    # --------
        video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), custom_fn=None)
        for fnos, imgs in video.chunk(4):
            i1 = photoframe(imgs, nCol=None)
            i2 = photoframe(imgs, nCol=4)
            i3 = photoframe(imgs, nCol=4, rcsize=(200,300),nimgs=7)
            i4 = photoframe(imgs, nCol=3, nimgs=7)
            i5 = photoframe(imgs, nCol=4, rcsize=imgs[0].shape)
            i6 = photoframe(imgs, nCol=6, rcsize=imgs[0].shape, fit=True)
            i7 = photoframe(imgs, nCol=4, rcsize=imgs[0].shape, fit=True, asgray=True)
            for i, oldFeature in enumerate([i1, i2, i3, i4, i5, i6, i7], 1):
                print(i, oldFeature.shape)
                win('i%s' % i, )(oldFeature)
            win('totoal')(photoframe([i1, i2, i3, i4, i5, i6, i7]))
            if win().__wait(waittime) == 'esc':
                break
    '''
    if len(imgs):
        if chFirst:
            imgs = np.array([np.transpose(img, [1, 2, 0]) for img in imgs])
        if rcsize is None:
            rcsize = imgs[0].shape
        imrow, imcol = rcsize[:2]  # fetch first two vals
        nimgs = len(imgs)
        nRow = int(np.ceil(nimgs ** .5)) if nRow is None else int(nRow)
        nCol = nimgs / nRow
        nCol = int(np.ceil(nCol + 1)) if (nRow * nCol) - nimgs else int(np.ceil(nCol))
        if fit:
            imrow /= nRow
            imcol /= nCol
        imrow, imcol = int(imrow), int(imcol)
        resshape = (imrow, imcol) if asgray else (imrow, imcol, 3)
        imgs = zip_longest(list(range(nRow * nCol)), imgs, fillvalue=np.zeros(resshape, imgs[0].dtype))
        resimg = []
        for i, imggroup in groupby(imgs, lambda k: k[0] // nCol):
            rowimg = []
            for i, img in imggroup:
                if img.dtype != np.uint8:
                    img = float2img(img)
                if asgray:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 1:
                    img = img.reshape(*img.shape[:2])
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if tuple(img.shape) != resshape:
                    img = cv2.resize(img, (imcol, imrow), interpolation=resize_method)
                rowimg.append(img)
            resimg.append(cv2.hconcat(rowimg))
        return cv2.vconcat(resimg)


def getTimeStamp():
    return dt.now().strftime("%d%b%H%M%S_%f")


def replaces(path, *words):
    path = str(path)
    for word in words:
        path = path.replace(*word)
    return path
