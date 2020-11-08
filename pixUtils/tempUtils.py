import logging
from pixUtils.pixCommon import *


def videoPlayer(vpath, startSec=0.0, stopSec=np.inf):
    cam = vpath if type(vpath) == cv2.VideoCapture else cv2.VideoCapture(vpath)
    ok, fno = True, startSec
    cam.set(cv2.CAP_PROP_POS_MSEC, fno * 1000)
    videoPlayer.cam = cam
    while ok:
        ok, img = cam.read()
        ok = img is not None and fno < stopSec
        if ok:
            fno = round(cam.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
            '''pre processing'''
            yield fno, img


def video2wav(vpath, apath, startStop=(0, 999999)):
    start, stop = startStop
    duration = stop - start
    cmd = 'ffmpeg -ss {start} -t {duration} -i {vpath} -acodec pcm_s16le -ac 2 {apath}'
    os.system('%s -y -loglevel warning' % cmd.format(**locals()))


def stitchVideoAudio(video, audio, output):
    opath, ext = os.path.splitext(output)
    opath = opath + '_temp' + ext
    apath = audio
    if not apath.endswith('.wav'):
        apath = 'temp.wav'
        cmd = 'ffmpeg -i {audio} -acodec pcm_s16le -ac 2 {apath}'
        os.system('%s -y -loglevel warning' % cmd.format(**locals()))
    cmd = 'ffmpeg -i {video} -i {apath} -strict -2 {opath}'
    os.system('%s -y -loglevel warning' % cmd.format(**locals()))
    if os.path.exists('temp.wav'):
        os.remove('temp.wav')
    shutil.move(opath, output)


def compareVersions(versions, compareBy, putTitle=bboxLabel, bbox=None, showDiff=True):
    vpaths = [compareBy] + [version for version in versions if version != compareBy]
    vplayers = [videoPlayer(version) for version in vpaths]
    for ix, data in enumerate(zip(*vplayers)):
        imgs = []
        for vpath, (fno, img) in zip(vpaths, data):
            if bbox is True:
                winname = "select roi"
                cv2.namedWindow(winname, 0)
                bbox = cv2.selectROI(winname, img)
                cv2.destroyWindow(winname)
            if bbox is not None:
                img = getSubImg(img, bbox)
            # img = bboxLabel(img, basename(vpath))
            imgs.append(img)
        res = []
        for ix, img in enumerate(imgs[1:], 1):
            res.append(putTitle(imgs[0].copy(), basename(vpaths[0])))
            res.append(putTitle(imgs[ix].copy(), basename(vpaths[ix])))
            if showDiff:
                diff = cv2.absdiff(imgs[0], img)
                res.append(diff)
                res.append(cv2.inRange(diff.min(axis=-1), 10, 300))
        yield res


#
#
# def rotateVideo(vpath, angle, opath):
#     angleMap = {90: "transpose=1",180: "transpose=1,transpose=1", 270: "transpose=2"}
#     angle = angleMap[angle]
#     cmd = 'ffmpeg -i {vpath} -vf {angle} -strict -2 {opath}'
#     os.system('%s -y -loglevel warning' % cmd.format(**locals()))
#
#
# def stitchVideos(inputs, output, videoPlayer2=videoPlayer):
#     vWriter = VideoWrtier(output, cv2.VideoCapture(inputs[0]))
#     cams = [videoPlayer2(cam) for cam in inputs]
#     for datas in zip(*cams):
#         img = photoframe([data for _, data in datas])
#         vWriter.write(img)
#     vWriter.close()
#
#
#
# def makeWordVideo(layVideoPath, wordVideoPath, outputPath, args):
#     dirop(outputPath, remove=True)
#     personalizationStart, personalizationEnd = args.personalizationStartStop[0]
#     sceneStart, sceneEnd = args.sceneStartStop
#
#     sceneDuration = personalizationStart - sceneStart
#     personalizationDuration = sceneEnd - personalizationEnd
#     personalizationVideoPath = layVideoPath
#     print("writing", outputPath)
#
#     command = '''
#     ffmpeg
#     -ss {sceneStart} -t {sceneDuration} -i "{layVideoPath}" -i "{wordVideoPath}"
#     -ss {personalizationEnd} -t {personalizationDuration} -i "{personalizationVideoPath}"
#     -filter_complex "[0:v:0][0:a:0][1:v:0][1:a:0][2:v:0][2:a:0]concat=n=3:v=1:a=1[outv][outa]"
#     -map "[outv]" -map "[outa]" {outputPath}  -y -hide_banner
#     '''
#     command = [c for c in command.split('\n') if c and not c.lstrip().startswith('#')]
#     command = ' '.join(command)
#     command = command.format(**locals())
#     print(command)
#     os.system(command)
#
#
# def selectSubImg(img):
#     cv2.namedWindow("select roi", 0)
#     bbox = cv2.selectROI("select roi", img)
#     return img, bbox
#
#
# def cropImg(src, des):
#     img = cv2.imread(src)
#     img, bbox = selectSubImg(img)
#     img = getSubImg(img, bbox)
#     cv2.imwrite(des, img)
#
#
# def cropVideo(vpath, facePath):
#     bbox = None
#     cam = cv2.VideoCapture(vpath)
#     vWriter = VideoWrtier(facePath, cam)
#     for fno, img in videoPlayer(cam):
#         if not bbox:
#             img, bbox = selectSubImg(img)
#         img = getSubImg(img, bbox)
#         vWriter.write(img)
#         key = showImg('demoUtils_img', img, 1)
#         if key == 27:
#             break
#     vWriter.close()
#     stitchVideoAudio(facePath, vpath, facePath)


def mergeVideos(vpaths, outputPath, fps=25, outSize=(720, 1280)):
    dirop(outputPath, remove=True)
    vwrite = VideoWrtier(outputPath, fps)
    vplayers = [videoPlayer(vpath, start, end) for vpath, start, end in vpaths]
    for vplayer in vplayers:
        for fno, img in vplayer:
            img = imResize(img, outSize)
            vwrite.write(img)


def compareFolders(f1, f2, skipExts=[]):
    def getBooks(f1):
        b1s = []
        for root, dirs, books in os.walk(f1):
            for book in books:
                src = join(root, book)
                b1s.append(src)
        return b1s

    def filterBook(b1s):
        res = []
        for b1 in b1s:
            ok = True
            for skipExt in skipExts:
                if b1.endswith(skipExt):
                    ok = False
                    break
            if ok:
                res.append(b1)
        return res

    def readBook(b1):
        with open(b1, 'r') as book:
            lines = book.read()
        return lines

    def findMissing(b1s, f1, f2):
        res = []
        for b1 in b1s:
            b2 = b1.replace(f1, f2)
            if not exists(b2):
                res.append(b2)
        return res

    def compare(b1s, f1, f2):
        res = []
        for b1 in b1s:
            b2 = b1.replace(f1, f2)
            if not exists(b2):
                continue
            l1 = readBook(b1)
            l2 = readBook(b2)
            if l1 != l2:
                res.append(['miss match', b1, b2])
        return res

    b1s = getBooks(f1)
    b2s = getBooks(f2)
    b1s = filterBook(b1s)
    b2s = filterBook(b2s)
    print(f"__________________________________________not found__________________________________________")
    for b in findMissing(b1s, f1, f2):
        if '.idea' not in b:
            print(b)
    for b in findMissing(b2s, f2, f1):
        if '.idea' not in b:
            print(b)

    for status, b1, b2 in compare(b1s, f1, f2):
        print(f"__________________________________________{status}__________________________________________")
        print(b1)
        print(b2)


if __name__ == '__main__':
    # getCondaPackages()
    compareFolders('/home/hippo/Downloads/vishnuChand-text2face-1c6a69597bb0/text2faceCode', '/home/hippo/awsBridge/text2face/text2faceCode', ['.pyc'])
