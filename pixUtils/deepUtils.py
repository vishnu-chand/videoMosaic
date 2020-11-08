from pixUtils import *


def getSession(tf, tarfile, FastGFile, GraphDef, wpath):
    wpath = str(wpath)
    if wpath.endswith('.pb'):
        with FastGFile(wpath, "rb") as f:
            graph_def = GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)
    else:
        graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(wpath)
        for tar_info in tar_file.getmembers():
            if tar_info.name.endswith('.pb'):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with graph.as_default():
            tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=graph)
    return sess


def isUsingGpu():
    try:
        import tensorflow as tf
        with tf.device('/gpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)
        with tf.Session() as sess:
            print(sess.run(c))
    except:
        pass
    try:
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
        print(devices)
    except:
        pass


def displayModel(model, returnImg=False):
    png = 'model_plot.png'
    try:
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file=png, show_shapes=True, show_layer_names=True)
    except:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file=png, show_shapes=True, show_layer_names=True)
    png = cv2.imread(png)
    dirop(png, remove=True)
    if returnImg:
        return png
    else:
        showImg('tempUtils_png', png, 0)
