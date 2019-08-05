import base64
import datetime
import io
import json
import os
import time
from threading import Thread

import cv2
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from scipy import misc

from align import detect_face

PATH_TO_MODEL = 'MODEL_FACE_NET/'
URI = 'https://face-detect-229308.appspot.com/add_camera_logs'


def detect_thread():
    time.sleep(4)
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    sess = tf.Session()
    # read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
    pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

    with tf.Graph().as_default():
        with tf.Session() as recognition_sess:
            # Load the model
            load_model(PATH_TO_MODEL)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            while True:
                image = frame_img
                time_of_frame = datetime.datetime.now()
                # bounding_boxes = xmin ymin xmax ymax accuracy
                bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)

                if len(bounding_boxes) == 0:
                    continue

                img_list = []
                image_size = 160

                img_base_64 = []
                for box in bounding_boxes:
                    xmin, ymin, xmax, ymax, _ = box

                    cropped = crop_img(image, xmin, ymin, xmax, ymax)
                    aligned = misc.imresize(cropped, (image_size, image_size))
                    prewhitened = prewhiten(aligned)
                    img_list.append(prewhitened)

                    img_base_64.append("data:image/png;base64," + to_base_64(cropped)[2:-1])
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                  (0, 0, 255), 1)

                images = np.stack(img_list)
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb = recognition_sess.run(embeddings, feed_dict=feed_dict)

                # send emb to google cloud
                logs = []
                for idx in range(len(emb)):
                    logs.append(
                        {
                            "camera_id": "CCTV01",
                            "face_vector": str(emb[idx].tolist()),
                            "time_detect": str(time_of_frame),
                            "face_image": img_base_64[idx]
                        })
                data = {
                    "logs": logs,
                    "full_image": "data:image/png;base64," + to_base_64(image)[2:-1]
                }
                data_json = json.dumps(data)
                headers = {'Content-type': 'application/json'}
                response = requests.post(URI, data=data_json, headers=headers)


def capture_thread():
    global frame_img
    # cap from notebook camera
    cap = cv2.VideoCapture(0)
    while True:
        # image format should be RGB format, but cap.read() get BGR format
        _, frame_img_tmp = cap.read()
        frame_img = cv2.cvtColor(frame_img_tmp, cv2.COLOR_BGR2RGB)


def crop_img(img, xmin, ymin, xmax, ymax):
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    return img[int(ymin):int(ymax), int(xmin):int(xmax)]


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        import re
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def to_base_64(img):
    img = Image.fromarray(img, 'RGB')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return str(base64.b64encode(img_byte_arr))


if __name__ == '__main__':
    global frame_img
    Thread(target=detect_thread).start()
    Thread(target=capture_thread).start()
