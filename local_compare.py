import cv2
import numpy as np
import os
import scipy
import tensorflow as tf
import time
from threading import Thread

from align import detect_face

EMB_THRESHOLD = 0.85
DEFALUT_DISTANCE = 2.0
PATH_TO_MODEL = 'MODEL_FACE_NET/'


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

            count_face = []
            centroid = []
            while True:
                # bounding_boxes = xmin ymin xmax ymax accuracy
                # copy lastest image from camera
                image = frame_img
                bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold,
                                                            factor)

                img_list = []
                image_size = 160

                if len(bounding_boxes) == 0:
                    continue

                for box in bounding_boxes:
                    xmin, ymin, xmax, ymax, _ = box

                    cropped = crop_img(image, xmin, ymin, xmax, ymax)
                    aligned = cv2.resize(cropped, (image_size, image_size))

                    prewhitened = prewhiten(aligned)
                    img_list.append(prewhitened)

                images = np.stack(img_list)
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb = recognition_sess.run(embeddings, feed_dict=feed_dict)
                total_face_id_min = []
                for e in emb:
                    min_distance = DEFALUT_DISTANCE
                    face_id_min = 0
                    idx = 0
                    for c in centroid:
                        distance_compared = scipy.spatial.distance.euclidean(e, c)
                        if distance_compared < min_distance:
                            min_distance = distance_compared
                            face_id_min = idx
                        idx += 1
                    # add new face id
                    if min_distance >= EMB_THRESHOLD:
                        count_face.append(1)
                        centroid.append(e)
                        face_id_min = idx
                    else:
                        count_face[face_id_min] += 1
                        centroid[face_id_min] = (count_face[face_id_min] * centroid[face_id_min] + e) / (
                                count_face[face_id_min] + 1)
                    total_face_id_min.append(face_id_min)

                for idx in range(len(emb)):
                    xmin, ymin, xmax, ymax, face_acc = bounding_boxes[idx]

                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                    face_id_text = str(total_face_id_min[idx]) + ' ' + str(face_acc)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, face_id_text, (int(xmax), int(ymin)), font, 2, (0, 0, 255), 2)

                for i in emb:
                    j = centroid[0]
                    print("{:^8.4f}".format(scipy.spatial.distance.euclidean(i, j)), end="")
                    print()
                cv2.imshow('img', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    out = cv2.imwrite('capture.jpg', image)


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
    # Check if the model is local_embedding_send_to_cloud.py model directory (containing local_embedding_send_to_cloud.py metagraph and local_embedding_send_to_cloud.py checkpoint file)
    #  or if it is local_embedding_send_to_cloud.py protobuf file with local_embedding_send_to_cloud.py frozen graph
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


if __name__ == '__main__':
    global frame_img
    Thread(target=detect_thread).start()
    Thread(target=capture_thread).start()
