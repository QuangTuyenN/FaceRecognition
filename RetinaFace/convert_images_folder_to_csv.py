from PIL import Image
import argparse
import hashlib
import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from layers.functions.prior_box import PriorBox
from models.retinaface import RetinaFace
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms
from data import cfg_mnet
import insightface

# Load the InsightFace model
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=0)  # -1 for CPU, use GPU ID if available (e.g., 0 for GPU)

path = 'ImageFolder'

f = open("faces.csv", "a")
# first_row = str()
# first_row = first_row + "id"
# for bien in range(1, 513):
#     first_row = first_row + "," + str(bien)
# first_row = first_row + "\n"
# f.write(first_row)


# Tải ảnh khẩu trang
mask = Image.open('face_mask.png')

# parser = argparse.ArgumentParser(description='Retinaface')
# parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


torch.set_grad_enabled(False)
cfg = cfg_mnet
net = RetinaFace(cfg=cfg, phase='test')
net = load_model(net, "./weights/mobilenet0.25_Final.pth", True)
net.eval()
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

resize = 1

#
#
# # Hiển thị ảnh đã có khẩu trang
# img_with_mask = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# cv2.imshow('Face with Mask', img_with_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#


def read_file(path):
    img_path = []
    for paths, subdirs, files in os.walk(path):
        for name in files:
            img_path.append(os.path.join(paths, name))
    return img_path


train_imgs = read_file(path)
count = 1
for path in train_imgs:
    print(count)
    ############################## NO MASK #################################
    img_no_mask = cv2.imread(path)
    img_no_mask = cv2.cvtColor(img_no_mask, cv2.COLOR_BGR2RGB)

    # Detect faces and extract features (embeddings)
    faces = model.get(img_no_mask)

    # If a face is detected, extract the 512-dimensional vector
    if len(faces) > 0:
        face = faces[0]  # Get the first detected face
        embedding = face.normed_embedding  # The 512-dimensional vector
        string = str()
        name = path.split('\\')[-1]
        name = name[:-4]
        string = string + name
        for i in embedding:
            string = string + "," + str(i)
        string = string + "\n"
        f.write(string)
    else:
        print(f"No face detected in path no mask {path}")
    ############################## MASK #################################
    image_path = path
    img_raw = cv2.imread(path, cv2.IMREAD_COLOR)

    if img_raw is None:
        print("Failed to load image.")

    scale = max(img_raw.shape[1] / 384.0, img_raw.shape[0] / 512.0)
    new_size = (int(img_raw.shape[1] / scale), int(img_raw.shape[0] / scale))
    img_raw = cv2.resize(img_raw, new_size)
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img_ori = img.copy()  # Copy to preserve the original image
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, _ = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # Ignore low scores
    inds = np.where(scores > 0.02)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]

    # Do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.4)
    dets = dets[keep, :]

    dets = dets[:750, :]

    # Convert img_ori to uint8
    img_ori = np.clip(img_ori.transpose(1, 2, 0), 0, 255).astype(np.uint8)

    # Chuyển đổi ảnh sang định dạng PIL để có thể dễ dàng thao tác
    img_pil = Image.fromarray(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))

    for b in dets:
        if b[4] < 0.9:
            continue

        b = list(map(int, b))

        x0, y0 = b[0], b[1]
        x1, y1 = b[2], b[3]
        # Ensure coordinates are within the image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img_ori.shape[1], x1), min(img_ori.shape[0], y1)

        x, y = x0, y0
        w, h = x1 - x0, y1 - y0

        # Resize khẩu trang sao cho phù hợp với khuôn mặt
        resized_mask = mask.resize((w, int(1.1*(h / 2))))

        # Tính toán vị trí để vẽ khẩu trang (khoảng vị trí miệng)
        img_pil.paste(resized_mask, (x, y + int(h / 2)), resized_mask)

        img_pil = np.array(img_pil)

        # Detect faces and extract features (embeddings)
        faces_mask = model.get(img_pil)

        # If a face is detected, extract the 512-dimensional vector
        if len(faces_mask) > 0:
            face_mask = faces_mask[0]  # Get the first detected face
            embedding = face_mask.normed_embedding  # The 512-dimensional vector
            string = str()
            name = path.split('\\')[-1]
            name = name[:-4]
            string = string + name
            for i in embedding:
                string = string + "," + str(i)
            string = string + "\n"
            f.write(string)
        else:
            print(f"No face detected in path with mask {path}")
    ##############################      #################################
    count += 1
f.close()
