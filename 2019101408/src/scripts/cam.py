import cv2 as cv
import torch
import json
import torchvision.models as models
import numpy as np
import augmentation
import torchvision.transforms as transforms
import copy
import os
class Drawer(object):
    def __init__(self, model, device=None):
        self.model = model
        if device:
            self.model = self.model.to(device)
        self.weights = self.get_fc_weights(models)
        self.weights = self.weights.cpu().detach().numpy()
        self.weights = self.weights[:, :, np.newaxis, np.newaxis]
        print(self.weights.shape)

        self.device = device

    def get_fc_weights(self, models):
        # return model.classifier.weight.data
        return self.model.fc.weight.data

    def get_heat_map(self, imgpath, camtype=0, class_num=1000, weight=0.1, rtscore=False):
 
        label = os.path.split(imgpath)[-1][2]

        raw_img, inputs = preprocess(imgpath)
        H, W = raw_img.shape[:2]
        gray = cv.cvtColor(raw_img, cv.COLOR_BGR2GRAY)
        grey_raw_img = np.dstack((gray, gray, gray))
        raw_img_RGB = cv.cvtColor(raw_img, cv.COLOR_BGR2RGB)

        #if self.device:
            #inputs = inputs.to(self.device)
        inputs = inputs.cuda()

        with torch.no_grad():
            outputs, feature = self.model(inputs)

        class_num_pre = np.argmax(outputs[0].cpu().numpy())

        feature_maps = feature
        feature_maps = feature_maps[0].cpu().numpy()
        grey_map = feature_maps * self.weights[camtype]
        grey_map = grey_map.sum(axis=0)
        grey_map = weighted_sigmoid(grey_map, weight) * 255
        grey_map2 = copy.deepcopy(grey_map)
        
        grey_map2[np.where(grey_map2<144)] = 0
        h, w = grey_map.shape
        heat_map = np.zeros((h, w, 3), np.uint8)
        for i in range(h):
            for j in range(w):
                heat_map[i, j] = grey2heat(grey_map[i, j])

        heat_map_fundnus = cv.resize(heat_map, (H, W))
        
        heat_map2 = np.zeros((h, w, 3), np.uint8)
        for i in range(h):
            for j in range(w):
                heat_map2[i, j] = grey2heat(grey_map2[i, j])

        heat_map2_fundus = cv.resize(heat_map2, (H, W))

        if rtscore:
            return class_num_pre, cls_num, np.squeeze(torch.softmax(outputs,dim=1).cpu().numpy())
        return class_num_pre, raw_img_RGB, fusion(raw_img_RGB, heat_map2_fundus[:, :, [2, 1, 0]], 0.7), 0


def preprocess(imgpath):

    transform = transforms.Compose([transforms.ToTensor()])

    raw_img = cv.imread(imgpath)
    image = transform(raw_img)
    image = image.reshape((-1, 3, 448, 448))

    inputs = image

    return raw_img, inputs


def weighted_sigmoid(arr, w=1):
    return 1. / (1 + np.exp(-arr * w))


# convert a grey scale color into RGB
def grey2heat(grey):
    #heat_stages_grey = np.array((0, 64, 128, 192, 256))
    #heat_stages_color = np.array(((0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)))
    heat_stages_grey = np.array((0, 144, 160, 176, 192, 224, 256))
    heat_stages_color = np.array(((0, 0, 0), (0, 0, 255), (165, 0, 255),(255, 0, 255), (255, 0, 0), (255, 255, 0), (255, 255, 255)))

    np.clip(grey, 0, 255)

    for i in range(1, len(heat_stages_grey)):
        if heat_stages_grey[i] > grey >= heat_stages_grey[i - 1]:
            weight = (grey - heat_stages_grey[i - 1]) / float(heat_stages_grey[i] - heat_stages_grey[i - 1])
            color = weight * heat_stages_color[i] + (1 - weight) * heat_stages_color[i - 1]
            break
    return color.astype(np.int)


def fusion(im1, im2, weight=0.5):
    f_im = im1 * weight + im2 * (1 - weight)
    f_im = f_im.astype(np.uint8)
    return f_im
