import cv2
import numpy as np
import torch
from torch.nn import AvgPool2d, Conv2d, Linear, ReLU, MaxPool2d, BatchNorm2d, SiLU
import torch.nn.functional as F

class GradCam():
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.handles = []
        self.gradients = None
        self.conv_outputs = None
        self.hook_layers(target_layer)

    def hook_layers(self, target_layer):
        def store_grads(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        def store_outputs(module, input, outputs):
            self.conv_outputs = outputs

        if target_layer is None:
            layers = get_layers(self.model)
            for layer in layers:
                if isinstance(layer, Conv2d):
                    target_layer = layer

        self.handles.append(target_layer.register_forward_hook(store_outputs))
        self.handles.append(target_layer.register_backward_hook(store_grads))

    def __call__(self, input_image, target_class=None):
        input_image = input_image.unsqueeze(0)
        predictions = self.model(input_image)

        if target_class is None:
            _, target_class = torch.max(predictions, dim=1)

        self.model.zero_grad()
        if predictions.size(1) > 1:
            predictions[0][target_class].backward()
        else:
            predictions.backward(gradient=target_class)

        with torch.no_grad():
            avg_channel_grad = F.adaptive_avg_pool2d(self.gradients.data, 1)
            cam = F.relu(torch.sum(self.conv_outputs[0] * avg_channel_grad[0], dim=0))
            cam = cam.detach().cpu().numpy()
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))

        return cam, {'prediction': target_class}


class SaliencyMap():
    def __init__(self, model, target_layer=None, guided=False):
        self.model = model.eval()
        self.handles = []
        self.gradients = None
        self.hook_layers(target_layer)
        if guided:
            self.update_relus()

    def hook_layers(self, target_layer):
        def store_grad(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        
        if target_layer is None:
            target_layer = get_layers(self.model)[0]
        self.handles.append(target_layer.register_backward_hook(store_grad))


    def update_relus(self):
        def guide_relu(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        modules = get_layers(self.model)
        for module in modules:
            if isinstance(module, ReLU):
                self.handles.append(module.register_backward_hook(guide_relu))
            if isinstance(module, SiLU):
                self.handles.append(module.register_backward_hook(guide_relu))

    def __call__(self, input_image, target_class=None):
        input_image = input_image.unsqueeze(0)
        input_image.requires_grad_()
        predictions = self.model(input_image)

        if target_class is None:
            _, target_class = torch.max(predictions, dim=1)

        self.model.zero_grad()
        if predictions.size(1) > 1:
            predictions[0][target_class].backward()
        else:
            predictions.backward(gradient=target_class)

        hm = self.gradients.data[0].permute(1,2,0).cpu().numpy()
        return hm, { 'prediction': target_class }

def get_layers(module):
    children = list(module.children())
    flatt_children = []
    if children == []:
        return module
    else:
       for child in children:
            try:
                flatt_children.extend(get_layers(child))
            except TypeError:
                flatt_children.append(get_layers(child))
    return flatt_children

def image_with_colormap(img, hm):
    img = np.array(img, dtype=float)
    hm = np.uint8(hm*255)
    cm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)
    img_with_cam = cm + img
    img_with_cam /= np.max(img_with_cam)
    return img_with_cam

def convert_to_grayscale(cv2im):
    grayscale_im = np.sum(np.abs(cv2im), axis=2)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    return grayscale_im

def get_positive_negative_saliency(gradient):
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency