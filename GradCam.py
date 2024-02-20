import torch.nn.functional as F
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn

def replace_relu_inplace_with_outplace(model, module):
    """
    function is needed, when model uses inplace operation in ReLu
    """
    for name, submodule in module.named_children():
        if isinstance(submodule, nn.ReLU) and submodule.inplace:
            setattr(module, name, nn.ReLU(inplace=False))
        elif len(list(submodule.children())) > 0:
            replace_relu_inplace_with_outplace(submodule)

class GradCAM:
    def __init__(self, model, layer, device = 'cuda'):
        """
        Calculates class discriminative localization map using Grad-CAM algorithm.
        model: the model to be used
        layer: the layer to be used (normaly the last convolutional layer)
        device: the device to be used (default: 'cuda')
        """
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.feature_map = None
        self.gradients = None
        self.device = device
        final_conv_layer = layer
        self.forward_hook = final_conv_layer.register_forward_hook(self.save_feature_map)
        self.backward_hook = final_conv_layer.register_full_backward_hook(self.save_gradient)
        
    def save_feature_map(self, module, input, output):
        self.feature_map = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def __call__(self, inputs, resize = True):
        inputs = inputs.to(self.device)
        output = self.model(inputs.unsqueeze(0))
        self.model.zero_grad()
        output.backward(retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])  # Global Average Pooling
        for i in range(pooled_gradients.shape[1]):
            self.feature_map[:, i, :, :] *= pooled_gradients[:, i].unsqueeze(1).unsqueeze(1)
            
        heatmap = torch.mean(self.feature_map, dim=1).squeeze()
        heatmap = F.relu(heatmap)  # ReLU step

        if resize == True:
            resized_heatmap = torch.nn.functional.interpolate(
                                                                heatmap.unsqueeze(0).unsqueeze(0),
                                                                size=(inputs.size()[1],inputs.size()[2], inputs.size()[3]),
                                                                mode='trilinear',
                                                                align_corners=True).squeeze()
            heatmap = resized_heatmap
        
        with torch.no_grad():
            self.forward_hook.remove()
            self.backward_hook.remove()
            self.feature_map = None
            self.gradients = None
        return heatmap
    
class GuidedCradCAM:
    def __init__(self, model, device = 'cuda'):
        """
        Calculates non-class discriminative fine grained localization map using Guided Backpropagation algorithm.
        model: the model to be used
        device: the device to be used (default: 'cuda')
        """
         
        self.model = model
        self.device = device
        self.model.eval()

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0), )

        for layer in self.model.modules():
            if isinstance(layer, torch.nn.ReLU):
                layer.register_backward_hook(backward_hook)

    def __call__(self, inputs):
        inputs = inputs.to(self.device)
        inputs.requires_grad = True
        self.model(inputs.unsqueeze(0)).backward()
        gradients = inputs.grad.data
        gradients = torch.sum(gradients, dim = 0)
        return torch.abs(gradients)

class CAM:
    def __init__(self, model, layer, device = 'cuda'):
        "Combines GradCAM and Guided Backpropagation to produce class discriminative fine grained localization map."
        self.guided_grad_cam = GuidedCradCAM(model, device=device)
        self.grad_cam = GradCAM(model, layer, device=device)
        self.device = device
    def __call__(self, inputs, type = 'raw'):
        if type == 'grad_cam':
            """
            Only outputs the result from GradCAM
            """
            grad_cam_output = self.grad_cam(inputs)
            im = plt.imshow(torch.mean(grad_cam_output.to('cpu'),dim = (0)))
            plt.colorbar(im)
            heatmap = grad_cam_output
        elif type == 'guided':
            """
            Only outputs the result from Guided Backpropagation
            """
            guided_grad_cam_output = self.guided_grad_cam(inputs)
            guided_grad_cam_output = torch.mean(guided_grad_cam_output,dim = 0)
            im = plt.imshow(torch.mean(guided_grad_cam_output.to('cpu'),dim = (0)))
            plt.colorbar(im)
            heatmap = guided_grad_cam_output
        elif type == "overlay":
            """
            Outputs the overlay for GradCAM and Guided Backpropagation
            """
            grad_cam_output = self.grad_cam(inputs, resize=False)
            guided_grad_cam_output = self.guided_grad_cam(inputs)
            guided_grad_cam_output = torch.mean(guided_grad_cam_output,dim = 0)

            cam = torch.mul(grad_cam_output, guided_grad_cam_output)
            cam = torch.div(cam,torch.max(cam))
            im = plt.imshow(torch.mean(cam.to('cpu'),dim = (0)))
            plt.colorbar(im)
            heatmap = cam
        elif type == "raw":
            """
            Outputs the raw heatmap from GradCAM and Guided Backpropagation
            """
            grad_cam_output = self.grad_cam(inputs, resize=True)
            # guided_grad_cam_output = torch.sum(self.guided_grad_cam(inputs), dim = 0)
            guided_grad_cam_output = self.guided_grad_cam(inputs)
            heatmap = torch.mul(grad_cam_output, guided_grad_cam_output.to(self.device))
        else:
            NotImplemented
        return heatmap