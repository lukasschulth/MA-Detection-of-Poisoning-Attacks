
import cv2
import numpy as np

from utils import toconv, imgclasses, newlayer, heatmap

img = np.array(cv2.imread('castle.jpg'))[..., ::-1]/255.0

import torch
mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
std = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

X = (torch.FloatTensor(img[np.newaxis].transpose([0,3,1,2])*1) - mean) / std

import torchvision

model = torchvision.models.vgg16(pretrained=True)
model.eval()


model = torch

print(model)
layers = list(model._modules['features']) + toconv(list(model._modules['classifier']))
L = len(layers)

A = [X] + [None]*L

for l in range(L):
    A[l+1] = layers[l].forward(A[l])

scores = np.array(A[-1].data.view(-1))
ind = np.argsort(-scores)

for i in ind[:10]:
    print('%20s (%3d): %6.3f'%(imgclasses[i][:20], i, scores[i]))

# Top-layer activations are first multiplied by the mask
# to retain only the predicted evidence for the class "castle".

T = torch.FloatTensor((1.0*(np.arange(1000)==483).reshape([1, 1000, 1, 1])))
R = [None] * L + [(A[-1] * T).data ]

for l in range(1,L)[::-1]:

    A[l] = (A[l].data).requires_grad_(True)

    if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

    if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

        if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
        if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
        if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

        z = incr(newlayer(layers[l], rho).forward(A[l]))  # step 1
        s = (R[l+1]/z).data                                    # step 2
        (z*s).sum().backward(); c = A[l].grad                  # step 3
        R[l] = (A[l]*c).data                                   # step 4

    else:

        R[l] = R[l+1]


for i,l in enumerate([31,21,11,1]):
    heatmap(np.array(R[l][0]).sum(axis=0), 0.5*i+1.5,0.5*i+1.5)

A[0] = (A[0].data).requires_grad_(True)

lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

z = layers[0].forward(A[0]) + 1e-9                                      # step 1 (a)
z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)           # step 1 (b)
z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)           # step 1 (c)
s = (R[1]/z).data                                                       # step 2
(z*s).sum().backward(); c, cp, cm = A[0].grad, lb.grad, hb.grad         # step 3
R[0] = (A[0]*c+lb*cp+hb*cm).data                                        # step 4


heatmap(np.array(R[0][0]).sum(axis=0),3.5,3.5)
