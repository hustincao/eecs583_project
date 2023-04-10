import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata

import torch
import torchvision

from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms
# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

# Load a pre-trained PyTorch model
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
# model.eval()

# Get Pytorch Model
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Convert the PyTorch model to a Relay function
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
feat = relay.analysis.detect_feature(mod)
print(feat)
# print(mod)
fold_const = relay.transform.FoldConstant()
mod_fold_const = fold_const(mod)
feat_fc = relay.analysis.detect_feature(mod_fold_const)
# print(type(mod_fold_const))
# print(relay.analysis.check_constant(mod_fold_const))
print(feat_fc)
# print(mod_fold_const)
# print(mod_fold_const[0])
print(tvm.relay.analysis.count_layers(mod,['add'] ))
print(len(tvm.relay.analysis.extract_fused_functions(mod_fold_const)))
# fog =  relay.transform.FirstOrderGradient()
# mod_fog = fog(mod)
# feat_fog = relay.analysis.detect_feature(mod_fog)
# print(feat_fog)

# # Compile the model
# target = tvm.target.Target("llvm", host="llvm")
# dev = tvm.cpu(0)
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)

llvm_ver = tvm.build(mod, 'llvm')
print(llvm_ver)