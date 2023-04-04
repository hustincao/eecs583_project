import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata

# PyTorch imports
import torch
import torchvision

from PIL import Image

# Get Test Image Data
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

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

print(mod) // Computation graph

# Compile the model
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
