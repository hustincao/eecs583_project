import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata

import torch
import torchvision

from PIL import Image


import logging

# Create a logging instance
logger = logging.getLogger('my_application')
logger.setLevel(logging.DEBUG) # you can set this to be DEBUG, INFO, ERROR

# Assign a file-handler to that instance
fh = logging.FileHandler(
    "file_dir.txt")
fh.setLevel(logging.DEBUG) # again, you can set this differently



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

# Get Pytorch Model
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Convert the PyTorch model to a Relay function
input_name = "input0"
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

def get_features(mod):
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}

    feature_dict = {
        "num_expr": [],
        "num_fv": [],
        "num_constant": [],
        "num_fused_function":[],
        "count_layers_add":[],
        "count_layers_mul":[],
        "count_layers_div":[],
        "count_layers_transpose":[],
        "count_layers_reshape":[],
    }

    tvm.relay.analysis.post_order_visit(mod['main'], lambda x: _traverse_expr(x, node_dict))

    feature_dict['num_expr'] = len(node_dict)
    feature_dict['num_fv'] = np.sum([len(tvm.relay.analysis.free_vars(i)) for i in node_dict.keys()])
    feature_dict['num_constant'] = np.sum([tvm.relay.analysis.check_constant(i) for i in node_dict.keys()])
    feature_dict['num_bound_var'] = np.sum([len(tvm.relay.analysis.bound_vars(i)) for i in node_dict.keys()])
    # feature_dict['mac'] = np.mean([tvm.relay.analysis.get_total_mac_number(i)for i in node_dict.keys()])
    feature_dict['count_layers_add'] = np.sum([tvm.relay.analysis.count_layers(i, ['add'])for i in node_dict.keys()])
    feature_dict['count_layers_mul'] = np.sum([tvm.relay.analysis.count_layers(i, ['multiply'])for i in node_dict.keys()])
    feature_dict['count_layers_divide'] = np.sum([tvm.relay.analysis.count_layers(i, ['divide'])for i in node_dict.keys()])
    feature_dict['count_layers_transpose'] = np.sum([tvm.relay.analysis.count_layers(i, ['transpose'])for i in node_dict.keys()])
    feature_dict['count_layers_reshape'] = np.sum([tvm.relay.analysis.count_layers(i, ['reshape'])for i in node_dict.keys()])
    feature_dict['num_fused_function'] = len(tvm.relay.analysis.extract_fused_functions(mod))
    return feature_dict

pass_dict = {
    'BatchingOps': relay.transform.BatchingOps(),
    'CanonicalizeOps':relay.transform.BatchingOps(),
    'FoldConstant': relay.transform.FoldConstant(),
    'SimplifyExpr':relay.transform.SimplifyExpr(),
    # 'FuseOps': relay.transform.FuseOps()
    # 'DivToMul()':  tvm.relay.transform.DivToMul(),
    # 'DeadCodeElimination': relay.transform.DeadCodeElimination()

}

feat = relay.analysis.detect_feature(mod)



# fold_const = relay.transform.AlterOpLayout()
fold_const = relay.transform.SimplifyExpr()

mod_fold_const = fold_const(mod)
feat_fc = relay.analysis.detect_feature(mod_fold_const)

pass_feature_dict = {}
for k, v in pass_dict.items():
    mod_pass = v(mod)
    feature = get_features(mod_pass)
    pass_feature_dict[k] = feature


print(pass_feature_dict)