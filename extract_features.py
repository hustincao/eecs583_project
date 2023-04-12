import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata

import torch
import torchvision
from transformers import AutoImageProcessor, ResNetForImageClassification


from PIL import Image
import pandas as pd
import logging
from optparse import OptionParser

import os


# Preprocess the image and convert to tensor
from torchvision import transforms

def GenerateComputationGraph(model_name):
    # model = model.eval()
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))


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

    # dataset = load_dataset("huggingface/cats-image")
    # image = dataset["test"]["image"][0]

    # image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained(model_name ,torchscript=True)

    # inputs = image_processor(image, return_tensors="pt")
    # m.build([None, 224, 224, 3])  # Batch input shape.
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    # print(model)
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    model = torch.jit.trace(model, input_data).eval()
    # scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = "input0"
    shape_list = [(input_name, img.shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)
    return mod, params



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





def get_feature_per_pass(mod):
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
    pass_feature_dict = {}
    for k, v in pass_dict.items():
        mod_pass = v(mod)
        feature = get_features(mod_pass)
        pass_feature_dict[k] = feature
    return pass_feature_dict


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--datadir",
                      type = 'string',        
                      help="name of data dir")
    (options, args) = parser.parse_args()
    model_names = ["microsoft/resnet-50", "fxmarty/resnet-tiny-beans"]
    # model_names = ["microsoft/resnet-50"]

    # datapath = '/Users/shinkamori/Documents/eecs583_project/data'
    for model_name in model_names:

        mod, params = GenerateComputationGraph(model_name)
        model_name = model_name.replace('/', '_')

        if not os.path.exists(options.datadir + '/' + model_name):
            os.makedirs(options.datadir + '/' + model_name)
        feat = get_feature_per_pass(mod)

       
        feats = []
        for k,v in feat.items():
            # df_path = options.datadir + '/' + model_name + '/' + k

            feats.append(pd.DataFrame.from_dict(v, orient='index'))
        feats_df = pd.concat(feats, axis=1, ignore_index=True).transpose()
        # print(feats)
        # print(feats_df)
        # feat_df['pass'] = list(feat.keys())
        feats_df.index = list(feat.keys())
        # print(len(feat.keys()))
        # print(len(feats_df))
        # print(feats_df)
        feats_df.to_csv(options.datadir +'/' + model_name + '/features.csv')

        
       
