import tvm
from tvm import relay
import numpy as np
from tvm.relay.build_module import bind_params_by_name
from tvm.ir.instrument import (
    PassTimingInstrument,
    pass_instrument,
)
from tvm.contrib.download import download_testdata
from pprint import pprint


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
        "count_layers_dense":[]
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
    feature_dict['count_layers_dense'] = np.sum([tvm.relay.analysis.count_layers(i, ['reshape'])for i in node_dict.keys()])

    feature_dict['num_fused_function'] = len(tvm.relay.analysis.extract_fused_functions(mod))
    return feature_dict

def collect_ops(node):
    ops = set()
    def visitor(e):
        if isinstance(e, tvm.ir.Op):
            ops.add(e.name)
    tvm.relay.analysis.post_order_visit(node, visitor)
    return ops



def get_feature_per_pass(mod):
    pass_dict = {
    'None': [        relay.transform.FoldConstant(),
        relay.transform.ConvertLayout(desired_layouts),
        relay.transform.FoldConstant(),
    ],
    'BatchingOps': [relay.transform.BatchingOps()],
    'CanonicalizeOps':[relay.transform.BatchingOps()],
    'FoldConstant': [relay.transform.FoldConstant()],
    'SimplifyExpr':[relay.transform.SimplifyExpr()],
    'DefuseOps': [relay.transform.DefuseOps()],
    # 'FuseOps': relay.transform.FuseOps()
    # 'DivToMul()':  tvm.relay.transform.DivToMul()
    # 'DeadCodeElimination': [relay.transform.InferType(), relay.transform.DeadCodeElimination()]
    # }
    'InferType': [relay.transform.InferType()],

    }
    # feat = relay.analysis.detect_feature(mod)
    pass_feature_dict = {}
    # for k, v in pass_dict.items():
        # mod_pass = v(mod)
        # feature = get_features(mod_pass)
    # passes = [
    # relay.transform.InferType(),
    # relay.transform.DeadCodeElimination()
    
    # ]
    # seq = tvm.transform.Sequential(passes)

    # mod_pass = seq(mod)
    # feature = get_features(mod_pass)
    # pass_feature_dict[0] = feature

    for k, v in pass_dict.items():
         if k == 'None':
             feature = get_features(mod)
         else:
            seq = tvm.transform.Sequential(v)
            mod_pass = seq(mod)
            feature = get_features(mod_pass)
         pass_feature_dict[k] = feature
    return pass_feature_dict

@pass_instrument
class RelayCallNodeDiffer:
    def __init__(self):
        self._op_diff = []
        # Passes can be nested.
        # Use stack to make sure we get correct before/after pairs.
        self._op_cnt_before_stack = []
        self._before = []
        self._after = []

    def enter_pass_ctx(self):
        self._op_diff = []
        self._op_cnt_before_stack = []

    def exit_pass_ctx(self):
        assert len(self._op_cnt_before_stack) == 0, "The stack is not empty. Something wrong."

    def run_before_pass(self, mod, info):
        self._before.append((info.name, self._count_nodes(mod)))
        self._op_cnt_before_stack.append((info.name, self._count_nodes(mod)))

    def run_after_pass(self, mod, info):
        # Pop out the latest recorded pass.
        name_before, op_to_cnt_before = self._op_cnt_before_stack.pop()
        assert name_before == info.name, "name_before: {}, info.name: {} doesn't match".format(
            name_before, info.name
        )
        cur_depth = len(self._op_cnt_before_stack)
        op_to_cnt_after = self._count_nodes(mod)
 
        op_diff = self._diff(op_to_cnt_after, op_to_cnt_before)
        # only record passes causing differences.
        self._after.append((info.name, op_to_cnt_after))
        if op_diff:
            self._op_diff.append((info.name, op_diff))

    def get_pass_to_op_diff(self):
        """
        return [
          (depth, pass_name, {op_name: diff_num, ...}), ...
        ]
        """
        return self._op_diff

    def get_before_after(self):
        return (self._before, self._after)

    @staticmethod
    def _count_nodes(mod):
        """Count the number of occurrences of each operator in the module"""
        ret = {}

        def visit(node):
            if isinstance(node, relay.expr.Call):
                if hasattr(node.op, "name"):
                    op_name = node.op.name
                else:
                    # Some CallNode may not have 'name' such as relay.Function
                    return
                ret[op_name] = ret.get(op_name, 0) + 1

        relay.analysis.post_order_visit(mod["main"], visit)
        return ret

    @staticmethod
    def _diff(d_after, d_before):
        """Calculate the difference of two dictionary along their keys.
        The result is values in d_after minus values in d_before.
        """
        ret = {}
        key_after, key_before = set(d_after), set(d_before)
        for k in key_before & key_after:
            tmp = d_after[k] - d_before[k]
            if tmp:
                ret[k] = d_after[k] - d_before[k]
        for k in key_after - key_before:
            ret[k] = d_after[k]
        for k in key_before - key_after:
            ret[k] = -d_before[k]
        return ret


def get_features(mod, param, seq):

    pass_seq = tvm.transform.Sequential(
        seq
    )
    call_node_inst = RelayCallNodeDiffer()
    timing_inst = PassTimingInstrument()

    mod["main"] = bind_params_by_name(mod["main"], params)
    with tvm.transform.PassContext(opt_level=3, instruments=[call_node_inst, timing_inst]):
        relay_mod = pass_seq(mod)
        profiles = timing_inst.render()
    # print(profiles)
    diff = call_node_inst.get_pass_to_op_diff()
    before, after = call_node_inst.get_before_after()
    # print('before\n')
    # pprint(before)
    # print('after\n')
    # pprint(after)
    before_feat = [k[1] for k in before]
    before_df = pd.DataFrame.from_dict(before_feat)
    before_df['pass'] = [k[0] for k in before]

    after_feat = [k[1] for k in after]
    after_df = pd.DataFrame.from_dict(after_feat)
    after_df['pass'] = [k[0] for k in after]
    print('diff\n', diff)
    diff_feat = [k[1] for k in diff]
    diff_df = pd.DataFrame.from_dict(diff_feat)
    diff_df['pass'] = [k[0] for k in diff]
    print('diffdf \n', diff_df)

    return before_df, after_df, diff_df, profiles

def get_features_for_pass(mod, param, datadir, model_name):
    desired_layouts = {
    "nn.conv2d": ["NHWC", "HWIO"],
    }
    pass_dict = {
    'FoldConstant': [relay.transform.FoldConstant()],
    'BatchingOps': [relay.transform.BatchingOps()],
    'CanonicalizeOps':[relay.transform.BatchingOps()],
    'FoldConstant': [relay.transform.FoldConstant()],
    'SimplifyExpr':[relay.transform.SimplifyExpr()],
    'DefuseOps': [relay.transform.DefuseOps()],
    'BL': [ relay.transform.AlterOpLayout(),
            relay.transform.CanonicalizeCast(),
            relay.transform.CanonicalizeOps(),
            relay.transform.ConvertLayout(desired_layouts),
            relay.transform.DefuseOps(),
            relay.transform.EliminateCommonSubexpr()
            ],
    'AS-0': [relay.transform.AlterOpLayout(),
             relay.transform.FuseOps(),
             relay.transform.SimplifyExpr(),
             relay.transform.FoldConstant(),
             relay.transform.DeadCodeElimination(),
            #  relay.transform.MergeComposite().
             relay.transform.FastMath(),
             relay.transform.RemoveUnusedFunctions()
    ],
    'AS-1': [
            relay.transform.SimplifyExpr(),
            relay.transform.FuseOps(),
            relay.transform.AlterOpLayout(),
            # relay.transform.MergeComposite(),
            relay.transform.FastMath(),
            relay.transform.DeadCodeElimination(),
            relay.transform.FoldConstant(),
            relay.transform.RemoveUnusedFunctions()
            ],
    'PO-0': [
            relay.transform.AlterOpLayout(),
            relay.transform.CombineParallelConv2D(),
            relay.transform.DefuseOps(),
            relay.transform.DynamicToStatic(),
            relay.transform.CanonicalizeOps(),
            relay.transform.CanonicalizeCast(),
    ],
    'PO-1': [
            relay.transform.CanonicalizeCast(),
            relay.transform.AlterOpLayout(),
            relay.transform.DefuseOps(),
            relay.transform.CombineParallelConv2D(),
            relay.transform.FakeQuantizationToInteger()
        
    ],
    'PO-2':[
        relay.transform.CanonicalizeCast(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.EliminateCommonSubexpr(),
        # relay.transform.SimplifyFCTranspose(),
        relay.transform.CanonicalizeOps(),
        relay.transform.DefuseOps(),
        relay.transform.ToGraphNormalForm(),
        relay.transform.ToGraphNormalForm()
    ],
    'PO-3':[
        relay.transform.CombineParallelDense(),
        relay.transform.FakeQuantizationToInteger(),
        relay.transform.AlterOpLayout(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.ToGraphNormalForm(),
        relay.transform.CanonicalizeOps()
    ]

    }



    features_bef = {}
    features_after = {}
    features_time = {}
    for k, v in pass_dict.items():
        before_df, after_df, diff_df, profiles = get_features(mod, param, v)
        datapath = datadir + '/' + model_name + '/' + k + '/'
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        before_df.to_csv(datapath + 'before.csv')
        after_df.to_csv(datapath + 'after.csv')
        diff_df.to_csv(datapath + 'diff.csv')

    # return (features_bef, features_after, features_time)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--datadir",
                      type = 'string',        
                      help="name of data dir")
    (options, args) = parser.parse_args()
    model_names = ["microsoft/resnet-50", "fxmarty/resnet-tiny-beans"]
    # model_names = ["microsoft/resnet-50"]
   
    for model in model_names:
        mod, params = GenerateComputationGraph(model)
        model_name = model.replace('/', '_')
        get_features_for_pass(mod, params, options.datadir, model_name,)
    # print('before\n')
    # print(before)
    # print('after\n')

    # print(after)
    # print('profiles\n')

    # print(profiles)



    # for model_name in model_names:

    #     mod, params = GenerateComputationGraph(model_name)
    #     model_name = model_name.replace('/', '_')

    #     if not os.path.exists(options.datadir + '/' + model_name):
    #         os.makedirs(options.datadir + '/' + model_name)
        
    #     feat = get_feature_per_pass(mod)
        # node_dict = {}
        # def _traverse_expr(node, node_dict):
        #     if node in node_dict:
        #         return
        #     node_dict[node] = len(node_dict)
        # tvm.relay.analysis.post_order_visit(mod['main'], lambda x: _traverse_expr(x, node_dict))
        # for node, node_id in sorted(node_dict.items(), key=lambda x: x[1]):
        #     ops = collect_ops(node)
        #     # print(ops.list_op_names())
        #     for o in ops:
        #         print(o, type(o))

        #track time
       
        # feats = []
        # for k,v in feat.items():
        #     feats.append(pd.DataFrame.from_dict(v, orient='index'))
        # feats_df = pd.concat(feats, axis=1, ignore_index=True).transpose()
        # feats_df.index = list(feat.keys())
        # feats_df.to_csv(options.datadir +'/' + model_name + '/features.csv')

        
       
