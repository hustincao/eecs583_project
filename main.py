import tvm
from tvm import relay
import numpy as np
from tvm.contrib.download import download_testdata
from tvm.contrib import relay_viz
from graphviz import Digraph
import json
# PyTorch imports
from datasets import load_dataset

import torch
import torchvision
from transformers import ResNetForImageClassification, BertModel, BertTokenizer

from PIL import Image

# model_name: used to get model form pytorch
def GenerateComputationGraph(model, nn_arch):
    
    # Create different input data for different nn_arch
    if nn_arch == "resnet":
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


        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)

        traced_model = torch.jit.trace(model, input_data).eval()
        input_name = "input0"
        shape_list = [(input_name, img.shape)]

    elif nn_arch == "bert":
        enc = BertTokenizer.from_pretrained("bert-base-uncased")

        # Tokenizing input text
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = enc.tokenize(text)

        # Masking one of the input tokens
        masked_index = 8
        tokenized_text[masked_index] = '[MASK]'
        indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Creating a dummy input
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        dummy_input = [tokens_tensor, segments_tensors]

        # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
        model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])
        traced_model.eval()
        for p in traced_model.parameters():
            p.requires_grad_(False)
        
        shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]

        mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
        # enc = BertTokenizer.from_pretrained("bert-base-uncased")
        # # Tokenizing input text
        # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        # tokenized_text = enc.tokenize(text)

        # # Masking one of the input tokens
        # masked_index = 8
        # tokenized_text[masked_index] = "[MASK]"
        # indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
        # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # # Creating a dummy input
        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])
        # dummy_input = [tokens_tensor, segments_tensors]

        # tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])
        # input_data = [tokens_tensor, segments_tensors]

        # model = torch.jit.trace(model, input_data).eval()
        # for p in model.parameters():
        #     p.requires_grad_(False)
        # shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(model.graph.inputs())[1:]]
        # print(shape_list)


    # scripted_model = torch.jit.trace(model, input_data).eval()
    mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

    return mod, params

# model_name: used to save graph to file
# mod: computation graph generated by 
def VisualizeGraph(mod, model_name, nn_arch):
    # graphviz attributes
    graph_attr = {"color": "red"}
    node_attr = {"color": "blue"}
    edge_attr = {"color": "black"}

    # VizNode is passed to the callback.
    # We want to color NCHW conv2d nodes. Also give Var a different shape.
    def get_node_attr(node):
        if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
            return {
                "fillcolor": "green",
                "style": "filled",
                "shape": "box",
            }
        if "Var" in node.type_name:
            return {"shape": "ellipse"}
        return {"shape": "box"}

    # Create plotter and pass it to viz. Then render the graph.
    dot_plotter = relay_viz.DotPlotter(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        get_node_attr=get_node_attr)

    viz = relay_viz.RelayVisualizer(
        mod,
        relay_param=params,
        plotter=dot_plotter,
        parser=relay_viz.DotVizParser())
    viz.render("viz_graphs/"+nn_arch+"/"+model_name)

#, target, passes
def CompileModel(mod, params, model_name, nn_arch):
    print(nn_arch)
    target = tvm.target.Target("llvm", host="llvm")
    # dev = tvm.cpu(0)

    # Apply pass using opt_level=3
    # with tvm.transform.PassContext(opt_level=3):
    #     lib = relay.build(mod, target=target, params=params)
    
    # Apply passes sequentially
    # passes = [
    #         relay.transform.FoldConstant(),
    #         # tvm.transform.PrintIR(),
    #         relay.transform.EliminateCommonSubexpr(),
    #         relay.transform.FuseOps(),
    #     ]
    # seq = tvm.transform.Sequential(passes)

    # mod1 = seq(mod)

    llvm_ir = tvm.build(mod, target="llvm")

    with open("llvm/"+nn_arch+"/"+model_name+".ll", "w") as f:
        f.write(llvm_ir.get_source())
    

    # performance measuring: https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html#compile-the-model-with-relay
    # compiling DL models: https://tvm.apache.org/docs/tutorial/relay_quick_start.html

if __name__ == '__main__':

  
    #resnet, bert, Potential todo: suppoprt ssd and mobilenet
    nn_arch = "bert"

    # Config to easily change from one neural net architecture to another
    model_config = {
        "bert":{
            "file": "bert_models.json",
            "loader": BertModel
        },
        "resnet":{
            "file": "resnet_models.json",
            "loader": ResNetForImageClassification
        }
    }

    # Get list of models to load
    with open(model_config[nn_arch]["file"]) as user_file:
        model_names = json.load(user_file)

    for model_name in model_names:
        model = model_config[nn_arch]["loader"].from_pretrained(model_name ,torchscript=True)
        # model = ResNetForImageClassification.from_pretrained(model_name ,torchscript=True)
        # model = BertModel.from_pretrained(model_name ,torchscript=True)

        # Create output file names that doesn't have a / in it
        output_file = model_name[model_name.rindex("/")+1:] if "/" in model_name else model_name

        mod, params = GenerateComputationGraph(model, nn_arch)
        VisualizeGraph(mod, output_file, nn_arch)
        CompileModel(mod, params, output_file, nn_arch)
        

# Compile the model
# target = tvm.target.Target("llvm", host="llvm")
# dev = tvm.cpu(0)
# with tvm.transform.PassContext(opt_level=3):
#     lib = relay.build(mod, target=target, params=params)
