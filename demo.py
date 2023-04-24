import time
from knn import *
from extract_features import *
from pipeline import *
import random
import pandas as pd
import os
from transformers import ResNetForImageClassification, BertModel, BertTokenizer


def main():
    # Settings
    data_folder_name = os.getcwd() + '/data/'
    model_names = os.listdir(data_folder_name)

    # Select test model
    # test_model = "fxmarty_resnet-tiny-beans"
    test_model = "microsoft_resnet-50"

    # Remove test model from training set
    index = model_names.index(test_model)
    train_models = model_names[:index] + model_names[index+1:]

    train_models_vec_dict = {} # in format key=model_name,val=vector representation of model_name
    test_model_vec_dict = {test_model:[]}
    
    # get feature order for the vectors
    feature_order = get_feature_order([(data_folder_name + x) for x in train_models])
    
    # save init model feature embeddings -- training set
    for model_name in train_models:
        # need to pass in the full filepath name here
        model_filename = data_folder_name + model_name
        train_models_vec_dict[model_name] = get_vector_representation(model_filename, feature_order)

    # save init model feature embedding -- test data
    test_model_vec_dict[test_model] = get_vector_representation((data_folder_name + test_model),feature_order)

    #### USE KNN TO GET NEAREST TRAINING DATA TO TEST ####
    k = 5
    train_test_sims = knn(train_models_vec_dict, test_model_vec_dict, k=5)
    # get the best passes for the k nearest neighbors
    nearest_neighbors = []
    for item in train_test_sims[test_model]:
        nearest_neighbors.append(item['train_vec'])
    models_best_passes_dict = get_best_passes([(data_folder_name + x) for x in nearest_neighbors])

    # get sequence of passes to apply to test sample
    sequence = []
    passes_added = [] # remove duplicates
    for model_name in models_best_passes_dict.keys():
        # get more than 1st level of features in case there are non-unique ones in first level
        for i in range(2):
            if models_best_passes_dict[model_name][0][0] not in passes_added:
                sequence.append(models_best_passes_dict[model_name][0])
                passes_added.append(models_best_passes_dict[model_name][0][0])
            if len(models_best_passes_dict[model_name]) > 1:
                if models_best_passes_dict[model_name][1][0] not in passes_added:
                    sequence.append(models_best_passes_dict[model_name][1])
                    passes_added.append(models_best_passes_dict[model_name][1][0])

    # order sequence by execution time
    sequence = sorted(sequence, key=lambda x: x[1], reverse=False)

    # Convert pass names into tvm FunctionPasses
    pass_sequence = [get_pass(x)() for x in sequence]

    test_model_name = test_model.replace('_', '/')

    # Download model
    model = ResNetForImageClassification.from_pretrained(test_model_name, torchscript=True)
    mod, params = GenerateComputationGraph(model, "resnet")

    # Compile with KNN passes
    compile_time, exec_time = CompileModel(mod, params, pass_sequence) # Compile and get execution time
    
    print("KNN Compile time: ", compile_time)
    print("KNN Runtime: ", exec_time)

    # with open(f'results/knn/{test_model}_compile_profile.txt', 'w') as f:
    #     f.writelines(compile_profile)

    # with open(f'results/knn/{test_model}_execution_time.txt', 'w') as f:
    #     f.writelines(exec_time)
    
    # Compile with baseline opt_level=3
    compile_time, exec_time = CompileModelBaseline(mod, params)
    print("Baseline Compile time: ", compile_time)
    print("Baseline Runtime: ", exec_time)
    # with open(f'results/o3/{test_model}_compile_profile.txt', 'w') as f:
    #     f.writelines(compile_profile)

    # with open(f'results/o3/{test_model}_execution_time.txt', 'w') as f:
    #     f.writelines(exec_time)
if __name__=='__main__':
    main()


