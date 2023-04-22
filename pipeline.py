from knn import *
from extract_features import *
import random
import pandas as pd
import os

''' provided an input code, output the feature vector (init) '''
def get_vector_representation(model_name, feature_order):
    # all of them have same value for 'pass'
    feature_order = [x for x in feature_order if x!='pass']
    # TODO --> directly compute features intead of getting from pre-computed
    model_folder = model_name + '/'
    vec = []
    for pass_folder in os.listdir(model_folder):
        # checks for the feature filename
        before_file = model_folder + pass_folder + '/before.csv'
        if os.path.isdir(model_folder + pass_folder):
            before_df = pd.read_csv(before_file)
            # create the feature vector
            for feature in feature_order:
                # not all the codes have the same feature vectors 
                if feature in before_df.columns:
                    vec.append(before_df.iloc[0][feature])
                else:
                    vec.append(0)
            # the 'before' will be same across passes so just take first one and go
            break

    return vec

''' provided model file names, '''
def get_feature_order(model_filenames):
    # save all possible features
    feature_order = []
    for model_filename in model_filenames:
        for pass_foldername in os.listdir(model_filename):
            if os.path.isdir(model_filename + '/' + pass_foldername):
               # print('pass folder name ', model_filename + '/' + pass_foldername)
                feature_filename = model_filename + '/' + pass_foldername + '/before.csv'
                features_csv = pd.read_csv(feature_filename)
                features = features_csv.columns
                feature_order.extend(features)
    # return the full set of features
    return list(set(feature_order))

def get_pass(pass_name):
    if pass_name == 'AlterOpLayout':
        return relay.transform.AlterOpLayout()
    if pass_name == 'BatchingOps':
        return relay.transform.BatchingOps()
    if pass_name == 'CanonicalizeOps':
        return relay.transform.CanonicalizeOps() 
    if pass_name == 'CombineParallelConv2D':
        return relay.transform.CombineParallelConv2D() 
    if pass_name == 'CombineParallelDense':
        return relay.transform.CombineParallelDense() 
    if pass_name == 'DeadCodeElimination':
        return relay.transform.DeadCodeElimination() 
    if pass_name == 'DefuseOps':
        return relay.transform.DefuseOps() 
    if pass_name == 'DynamicToStatic':
        return relay.transform.DynamicToStatic() 
    if pass_name == 'EliminateCommonSubexpr':
        return relay.transform.EliminateCommonSubexpr() 
    if pass_name == 'FakeQuantizationToInteger':
        return relay.transform.FakeQuantizationToInteger() 
    if pass_name == 'FastMath':
        return relay.transform.FastMath() 
    if pass_name == 'FoldConstant':
        return relay.transform.FoldConstant() 
    if pass_name == 'RemoveUnusedFunctions':
        return relay.transform.RemoveUnusedFunctions() 
    if pass_name == 'SimplifyExpr':
        return relay.transform.SimplifyExpr() 
    if pass_name == 'ToGraphNormalForm':
        return relay.transform.ToGraphNormalForm()


''' get the best passes for the models provided '''
def get_best_passes(model_filenames):
   # models_best_passes_dict = {}
    cur_best_passes_dict = {}
    for model_filename in model_filenames:
        #cur_best_passes_dict = {}
        print(model_filename)
        for pass_foldername in os.listdir(model_filename):
            # get the actul pass
            if os.path.isdir(model_filename + '/' + pass_foldername):
                profile_filename = model_filename + '/' + pass_foldername + '/profile.txt'
                if 'profile.txt' in os.listdir(model_filename + '/' + pass_foldername):
                    # save each pass execution time
                    time_diff = 0
                    with open(profile_filename, 'r') as inf:
                        file_text = inf.read().replace('\n','')
                        init_ind = file_text.find('sequential:')
                        if init_ind != -1:
                            ind = init_ind + len('sequential:') # 'passfoldername' + ':'
                            file_toks = (file_text[ind:]).split(' ')
                            file_toks = [x for x in file_toks if x!='']
                            time_diff = file_toks[0]

                        # save pass info
                        if 'us' in str(time_diff):
                            if model_filename not in cur_best_passes_dict.keys():
                                cur_best_passes_dict[model_filename] = []
                            time_diff = time_diff.replace('us','')
                            time_diff = int(time_diff)
                            cur_best_passes_dict[model_filename].append((pass_foldername,time_diff))
       # models_best_passes_dict[model_filename] = cur_best_passes_dict

    # order passes
    for model_name in cur_best_passes_dict.keys():
        cur_best_passes_dict[model_name] = sorted(cur_best_passes_dict[model_name], key=lambda x: x[1], reverse=False)
        print('model name: ',model_name,cur_best_passes_dict[model_name])

    return cur_best_passes_dict

def CompileModel(mod, passes):
    target = tvm.target.Target("llvm", host="llvm")
    
    pass_seq = tvm.transform.Sequential(passes)

    timing_inst = PassTimingInstrument()

    with tvm.transform.PassContext(instruments=[timing_inst]):
        mod = pass_seq(mod)
        profiles = timing_inst.render()

    return profiles

def main():

    data_folder_name = os.getcwd() + '/data/'

    # small sample size so we'll do n-1 training
    for i in range(len(os.listdir(data_folder_name))):
        model_names = os.listdir(data_folder_name)

        #### GET INITIAL FEATURE VECTORS FOR MODELS ####
        # split into train and test data
        train_data = model_names[:i] + model_names[i+1:]
        test_data = model_names[i]
        train_data_vec_dict = {} # in format key=model_name,val=vector representation of model_name
        test_data_vec_dict = {test_data:[]}
        # get feature order for the vectors
        feature_order = get_feature_order([(data_folder_name + x) for x in train_data])
        # save init model feature embeddings -- training set
        for model_name in train_data:
            # need to pass in the full filepath name here
            model_filename = data_folder_name + model_name
            train_data_vec_dict[model_name] = get_vector_representation(model_filename, feature_order)

        # save init model feature embedding -- test data
        test_data_vec_dict[test_data] = get_vector_representation((data_folder_name + test_data),feature_order)

        #### USE KNN TO GET NEAREST TRAINING DATA TO TEST ####
        k = 5
        train_test_sims = knn(train_data_vec_dict, test_data_vec_dict, k=5)
        # get the best passes for the k nearest neighbors
        nearest_neighbors = []
        for item in train_test_sims[test_data]:
            nearest_neighbors.append(item['train_vec'])
        models_best_passes_dict = get_best_passes([(data_folder_name + x) for x in nearest_neighbors])

        # print('test sample: ', test_data)
        # print('nearest neighbors: ', nearest_neighbors)
        # print('best passes for nearest neighbors: ')
        # for model_name in models_best_passes_dict.keys():
        #     print(model_name)
        #     for item in models_best_passes_dict[model_name]:
        #         print(item)
          #  print(models_best_passes_dict[model_name])

        #### APPLY BEST PASSES TO TEST SAMPLE ####
        # get sequence of passes to apply to test sample
        sequence = []
        for model_name in models_best_passes_dict.keys():
            # add the (pass_name, time) tuple to sequence
            sequence.append(models_best_passes_dict[model_name][0])
        # order sequence by execution time
        sequence = sorted(sequence, key=lambda x: x[1], reverse=False)

        # apply the passes to the test sample and get execution time
        for pass_name in sequence:
            pass_obj = get_pass(pass_name)
            # TODO lol ????? 
        
        #### APPLY BASELINE PASSES TO TEST SAMPLE ####
        # apply baseline passes to test sample and get execution time


if __name__=='__main__':
    main()




