
import math


#https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

'''get cosine similarity between two vectors'''
def compute_cosine_sim(vec1, vec2):
    if len(vec1)!=len(vec2):
        print('can only compute cosine similarity if vectors are same length')
        return -1
    # compute magnitudes of each vector
    mag1 = 0
    mag2 = 0
    for item in vec1:
        mag1 += math.pow(item,2)
    for item in vec2:
        mag2 += math.pow(item,2)
    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)
    # compute numerator
    numerator = 0
    for i in range(len(vec1)):
        numerator += (vec1[i] * vec2[i])
    # check case for 0 in-common terms
    if mag1 * mag2 == 0:
        return 0

    return float(numerator / (mag1*mag2))


# training_vecs, test_vecs are dictionaries
def knn(training_vecs, test_vecs, k):
    # get all the similarity scores between train and test vecs
    train_test_sims = {}
    for test_vec in test_vecs.keys():
        for training_vec in training_vecs.keys():
            cosine_sim = compute_cosine_sim(test_vecs[test_vec],training_vecs[training_vec])
            if test_vec not in train_test_sims.keys():
                train_test_sims[test_vec] = [{'train_vec':training_vec, 'cosine_sim':cosine_sim}]
            else:
                train_test_sims[test_vec].append({'train_vec':training_vec, 'cosine_sim':cosine_sim})

    # sort neighbors by similarity
    for key in train_test_sims.keys():
        train_test_sims[key] = sorted(train_test_sims[key], key=lambda x: x['cosine_sim'], reverse=True )
        # trim to k neighbors for each test vec
        train_test_sims[key] = (train_test_sims[test_vec])[:k]

    return train_test_sims
    

def main():
    # TODO put the feature extraction code here, convert to vectors
    training_vecs = {0:[1,2,3], 1:[4,5,6], 2:[8, 9, 10]}
    test_vecs = {0:[2,2,3]}
    test_vecs_knn_results = knn(training_vecs, test_vecs, k=2)
    print(test_vecs_knn_results)

    # TODO map the training vec ids per test vec to their assoc passes
    # and apply those passes to the target test program


    




if __name__=='__main__':
    main()

