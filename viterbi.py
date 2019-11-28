import sys
import numpy as np
import math

def data_format(test_input):
    with open(test_input, 'r') as ti:
        contents = [k.strip('\n') for k in ti.readlines()]
    return contents

def total_keys(index_to_tag):
    with open(index_to_tag) as itt:
        keys = [k.strip('\n') for k in itt.readlines()]
    size_tags = len(keys)
    return size_tags, keys

def sentence_format(words,index_to_word):
    with open(index_to_word) as itw:
        values = [k.strip('\n') for k in itw.readlines()]
    formatted = []
    for w in words:
        info = w.split("_")
        word = info[0]
        index = values.index(word)
        formatted.append(index+1)
    return formatted

def predict_line(words,hmmemit,hmmtrans,hmmprior,index_to_word,index_to_tag):
    formatted = sentence_format(words, index_to_word)
    size_tags, tags = total_keys(index_to_tag)
    w_matrix = np.zeros((size_tags, len(formatted)))
    b_matrix = np.zeros((size_tags, len(formatted)))
    prior_index = formatted[0] - 1
    for t in range(size_tags):
        w_matrix[t][0] = math.log(hmmprior[t]) + math.log(hmmemit[t][prior_index])
        b_matrix[t][0] = t

    transition = []
    for i in range(1,len(formatted)):
        prior_index = formatted[i]-1
        for t in range(size_tags):
            transition = [(w_matrix[k][i-1] + math.log(hmmtrans[k][t])) for k in range(size_tags)]
            max_prob = max(transition)
            max_tag = int(transition.index(max_prob))
            w_matrix[t][i] = math.log(hmmemit[t][prior_index]) + max_prob
            b_matrix[t][i] = int(max_tag)

    for t in range(size_tags):
        transition = [(w_matrix[k][len(formatted)-1] + math.log(hmmtrans[k][t])) for k in range(size_tags)]
        max_prob = max(transition)
        max_tag = transition.index(max_prob)

    t_seq = []
    t_seq.append(max_tag)

    for w in range(len(formatted)-1,0,-1):
        curr_tag = t_seq[-1]
        t_seq.append(int(b_matrix[curr_tag][w]))

    t_seq.reverse()
    final = []
    for t in t_seq:
        final.append(tags[t])

    return final

def convert_sentence(words,t_seq):
    new = []
    final = ""
    for i in range(len(words)):
        info = (words[i]).split("_")
        info[1] = "_" + t_seq[i]
        final = "".join(info)
        new.append(final)
    result = " ".join(new)
    return result

def predict_file(predictedtest,testinput,hmmemit,hmmtrans,hmmprior,index_to_word,index_to_tag):
    contents = data_format(testinput)
    predict = []
    for sentence in contents:
        words = sentence.split()
        t_seq = predict_line(words,hmmemit,hmmtrans,hmmprior,index_to_word,index_to_tag)
        result = convert_sentence(words,t_seq)
        predict.append(result)
    with open(predictedtest, 'w') as ptest:
        for s in predict:
            ptest.write("%s\n" % s)

def metrics(predictedtest,testinput,metric_file):
    ptest = data_format(predictedtest)
    ttest = data_format(testinput)
    error = 0
    total = 0
    for i in range(len(ptest)):
        s1 = ptest[i]
        s2 = ttest[i]
        words1 = s1.split()
        words2 = s2.split()
        total += len(words1)
        for i in range(len(words1)):
            info1 = (words1[i]).split("_")
            info2 = (words2[i]).split("_")
            if(info1[1] != info2[1]):
                error += 1
    result = (total-error)/(total)
    final = "Accuracy: " + str(result)
    with open(metric_file, 'w') as metrics:
        metrics.write("%s\n" % final)

if __name__ == "__main__":
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]

    hmmprior = np.loadtxt(hmmprior)
    hmmtrans = np.loadtxt(hmmtrans)
    hmmemit = np.loadtxt(hmmemit)

    predict_file(predicted_file,test_input,hmmemit,hmmtrans,hmmprior,index_to_word,index_to_tag)
    metrics(predicted_file, test_input, metric_file)








