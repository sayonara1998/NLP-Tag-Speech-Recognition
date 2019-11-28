import sys
import numpy as np


def prior(train_input,index_to_tag,hmmprior):
    keys = []
    with open(index_to_tag) as itt:
        keys = [k.strip('\n') for k in itt.readlines()]
    first_words = []
    with open(train_input, 'r') as ti:
        first_words = [line.split(None, 1)[0] for line in ti]
    d = {}
    for k in keys:
        d.update({k:1})
    for word in first_words:
        info = word.split("_")
        tag = info[1]
        d[tag] += 1
    total = 0
    for k,v in d.items():
        total += v

    occur = 0
    final = np.zeros((len(keys),1))
    for i in range(len(keys)):
        occur = d[keys[i]]/total
        final[i][0] = occur

    with open(hmmprior, 'wb') as hmm_prior:
        np.savetxt(hmm_prior, final)
        hmm_prior.seek(-1, 2)
        hmm_prior.truncate()

def trans(train_input, index_to_tag,hmmtrans):
    with open(index_to_tag) as itt:
        keys = [k.strip('\n') for k in itt.readlines()]
    size = len(keys)
    with open(train_input, 'r') as ti:
        contents = [k.strip('\n') for k in ti.readlines()]
    temp = np.ones((size,size))
    for sentence in contents:
        words = sentence.split()
        for i in range(len(words)-1):
            info1 = (words[i]).split("_")
            info2 = (words[i+1]).split("_")
            tag1 = info1[1]
            tag2 = info2[1]
            index1 = keys.index(tag1)
            index2 = keys.index(tag2)
            temp[index1][index2] += 1
    total = 0
    totallist = []
    for i in range(size):
        for j in range(size):
            total += temp[i][j]
        totallist.append(total)
        total = 0
    for i in range(size):
        for j in range(size):
            temp[i][j] /= totallist[i]
    count = len(keys)-1
    with open(hmmtrans, 'wb') as hmm_trans:
        np.savetxt(hmm_trans, temp)
        hmm_trans.seek(-1, 2)
        hmm_trans.truncate()

def emit(train_input, index_to_tag,index_to_word,hmmemit):
    with open(index_to_tag) as itt:
        keys = [k.strip('\n') for k in itt.readlines()]
    size_tags = len(keys)
    with open(index_to_word) as itw:
        values = [k.strip('\n') for k in itw.readlines()]
    size_words = len(values)
    temp = np.ones((size_tags, size_words))

    with open(train_input, 'r') as ti:
        contents = [k.strip('\n') for k in ti.readlines()]
    for sentence in contents:
        words = sentence.split()
        for i in range(len(words)):
            info = (words[i]).split("_")
            word = info[0]
            tag = info[1]
            index1 = keys.index(tag)
            index2 = values.index(word)
            temp[index1][index2] += 1
    total = 0
    totallist = []
    for i in range(size_tags):
        for j in range(size_words):
            total += temp[i][j]
        totallist.append(total)
        total = 0
    for i in range(size_tags):
        for j in range(size_words):
            temp[i][j] /= totallist[i]
    with open(hmmemit, 'wb') as hmm_emit:
        np.savetxt(hmm_emit, temp)
        hmm_emit.seek(-1, 2)
        hmm_emit.truncate()


if __name__ == "__main__":
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]

    prior(train_input,index_to_tag,hmmprior)
    trans(train_input,index_to_tag,hmmtrans)
    emit(train_input,index_to_tag,index_to_word,hmmemit)
