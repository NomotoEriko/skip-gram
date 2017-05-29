# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from pprint import pprint

idx_to_word = None
Win = None
word_to_idx = None



def _load_model():
    global idx_to_word, word_to_idx, Win
    model_num = int(input('>>>>>>>model epoch: '))
    model_num = (model_num + 2) // 5 * 5 - 1
    path = 'results2/epoch%d_model' % model_num
    if not os.path.isdir(path):
        print("%s can't be found" % path)
        return None
    else:
        with open(os.path.join(path, 'idx_to_word.pickle'), 'rb') as f:
            idx_to_word = pickle.load(f)
        with open(os.path.join(path, 'word_to_idx.pickle'), 'rb') as f:
            word_to_idx = pickle.load(f)
            print('vocabulary:')
            print(list(word_to_idx.keys())[:50])
        with open(os.path.join(path, 'Win.pickle'), 'rb') as f:
            Win = pickle.load(f)
        return None


def get_word_embwdding(word):
    if word in word_to_idx.keys():
        return Win[:, word_to_idx[word]]
    else:
        print('sorry, %s is not in the vocabulary.' % word)
        return None


def get_near_words(word, k=10):
    '''
    :param word: 単語
    :param k: 取ってくる近い単語の数
    :return:　wordとベクトルの近い上位k件の単語-cos類似度リスト
    '''
    embedding = get_word_embwdding(word)
    if embedding is None:
        return None
    bunnsi = Win.T.dot(embedding)
    bunnbo = np.sqrt(np.sum(Win**2, axis=0))
    res = bunnsi/bunnbo
    idx = (res).argsort()[:-(k+1):-1]
    norm_em = np.sqrt(np.sum(embedding**2))
    result = [(idx_to_word[i], res[i]/norm_em) for i in idx]
    return result


if idx_to_word is None:
    _load_model()


if __name__ == '__main__':
    word = input('>>>>')
    while 'q' != word:
        pprint(get_near_words(word))
        word = input('>>>>')
