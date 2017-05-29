# -*- coding: utf-8 -*-
import MeCab
from itertools import chain
from collections import Counter
# ここのパスを自分のデータセットのパスに書き換えてください。
courpas_PATH = '/path/to/corpus/neko.txt'


def load_courpas():
    m = MeCab.Tagger('-F"%f[6] " -U"%m " -E"\n"')
    sentences = []
    for line in open(courpas_PATH, 'r'):
        wakachi = m.parse(line).strip().split('"')
        wakachi = [w for w in filter(lambda x: x != '' and x != '\u3000', wakachi)]
        if len(wakachi) > 2:
            sentences.append(wakachi)
    return sentences


def mk_dictionaries():
    sentences = load_courpas()
    words = list(chain(*sentences))
    word_frequency = Counter(words)

    # 上位10%を語彙から外す
    stop_words = set(w for w, c in word_frequency.most_common(len(word_frequency)//10))
    vocab = set(word_frequency.keys()) - stop_words
    words = [w for w in filter(lambda x: x not in stop_words, words)]
    sentences = [[w for w in filter(lambda x: x not in stop_words, sentence)] for sentence in sentences]

    word_to_idx = {w:i for i, w in enumerate(vocab)}
    idx_to_word = {i:w for i, w in enumerate(vocab)}
    return sentences, words, vocab, word_to_idx, idx_to_word
