# -*- coding: utf-8 -*-
import numpy as np
from preparate_courpas import mk_dictionaries
from functions import sigmoid, error
from random import shuffle, sample
import pickle
import os

sentences, words, vocab, word_to_idx, idx_to_word = mk_dictionaries()
Win = Wout = None
vocab_size = len(vocab)
learning_ratio = 0.05


def init_model(embedding_size=1024):
    global Win, Wout
    if Win:
        pass
    else:
        Win = np.random.random((embedding_size, vocab_size))
        Wout = np.random.random((vocab_size, embedding_size))


def feed_forward(in_vector):
    hidden_vector = Win.dot(in_vector)
    out_vector = Wout.dot(hidden_vector)
    out_vector = sigmoid(out_vector)
    return hidden_vector, out_vector


def feed_back(out_vector, label):
    # delta_out = ((1 - label) / (1 - out_vector) - label / out_vector) * (1 - out_vector) * out_vector
    delta_out = out_vector - label
    delta_out[np.isnan(delta_out)] = 0
    delta_hidden = Wout.T.dot(delta_out)
    return delta_hidden, delta_out


def update_parameters(in_vector, hidden_vector, delta_hidden, delta_out):
    global Win, Wout
    dEin = np.array(np.matrix(delta_hidden).T * in_vector)
    dEout = np.array(np.matrix(delta_out).T * hidden_vector)
    Win -= learning_ratio * dEin
    Wout -= learning_ratio * dEout


def train_one_step(word, pos_words, neg_words):
    in_vector = np.zeros(vocab_size)
    in_vector[word_to_idx[word]] = 1
    label = np.array([np.nan for _ in range(vocab_size)])
    label[[word_to_idx[w] for w in pos_words]] = 1
    label[[word_to_idx[w] for w in neg_words]] = 0

    hidden_vector, out_vector = feed_forward(in_vector)
    delta_hidden, delta_out = feed_back(out_vector, label)
    update_parameters(in_vector, hidden_vector, delta_hidden, delta_out)
    return error(out_vector, label)


def estimate(word, pos_words=[], neg_words=[]):
    in_vector = np.zeros(vocab_size)
    in_vector[word_to_idx[word]] = 1
    label = np.array([np.nan for _ in range(vocab_size)])
    label[[word_to_idx[w] for w in pos_words]] = 1
    label[[word_to_idx[w] for w in neg_words]] = 0

    embedding, out_vector = feed_forward(in_vector)
    return embedding, error(out_vector, label)


if __name__ == '__main__':
    # 各種パラメータ入力
    max_epoch = int(input('>>>>>> max epoch:'))
    window_size = int(input('>>>>>> window size:'))
    neg_sample_num = int(input('>>>>>> negative sample number:'))
    embedding_size = int(input('>>>>>> embedding size: '))
    with open('results/parameters.txt', 'w') as f:
        f.write("max_epoch: %d\n" % max_epoch)
        f.write("window size: %d\n" % window_size)
        f.write("negative sample: %d\n" % neg_sample_num)
        f.write("embedding size: %d\n" % embedding_size)


    # データセットを準備
    train_error = []
    val_error = []
    shuffle(sentences)
    i1 = int(len(sentences) * 0.6)
    i2 = int(len(sentences) * 0.8)
    train_sentences = sentences[:i1]
    val_sentences = sentences[i1:i2]
    test_sentences = sentences[i2:]

    # 重み行列を作成
    init_model(embedding_size)

    # テスト用のデータを保存しコードから抹消
    with open('results/test_sentences.pickle', 'wb') as f:
        pickle.dump(test_sentences, f)
    del test_sentences

    # 学習
    for epoch in range(max_epoch):
        print('epoch', epoch)
        shuffle(train_sentences)
        error_tmp = []

        for sentence in train_sentences:
            # train_sentence内の全ての単語について学習する。train_sentenceはsentenceのリスト。sentenceは単語のリスト。
            for i in range(len(sentence)):
                # sentence内の全ての単語について、その前後window_size個の単語を正例(pos_words)とする。
                start = i - window_size if i - window_size > 0 else 0
                end = i + window_size + 1 if i + window_size + 1 < len(sentence) else len(sentence)
                pos_words = sentence[start:i] + sentence[i + 1:end]
                # 負例をneg_sample_num個ランダムに取ってくる。
                neg_words = sample(words, neg_sample_num)
                word = sentence[i]
                # これらのword, pos_words, neg_wordsで一回学習し、その時の誤差をerror_tmpに保持
                error_tmp.append(train_one_step(word, pos_words, neg_words))
        # 1epoch文の学習が終わったので、今回の訓練誤差を記憶
        train_error.append(np.average(error_tmp))

        error_tmp = []
        for sentence in val_sentences:
            # 訓練時と同様にword, pos_words, neg_wordsのセットを作り、誤差を得る。※重みの更新はしない。
            for i in range(len(sentence)):
                start = i - window_size if i - window_size > 0 else 0
                end = i + window_size + 1 if i + window_size + 1 < len(sentence) else len(sentence)
                pos_words = sentence[start:i] + sentence[i + 1:end]
                neg_words = sample(words, neg_sample_num)
                word = sentence[i]
                _, val_err = estimate(word, pos_words, neg_words)
                error_tmp.append(val_err)
        # 検証誤差を記憶
        val_error.append(np.average(error_tmp))

        # 重み行列を5epochごとに記憶
        if epoch % 5 == 4 or epoch == max_epoch-1:
            if not os.path.isdir('results/epoch%d_model' % epoch):
                os.system('mkdir results/epoch%d_model' % epoch)
            with open('results/epoch%d_model/Win.pickle' % epoch, 'wb') as f:
                pickle.dump(Win, f)
            with open('results/epoch%d_model/Wout.pickle' % epoch, 'wb') as f:
                pickle.dump(Wout, f)
            with open('results/epoch%d_model/word_to_idx.pickle' % epoch, 'wb') as f:
                pickle.dump(word_to_idx, f)
            with open('results/epoch%d_model/idx_to_word.pickle' % epoch, 'wb') as f:
                pickle.dump(idx_to_word, f)

    # 訓練誤差、検証誤差を保存
    with open('results/learning_curve.csv', 'w') as f:
        f.write('epoch,train,val\n')
        for epo, err in enumerate(zip(train_error, val_error)):
            f.write('%.4f,%.4f,%.4f\n' % (epo, err[0], err[1]))

