#!/usr/bin/python
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from entities import find_feature


def get_vecs(vocab_size, train_toks, test_toks, train_res, ef_train, ef_test):
    """
    Given an array of training files (tokens), an array of test files (tokens)
    and an array of classes - each corresponding to one training example -
    returns two arrays of binary vectors - the first corresponds to training
    examples and the latter to test set

    :param vocab_size: int
    :param train_toks: 2D nparray
    :param test_toks: 2D nparray
    :param train_res: nparray
    :param ef_test: list - features extracted from entity grid, can be None
    :param ef_train: list - features extracted from entity grid, can be None
    :return: 2D nparray, 2D nparray
    """

    vocab = retrieve_full_vocab(train_toks)

    # get vectors for full vocabulary
    train_vecs = get_vectors_for_all_docs(train_toks, vocab)
    test_vecs = get_vectors_for_all_docs(test_toks, vocab)

    if ef_train is not None and ef_test is not None:
        train_vecs = np.array(
            [np.append(e, d) for e, d in zip(train_vecs, ef_train)])
        test_vecs = np.array(
            [np.append(e, d) for e, d in zip(test_vecs, ef_test)])

    # prune vocabulary
    mi = SelectKBest(mutual_info_classif, k=vocab_size)
    train_vecs = mi.fit_transform(train_vecs, train_res)
    test_vecs = mi.transform(test_vecs)

    if ef_train is not None and ef_test is not None:
        efs_chosen(mi, ef_train, vocab)
    return train_vecs, test_vecs


def efs_chosen(mi, ef_train, vocab):
    """
    Test weather any of the entity related features was chosen to the
    'most meaningful' feature set
    :param mi:
    :param ef_train:
    :return:
    """
    if ef_train is None:
        return
    vlen = len(vocab)
    last_normal_feature = vlen - 1
    test_len = vlen + len(ef_train[0])
    test = np.array([list(range(0, test_len)), list(range(0, test_len))])
    res = mi.transform(test)

    ent_feats = []
    for i in res[0]:
        if i > last_normal_feature:
            ent_feats.append(i)

    print "Number of entity features in chosen vocab: ", len(ent_feats)
    for ef in ent_feats:
        index = ef - vlen
        find_feature(index)


def get_vectors_for_all_docs(docs, vocab):
    """
    Create binary vector for each document in given documents array. The vector
    length is the same as the given vocabulary size - for each word in the vocab
    a corresponding vector dimension holds either 0 (if a word is in not in the
    document) or 1 (if the word is in the document)

    :param docs: 2D nparray - each document is tokenized
    :param vocab: dict
    :return: 2D nparray
    """
    docs_vectors = [get_feature_vector(doc, vocab) for doc in docs]
    return np.array(docs_vectors)


def get_feature_vector(doc, vocab):
    vec = [0] * len(vocab)
    for word in doc:
        if word in vocab:   # set the value of the word's feature to 1
            index = vocab[word]
            vec[index] = 1
    return np.array(vec)


def retrieve_full_vocab(docs):
    """
    Retrieve all words that occur in all documents, record them in the global
    full_vocab_dic. Their values correspond to features ids/indexes.
    The passed docs should be the set of training data only.

    :param docs: the list of all documents
    """
    full_vocab_dic = {}

    # keys are words, values are their counts
    full_vocab_counts = {}
    for doc in docs:
        for word in doc:
            if word not in full_vocab_counts:
                full_vocab_counts[word] = 1
            else:
                full_vocab_counts[word] += full_vocab_counts[word] + 1

    # remove all words which occurred only once form a feature set
    todel = []
    for v in full_vocab_counts:
        if full_vocab_counts[v] == 1:
            todel.append(v)
    for d in todel:
        del full_vocab_counts[d]

    # set all the indexes/ids of features
    i = 0
    for v in full_vocab_counts:
        full_vocab_dic[v] = i
        i += 1

    print "full vocab: ", len(full_vocab_dic)
    return full_vocab_dic

