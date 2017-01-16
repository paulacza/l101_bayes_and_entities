#!/usr/bin/python
from scripts import data_retrieval
from scripts import docs_to_vecs
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.metrics import f1_score
import sys
from scripts import entities
import numpy as np
from scripts import text_processing
from sklearn.model_selection import StratifiedKFold

k = 3
vocab_size = 100
include_entity_feats = True
force_using_efs = False
entity_feats_len = 0


def main(args):
    global test_runs, include_entity_feats, force_using_efs, vocab_size

    if len(args) == 7:
        vocab_size = int(args[0])
        stoplist = bool(int(args[1]))
        stem = bool(int(args[2]))
        process_html = bool(int(args[3]))
        remove_headers = bool(int(args[4]))
        include_entity_feats = bool(int(args[5]))
        force_using_efs = bool(int(args[6]))

        files, target = data_retrieval.get_webkb_files()
        print"\n============================================\nDocs:", len(files)

        tokens = text_processing.process_texts(files, stoplist, stem,
                                               process_html, remove_headers)
        run_tests(files, tokens, target)
    else:
        print("Arguments need to be specified:\nvocabulary size (int), stoplist "
              "use (0/1),\nstemming (0/1), html processing (0/1),\n"
              "header removal (0/1), use of entity grid features (0/1)")


####################################
#       RUNNING TESTS              #
####################################


def run_tests(files, tokens, target):
    """
    Do k-fold cross validation tests on the given data.
    :param files:
    :param tokens:
    :param target:
    :return:
    """
    global entity_feats_len
    if include_entity_feats:
        entity_feats = entities.get_features(files)
        entity_feats_len = len(entity_feats[0])
    else:
        entity_feats = None

    average_acc = 0
    average_fmac = 0
    average_fmic = 0
    kf = StratifiedKFold(n_splits=k)
    i = 1
    for train_indexes, test_indexes in kf.split(X=tokens, y=target):
        print "\n--> TEST ", i
        acc, macro_f, micro_f = run_test(tokens, target, entity_feats,
                                train_indexes, test_indexes)
        print "acc: %0.3f" % acc
        print "macro_f: %0.3f" % macro_f
        print "micro_f: %0.3f" % micro_f
        average_acc += acc
        average_fmic += micro_f
        average_fmac += macro_f
        i += 1

    average_acc /= float(k)
    average_fmic /= float(k)
    average_fmac /= float(k)
    print_results(average_acc, average_fmac, average_fmic)


def run_test(tokens, target, entity_feats, train_indexes, test_indexes):
    """
    Obtain vectors an run the one test.
    :param tokens:
    :param target:
    :param entity_feats:
    :param train_indexes:
    :param test_indexes:
    :return:
    """
    tok_train = tokens[train_indexes]
    tok_test = tokens[test_indexes]
    target_train = target[train_indexes]
    target_test = target[test_indexes]

    if include_entity_feats:
        ef_train = entity_feats[train_indexes]
        ef_test = entity_feats[test_indexes]
    else:
        ef_test = ef_train = None

    if not include_entity_feats or force_using_efs:
        data_train, data_test = docs_to_vecs.get_vecs(vocab_size, tok_train,
                                tok_test, target_train, None, None)
    else:
        data_train, data_test = docs_to_vecs.get_vecs(vocab_size, tok_train,
                                tok_test, target_train, ef_train, ef_test)

    if force_using_efs:
        data_train = np.array(
            [np.append(e, d) for e, d in zip(data_train, ef_train)])
        data_test = np.array(
            [np.append(e, d) for e, d in zip(data_test, ef_test)])

    return train_and_test(data_train, target_train, data_test, target_test)


def train_and_test(data_train, target_train, data_test, target_test):
    clf = BernoulliNB(binarize=None)
    clf.fit(data_train, target_train)
    test_res = clf.predict(data_test)
    macro_f = f1_score(target_test, test_res, average='macro')
    micro_f = f1_score(target_test, test_res, average='micro')
    acc = metrics.accuracy_score(target_test, test_res)
    return acc, macro_f, micro_f


####################################
#       OTHER                      #
####################################

def print_results(acc, f_mac, f_mic):
    print "\n--------- RESULTS ---------"
    print "test runs:", k, "\n"
    if force_using_efs:
        print "vocab size: ", (vocab_size + entity_feats_len)
    else:
        print "vocab size: ", vocab_size
    print "stoplist: ", text_processing.stoplist
    print "stem: ", text_processing.stem
    print "html parse: ", text_processing.process_html
    print "headers removed: ", text_processing.remove_headers
    print "entity features used: ", include_entity_feats
    print "forces using entity feats: ", force_using_efs
    print "~~~~~~~~~~~~~~~~~~~~~~~~~~"
    print "average accuracy: %0.3f" % acc
    print "average F score (macro): %0.3f" % f_mac
    print "average F score (micro): %0.3f" % f_mic


if __name__ == "__main__":
    main(sys.argv[1:])
