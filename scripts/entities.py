#!/usr/bin/python
import spacy
from scripts import text_processing
from nltk.stem import WordNetLemmatizer
import numpy as np
nlp = spacy.load('en')
lemmatizer = WordNetLemmatizer()
seq_len = 2
roles = ["s", "o", "x", "-"]


############################################
#       MODULE LOADING FUNCTIONS           #
#       - CREATING TRANSITION TYPES        #
############################################

def get_trans(trans, trans_len):
    """
    Retrieve all possible types of transitions of length specified by the global
    seq variable. It's a recursive function. The first variable is a list -
    at each call of the function is contains all types of transitions of length
    trans_len. If this is not the desired length the function will create
    new, longer transitions based on the ones it was given and call itself
    with these new transitions and trans_len increased by one.

    :param trans: []
    :param trans_len: int
    :return: []
    """
    if trans_len is seq_len:
        return trans

    longer_trans = []
    for i in range(0, len(trans)):
        for j in range(0, len(roles)):
            extended_trans = trans[i] + roles[j]
            longer_trans.append(extended_trans)
    return get_trans(longer_trans, trans_len + 1)


def get_transition_seqs_mapping():
    """
    Get a dictionary which maps types of all possible transitions to their id.
    :return:
    """
    trans = get_trans(roles, 1)
    return dict(zip(trans, range(0, len(trans))))

trans_seqs = get_transition_seqs_mapping()


#######################################
#       MAIN FUNCTIONALITY            #
#       - GETTING FEATURES            #
#######################################

def get_features(files):
    """
    From each file retrieve a special set of features which encode information
    relating how entities are discussed throughout the text.
    :param files:
    :return:
    """
    files = files.tolist()
    return np.array([pipeline(file) for file in files])


def pipeline(file):
    """
    Pipeline involves parsing a text in order to obtain POS for all words
    and dependency relations, creating an entity-grid from all the nouns, proper
    nouns and pronouns, turning that grid into a distribution over sequences
    of entities and finally turning that distribution into a number of features
    :param file:
    :return:
    """
    # special processing is performed to avoid sentence boundaries after abbrevs
    doc = nlp(text_processing.preprocess_text_ents(file))
    grid = get_grid(doc)
    distrib = get_distrib(grid, doc)
    return get_feats(distrib)


def get_feats(distrib):
    """
    Turn the given binary distribution into a set of binary features.
    :param distrib: list
    :return: list
    """
    feats = []
    for i in range(0, len(distrib)):
        for j in range(i + 1, len(distrib)):
            if distrib[i] > distrib[j]:
                feats.append(1)
            else:
                feats.append(0)
    return np.array(feats)


def get_distrib(grid, doc):
    """
    From the entity grid retrieve the distribution of specific sequences of entity
    transitions - the sequence length is chosen to be 2.
    :param grid:
    :param doc:
    :return:
    """
    global trans_seqs
    total_count = 0
    distrib = [0] * len(trans_seqs)
    for ent in grid:
        ent_info = grid[ent]

        # start checking from 1st sent, at each sent look back
        for i, s in enumerate(doc.sents):
            if i is 0:
                continue
            if i in ent_info:
                current_label = ent_info[i]
            else:
                current_label = "-"
            if (i-1) in ent_info:
                prev_label = ent_info[i-1]
            else:
                prev_label = "-"

            seq_index = trans_seqs[current_label + prev_label]
            total_count += 1
            distrib[seq_index] = distrib[seq_index] + 1
    return distrib


################################
#       GETTING GRID           #
################################

def get_grid(doc):
    """
    Analyse the text sentence by sentence and record how the grammatical roles of
    the entities are changing. Record all information in a entity-grid,
    implemented using a dictionary. The keys are words (lemmatized) and the
    values are dictionaries, in which keys are sentence indexes and values are
    the word's grammatical roles. If a word does not occur in some sentence
    it does not have an entry associated with that sentence's id.
    :param doc: spacy parsed document
    :return: {{}}
    """
    grid = {}
    i = 0
    for sent in doc.sents:
        if len(sent) > 2:
            for token in sent:
                if token.pos_ == "NOUN" or token.pos_ == "PRON"\
                            or token.pos_ == "PROPN":
                    add_to_grid(grid, i, token)
            i += 1
    return grid


def add_to_grid(grid, i, token):
    global lemmatizer
    cat = get_category(token.dep_)
    word = lemmatizer.lemmatize(token.text)

    if word in grid:
        grid[word][i] = cat
    else:
        grid[word] = {i: cat}


def get_category(dep):
    if dep == "pobj" or dep == "dobj":
        return "o"
    elif dep == "nsubj":
        return "s"
    else:
        return "x"


def print_grid(grid):
    for k in grid:
        print "\n"
        print(k)
        for y in grid[k]:
            print(y, ":", grid[k][y])


#########################
#       OTHER           #
#########################

def find_feature(index):
    count = 0
    for i in range(0, len(trans_seqs)):
        for j in range(i + 1, len(trans_seqs)):
            if count == index:
                print find_trans(i), " > ", find_trans(j)
                return
            count += 1
    print "didn't find feature ", index


def find_trans(index):
    for t in trans_seqs:
        if trans_seqs[t] is index:
            return t

# CORENLP - anaphora resolution attempt
#def get_features2(files):
#    server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
#                                 jsonrpc.TransportTcpIp(addr=("127.0.0.1", 8080)))
#
#    files = files.tolist()
#
#    f = process_text.preprocess_text_ents(files[0])
#    sents = nltk.sent_tokenize(f)
#    #for sent in sents:
#    #    print sent
#    result = loads(server.parse(sents[0] + sents[1]))
#    print "Result", result
