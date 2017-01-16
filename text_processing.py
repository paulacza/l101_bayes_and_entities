#!/usr/bin/python
from nltk.stem.snowball import EnglishStemmer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import re

stoplist = False
stem = False
process_html = False
remove_headers = False


def process_texts(docs, sl, s, ph, rh):
    """
    Preprocess and tokenize all given texts.

    :param docs: nparray (strings)
    :param sl: bool
    :param s: bool
    :param ps: bool
    :param rh: bool
    :return: 2D nparray (array of tokens)
    """
    global stoplist, stem, process_html, remove_headers
    stoplist = sl
    stem = s
    process_html = ph
    remove_headers = rh
    return np.array([tokenize(preprocess_text(file)) for file in docs])


def preprocess_text(doc):
    """
    Preprocess text - remove html tags, skip headers
    (if the corresponding variables are set to True)

    :param doc: string
    :return: string
    """
    doc = doc.lower()
    if remove_headers:
        doc = doc[doc.find("<html>"):]
    if re.search("<html>", doc) is not None and process_html:
        doc = proc_html(doc)
    return doc


def preprocess_text_ents(doc):
    return proc_html(re.sub(r"\.([a-z])\.", r".\1", doc))


def proc_html(doc):
    if "<html>" not in doc:
        return doc
    return re.sub("(\s)+", " ", BeautifulSoup(doc, "html.parser").get_text())


def tokenize(doc):
    """
    Tokenize given document, replacing the digits with a special token.
    Stemming and stoplist are also performed at this stage (if at all).

    :param doc: string
    :param stoplist: bool
    :param stem: bool
    :return: nparray - tokens
    """
    stopset = set(stopwords.words('english'))
    tokens = word_tokenize(doc)
    digit_pattern = re.compile(r'\d+')
    word_pattern = re.compile(r"[a-z'-]+")
    stemmer = EnglishStemmer()

    toks = []
    for t in tokens:
        if not t in stopset or not stoplist:
            if digit_pattern.match(t):
                toks.append("__digits__")
            elif word_pattern.match(t):
                if stem:
                    toks.append(stemmer.stem(t))
                else:
                    toks.append(t)
    return np.array(toks)
