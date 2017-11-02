#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    SearchWithInference - Proof of concept for PerSea (Personal Search)
    Copyright (C) 2017  Matthew A. Snell (Astro7467@XLTech.io)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import collections
import contextlib
import csv
import itertools
import math
import os
import re
import random
import sys
import time
import shelve
import uuid

import numpy as np
import tensorflow as tf

# from pathlib import Path
# from simple_shelve import SimpleShelf, SimpleMultiShelf

from bisect import bisect_left
from sklearn.feature_extraction.text import CountVectorizer  #, TfidfVectorizer
from tqdm import tqdm

_storeDocMeta_ngrams = True # Storage of ngram list in dbstores['docmeta'] per srcID

# Current Suggested Global Variables for use with traceLog
_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace
# _logSysLevel = _logStatus

# see https://pypi.python.org/pypi/tqdm for how to use tqdm

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err



def tf_word2vec(dbstores, swicfg):
    # Tensorflow Word2Vec Model Training
    # Backtrack a little bit to avoid skipping words in the end of a batch

    w2vdbstore = shelve.open(swicfg['w2vdata'],  flag='r', protocol=4, writeback=False)

    vectors_size = len(w2vdbstore['vectors'])
    vocabulary_size = max(w2vdbstore['revdict'].keys()) + 1
    #vector_set_size = len(w2vdbstore['revdict'])
    dataIndex = 0

    # Function to generate a training batch for the skip-gram model.
    def tf_w2v_generate_batch(dataIndex, batchSize, numSkips, skipWindow):
        assert batchSize % numSkips == 0
        assert numSkips <= 2 * skipWindow
        batch = np.ndarray(shape=(batchSize), dtype=np.int32)
        labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
        span = 2 * skipWindow + 1  # [ skipWindow target skipWindow ]
        buffer = collections.deque(maxlen=span)
        if dataIndex + span > vectors_size:
            dataIndex = 0
        buffer.extend(w2vdbstore['vectors'][dataIndex:dataIndex + span])
        dataIndex += span
        for i in range(batchSize // numSkips):
            contextWords = [w for w in range(span) if w != skipWindow]
            random.shuffle(contextWords)
            wordsToUse = collections.deque(contextWords)
            for j in range(numSkips):
                batch[i * numSkips + j] = buffer[skipWindow]
                contextWord = wordsToUse.pop()
                labels[i * numSkips + j, 0] = buffer[contextWord]
            if dataIndex == vectors_size:
                buffer[:] = w2vdbstore['vectors'][:span]
                dataIndex = span
            else:
                buffer.append(w2vdbstore['vectors'][dataIndex])
                dataIndex += 1
        dataIndex = (dataIndex + vectors_size - span) % vectors_size
        return dataIndex, batch, labels


    dataIndex, batch, labels = tf_w2v_generate_batch(dataIndex,
                                                     batchSize=8, numSkips=2, skipWindow=1)
    for i in range(8):
        logStr = str(batch[i])+' '+str(w2vdbstore['revdict'][batch[i]])+ \
                 '->'+str(labels[i, 0])+' '+str(w2vdbstore['revdict'][labels[i, 0]])
        trace_log(_logSysLevel, _logInfo, logStr, context='Word2Vec')

    # Build and train a skip-gram model.
    batchSize = 128
    embeddingSize = 128  # Dimension of the embedding vector.
    skipWindow = 1       # How many words to consider left and right.
    numSkips = 2         # How many times to reuse an input to generate a label.
    numSampled = 64      # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    validSize = 16     # Random set of words to evaluate similarity on.
    validWindow = 100  # Only pick dev samples in the head of the distribution.

    validExampleSet = w2vdbstore['counts'][:validWindow]
    random.shuffle(validExampleSet)
    validExampleSet = validExampleSet[:validSize]
    trace_log(_logSysLevel, _logInfo, validExampleSet,
              context='Word2Vec - 16 Random Validation Items from Top 100')

    validExamples = list()
    for wordcountpair in validExampleSet:
        validExamples.append(w2vdbstore['dict'][wordcountpair[0]])

    trace_log(_logSysLevel, _logInfo, validExamples, context='Word2Vec - Valid Examples')

    graph = tf.Graph()

    with graph.as_default():

        # Input data.
        trainInputs = tf.placeholder(tf.int32, shape=[batchSize])
        trainLabels = tf.placeholder(tf.int32, shape=[batchSize, 1])
        validDataset = tf.constant(validExamples, dtype=tf.int32)
        fullDataset = tf.constant(sorted(w2vdbstore['vectorset']), dtype=tf.int32)

        # Ops and variables pinned to the CPU or GPU
        # change to CPU if not on tensorflow-gpu with CDDN & CUDA support
        #with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embeddingSize], -1.0, 1.0))

        embed = tf.nn.embedding_lookup(embeddings, trainInputs)

        # Construct the variables for the NCE loss
        nceWeights = tf.Variable(
            tf.truncated_normal(
                [vocabulary_size, embeddingSize], stddev=1.0 / math.sqrt(embeddingSize)))

        nceBiases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nceWeights,
                biases=nceBiases,
                labels=trainLabels,
                inputs=embed,
                num_sampled=numSampled,
                num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalizedEmbeddings = embeddings / norm

        validEmbeddings = tf.nn.embedding_lookup(
            normalizedEmbeddings, validDataset)
        similarity = tf.matmul(
            validEmbeddings, normalizedEmbeddings, transpose_b=True)

        fullEmbeddings = tf.nn.embedding_lookup(
            normalizedEmbeddings, fullDataset)
        fullSimilarity = tf.matmul(
            fullEmbeddings, normalizedEmbeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    # Begin training.
    numSteps = 10001
    #numSteps = 101

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        trace_log(_logSysLevel, _logInfo, 'Initialized', context='Word2Vec')

        # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
        # Enables tqdm to control progress bar on screen location
        with std_out_err_redirect_tqdm() as orig_stdout:
            # tqdm needs the original stdout
            # and dynamic_ncols=True to autodetect console width
            averageLoss = 0
            for step in tqdm(range(numSteps), file=orig_stdout, dynamic_ncols=True):
                dataIndex, batchInputs, batchLabels = tf_w2v_generate_batch(dataIndex, batchSize, numSkips, skipWindow)
                feedDict = {trainInputs: batchInputs, trainLabels: batchLabels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, lossVal = session.run([optimizer, loss], feed_dict=feedDict)
                averageLoss += lossVal

                if step % 2000 == 0:
                    if step > 0:
                        averageLoss /= 2000

                    # The average loss is an estimate of the loss over the last 2000 batches.
                    trace_log(_logSysLevel, _logInfo,
                              'Average loss at step '+str(step)+': '+str(averageLoss), context='Word2Vec')
                    averageLoss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 1000 == 0:
                    trace_log(_logSysLevel, _logInfo, 'Validation Set Similarity Comparisions', context='Word2Vec')
                    sim = similarity.eval()
                    ngramList = list()

                    for i in range(validSize):
                        validWordList = list()
                        validWord = w2vdbstore['revdict'][validExamples[i]]
                        logStr = 'Nearest to %s:' % validWord
                        topK = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:topK + 1]

                        for k in range(topK):
                            if nearest[k] in w2vdbstore['revdict']:
                                closeWord = w2vdbstore['revdict'][nearest[k]]
                                validWordList.append(closeWord)
                                logStr = '%s %s,' % (logStr, closeWord)
                            elif nearest[k] in dbstores['vectors']['vectors']:
                                closeWord = dbstores['vectors']['vectors'][nearest[k]]
                                validWordList.append(closeWord)
                                logStr = '%s %s*,' % (logStr, closeWord)
                            else:
                                logStr = '%s (%s),' % (logStr, nearest[k])

                        trace_log(_logSysLevel, _logInfo, logStr, context='Word2Vec')

                        for word in validWordList:
                            ngramList.append(str(validWord + ' ' + word))
                            ngramList.append(str(word + ' ' + validWord))

                    ngramCounts, unseenNgrams, _ = ngram_counts(dbstores, ngramList)
                    trace_log(_logSysLevel, _logTrace, ngramCounts, context='Word2Vec ngram Counts')
                    trace_log(_logSysLevel, _logTrace, unseenNgrams, context='Word2Vec Unseen ngram')

        trace_log(_logSysLevel, _logStatus, 'Calculating Full Similarity Data...', context='Word2Vec - Similarity')
        # Store full similarity data calcs for extraction
        fullSim = fullSimilarity.eval()

    # Store Similiarity Results
    trace_log(_logSysLevel, _logStatus,
              'Storing Similarity Data', context='Word2Vec - Similarity')
    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    # Enables tqdm to control progress bar on screen location
    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width

        for vector in tqdm(w2vdbstore['vectorset'], file=orig_stdout, dynamic_ncols=True):
            if vector in dbstores['vectors']['vectors']:
                wordList = list()
                word = dbstores['vectors']['vectors'][vector]
                logStr = 'Nearest to %s:' % word
                topK = 8  # number of nearest neighbors we will keep
                try:
                    nearest = (-fullSim[vector, :]).argsort()[1:topK + 1]
                    for distance in range(topK):
                        if nearest[distance] in dbstores['vectors']['vectors']:
                            closeWord = dbstores['vectors']['vectors'][nearest[distance]]
                            wordList.append(closeWord)
                            logStr = '%s %s,' % (logStr, closeWord)
                        else:
                            logStr = '%s (%s),' % (logStr, nearest[distance])

                    trace_log(_logSysLevel, _logTrace, logStr, context='Word2Vec - Similarity')

                except:
                    trace_log(_logSysLevel, _logTrace,
                              {'msg': 'Invalid Similarity Vector', 'vector': vector},
                              context='Word2Vec - Similarity')

                # Only save / overwrite existing data if we have something
                # expensive to get data, so don't destroy previous learnings
                if not len(wordList) == 0:
                    dbstores['similarity'][word] = wordList
                    dbstores['similarity'].sync()
            else:
                trace_log(_logSysLevel, _logTrace,
                          {'msg': 'Invalid Training Vector', 'vector': vector}, context='Word2Vec - Similarity')

    w2vdbstore.close()
    return


def ngram_counts(dbstores, ngramList):
    # Parse ngramList for unseen ngrams and remove
    ngramCounts = dict()
    unseenNgrams = list()
    srcNgramCounts = dict()

    for ngram in ngramList[:]:
        if ngram not in dbstores['ngram']:
            ngramList.remove(ngram)
            unseenNgrams += [ngram]

    # take verified ngram list and build counts
    for ngram in ngramList:
        # cycle thru srcID with ngram, adding to srcNgramCounts & ngramCounts
        for srcID in dbstores['ngram'][ngram].keys():
            if srcID not in srcNgramCounts:
                # Create and add src's count of ngram
                srcNgramCounts[srcID] = {ngram: dbstores['ngram'][ngram][srcID]}
            else:
                # increment ngram count for src (in theory, this should never happen)
                if ngram not in srcNgramCounts[srcID]:
                    # Create and add src's ngram count of ngram
                    srcNgramCounts[srcID][ngram] = dbstores['ngram'][ngram][srcID]
                else:
                    # increment ngram count for src (in theory, this should never happen)
                    srcNgramCounts[srcID][ngram] += dbstores['ngram'][ngram][srcID]

            # Record total ngram appearances so we can use as weight later
            if ngram not in ngramCounts:
                # Create and add src's count of ngram
                ngramCounts[ngram] = dbstores['ngram'][ngram][srcID]
            else:
                # Increment Total instances of ngram's occurance
                ngramCounts[ngram] += dbstores['ngram'][ngram][srcID]

    return ngramCounts, unseenNgrams, srcNgramCounts


def dictionary_vector(dbstores, lowest=False):
    # increment vector available and return

    nextVector = dbstores['config']['nextvector']

    if lowest:
        nextVector = 0

    # Validate we have a free Vector
    while nextVector in dbstores['vectors']['vectors']:
        nextVector += 1

    dbstores['config']['nextvector'] = nextVector
    dbstores['config'].sync()

    return nextVector


def dict_parse_words(dbstores, swicfg, words, xcheck=False):
    # Parse a given word list, add to dictionary & vectors if new

    # reduce to unique words
    wordList = set(words)

    # find words not already in dictionary, and add them to dict & vectors
    for word in wordList:
        if word not in dbstores['dict']:
            vector = dictionary_vector(dbstores)

            dbstores['dict'][word] = vector
            dbstores['dict'].sync()

            dbstores['vectors']['vectors'][vector] = word
            dbstores['vectors'].sync()

            trace_log(_logSysLevel, _logTrace, {word:dbstores['dict'][word], vector:dbstores['vectors']['vectors'][vector]}, context='Dictionary New Word: Vector')

        if xcheck:
            if dbstores['dict'][word] not in dbstores['vectors']['vectors']:
                # Missing Vector -> Word
                vector = dbstores['dict'][word]
                dbstores['vectors']['vectors'][vector] = word

                dbstores['dict'].sync()
                dbstores['vectors'].sync()

                trace_log(_logSysLevel, _logStatus, {word: vector}, context='Dictionary Fixed - Missing Vector Key')

            if not isinstance(dbstores['dict'][word], int):
                # found an invalid vector word/vector pair???
                vector = dictionary_vector(dbstores)
                dbstores['dict'][word] = vector
                dbstores['vectors']['vectors'][vector] = word

                dbstores['dict'].sync()
                dbstores['vectors'].sync()

                trace_log(_logSysLevel, _logStatus, {word: vector}, context='Dictionary Fixed - Bad Vector')

            if not dbstores['vectors']['vectors'][dbstores['dict'][word]] == word:
                # found an invalid word->vector->word???
                oldword = dbstores['vectors']['vectors'][dbstores['dict'][word]]

                # fix current word & vector maps
                currVector = dbstores['dict'][word]
                dbstores['vectors']['vectors'][currVector] = word

                # assign new vector to oldword
                newVector = dictionary_vector(dbstores)
                dbstores['dict'][oldword] = newVector
                dbstores['vectors']['vectors'][newVector] = oldword

                dbstores['dict'].sync()
                dbstores['vectors'].sync()

                trace_log(_logSysLevel, _logStatus, {word: dbstores['dict'][word], oldword: newVector}, context='Dictionary Fixed - Bad X Vector')

            if not normalise_text(word) == word:
                trace_log(_logSysLevel, _logStatus, {word: dbstores['dict'][word], dbstores['dict'][word]: dbstores['vectors']['vectors'][dbstores['dict'][word]]}, context='Dictionary Warning - Word Failed Normalised Scan')

    dbstores['dict'].sync()
    dbstores['vectors'].sync()
    return


def trace_log(sysLogLevel=_logSysLevel, logType=_logInfo, logData='', context=''):
    # Common Logging/Trace Routine

    logTypes = ('ERROR', 'STATUS', 'CONFIG', 'INFO', 'TRACE')

    if logType < 0 : logType = 0
    if logType > len(logTypes) -1 : logType = len(logTypes) -1

    if logType <= sysLogLevel:
        logText = time.strftime("%Y-%m-%d %H:%M:%S UTC%Z  ") + logTypes[logType].ljust(8)
        newLinePrefix = ' ' * len(logText)
        if context not in [None, ' ', '']: logText += str(context) + '; '

        logTextList = []
        logMaxLines = 12
        maxLineLen = 180

        if (isinstance(logData, str) or isinstance(logData, int) or isinstance(logData, float)):
            logTextList = [logText + str(logData).strip()[:maxLineLen]]
        elif (isinstance(logData, list) or isinstance(logData, tuple)):
            firstItem = True
            logText += ' Length '+str(len(logData))+', Type '+str(type(logData))+' - '

            for item in logData:
                if not firstItem:
                    logText += ', '
                if len(logText) > maxLineLen:
                    if len(logTextList) > logMaxLines:
                        break
                    else:
                        logTextList += [str(logText)]
                        logText = newLinePrefix

                logText += str(item).strip()[:maxLineLen]
                firstItem = False

            logTextList += [str(logText)]
        elif isinstance(logData, dict):
            firstItem = True
            logText += ' Length: '+str(len(logData))+', Type: '+str(type(logData))+' - '
            for key in logData.keys():
                if not firstItem:
                    logText = logText + ', '
                    if len(logText) > maxLineLen:
                        if len(logTextList) > logMaxLines:
                            break
                        else:
                            logTextList += [str(logText)]
                            logText = newLinePrefix

                logText += str(key).strip() + ': ' + str(logData[key]).strip()[:maxLineLen]
                firstItem = False

            logTextList += [str(logText)]
        else:
            logText += ' Length: ' + str(len(logData)) + ', Type: ' + str(type(logData))
            logTextList += [str(logText)]
            logText = newLinePrefix + str(logData).strip()[:maxLineLen]
            logTextList += [str(logText)]

        for line in logTextList: print(line)
    return


def calc_matches(dbstores, ngramList, maxResults=12):
    # find N best matches of given Ngrams vs Ngram Index

    srcNgramCounts = {}     # Store Raw per SrcID, by ngram counts
    ngramCounts = {}        # Per ngram how many unique (srcID) counts
    ngramWeights = {}       # Weight of each ngram in calcing srcID's value
    srcWeightedScore = {}   # Weight of each ngram in calcing srcID's value
    ngramWeightBase = 0.0   # Count we will base weight on (i.e. total ngram matches)
    srcNgramScore = {}      # Weighted score of match for each srcID
    unseenNgrams = []

    ngramCounts, unseenNgrams, srcNgramCounts = ngram_counts(dbstores, ngramList)

    for ngramCount in ngramCounts:
        ngramWeightBase += ngramCounts[ngramCount]

    # Assign a weight to each based on 'uniquiness'
    normaliseBaseMax = 1
    normaliseBaseMin = 0

    #for ngram, ngramCount in sorted(ngramCounts.interitems(), reverse=True, key=lambda(k, v): (v, k)):
    for ngram, ngramCount in [(ngram, ngramCounts[ngram]) for ngram in sorted(ngramCounts, key=ngramCounts.get, reverse=True)]:
        try:
            # More common ngrams get lower scores, longer ngrams get higher score
            ngramWeight = math.log(float(ngramWeightBase), float(ngramCount))
        except:
            ngramWeight = 1.0

        ngramWeight = ngramWeight * (float(ngramWeightBase) - float(ngramCount)) * float(len(ngram.split()))
        ngramWeights[ngram] = ngramWeight
        normaliseBaseMax = max(ngramWeight, normaliseBaseMax)
        normaliseBaseMin = min(ngramWeight, normaliseBaseMin)

    # Normalise ngramWeights
    for ngram in ngramWeights.keys():
        ngramWeights[ngram] /= (normaliseBaseMax - normaliseBaseMin)

    trace_log(_logSysLevel, _logTrace, normaliseBaseMax, context='normaliseBaseMax')
    trace_log(_logSysLevel, _logTrace, normaliseBaseMin, context='normaliseBaseMin')
    trace_log(_logSysLevel, _logTrace, ngramWeights, context='ngramWeights')

    # Score each srcID/doc based on ngram Weights
    normaliseBaseMax = 1
    normaliseBaseMin = 0
    for srcID in srcNgramCounts:
        for ngram in srcNgramCounts[srcID]:
            WeightedScore = ngramWeights[ngram] * srcNgramCounts[srcID][ngram] * len(ngram.split())
            if srcID not in srcWeightedScore:
                srcWeightedScore[srcID] = WeightedScore
            else:
                srcWeightedScore[srcID] += WeightedScore

        normaliseBaseMax = max(srcWeightedScore[srcID], normaliseBaseMax)
        normaliseBaseMin = min(srcWeightedScore[srcID], normaliseBaseMin)

    # Normalise srcWeightedScore
    for srcID in srcWeightedScore.keys():
        srcWeightedScore[srcID] /= (normaliseBaseMax - normaliseBaseMin)

    trace_log(_logSysLevel, _logTrace, normaliseBaseMax, context='normaliseBaseMax')
    trace_log(_logSysLevel, _logTrace, normaliseBaseMin, context='normaliseBaseMin')
    trace_log(_logSysLevel, _logInfo, srcWeightedScore, context='srcWeightedScore')

    topSrcMatches = []
    topSrcInfo = {}
    for srcID, WeightedScore in [(srcID, srcWeightedScore[srcID]) for srcID in sorted(srcWeightedScore, key=srcWeightedScore.get, reverse=True)]:
        #for srcID, WeightedScore in sorted(srcWeightedScore.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        topSrcMatches.append(srcID)
        topSrcInfo[srcID] = {'score': WeightedScore, 'ngrams': len(srcNgramCounts[srcID])}

    trace_log(_logSysLevel, _logInfo, topSrcMatches, context='topSrcMatches')

    return topSrcMatches[:maxResults], topSrcInfo, unseenNgrams


def open_datastores(setWriteback=True):
    # Open Datastores and return handles
    # TODO: Separate W/R opens from RO opens

    dbstores = dict()

    trace_log(_logSysLevel, _logInfo, 'Started Opening dbstores...')

    # Config Datastore
    dbstores['config'] = shelve.open('./data/config',
                                     flag='c', protocol=4, writeback=setWriteback)

    # Dictionary to Vectore Datastore
    dbstores['dict'] = shelve.open('./data/dictionary',
                                   flag='c', protocol=4, writeback=setWriteback)

    # Document Meta Data Datastore
    dbstores['docmeta'] = shelve.open('./data/docmeta',
                                      flag='c', protocol=4, writeback=setWriteback)

    # Document Stats (file ngrams) Datastore
    dbstores['docstat'] = shelve.open('./data/docstat',
                                      flag='c', protocol=4, writeback=setWriteback)

    # Ngram Datastore (Core index)
    dbstores['ngram'] = shelve.open('./data/ngram',
                                    flag='c', protocol=4, writeback=setWriteback)

    #file Type/Path to UUID Datastore
    dbstores['sources'] = shelve.open('./data/sources',
                                      flag='c', protocol=4, writeback=setWriteback)

    # Stopword Lists by Language (eg en) Datastore
    dbstores['similarity'] = shelve.open('./data/similarity',
                                        flag='c', protocol=4, writeback=setWriteback)

    # Stopword Lists by Language (eg en) Datastore
    dbstores['stopwords'] = shelve.open('./data/stopwords',
                                        flag='c', protocol=4, writeback=setWriteback)

    # Vector to Dictionary Datastore
    dbstores['vectors'] = shelve.open('./data/vectors',
                                      flag='c', protocol=4, writeback=setWriteback)

    # As shelves mandates str for keys, we create a dict inside with int keys for vectors
    # open()/close()/sync() need to occur against 'vectors' not ['vectors]['vectors']
    if 'vectors' not in dbstores['vectors']:
        dbstores['vectors']['vectors'] = dict()

    # Vectorized version of normalizeds and de-stopworded source document
    dbstores['vectorized'] = shelve.open('./data/vectorized',
                                         flag='c', protocol=4, writeback=setWriteback)

    trace_log(_logSysLevel, _logInfo, 'Finished Opening dbstores')

    return dbstores


def close_datastores(dbstores):
    # Close/Cleanup datastores

    for fileHandle in dbstores.keys():
        dbstores[fileHandle].sync()
        dbstores[fileHandle].close()
        trace_log(_logSysLevel, _logInfo, 'Closed dbStore: ' + fileHandle)
    return


def sys_config(dbstores):
    # UUID Management

    swicfg = dict()

    try:
        swicfg['uuid'] = dbstores['config']['uuid']
    except:
        dbstores['config']['uuid'] = str(uuid.uuid1())
        dbstores['config'].sync()
        swicfg['uuid'] = dbstores['config']['uuid']

    try:
        swicfg['corpus'] = dbstores['config']['corpus']
    except:
        dbstores['config']['corpus'] = './corpus'
        dbstores['config'].sync()
        swicfg['corpus'] = dbstores['config']['corpus']

    try:
        swicfg['ngram'] = dbstores['config']['ngram']
    except:
        dbstores['config']['ngram'] = 3
        dbstores['config'].sync()
        swicfg['ngram'] = dbstores['config']['ngram']

    try:
        swicfg['lang'] = dbstores['config']['lang']
    except:
        dbstores['config']['lang'] = 'en'
        dbstores['config'].sync()
        swicfg['lang'] = dbstores['config']['lang']

    try:
        swicfg['nextvector'] = dbstores['config']['nextvector']
    except:
        dbstores['config']['nextvector'] = 1
        dbstores['config'].sync()
        swicfg['nextvector'] = dbstores['config']['nextvector']

    try:
        swicfg['dbstores'] = dbstores['config']['dbstores']
    except:
        dbstores['config']['dbstores'] = './data'
        dbstores['config'].sync()
        swicfg['dbstores'] = dbstores['config']['dbstores']

    try:
        swicfg['word2vec'] = dbstores['config']['word2vec']
    except:
        dbstores['config']['word2vec'] = 5
        dbstores['config'].sync()
        swicfg['word2vec'] = dbstores['config']['word2vec']

    try:
        swicfg['w2vdata'] = dbstores['config']['w2vdata']
    except:
        dbstores['config']['w2vdata'] = './data/w2vdata'
        dbstores['config'].sync()
        swicfg['w2vdata'] = dbstores['config']['w2vdata']

    try:
        swicfg['srcdata'] = dbstores['config']['srcdata']
    except:
        dbstores['config']['srcdata'] = './data/srcdata'
        dbstores['config'].sync()
        swicfg['srcdata'] = dbstores['config']['srcdata']

    try:
        swicfg['version'] = dbstores['config']['version']
    except:
        dbstores['config']['version'] = 0.1
        swicfg['version'] = dbstores['config']['version']

    dbstores['config'].sync()

    trace_log(_logSysLevel, _logInfo, 'Read swicfg in from dbStore')

    return swicfg


def uuid_source(dbstores, swicfg, srcPath, srcCat, srcSubCat):
    # Match or Create a UUID for data sources

    fullSrcPath = str(srcCat+':'+srcSubCat+':'+srcPath)
    if fullSrcPath in dbstores['sources']:
        srcID = dbstores['sources'][fullSrcPath]
    else:
        dbstores['sources'][fullSrcPath] = str(uuid.uuid5(uuid.UUID(swicfg['uuid']), fullSrcPath))
        dbstores['sources'].sync()
        srcID = dbstores['sources'][fullSrcPath]

    init_source(dbstores, swicfg, srcID, srcPath, srcCat, srcSubCat)
    return srcID


def init_source(dbstores, swicfg, srcID, srcPath, srcCat='UNK', srcSubCat='UNK'):
    # Initialize Src Records

    # Create srcID for new sources Meta Data Record, and set minimum values
    if srcID not in dbstores['docmeta']:
        dbstores['docmeta'][srcID] = {
            'path'      :str(srcPath),
            'cat'       :str(srcCat),
            'subcat'    :str(srcSubCat)
        }
        dbstores['docmeta'].sync()
        chk_coredb_keys(dbstores, swicfg)
    return


def tuples2text(tuple):
    text = ''
    for item in tuple:
        if item not in [None, '', ' ',]:
            text += " " + item

    return re.sub('[ \t\n\r]+', ' ', text).strip(' \t\n\r')


def build_ngrams(inputList, width=2):
    # Build ngram list from given word list
    # Used for search string - assume users write sentences or keywords
    # cover all scenarios with comprehensive ngram

    outputList = []
    for length in range(1, width + 1):
        for tuple in itertools.permutations(inputList, length):
            outputList.append(tuples2text(tuple))

    return sorted([item for item in outputList if item != ''])


def normalise_text(text, stopwords=None, recheck=True):
    # Normalize Text Convert text to lower-case and strip punctuation/symbols from words

    normText = str(text).lower().strip()

    # Replace spaces, underscores, tabs, newlines and return sequences with a space
    normText = re.sub(r'[\s,._]+', r' ', normText, count=0, flags=re.MULTILINE).strip()

    # Process against stopwords to cover punctuated words
    if isinstance(stopwords, list):
        if recheck:
            trace_log(_logSysLevel, _logTrace, normText, context='Normalising with Stopwords')

        newText = [str(word) for word in normText.split() if word not in stopwords]
        normText = ' '.join([str(word) for word in newText])

        if recheck:
            trace_log(_logSysLevel, _logTrace, normText, context='Normalising with Stopwords - newText')

    # remove apostrophe in words: [alpha]'[alpha]=[alpha][alpha] eg. don't = dont
    normText = re.sub(r'(\w+)[\`\'](\w+)', r'\1\2', normText, count=0, flags=re.MULTILINE).strip()

    # Replace non-AlphaNumeric sequences with Space
    normText = re.sub(r'[^\w]+', r' ', normText, count=0, flags=re.MULTILINE).strip()

    # Replace pure digits with space eg 1234, but not 4u or best4u
    normText = re.sub(r'^\d+$|^\d+\s+|\s+\d+\s+|\s+\d+$', r' ', normText, count=0, flags=re.MULTILINE).strip()

    # As RegEx is not hungry, we need to interate over this to ensure all conditions are meet
    if recheck:
        while normText != normalise_text(normText, stopwords=stopwords, recheck=False):
            normText = normalise_text(normText, stopwords=stopwords, recheck=False)

    if normText in ['', ' ', None]: normText = ''

    return normText


def normalise_file(dbstores, swicfg, fileName, srcID):
    # Take a file and produces a stored normalised file

    try:
        trace_log(_logSysLevel, _logTrace,
                  {'Directory':os.path.join(swicfg['srcdata'], srcID), 'Filename':fileName}, context='Normalise Init mkdir')
        os.makedirs(os.path.join(swicfg['srcdata'], srcID), exist_ok=True, mode=0o777)
    finally:
        trace_log(_logSysLevel, _logTrace,
                  {'Directory':os.path.join(swicfg['srcdata'], srcID), 'Filename':fileName}, context='Normalise Init mkdir Failure')

    outFile = os.path.join(swicfg['srcdata'], srcID, 'normal.dat')
    with open(outFile, mode='wt', errors='replace') as writeFile:
        normalisedText = normalise_text(fileName)

        if normalisedText not in [None, '', ' ']:
            writeFile.write(normalisedText + '\n')

        with open(fileName, mode='rU', errors='ignore') as readFile:
            # read each line, normalise it, and send to temp file
            if dbstores['docmeta'][srcID]['subcat'] == 'TXT':
                for line in readFile:
                    normalisedText = normalise_text(line,
                                                    stopwords=dbstores['stopwords']['en'])
                    if normalisedText not in [None, '', ' ']:
                        writeFile.write(normalisedText + '\n')

            elif dbstores['docmeta'][srcID]['subcat'] == 'CSV':
                for line in csv.reader(readFile, delimiter=','):
                    txtLine = ' '.join([str(item) for item in line])
                    normalisedText = normalise_text(txtLine,
                                                    stopwords=dbstores['stopwords']['en'])
                    if normalisedText not in [None, '', ' ']:
                        writeFile.write(normalisedText + '\n')

    trace_log(_logSysLevel, _logTrace,
              {'outFile':outFile, 'Filename':fileName}, context='Normalised File Finished')

    return


def parse_file_txt(dbstores, swicfg, fileList, srcCat, srcSubCat):
    # Process given file as raw text line by line

    # for each file, get a UID and parse
    for fileName in fileList:
        trace_log(_logSysLevel, _logTrace,
                  {'Filename':fileName, 'SrcCat':srcCat, 'SrcSubCat': srcSubCat}, context='Examining')
        srcID = uuid_source(dbstores, swicfg, fileName, srcCat, srcSubCat)

        # Build normalised versions of files
        if not dbstores['docmeta'][srcID]['normalised'] \
                or not os.path.isfile(os.path.join(swicfg['srcdata'], srcID, 'normal.dat')):
            normalise_file(dbstores, swicfg, fileName, srcID)
            dbstores['docmeta'][srcID]['normalised'] = True
            dbstores['docmeta'].sync()
            dbstores['docstat'].sync()
            dbstores['ngram'].sync()
            trace_log(_logSysLevel, _logInfo,
                      {'SrcID':srcID, 'Filename':fileName, 'SrcCat':srcCat, 'SrcSubCat': srcSubCat}, context='Finished')

    return


def import_stopwords():
    if os.path.isfile("./stopwords-en.txt"):
        dbstores = open_datastores()
        swicfg = sys_config(dbstores)
        trace_log(_logSysLevel, _logInfo, "./stopwords-en.txt", context='Stopwords Loading')
        with open("./stopwords-en.txt") as stopFile:
            dbstores['stopwords']['en'] = stopFile.read().lower().split()
            trace_log(_logSysLevel, _logInfo,
                      dbstores['stopwords']['en'], context='Stopwords Loaded')

        dbstores['stopwords']['en-norm'] = [normalise_text(word) for word in dbstores['stopwords']['en']]
        trace_log(_logSysLevel, _logInfo,
                  dbstores['stopwords']['en-norm'], context='Stopwords Normalised')
        dbstores['stopwords'].sync()
        close_datastores(dbstores)
    else:
        trace_log(_logSysLevel, _logError,
                  "./stopwords-en.txt", context='Stopwords File Missing')
    return


def chk_coredb_keys(dbstores, swicfg):
    trace_log(_logSysLevel, _logInfo, "Checking DocMeta SrcID Keys...")
    for srcID in dbstores['docmeta']:
        trace_log(_logSysLevel, _logTrace, {'SrcID':srcID}, context='Parsing')
        if 'version' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['version'] = 0.1

        if 'path' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['path'] = ''

        if 'cat' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['cat'] = 'UNK'

        if 'subcat' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['subcat'] = 'UNK'

        if 'indexed' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['indexed'] = False

        if 'staged' not in dbstores['docmeta'][srcID].keys():
            dbstores['docmeta'][srcID]['staged'] = False

        if 'normalised' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['normalised'] = False

        if 'vector' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['vector'] = False

        if 'verdate' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['verdate'] = None

        if 'lastidx' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['lastidx'] = None

        if 'qltyscore' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['qltyscore'] = 0

        if 'indexscore' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['indexscore'] = 0

        if 'xrefscore' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['xrefscore'] = 0

        if 'ngrams' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['ngrams'] = list()

        if 'wordcount' not in dbstores['docmeta'][srcID]:
            dbstores['docmeta'][srcID]['wordcount'] = list()

        dbstores['docmeta'].sync()

        if srcID not in dbstores['docstat']:
            dbstores['docstat'][srcID] = dict()

        dbstores['docstat'].sync()

        if srcID not in dbstores['vectorized']:
            dbstores['vectorized'][srcID] = list()

        dbstores['vectorized'].sync()

    return


def vectorize_src(dbstores, swicfg):
    # Scans srcID for missing WordCount and/or Empty Vectorized lists
    # Build the data and populates each

    trace_log(_logSysLevel, _logInfo, 'Starting WordCount & Vectorize List...')

    # for each srcID, if not indexed/parsed - then extract ngrams
    for srcID in dbstores['docmeta']:
        trace_log(_logSysLevel, _logTrace, {'SrcID':srcID}, context='Vectoring')

        # Check Preconditions to trigger wordcount and vectorization
        if srcID not in dbstores['vectorized']:
            bldVector = True
            dbstores['vectorized'][srcID] = list()
        else:
            bldVector = dbstores['docmeta'][srcID]['normalised'] and dbstores['docmeta'][srcID]['indexed']
            bldVector = bldVector and not dbstores['docmeta'][srcID]['vector']
            bldVector = bldVector and (len(dbstores['vectorized'][srcID]) == 0 or len(dbstores['docmeta'][srcID]['wordcount']) == 0)

        fileName = os.path.join(swicfg['srcdata'], srcID, 'normal.dat')
        if bldVector and os.path.isfile(fileName):
            trace_log(_logSysLevel, _logTrace, {'Filename':fileName},
                      context='Vectoring Starting')

            with open(fileName, mode='rt', errors='ignore') as readFile:
                wordCount = collections.Counter()
                vectorList = list()

                # read each line, count each word & append to vector list as vectors
                wordCount.update(readFile.read().split())
                readFile.seek(0)
                for word in readFile.read().split():
                    try:
                        vectorList.append(dbstores['dict'][word])
                    except:
                        trace_log(_logSysLevel, _logInfo,
                                  {'Filename': fileName, 'word': word},
                                  context='Building Vector List - Word not in Dictionary')
                        dict_parse_words(dbstores, swicfg, word, xcheck=True)
                        vectorList.append(dbstores['dict'][word])

            dbstores['docmeta'][srcID]['wordcount'] = wordCount
            dbstores['vectorized'][srcID] = vectorList
            dbstores['docmeta'][srcID]['vector'] = True

            dbstores['docmeta'].sync()
            dbstores['vectorized'].sync()

            trace_log(_logSysLevel, _logTrace,
                      dbstores['docmeta'][srcID]['wordcount'], context='Producted wordCount')
            trace_log(_logSysLevel, _logTrace,
                      dbstores['vectorized'][srcID], context='Producted vectorList')
            trace_log(_logSysLevel, _logInfo,
                      'Filename : '+fileName, context='Finished WordCount & Vectorize Lists')

    return


def ngram_srcdoc(dbstores, swicfg):
    # Process given file as raw text line by line

    # for each srcID, if not indexed/parsed - then extract ngrams
    for srcID in dbstores['docmeta']:
        trace_log(_logSysLevel, _logTrace, {'SrcID':srcID}, context='Parsing')

        # extract ngrams from normalised files
        if not (not dbstores['docmeta'][srcID]['normalised'] or dbstores['docmeta'][srcID]['indexed']):
            fileName = os.path.join(swicfg['srcdata'], srcID, 'normal.dat')
            trace_log(_logSysLevel, _logTrace, {'Filename':fileName}, context='Index Starting')

            if os.path.isfile(fileName):
                # Generate Vectorization of ngrams and strip stop words
                vectorizer = CountVectorizer(ngram_range=(1, swicfg['ngram']))
                ngramAnalyzer = vectorizer.build_analyzer()

                # with open(fileName, mode='rU', errors='ignore') as readFile:
                #     # read each line, process ngrams & check for vector dictionary
                #     lineID = 0
                #     for line in readFile:
                #         trace_log(_logSysLevel, _logTrace, {'LineID': lineID, 'Text': line},
                #                    context='Index Processing')
                #         src_line_ngram_storage(dbstores, srcID, lineID, ngramAnalyzer(line))
                #         dict_parse_words(dbstores, swicfg, line.split(), xcheck=True)
                #         lineID += 1

                # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
                # Enables tqdm to control progress bar on screen location
                with std_out_err_redirect_tqdm() as orig_stdout:
                    # tqdm needs the original stdout
                    # and dynamic_ncols=True to autodetect console width
                    with open(fileName, mode='rt', errors='ignore') as readFile:
                        readFileInMem = readFile.read()

                    # Build a list of line end (\n) locations before replacing them
                    # the index of the match is the line number/ LineID
                    trace_log(_logSysLevel, _logInfo, 'ngram - '+str(fileName)+' building line index from file...')
                    # readFile.seek(0)
                    # lineEndIndex = sorted([match.start() for match in re.finditer(r'\n', readFile.read())])
                    lineEndIndex = sorted([match.start() for match in re.finditer(r'\n', readFileInMem)])

                    # Grab every ngram in file and record of it's existence in srcID
                    # Will also create & record ngram's across line breaks
                    trace_log(_logSysLevel, _logInfo,  'ngram - '+str(fileName)+' builing list from file...')
                    # readFile.seek(0)
                    # ngramList = sorted(ngramAnalyzer(readFile.read().replace('\n', ' ')))
                    readFileInMem.replace('\n', ' ')
                    ngramList = sorted(list(set(ngramAnalyzer(readFileInMem))))

                    docmeta_src_ngrams_add(dbstores, srcID, ngramList)
                    trace_log(_logSysLevel, _logInfo,
                              {'Filename': fileName, 'nGramCount': len(ngramList), 'ngramSample': ngramList[-50:]},
                              context= 'ngram - '+str(fileName)+' builing list from file...finished')

                    trace_log(_logSysLevel, _logInfo,  'ngram - '+str(fileName)+' parsing ngram list for Line & Dictionary...')
                    srcIDngramCount = dict()
                    srcIDngramLineList = dict()
                    for ngram in tqdm(ngramList, file=orig_stdout, dynamic_ncols=True):

                        # if ngram is a single word, ensure it is in the dictionary
                        if len(ngram.split(' ')) == 1:
                            dict_parse_words(dbstores, swicfg, ngram, xcheck=False)

                        # Capture the starting index of every ngram, even if it crosses a line break
                        # readFile.seek(0)
                        # ngramIndex = [match.start() for match in re.finditer(re.escape(ngram), readFile.read().replace('\n', ' '))]
                        ngramIndex = [match.start() for match in re.finditer(re.escape(ngram), readFileInMem)]

                        # del readFileInMem
                        
                        # For every position, find the last indexed line end, so we are on the next line (+1)
                        # note; set returns results sorted when purely numeric values
                        # lineList = [bisect_left(lineEndIndex, index) + 1 for index in ngramIndex]
                        # docstat_src_ngram_lines(dbstores, srcID, ngram, list(set(lineList)))
                        srcIDngramLineList[ngram] = list(set([bisect_left(lineEndIndex, index) + 1 for index in ngramIndex]))

                        # ngram_store_add(dbstores, ngram, srcID, count=len(lineList))
                        srcIDngramCount[ngram] = len(srcIDngramLineList[ngram])


                    trace_log(_logSysLevel, _logInfo,  'ngram - '+str(fileName)+' parsing ngram list...finished')

                    ngram_full_src_update(dbstores, srcID, srcIDngramCount)
                    docstat_full_src_update(dbstores, srcID, srcIDngramLineList)

                # dbstores['docmeta'][srcID]['indexed'] = True
                dbstores['docmeta'].sync()
                dbstores['docstat'].sync()
                dbstores['ngram'].sync()
                trace_log(_logSysLevel, _logInfo, 'ngram - '+str(fileName)+' Indexing Finished')
            else:
                trace_log(_logSysLevel, _logError, 'ngram - '+str(fileName)+' Indexing File Missing')

    return


def ngram_full_src_update(dbstores, srcID, srcIDngramCount):
    # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    # Enables tqdm to control progress bar on screen location
    with std_out_err_redirect_tqdm() as orig_stdout:
    # tqdm needs the original stdout
    # and dynamic_ncols=True to autodetect console width
        for ngram in tqdm(srcIDngramCount, desc='Updating '+srcID, file=orig_stdout, dynamic_ncols=True):
            dbstores['ngram'][ngram][srcID] = srcIDngramCount[ngram]
        dbstores['ngram'].sync()
    return


def docstat_full_src_update(dbstores, srcID, srcIDngramLineList):
    dbstores['docstat'][srcID] = srcIDngramLineList
    dbstores['docstat'].sync()
    return


def src_line_ngram_storage(dbstores, srcID, lineID, lineNgrams):
    # For each word/ngram add to master dictionary with FileID & In FileDict

    for item in lineNgrams:
        #first Record Ngram is in File, then record which lines have the Ngram
        ngram_store_add(dbstores, item, srcID)
        src_ngram_add(dbstores, item, lineID, srcID)

    return


def ngram_store_add(dbstores, ngram, srcID, count=None):
    # Record/Add Source with Ngram usage, if count = None, then increment, else count = new total for srcID

    if ngram not in dbstores['ngram']:
        # initialize item if not already in the master dictionary
        dbstores['ngram'][ngram] = dict()
        trace_log(_logSysLevel, _logTrace, 'ngram - Created: '+ngram)

    if srcID not in dbstores['ngram'][ngram]:
        # SrcID isn't recorded as a viable match, then initialize
        dbstores['ngram'][ngram][srcID] = 0
        trace_log(_logSysLevel, _logTrace, 'ngram - New srcID: ' + ngram + ' / ' + srcID)

    if isinstance(count, int):
        # Absolute/Override call made
        dbstores['ngram'][ngram][srcID] = count
        trace_log(_logSysLevel, _logInfo, 'ngram - srcID: ' + ngram + ' / ' + srcID + ' Count Override ' + str(count))
    else:
        # Incremental Call made, then increment
        dbstores['ngram'][ngram][srcID] += 1
        trace_log(_logSysLevel, _logInfo, 'ngram - srcID Increment: ' + ngram + ' / ' + srcID)

    dbstores['ngram'].sync()

    return


def docmeta_src_ngrams_add(dbstores, srcID, ngramList):
    # Record Sources ngram list in DocMeta, and what lines/paragraphs inc ngram

    # Add ngram's existence into Meta Storage
    if srcID not in dbstores['docmeta']:
        dbstores['docmeta'][srcID] = dict()

    dbstores['docmeta'][srcID]['ngrams'] = ngramList

    trace_log(_logSysLevel, _logInfo, dbstores['docmeta'][srcID]['ngrams'][-50:],
              context='DocMeta updated ' + srcID)

    dbstores['docmeta'].sync()

    return


def docstat_src_ngram_lines(dbstores, srcID, ngram, lineList):
    # Record Sources ngram list in DocMeta, and what lines/paragraphs inc ngram

    # Add ngram's existence into Meta Storage
    # Add/initialize ngram and line(s) info Source Statistics
    if srcID not in dbstores['docstat']:
        dbstores['docstat'][srcID] = dict()

    dbstores['docstat'][srcID][ngram] = lineList

    trace_log(_logSysLevel, _logInfo, dbstores['docstat'][srcID][ngram][-50:],
              context='DocStat ' + srcID + ' / ' + ngram)

    dbstores['docstat'].sync()

    return


def src_ngram_add(dbstores, ngram, lineID, srcID):
    # Record Source includes ngram, and what lines/paragraphs inc ngram

    # Add ngram's existence into Meta Storage
    if srcID not in dbstores['docmeta']:
        dbstores['docmeta'][srcID] = dict()

    if 'ngrams' not in dbstores['docmeta'][srcID]:
        dbstores['docmeta'][srcID]['ngrams'] = list()

    if ngram not in dbstores['docmeta'][srcID]['ngrams']:
        dbstores['docmeta'][srcID]['ngrams'].append(ngram)
        trace_log(_logSysLevel, _logInfo, 'DocStat ' + srcID + ' / ' + ngram)

    dbstores['docmeta'].sync()

    # Add/initialize ngram and line(s) info Source Statistics
    if srcID not in dbstores['docstat']:
        dbstores['docstat'][srcID] = dict()

    if ngram not in dbstores['docstat'][srcID]:
        # if ngram hasn't been initialized
        dbstores['docstat'][srcID][ngram] = list()

    if lineID not in dbstores['docstat'][srcID][ngram]:
        # if line isn't recorded as a viable match, then add to list
        dbstores['docstat'][srcID][ngram].append(lineID)
        trace_log(_logSysLevel, _logInfo, dbstores['docstat'][srcID][ngram][-10:], context='DocStat Added to (' + srcID + '): ' + ngram)

    dbstores['docstat'].sync()

    return
