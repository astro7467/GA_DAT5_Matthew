#!/usr/bin/env python2
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
import itertools
import math
import os
import re
import random
import time
import shelve
import tempfile
import uuid

import numpy as np
import tensorflow as tf

#from pathlib import Path
#from simple_shelve import SimpleShelf, SimpleMultiShelf

from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer

_storeDocMeta_ngrams = True # Storage of ngram list in dbStores['docmeta'] per srcID

# Current Suggested Global Variables for use with traceLog
_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace
#_logSysLevel = _logStatus

# Tensorflow Word2Vec Model Training
def tf_word2vec(dbStores, sysConfig, wordCount, vectorList):
    dataIndex = 0
    batch, labels, dataIndex = w2v_generate_batch(vectorList, dataIndex, batchSize=8, numSkips=2, skipWindow=1)
    for i in range(8):
        print(batch[i], dbStores['vectors']['vectors'][batch[i]],
              '->', labels[i, 0], dbStores['vectors']['vectors'][labels[i, 0]])

    # Build and train a skip-gram model.

    batchSize = 128
    embeddingSize = 128  # Dimension of the embedding vector.
    skipWindow = 1       # How many words to consider left and right.
    numSkips = 2         # How many times to reuse an input to generate a label.
    numSampled = 64      # Number of negative examples to sample.
    vocabularySize = len(dbStores['vectors']['vectors'].keys())

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    validSize = 16     # Random set of words to evaluate similarity on.
    validWindow = 100  # Only pick dev samples in the head of the distribution.
    validExamples = np.random.choice(validWindow, validSize, replace=False)


    graph = tf.Graph()
    with graph.as_default():

        # Input data.
        trainInputs = tf.placeholder(tf.int32, shape=[batchSize])
        trainLabels = tf.placeholder(tf.int32, shape=[batchSize, 1])
        validDataset = tf.constant(validExamples, dtype=tf.int32)

        # Ops and variables pinned to the GPU
        # change to CPU if not on tensorflow-gpu with CDDN & CUDA support
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(tf.random_uniform([vocabularySize, embeddingSize], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, trainInputs)

            # Construct the variables for the NCE loss
            nceWeights = tf.Variable(tf.truncated_normal([vocabularySize, embeddingSize], stddev=1.0 / math.sqrt(embeddingSize)))
            nceBiases = tf.Variable(tf.zeros([vocabularySize]))

            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            loss = tf.reduce_mean( tf.nn.nce_loss(weights=nceWeights, biases=nceBiases, labels=trainLabels, inputs=embed, num_sampled=numSampled, num_classes=vocabularySize))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalizedEmbeddings = embeddings / norm
            validEmbeddings = tf.nn.embedding_lookup( normalizedEmbeddings, validDataset )
            similarity = tf.matmul( validEmbeddings, normalizedEmbeddings, transpose_b=True )

            # Add variable initializer.
            init = tf.global_variables_initializer()
        #htiw
    #htiw

    # builder = tf.saved_model.builder.SavedModelBuilder('./data/tfSavedModelBuilder')
    # with tf.Session(graph=tf.Graph()) as session:
    #     builder.add_meta_graph_and_variables(session, [tf.saved_model.tag_constants.TRAINING] )
    # builder.save()

    # Begin training.
    numSteps = 100001

    with tf.Session(graph=graph) as session:
    #with tf.Session(graph=tf.Graph()) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        averageLoss = 0
        for step in xrange(numSteps):
            batchInputs, batchLabels, dataIndex = w2v_generate_batch( vectorList, dataIndex, batchSize, numSkips, skipWindow)
            feedDict = {trainInputs: batchInputs, trainLabels: batchLabels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, lossVal = session.run([optimizer, loss], feed_dict=feedDict)
            averageLoss += lossVal

            if step % 2000 == 0:
                if step > 0:  averageLoss /= 2000

                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', averageLoss)
                averageLoss = 0
            #if

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(validSize):
                    validWord = dbStores['vectors']['vectors'][validExamples[i]]
                    topK = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:topK + 1]
                    logStr = 'Nearest to %s:' % validWord
                    for k in xrange(topK):
                        closeWord = dbStores['vectors']['vectors'][nearest[k]]
                        logStr = '%s %s,' % (logStr, closeWord)
                    #rof
                    print(logStr)
                #rof
            #fi
        #rof
        finalEmbeddings = normalizedEmbeddings.eval()
    #htiw

    return
#def

# Function to generate a training batch for the skip-gram model.
def w2v_generate_batch(vectorList, dataIndex, batchSize, numSkips, skipWindow):
    assert batchSize % numSkips == 0
    assert numSkips <= 2 * skipWindow
    batch = np.ndarray(shape=(batchSize), dtype=np.int32)
    labels = np.ndarray(shape=(batchSize, 1), dtype=np.int32)
    span = 2 * skipWindow + 1  # [ skipWindow target skipWindow ]
    buffer = collections.deque(maxlen=span)
    if dataIndex + span > len(vectorList):
        dataIndex = 0
    buffer.extend(vectorList[dataIndex:dataIndex + span])
    dataIndex += span
    for i in range(batchSize // numSkips):
        contextWords = [w for w in range(span) if w != skipWindow]
        random.shuffle(contextWords)
        wordsToUse = collections.deque(contextWords)
        for j in range(numSkips):
            batch[i * numSkips + j] = buffer[skipWindow]
            contextWord = wordsToUse.pop()
            labels[i * numSkips + j, 0] = buffer[contextWord]
        if dataIndex == len(vectorList):
            buffer[:] = vectorList[:span]
            dataIndex = span
        else:
            buffer.append(vectorList[dataIndex])
            dataIndex += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    dataIndex = (dataIndex + len(vectorList) - span) % len(vectorList)
    return batch, labels, dataIndex
#def

# Take a file and produces a temporary normalised file
def normalise_file(dbStores, sysConfig, fileName, tempDir):
    tempFile = "data/textstore.dat"
    tempFile = os.path.join(tempDir, os.path.basename(fileName))

    trace_log( _logSysLevel, _logTrace, {'tempDir':tempDir, 'tempFile':tempFile, 'Filename':fileName}, context='Normalised File Start')

    writeFile = open(tempFile, 'w+t')
    try:
        normalisedText = normalise_text(fileName)
        if not normalisedText in [None, '', ' ']:
            writeFile.write(normalisedText + ' ')
        #fi

        with open( fileName, mode = 'rU' ) as readFile:
            # read each line, normalise it, and send to temp file
            for line in readFile:
                normalisedText = normalise_text(line)
                if not normalisedText in [None, '', ' ']:
                    writeFile.write(normalisedText + ' ')
                #fi
            #rof
        #htiw
    finally:
        writeFile.close()
    #yrt

    trace_log( _logSysLevel, _logTrace, {'tempDir':tempDir, 'tempFile':tempFile, 'Filename':fileName}, context='Normalised File Finished')

    return tempFile
#fed

# Count occurances of words & Convert (normalised) file from words to list of vectors
def vector_word_count_file(dbStores, sysConfig, normFile):

    trace_log( _logSysLevel, _logInfo, {'Filename':normFile}, context='Starting WordCount & Vectorize List...')

    with open( normFile, mode = 'rU' ) as readFile:
        wordCount = list()
        vectorList = list()
        counter = collections.Counter()

        # read each line, count each word & append to vector list as vectors
        for line in readFile:
            counter.update(line.split())
            for word in line.split():
                try:
                    vectorList.append(dbStores['dict'][word])
                except:
                    trace_log( _logSysLevel, _logInfo, {'Filename':normFile, 'word':word}, context='Building Vector List - Word not in Dictionary')
                    dict_parse_words(dbStores, sysConfig, line, xcheck=True)
                    vectorList.append(dbStores['dict'][word])
                #yrt
            #rof
        #rof
    #htiw

    #wordCount.extend(counter)
    wordCount.extend(map(list, counter.items()))

    trace_log( _logSysLevel, _logTrace, wordCount[:50], context='Producted wordCount')
    trace_log( _logSysLevel, _logTrace, vectorList[:100], context='Producted vectorList')
    trace_log( _logSysLevel, _logInfo, {'Filename':normFile}, context='Finished WordCount & Vectorize Lists')

    return wordCount, vectorList
#def

# Takes a given document, and returns a list with the document vectorized,
# and counts for each word
def vector_file(dbStores, sysConfig, fileList, srcCat, srcSubCat):
    tempDir = tempfile.mkdtemp()

    for fileName in fileList:
        trace_log( _logSysLevel, _logTrace, {'Filename':fileName}, context='Vector File Examining')

        # Build a individual File Breakdown ngrams
        srcID = uuid_source(dbStores, sysConfig, srcCat + ':' + srcSubCat + ':' + fileName)
        init_source(dbStores, srcID, fileName, srcCat, srcSubCat)

        if not dbStores['docmeta'][srcID].has_key('word2vec'):
            dbStores['docmeta'][srcID]['word2vec'] = False
            dbStores['docmeta'].sync()
        #fi

        if not dbStores['docmeta'][srcID]['word2vec']:
            normFile = normalise_file(dbStores, sysConfig, fileName, tempDir)
            wordCount, vectorList = vector_word_count_file(dbStores, sysConfig, normFile)
            #tf_word2vec(dbStores, sysConfig, wordCount, vectorList)
            #dbStores['docmeta'][srcID]['word2vec'] = True
            trace_log( _logSysLevel, _logInfo, {'SrcID':srcID, 'Filename':fileName, 'Normalised Filename':normFile}, context='Vector File Finished')
        #fi
    #rof
    #os.removedirs(tempDir)
    return
#fed

# increment vector available and return
def dictionary_vector(dbStores):
    dbStores['config']['nextvector'] += 1
    dbStores['config'].sync()
    return dbStores['config']['nextvector']
#fed

# Parse a given word list, add to dictionary & vectors if new
def dict_parse_words(dbStores, sysConfig, words, xcheck=False):
    # reduce to unique words
    wordList = set(words)

    #if not dbStores['vectdict'].has_key('dict'): dbStores['vectdict']['dict'] = dict()
    #if not dbStores['vectdict'].has_key('vect'): dbStores['vectdict']['vect'] = dict()

    # find words not already in dictionary, and add them to dict & vectors
    for word in wordList:
        if not dbStores['dict'].has_key(word):
            vector = dictionary_vector(dbStores)

            dbStores['dict'][word] = vector
            dbStores['dict'].sync()

            dbStores['vectors']['vectors'][vector] = word
            dbStores['vectors'].sync()

            trace_log( _logSysLevel, _logTrace, {word:dbStores['dict'][word], vector:dbStores['vectors']['vectors'][vector]}, context='Dictionary New Word: Vector')
        #fi

        if xcheck:
            if not dbStores['vectors']['vectors'].has_key(dbStores['dict'][word]):
                # Missing Vector -> Word
                vector = dbStores['dict'][word]
                dbStores['vectors']['vectors'][vector] = word

                dbStores['dict'].sync()
                dbStores['vectors'].sync()

                trace_log( _logSysLevel, _logStatus, {word: vector}, context='Dictionary Fixed - Missing Vector Key')
            #fi

            if not isinstance(dbStores['dict'][word], int):
                # found an invalid vector word/vector pair???
                vector = dictionary_vector(dbStores)
                dbStores['dict'][word] = vector
                dbStores['vectors']['vectors'][vector] = word

                dbStores['dict'].sync()
                dbStores['vectors'].sync()

                trace_log( _logSysLevel, _logStatus, {word: vector}, context='Dictionary Fixed - Bad Vector')
            #fi

            if not dbStores['vectors']['vectors'][dbStores['dict'][word]] == word:
                # found an invalid word->vector->word???
                oldword = dbStores['vectors']['vectors'][dbStores['dict'][word]]

                # fix current word & vector maps
                currVector = dbStores['dict'][word]
                dbStores['vectors']['vectors'][currVector] = word

                # assign new vector to oldword
                newVector = dictionary_vector(dbStores)
                dbStores['dict'][oldword] = newVector
                dbStores['vectors']['vectors'][newVector] = oldword

                dbStores['dict'].sync()
                dbStores['vectors'].sync()

                trace_log( _logSysLevel, _logStatus, {word: dbStores['dict'][word], oldword: newVector}, context='Dictionary Fixed - Bad X Vector')
            #fi

            if not normalise_text(word) == word:
                trace_log( _logSysLevel, _logStatus, {word: dbStores['dict'][word], dbStores['dict'][word]: dbStores['vectors']['vectors'][dbStores['dict'][word]]}, context='Dictionary Warning - Word Failed Normalised Scan')
            #fi
        #fi
    #rof
    dbStores['dict'].sync()
    dbStores['vectors'].sync()
#def

# Step through Dictionary and validate Vector Mappings
def validate_dict():
    print line
    trace_log( _logSysLevel, _logStatus, 'Validating Dictionary...')
    dbStores = open_datastores()
    sysConfig = sys_config(dbStores)
    count = 0
    total = len(dbStores['dict'].keys())

    trace_log( _logSysLevel, _logStatus, 'Checking Vector...')
    minVector = max(dbStores['vectors']['vectors'].keys())
    vector = dictionary_vector(dbStores)

    if vector <= minVector:
        oldVector = vector
        dbStores['config']['nextvector'] = minVector + 1
        dbStores['config'].sync()
        vector = dictionary_vector(dbStores)
        trace_log( _logSysLevel, _logStatus, {'OldVector': oldVector, 'NewVector': vector, 'MinVector': minVector}, context='Bad Next Vector Found')
    #fi

    for word in list(dbStores['dict'].keys()):
        dict_parse_words(dbStores, sysConfig, [word], xcheck=True)
        if count % 100 == 0:
            trace_log( _logSysLevel, _logStatus, 'Progress: ' + str(count).rjust(len(str(total))+1) + ' of ' + str(total) + ' Last Vector: ' + str(dbStores['config']['nextvector']) + ' - ' + word)
        #fi
        count += 1
    #rof
    trace_log( _logSysLevel, _logStatus, 'Number of keys Dict: ' + str(len(dbStores['dict'].keys())) )
    trace_log( _logSysLevel, _logStatus, 'Number of keys Vect: ' + str(len(dbStores['vectors']['vectors'].keys())) )
    print line
    close_datastores(dbStores)
#def

# Common Logging/Trace Routine
def trace_log(sysLogLevel, logType, logData, context=''):
    logTypes = ['ERROR', 'STATUS', 'CONFIG', 'INFO', 'TRACE']

    if logType < 0 : logType = 0
    if logType > len(logTypes) -1 : logType = len(logTypes) -1

    # if (logType < 0 or logType > len(logTypes) -1):
    #     trace_log( sysLogLevel, 0, 'Following traceLog call changed to ERROR, made with invalid LogType: ' + str(logType))
    #     logType = 0
    # #fi

    if logType <= sysLogLevel:
        logText = time.strftime("%Y-%m-%d %H:%M:%S UTC%Z  ") + logTypes[logType].ljust(8)
        newLinePrefix = ' ' * len(logText)
        if context <> '': logText += str(context) + '; '

        # if context <> '': newLinePrefix += ' ' + context

        logTextList = []
        maxLineLen = 120

        if ( isinstance(logData, str) or isinstance(logData, int) or isinstance(logData, float) ):
            logTextList = [logText + str(logData).strip()[:120]]
        elif ( isinstance(logData, list) or isinstance(logData, tuple)):
            firstItem = True
            for item in logData:
                if not firstItem:
                    logText += ', '
                    if len(logText) > maxLineLen:
                        logTextList += [str(logText)]
                        logText = newLinePrefix
                    #fi
                #fi
                logText += str(item).strip()[:120]
                firstItem = False
            #rof
            logTextList += [str(logText)]
        elif isinstance(logData, dict):
            firstItem = True
            for key in logData.keys():
                if not firstItem:
                    logText = logText + ', '
                    if len(logText) > maxLineLen:
                        logTextList += [str(logText)]
                        logText = newLinePrefix
                    #fi
                #fi
                logText += str(key).strip() + ': ' + str(logData[key]).strip()[:120]
                firstItem = False
                if len(logTextList) > 10: break
            #rof
            logTextList += [str(logText)]
        #fi

        for line in logTextList: print line
    #fi
#def

#find N best matches of given Ngrams vs Ngram Index
def calc_matches(dbStores, ngramList, maxResults=12):
    srcNgramCounts = {}     # Store Raw per SrcID, by ngram counts
    ngramCounts = {}        # Per ngram how many unique (srcID) counts
    ngramWeights = {}       # Weight of each ngram in calcing srcID's value
    srcWeightedScore = {}   # Weight of each ngram in calcing srcID's value
    ngramWeightBase = 0.0   # Count we will base weight on (i.e. total ngram matches)
    srcNgramScore = {}      # Weighted score of match for each srcID

    # Parse ngramList for unseen ngrams and remove
    unseenNgrams = []
    for ngram in ngramList[:]:
        if not dbStores['ngram'].has_key(ngram):
            ngramList.remove(ngram)
            unseenNgrams += [ngram]
        #fi
    #rof

    # take verified ngram list and build counts
    for ngram in ngramList:
        # cycle thru srcID with ngram, adding to srcNgramCounts & ngramCounts
        for srcID in dbStores['ngram'][ngram].keys():
            if not srcNgramCounts.has_key(srcID):
                # Create and add src's count of ngram
                srcNgramCounts[srcID] = { ngram:dbStores['ngram'][ngram][srcID] }
            else:
                # increment ngram count for src (in theory, this should never happen)
                if not srcNgramCounts[srcID].has_key(ngram):
                    # Create and add src's ngram count of ngram
                    srcNgramCounts[srcID][ngram] = dbStores['ngram'][ngram][srcID]
                else:
                    # increment ngram count for src (in theory, this should never happen)
                    srcNgramCounts[srcID][ngram] += dbStores['ngram'][ngram][srcID]
                #fi
            #fi

            # Record total ngram appearances so we can use as weight later
            if not ngramCounts.has_key(ngram):
                # Create and add src's count of ngram
                ngramCounts[ngram] = dbStores['ngram'][ngram][srcID]
            else:
                # Increment Total instances of ngram's occurance
                ngramCounts[ngram] += dbStores['ngram'][ngram][srcID]
            #fi
            ngramWeightBase += dbStores['ngram'][ngram][srcID]
        #rof
    #rof

    # Assign a weight to each based on 'uniquiness'
    normaliseBaseMax = 1
    normaliseBaseMin = 0
    for ngram, ngramCount in sorted(ngramCounts.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        try:
            # More common ngrams get lower scores, longer ngrams get higher score
            ngramWeight = math.log(float(ngramWeightBase), float(ngramCount))
        except:
            ngramWeight = 1.0
        #yrt
        ngramWeight = ngramWeight * (float(ngramWeightBase) - float(ngramCount)) * float(len(ngram.split()))
        ngramWeights[ngram] = ngramWeight
        normaliseBaseMax = max(ngramWeight, normaliseBaseMax)
        normaliseBaseMin = min(ngramWeight, normaliseBaseMin)
    #rof

    # Normalise ngramWeights
    for ngram in ngramWeights.keys():
        ngramWeights[ngram] /= (normaliseBaseMax - normaliseBaseMin)
    #rof

    trace_log( _logSysLevel, _logTrace, normaliseBaseMax, context='normaliseBaseMax')
    trace_log( _logSysLevel, _logTrace, normaliseBaseMin, context='normaliseBaseMin')
    trace_log( _logSysLevel, _logTrace, ngramWeights, context='ngramWeights')

    # Score each srcID/doc based on ngram Weights
    normaliseBaseMax = 1
    normaliseBaseMin = 0
    for srcID in srcNgramCounts:
        for ngram in srcNgramCounts[srcID]:
            WeightedScore = ngramWeights[ngram] * srcNgramCounts[srcID][ngram] * len(ngram.split())
            if not srcWeightedScore.has_key(srcID):
                srcWeightedScore[srcID] = WeightedScore
            else:
                srcWeightedScore[srcID] += WeightedScore
            #fi
        #rof
        normaliseBaseMax = max(srcWeightedScore[srcID], normaliseBaseMax)
        normaliseBaseMin = min(srcWeightedScore[srcID], normaliseBaseMin)
    #rof

    # Normalise srcWeightedScore
    for srcID in srcWeightedScore.keys():
        srcWeightedScore[srcID] /= (normaliseBaseMax - normaliseBaseMin)
    #rof

    trace_log( _logSysLevel, _logTrace, normaliseBaseMax, context='normaliseBaseMax')
    trace_log( _logSysLevel, _logTrace, normaliseBaseMin, context='normaliseBaseMin')
    trace_log( _logSysLevel, _logInfo, srcWeightedScore, context='srcWeightedScore')

    topSrcMatches = []
    topSrcInfo = {}
    for srcID, WeightedScore in sorted(srcWeightedScore.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        topSrcMatches.append(srcID)
        topSrcInfo[srcID] = {'score': WeightedScore, 'ngrams': len(srcNgramCounts[srcID])}
    #rof

    trace_log( _logSysLevel, _logInfo, topSrcMatches, context='topSrcMatches')

    return topSrcMatches[:maxResults], topSrcInfo, unseenNgrams
#fed

# Normalize Text Convert text to lower-case and strip punctuation/symbols from words
def normalise_text(text):
    norm_text = str(text).lower()

    # remove apostrophe in words: [alpha]'[alpha]=[alpha][alpha] eg. don't = dont
    norm_text = re.sub(r'([\w]+)[\`\']([\w]+)', r'\1\2', norm_text)

    # Replace non-AlphaNumeric sequences with Space
    norm_text = re.sub(r'[^\w]+', r' ', norm_text)

    # Replace spaces, underscores, tabs, newlines and return sequences with a space
    norm_text = re.sub(r'[ _\t\n\r]+', r' ', norm_text)

    # Replace pure digits with space eg 1234, but not 4u or best4u
    norm_text = re.sub(r'^\d+$|^\d+\W+|\W+\d+\W+|\W+\d+$', r' ', norm_text)

    return norm_text.strip(' _\t\n\r')
#fed

### Open Datastores and return handles
# TODO: Serperate W/R opens from RO opens
def open_datastores(setWriteback=True):

    dbStores = dict()

    trace_log( _logSysLevel, _logInfo, 'Started Opening dbStores...')

    # Config Datastore
    dbStores['config'] = shelve.open('data/config', writeback=setWriteback)

    # Dictionary to Vectore Datastore
    dbStores['dict'] = shelve.open('data/dictionary', writeback=setWriteback)

    # Document Meta Data Datastore
    dbStores['docmeta'] = shelve.open('data/docmeta', writeback=setWriteback)

    # Document Stats (file ngrams) Datastore
    dbStores['docstat'] = shelve.open('data/docstat', writeback=setWriteback)

    # Ngram Datastore (Core index)
    dbStores['ngram'] = shelve.open('data/ngram', writeback=setWriteback)

    #file Type/Path to UUID Datastore
    dbStores['sources'] = shelve.open('data/sources', writeback=setWriteback)

    # (Original) Vector to Dictionary Datastore
    # dbStores['vectdict'] = shelve.open('data/vectdict', writeback=setWriteback)

    # Vector to Dictionary Datastore
    dbStores['vectors'] = shelve.open('data/vectors', writeback=setWriteback)

    # As shelves mandates str for keys, we create a dict inside with int keys for vectors
    # open()/close()/sync() need to occur against 'vectors' not ['vectors]['vectors']
    if not dbStores['vectors'].has_key('vectors'):
        dbStores['vectors']['vectors'] = dict()
    #fi

    trace_log( _logSysLevel, _logInfo, 'Finished Opening dbStores')

    return dbStores
#fed

# Close/Cleanup datastores
def close_datastores(dbStores):
    for fileHandle in dbStores.keys():
        dbStores[fileHandle].sync()
        dbStores[fileHandle].close()
        trace_log( _logSysLevel, _logInfo, 'Closed dbStore: ' + fileHandle)
    #rof
    return
#fed

# UUID Management
def sys_config(dbStores):
    sysConfig = dict()

    try:
        sysConfig['uuid'] = dbStores['config']['uuid']
    except:
        dbStores['config']['uuid'] = str(uuid.uuid1())
        sysConfig['uuid'] = dbStores['config']['uuid']
    #yrt

    try:
        sysConfig['corpus'] = dbStores['config']['corpus']
    except:
        dbStores['config']['corpus'] = './corpus'
        sysConfig['corpus'] = dbStores['config']['corpus']
    #yrt

    try:
        sysConfig['ngram'] = dbStores['config']['ngram']
    except:
        dbStores['config']['ngram'] = 3
        sysConfig['ngram'] = dbStores['config']['ngram']
    #yrt

    try:
        sysConfig['word2vec'] = dbStores['config']['word2vec']
    except:
        dbStores['config']['word2vec'] = 5
        sysConfig['word2vec'] = dbStores['config']['word2vec']
    #yrt

    # try:
    #     sysConfig['nextvector'] = dbStores['config']['nextvector']
    # except:
    #     dbStores['config']['nextvector'] = 0
    #     sysConfig['nextvector'] = dbStores['config']['nextvector']
    #yrt

    try:
        sysConfig['version'] = dbStores['config']['version']
    except:
        dbStores['config']['version'] = 0.1
        sysConfig['version'] = dbStores['config']['version']
    #yrt

    #sysConfig['cfgstorage'] = dbStores['config']
    dbStores['config'].sync()

    trace_log( _logSysLevel, _logInfo, 'Read sysConfig in from dbStore')

    return sysConfig
#fed

# Match or Create a UUID for data sources
def uuid_source(dbStores, sysConfig, srcPath):
    try:
        srcUUID = dbStores['sources'][srcPath]
    except:
        dbStores['sources'][srcPath] = str(uuid.uuid5(uuid.UUID(sysConfig['uuid']), srcPath))
        srcUUID = dbStores['sources'][srcPath]
    #yrt
    dbStores['sources'].sync()
    return str(srcUUID)
#fed

# Initialize Src Records (i.e. Delete for now)
def init_source(dbStores, srcID, srcPath, srcCat='UNK', srcSubCat='UNK'):
    srcID = str(srcID)
    # Create srcID for new sources Meta Data Record, and set minimum values
    if not dbStores['docmeta'].has_key(srcID):
        dbStores['docmeta'][srcID] = {
            'version'   :0.1,
            'path'      :str(srcPath),
            'cat'       :str(srcCat),
            'subcat'    :str(srcSubCat),
            'indexed'   :False,
            'verdate'   :'UNK',
            'lastidx'   :'UNK',
            'qltyscore' :0,
            'indexscore':0,
            'xrefscore' :0,
            'ngrams'    :[] }
        dbStores['docmeta'].sync()
    #fi

    # Flush Anagram Line/paragrph data
    if not dbStores['docstat'].has_key(srcID):
        dbStores['docstat'][srcID] = {}
        dbStores['docstat'].sync()
    #fi
    return
#fed

# Record/Add Source with Ngram usage
def ngram_store_add(dbStores, ngram, srcID):
    ngram = str(ngram)
    srcID = str(srcID)

    if not dbStores['ngram'].has_key(ngram):
        # initialize item if not already in the master dictionary
        dbStores['ngram'][ngram] = {srcID:1}
        trace_log( _logSysLevel, _logTrace, dbStores['ngram'][ngram], context='Created ngram: ' + ngram)
    elif srcID in dbStores['ngram'][ngram]:
        # Count finds of ngram in srcID
        dbStores['ngram'][ngram][srcID] += 1
        trace_log( _logSysLevel, _logTrace, dbStores['ngram'][ngram], context='Increased (' + ngram + '): ' + srcID)
    else:
        #file isn't recorded as a viable match, then add to list
        dbStores['ngram'][ngram][srcID] = 1
        trace_log( _logSysLevel, _logTrace, dbStores['ngram'][ngram], context='Added (' + ngram + '): ' + srcID)
    #fi
    return
#fed

# Record Source includes ngram, and what lines/paragraphs inc ngram
def src_ngram_add(dbStores, ngram, lineID, srcID):
    ngram = str(ngram)
    srcID = str(srcID)
    #lineID = str(lineID)

    if _storeDocMeta_ngrams:
        # Add ngram's existence into Meta Storage
        if dbStores['docmeta'][srcID]['ngrams'] == []:
            dbStores['docmeta'][srcID]['ngrams'] = [ngram]
            trace_log( _logSysLevel, _logTrace, dbStores['docmeta'][srcID]['ngrams'][-10:], context='DocMeta Created (' + srcID + '):')
        elif ngram not in dbStores['docmeta'][srcID]['ngrams']:
            dbStores['docmeta'][srcID]['ngrams'].append(ngram)
            trace_log( _logSysLevel, _logTrace, dbStores['docmeta'][srcID]['ngrams'][-10:], context='DocMeta Added (' + srcID + '):')
        #fi
    #fi

    # Add/initialize ngram and line(s) info Source Statistics
    if not dbStores['docstat'].has_key(srcID):
        dbStores['docstat'][srcID] = { ngram :[lineID] }
        trace_log( _logSysLevel, _logInfo, dbStores['docstat'][srcID][ngram][-10:], context='DocStat Created ' + srcID + ' with ngram ' + ngram)
    elif ngram not in dbStores['docstat'][srcID]:
        # if ngram hasn't been initialized
        dbStores['docstat'][srcID][ngram] = [lineID]
        trace_log( _logSysLevel, _logInfo, dbStores['docstat'][srcID][ngram][-10:], context='DocStat Created ' + srcID + ' / ' + ngram)
    elif lineID not in dbStores['docstat'][srcID][ngram]:
        # if line isn't recorded as a viable match, then add to list
        dbStores['docstat'][srcID][ngram].append(lineID)
        trace_log( _logSysLevel, _logInfo, dbStores['docstat'][srcID][ngram][-10:], context='DocStat Added to (' + srcID + '): ' + ngram)
    #fi
    return
#fed

def tuples2text(tuple):
    text = ''
    for item in tuple:
        if not item in [None, '', ' ',]:
            text += " " + item
        #fi
    #rof
    return re.sub('[ \t\n\r]+', ' ', text).strip(' \t\n\r')
#fed

# Build ngram list from given word list
# Used for search string - assume users write sentences or keywords
# cover all scenarios with comprehensive ngram
def build_ngrams(inputList, width=2):
    outputList = []
    for length in range(1, width + 1):
        for tuple in itertools.permutations(inputList, length):
            outputList.append(tuples2text(tuple))
        #rof
    #rof
    return sorted([item for item in outputList if item != ''])
#fed

# For each word/ngram add to master dictionary with FileID & In FileDict
def src_line_ngram_storage(dbStores, srcID, lineID, lineNgrams):
    for item in lineNgrams:
        #first Record Ngram is in File, then record which lines have the Ngram
        ngram_store_add(dbStores, item, srcID)
        src_ngram_add(dbStores, item, lineID, srcID)
    #rof
#fed

# Process given file as raw text line by line
def index_file_txt(dbStores, sysConfig, fileList, srcCat, srcSubCat):
    # Generate Vectorization of ngrams and strip stop words
    vectorizer = CountVectorizer(ngram_range=(1, sysConfig['ngram']), stop_words='english')
    ngramAnalyzer = vectorizer.build_analyzer()

    # for each file, get a UID and parse
    for fileName in fileList:
        trace_log( _logSysLevel, _logTrace, {'Filename':fileName, 'SrcCat':srcCat, 'SrcSubCat': srcSubCat}, context='Examining')

        # Build a individual File Breakdown ngrams
        srcID = uuid_source(dbStores, sysConfig, srcCat + ':' + srcSubCat + ':' + fileName)
        init_source(dbStores, srcID, fileName, srcCat, srcSubCat)

        if not dbStores['docmeta'][srcID].has_key('indexed'):
            dbStores['docmeta'][srcID]['indexed'] = False
        #fi

        if not dbStores['docmeta'][srcID]['indexed']:
            lineID = 0
            with open( fileName, mode = 'rU' ) as currFile:
                normalisedText = normalise_text(fileName)
                if not normalisedText in [None, '', ' ']:
                    trace_log( _logSysLevel, _logInfo, {'SrcID':srcID, 'Filename':fileName, 'SrcCat':srcCat, 'SrcSubCat': srcSubCat, 'LineID': lineID, 'NormalisedText':normalisedText}, context='Initializing')
                    dict_parse_words(dbStores, sysConfig, normalisedText.split())
                    src_line_ngram_storage(dbStores, srcID, lineID, ngramAnalyzer(normalisedText))
                #fi

                # for each line get a UID and parse line
                for lineID, line in enumerate(currFile, 1):
                    normalisedText = normalise_text(line)
                    if not normalisedText in [None, '', ' ']:
                        trace_log( _logSysLevel, _logTrace, {'SrcID':srcID, 'Filename':fileName, 'SrcCat':srcCat, 'SrcSubCat': srcSubCat, 'LineID': lineID, 'NormalisedText':normalisedText}, context='Processing')
                        dict_parse_words(dbStores, sysConfig, normalisedText.split())
                        src_line_ngram_storage(dbStores, srcID, lineID, ngramAnalyzer(normalisedText))
                    #fi
                #rof
            #htiw
            dbStores['docmeta'][srcID]['indexed'] = True
            dbStores['ngram'].sync()
            dbStores['docmeta'].sync()
            dbStores['docstat'].sync()
            trace_log( _logSysLevel, _logInfo, {'SrcID':srcID, 'Filename':fileName, 'SrcCat':srcCat, 'SrcSubCat': srcSubCat, 'Lines': lineID}, context='Finished')
        #fi
    #rof
#def
