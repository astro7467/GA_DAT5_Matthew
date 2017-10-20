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
import uuid

import numpy as np
import tensorflow as tf

#from pathlib import Path
#from simple_shelve import SimpleShelf, SimpleMultiShelf

from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer
from tempfile import gettempdir

_storeDocMeta_ngrams = True # Storage of ngram list in dbStores['docmeta'] per srcID

# Current Suggested Global Variables for use with traceLog
_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace
#_logSysLevel = _logStatus

# increment vector available and return
def dictionary_vector(dbStores):
    dbStores['config']['nextvector'] += 1
    dbStores['config'].sync()
    return dbStores['config']['nextvector']
#fed

# Parse a given word list, add to dictionary & vectors if new
def dict_parse_words(dbStores, sysConfig, words, xcheck=False):
    # reduce to unique words
    words = set(words)

    #if not dbStores['vectdict'].has_key('dict'): dbStores['vectdict']['dict'] = dict()
    #if not dbStores['vectdict'].has_key('vect'): dbStores['vectdict']['vect'] = dict()

    # find words not already in dictionary, and add them to dict & vectors
    for word in words:
        if not dbStores['dict'].has_key(word):
            vector = dictionary_vector(dbStores)
            dbStores['dict'][word] = vector
            dbStores['vector'][vector] = word

            dbStores['dict'].sync()
            dbStores['vectors'].sync()

            trace_log( _logSysLevel, _logTrace, {word: vector}, context='Added to Dictionary')
        #fi
        if xcheck:
            if not isinstance(dbStores['dict'][word], int):
                # found an invalid vector word/vector pair???
                vector = dictionary_vector(dbStores)
                dbStores['dict'][word] = vector
                dbStores['vector'][vector] = word

                dbStores['dict'].sync()
                dbStores['vectors'].sync()

                trace_log( _logSysLevel, _logStatus, {word: vector}, context='Fixed Dictionary - Bad Vector')
            #fi

            if not dbStores['vector'].has_key(dbStores['dict'][word]):
                # Missing Vector -> Word
                dbStores['vector'][dbStores['dict'][word]] = word

                dbStores['dict'].sync()
                dbStores['vectors'].sync()

                trace_log( _logSysLevel, _logStatus, {word: vector}, context='Fixed Dictionary - Missing Vector Key')
            #fi

            if not dbStores['vector'][dbStores['dict'][word]] == word:
                # found an invalid word->vector->word???
                oldword = dbStores['vector'][dbStores['dict'][word]]

                # fix current word & vector maps
                dbStores['vector'][dbStores['dict'][word]] = word

                # assign new vector to oldword
                vector = dictionary_vector(dbStores)
                dbStores['dict'][oldword] = vector
                dbStores['vector'][vector] = oldword

                dbStores['dict'].sync()
                dbStores['vectors'].sync()

                trace_log( _logSysLevel, _logStatus, {word: dbStores['dict'][word], oldword: vector}, context='Fixed Dictionary - Bad X Vector')
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
    minVector = max(dbStores['vector'].keys())
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
    trace_log( _logSysLevel, _logStatus, 'Number of keys Vect: ' + str(len(dbStores['vector'].keys())) )
    print line
    close_datastores(dbStores)
#def

def w2v_build_dataset(words):
    #Process raw inputs into a dataset ready for word2vec
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count: dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary
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
        topSrcInfo[srcID] = {'score': WeightedScore,
                             'ngrams': len(srcNgramCounts[srcID])}
    #rof

    trace_log( _logSysLevel, _logInfo, topSrcMatches, context='topSrcMatches')

    return topSrcMatches[:maxResults], topSrcInfo, unseenNgrams
#fed

# Normalize Text Convert text to lower-case and strip punctuation/symbols from words
def normalise_text(text):
    norm_text = text.lower()

    # remove apostrophe in words: [alpha]'[alpha]=[alpha][alpha] eg. don't = dont
    norm_text = re.sub(r'([\w]+)[\`\']([\w]+)', r'\1\2', norm_text)

    # Replace non-AlphaNumeric sequences with Space
    norm_text = re.sub(r'[^\w]+', r' ', norm_text)

    # Replace spaces, underscores, tabs, newlines and returns with a space
    norm_text = re.sub(r'[ _\t\n\r]+', r' ', norm_text)

    # Replace pure digits with space eg 1234, but not 4u or best4u
    norm_text = re.sub(r'^\d+\W+|\W+\d+\W+|\W+\d+$', r' ', norm_text)

    return norm_text.strip(' _\t\n\r')
#fed

### Open Datastores and return handles
# TODO: Serperate W/R opens from RO opens
def open_datastores():

    dbStores = dict()

    trace_log( _logSysLevel, _logInfo, 'Started Opening dbStores...')

    # Config Datastore
    dbStores['config'] = shelve.open('data/config', writeback=True)

    # Dictionary to Vectore Datastore
    dbStores['dict'] = shelve.open('data/dictionary', writeback=True)

    # Document Meta Data Datastore
    dbStores['docmeta'] = shelve.open('data/docmeta', writeback=True)

    # Document Stats (file ngrams) Datastore
    dbStores['docstat'] = shelve.open('data/docstat', writeback=True)

    # Ngram Datastore (Core index)
    dbStores['ngram'] = shelve.open('data/ngram', writeback=True)

    #file Type/Path to UUID Datastore
    dbStores['sources'] = shelve.open('data/sources', writeback=True)

    # (Original) Vector to Dictionary Datastore
    # dbStores['vectdict'] = shelve.open('data/vectdict', writeback=True)

    # Vector to Dictionary Datastore
    dbStores['vectors'] = shelve.open('data/vectors', writeback=True)

    # As shelves mandates str for keys, we create a dict inside with int keys for vectors
    # open()/close()/sync() need to occur against 'vectors' not vector
    if not dbStores['vectors'].has_key('vectors'): dbStores['vectors']['vectors'] = dict()
    dbStores['vector'] = dbStores['vectors']['vectors']

    trace_log( _logSysLevel, _logInfo, 'Finished Opening dbStores')

    return dbStores
#fed

# Close/Cleanup datastores
def close_datastores(dbStores):
    for fileHandle in dbStores.keys():
        if not fileHandle == 'vector':
            dbStores[fileHandle].sync()
            dbStores[fileHandle].close()
            trace_log( _logSysLevel, _logInfo, 'Closed dbStore: ' + fileHandle)
        #fi
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
