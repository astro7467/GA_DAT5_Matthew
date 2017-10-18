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

import itertools
import math
import re
import time
import shelve
import uuid

from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer

_trace = True # enable increased data reporting
_storeDocMeta_ngrams = True # Storage of ngram list in dbStores['docmeta'] per srcID

# Current Suggested Global Variables for use with traceLog
_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace
#_logSysLevel = _logStatus

# Common Logging/Trace Routine
def trace_log(sysLogLevel, logType, logData):
    logTypes = ['ERROR', 'STATUS', 'CONFIG', 'INFO', 'TRACE']

    if logType < 0 : logType = 0
    if logType > len(logTypes) -1 : logType = len(logTypes) -1

    # if (logType < 0 or logType > len(logTypes) -1):
    #     trace_log( sysLogLevel, 0, 'Following traceLog call changed to ERROR, made with invalid LogType: ' + str(logType))
    #     logType = 0
    # #fi

    if logType <= sysLogLevel:
        logText = time.strftime("%Y-%m-%d %H:%M:%S UTC%Z %a") + ' ' + logTypes[logType].ljust(8)
        newLinePrefix = ' ' * len(logText)
        logTextList = []
        maxLineLen = 120
        if ( isinstance(logData, str) or isinstance(logData, int) or isinstance(logData, float) ):
            logTextList = [logText + str(logData).strip()]
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
                logText += str(item).strip()
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
                logText += str(key).strip() + ': ' + str(logData[key]).strip()
                firstItem = False
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
        trace_log( _logSysLevel, _logTrace, ['ngramWeightBase, dbStores[ngram][ngram]:', ngramWeightBase, ngram, dbStores['ngram'][ngram]])
    #rof

    # Assign a weight to each based on 'uniquiness'
    for ngram, ngramCount in sorted(ngramCounts.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        try:
            # More common ngrams get lower scores, longer ngrams get higher score
            ngramWeight = math.log(float(ngramWeightBase), float(ngramCount))
        except:
            ngramWeight = 1.0
        #yrt
        ngramWeight = ngramWeight * (float(ngramWeightBase) - float(ngramCount)) * float(len(ngram.split()))
        ngramWeights[ngram] = ngramWeight
        trace_log( _logSysLevel, _logTrace, {'Routine': 'Weight', 'ngram': ngram, 'ngramWeight': ngramWeight, 'ngramCount': ngramCount, 'ngramWeightBase': ngramWeightBase})
    #rof

    for srcID in srcNgramCounts:
        for ngram in srcNgramCounts[srcID]:
            WeightedScore = ngramWeights[ngram] * srcNgramCounts[srcID][ngram] * len(ngram.split())
            if not srcWeightedScore.has_key(srcID):
                srcWeightedScore[srcID] = WeightedScore
            else:
                srcWeightedScore[srcID] += WeightedScore
            #fi
        #rof
        trace_log( _logSysLevel, _logInfo, ['srcWeightedScore[srcID]', srcID, srcWeightedScore[srcID]])
    #rof

    trace_log( _logSysLevel, _logInfo, ['srcWeightedScore', srcWeightedScore])

    topSrcMatches = []
    topSrcInfo = {}
    for srcID, WeightedScore in sorted(srcWeightedScore.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        topSrcMatches.append(srcID)
        topSrcInfo[srcID] = {'score': WeightedScore,
                             'ngrams': len(srcNgramCounts[srcID])}
        trace_log( _logSysLevel, _logTrace, ['topSrcMatches, WeightedScore, srcID', WeightedScore, srcID])
    #rof

    trace_log( _logSysLevel, _logInfo, ['topSrcMatches'] + topSrcMatches)

    return topSrcMatches[:maxResults], topSrcInfo, unseenNgrams
#fed

# Normalize Text Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # remove apostrophe in words: [alpha]'[alpha]=[alpha][alpha] eg. don't = dont
    norm_text = re.sub(r'([\w]+)\'([\w]+)', r'\1\2', norm_text)

    # Replace non-AlphaNumeric sequences with Space
    norm_text = re.sub(r'[^\w]+', r' ', norm_text)

    # Replace spaces, underscores, tabs, newlines and returns with a space
    return re.sub(r'[ _\t\n\r]+', r' ', norm_text).strip(' _\t\n\r')
#fed

### Open Datastores and return handles
# TODO: Serperate W/R opens from RO opens
def open_datastores():
    dbStores = {}
    # Config Datastore
    dbStores['config'] = shelve.open('data/config.db', writeback=True)

    # Ngram Datastore (Core index)
    dbStores['ngram'] = shelve.open('data/ngram.db', writeback=True)

    # Document Meta Data Datastore
    dbStores['docmeta'] = shelve.open('data/docmeta.db', writeback=True)

    # Document Stats (file ngrams) Datastore
    dbStores['docstat'] = shelve.open('data/docstat.db', writeback=True)

    #file Type/Path to UUID Datastore
    dbStores['sources'] = shelve.open('data/sources.db', writeback=True)

    return dbStores
#fed

# UUID Management
def sys_config(dbStores):
    sysConfig = {}

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
        sysConfig['version'] = dbStores['config']['version']
    except:
        dbStores['config']['version'] = 0.1
        sysConfig['version'] = dbStores['config']['version']
    #yrt

    dbStores['config'].sync()

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
        trace_log( _logSysLevel, _logInfo, ['dbStores[ngram][ngram]:', srcID, ngram, dbStores['ngram'][ngram]])
    elif srcID in dbStores['ngram'][ngram]:
        # Count finds of ngram in srcID
        dbStores['ngram'][ngram][srcID] += 1
        trace_log( _logSysLevel, _logInfo, ['Changing dbStores[ngram][ngram]:', srcID, ngram, dbStores['ngram'][ngram]])
    else:
        #file isn't recorded as a viable match, then add to list
        dbStores['ngram'][ngram][srcID] = 1
        trace_log( _logSysLevel, _logInfo, ['Adding dbStores[ngram][ngram]:', srcID, ngram, dbStores['ngram'][ngram]])
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
            trace_log( _logSysLevel, _logInfo, ['Creating dbStores[docmeta][srcID][ngrams]:', srcID, ngram, dbStores['docmeta'][srcID]['ngrams'][-10:]])
        elif ngram not in dbStores['docmeta'][srcID]['ngrams']:
            dbStores['docmeta'][srcID]['ngrams'].append(ngram)
            trace_log( _logSysLevel, _logInfo, ['Adding dbStores[docmeta][srcID][ngrams]:', srcID, ngram, dbStores['docmeta'][srcID]['ngrams'][-10:]])
        #fi
    #fi

    # Add/initialize ngram and line(s) info Source Statistics
    if not dbStores['docstat'].has_key(srcID):
        dbStores['docstat'][srcID] = { ngram :[lineID] }
        trace_log( _logSysLevel, _logInfo, ['Creating dbStores[docstat][srcID][ngram]:', srcID, ngram, dbStores['docstat'][srcID][ngram][-10:]])
    elif ngram not in dbStores['docstat'][srcID]:
        # if ngram hasn't been initialized
        dbStores['docstat'][srcID][ngram] = [lineID]
        trace_log( _logSysLevel, _logInfo, ['Creating dbStores[docstat][srcID][ngram]:', srcID, ngram, dbStores['docstat'][srcID][ngram][-10:]])
    elif lineID not in dbStores['docstat'][srcID][ngram]:
        # if line isn't recorded as a viable match, then add to list
        dbStores['docstat'][srcID][ngram].append(lineID)
        trace_log( _logSysLevel, _logInfo, ['Adding dbStores[docstat][srcID][ngram]:', srcID, ngram, dbStores['docstat'][srcID][ngram][-10:]])
    #fi
    return
#fed

# Close/Cleanup datastores
def close_datastores(dbStores):
    for fileHandle in dbStores.keys():
        dbStores[fileHandle].sync()
        dbStores[fileHandle].close()
    #rof
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
def build_ngrams(inputList, n=2):
    outputList = []
    for L in range(0, n + 1):
        for tuple in itertools.combinations(inputList, L):
            outputList.append(tuples2text(tuple))
        #rof
    #rof
    return [item for item in outputList if item != '']
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
        # Build a individual File Breakdown dictionary
        srcID = uuid_source(dbStores, sysConfig, '' + srcCat + ':' + srcSubCat + ':' + fileName + '')
        init_source(dbStores, srcID, fileName, srcCat, srcSubCat)

        trace_log( _logSysLevel, _logTrace, ['SrcID, Filename, SrcCat, SrcSubCat:', srcID, fileName, srcCat, srcSubCat])

        if not dbStores['docmeta'][srcID]['indexed']:
            lineID = 0
            with open( fileName, mode = 'rU' ) as currFile:
                trace_log( _logSysLevel, _logTrace, ['LineID, fileName:', lineID, normalize_text(fileName)])
                src_line_ngram_storage(dbStores, srcID, lineID, ngramAnalyzer(normalize_text(fileName)))

                # for each line get a UID and parse line
                for lineID, line in enumerate(currFile, 1):
                    normalizedText = normalize_text(line)
                    if not normalizedText in [None, '', ' ']:
                        trace_log( _logSysLevel, _logTrace, ['LineID, normalizedText:', lineID, normalizedText])
                        src_line_ngram_storage(dbStores, srcID, lineID, ngramAnalyzer(normalizedText))
                    #fi
                #rof
            #htiw
            dbStores['docmeta'][srcID]['indexed'] = True
            dbStores['ngram'].sync()
            dbStores['docmeta'].sync()
            dbStores['docstat'].sync()
        #fi
    #rof
#def
