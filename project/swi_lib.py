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
import shelve
import uuid

_trace = True # enable increased data reporting
_docMetaStore_ngrams = True # Storage of ngram list in docMetaStore per srcID

#find N best matches of given Ngrams vs Ngram Index
def calc_matches(ngramStore, ngramList, n=12):
    srcNgramCounts = {} # Store Raw per SrcID, by ngram counts
    ngramCounts = {}    # Per ngram how many unique (srcID) counts
    ngramWeights = {}   # Weight of each ngram in calcing srcID's value
    srcWeightedScore = {}   # Weight of each ngram in calcing srcID's value
    ngramWeightBase = 0.0 # Count we will base weight on (i.e. total ngram matches)
    srcNgramScore = {}  # Weighted score of match for each srcID

    # Parse ngramList for unseen ngrams and remove
    for ngram in ngramList[:]:
        if not ngramStore.has_key(ngram):
            ngramList.remove(ngram)
        #fi
    #rof

    # take verified ngram list and build counts
    for ngram in ngramList:
        # cycle thru srcID with ngram, adding to srcNgramCounts & ngramCounts
        for srcID in ngramStore[ngram].keys():
            if not srcNgramCounts.has_key(srcID):
                # Create and add src's count of ngram
                srcNgramCounts[srcID] = { ngram:ngramStore[ngram][srcID] }
            else:
                # increment ngram count for src (in theory, this should never happen)
                if not srcNgramCounts[srcID].has_key(ngram):
                    # Create and add src's ngram count of ngram
                    srcNgramCounts[srcID][ngram] = ngramStore[ngram][srcID]
                else:
                    # increment ngram count for src (in theory, this should never happen)
                    srcNgramCounts[srcID][ngram] += ngramStore[ngram][srcID]
                #fi
            #fi

            # Record total ngram appearances so we can use as weight later
            if not ngramCounts.has_key(ngram):
                # Create and add src's count of ngram
                ngramCounts[ngram] = ngramStore[ngram][srcID]
            else:
                # Increment Total instances of ngram's occurance
                ngramCounts[ngram] += ngramStore[ngram][srcID]
            #fi
            ngramWeightBase += ngramStore[ngram][srcID]
        #rof
        if _trace: print('UPDATE: ngramWeightBase, ngramStore[ngram]:', ngramWeightBase, ngram, ngramStore[ngram])
    #rof

    # Assign a weight to each based on 'uniquiness'
    for ngram, ngramCount in sorted(ngramCounts.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        ngramWeight = math.log(ngramWeightBase, ngramCount) * len(ngram.split())
        ngramWeights[ngram] = ngramWeight
        if _trace: print "INFO: ngramWeight, count, ngram", ngramWeight, ngramCount, ngram
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
        if _trace: print "INFO: srcWeightedScore[srcID]", srcID, srcWeightedScore[srcID]
    #rof

    if _trace: print "INFO: srcWeightedScore", srcWeightedScore

    topSrcMatches = []
    for srcID, WeightedScore in sorted(srcWeightedScore.iteritems(), reverse=True, key=lambda (k,v): (v,k)):
        topSrcMatches.append(srcID)
        if _trace: print "INFO: topSrcMatches, WeightedScore, srcID", WeightedScore, srcID
    #rof

    if _trace: print "INFO: topSrcMatches", topSrcMatches

    return topSrcMatches
#fed

# Normalize Text Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace html breaks with newline
    #norm_text = re.sub(r'<br *\/*>', '\n', norm_text)
    # Replace non-AlphaNumeric with Space
    norm_text = re.sub(r'[^\w]+', ' ', norm_text)
    return re.sub('[ \t\n\r]+', ' ', norm_text).strip(' \t\n\r')
#fed

### Open Datastores and return handles
# TODO: Serperate W/R opens from RO opens
def open_datastores():
    # Config Datastore
    rootConfig = shelve.open('data/config.db', writeback=True)

    # Ngram Datastore (Core index)
    rootNgram = shelve.open('data/ngram.db', writeback=True)

    # Document Meta Data Datastore
    rootDocMeta = shelve.open('data/docmeta.db', writeback=True)

    # Document Stats (file ngrams) Datastore
    rootDocStat = shelve.open('data/docstat.db', writeback=True)

    #file Type/Path to UUID Datastore
    rootSources = shelve.open('data/sources.db', writeback=True)

    return rootConfig, rootNgram, rootDocMeta, rootDocStat, rootSources
#fed

# UUID Management
def sys_config(configStore):

    try:
        uuidInstance = configStore['uuid']
    except:
        configStore['uuid'] = str(uuid.uuid1())
        uuidInstance = configStore['uuid']
    #yrt

    try:
        corpusDir = configStore['corpus']
    except:
        configStore['corpus'] = './corpus'
        corpusDir = configStore['corpus']
    #yrt

    try:
        ngramWidth = configStore['ngram']
    except:
        configStore['ngram'] = 3
        ngramWidth = configStore['ngram']
    #yrt

    try:
        StoreDataVersion = configStore['version']
    except:
        configStore['version'] = 0.1
        StoreDataVersion = configStore['version']
    #yrt

    configStore.sync()

    return uuidInstance, corpusDir, ngramWidth, StoreDataVersion
#fed

# Match or Create a UUID for data sources
def uuid_source(rootSources, srcPath, Instance):
    try:
        srcUUID = rootSources[srcPath]
    except:
        rootSources[srcPath] = str(uuid.uuid5(uuid.UUID(Instance), srcPath))
        srcUUID = rootSources[srcPath]
    #yrt
    rootSources.sync()
    return str(srcUUID)
#fed

# Initialize Src Records (i.e. Delete for now)
def init_source(docMetaStore, docStat, srcID, srcPath, srcCat = 'UNK', srcSubCat = 'UNK'):
    srcID = str(srcID)
    # Create srcID for new sources Meta Data Record, and set minimum values
    if not docMetaStore.has_key(srcID):
        #del docMetaStore[srcID]
        docMetaStore[srcID] = {
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
        docMetaStore.sync()
    #fi

    # Flush Anagram Line/paragrph data
    if not docStat.has_key(srcID):
        #del docStat[srcID]
        docStat[srcID] = {}
        docStat.sync()
    #fi
    return
#fed

# Record/Add Source with Ngram usage
def ngram_store_add(ngramStore, ngram, srcID):
    ngram = str(ngram)
    srcID = str(srcID)

    if not ngramStore.has_key(ngram):
        # initialize item if not already in the master dictionary
        ngramStore[ngram] = {srcID:1}
        if _trace: print('CREATE: ngramStore[ngram]:', srcID, ngram, ngramStore[ngram])
    elif srcID in ngramStore[ngram]:
        # Count finds of ngram in srcID
        ngramStore[ngram][srcID] += 1
        if _trace: print('CHANGE: ngramStore[ngram]:', srcID, ngram, ngramStore[ngram])
    else:
        #file isn't recorded as a viable match, then add to list
        ngramStore[ngram][srcID] = 1
        if _trace: print('ADD: ngramStore[ngram]:', srcID, ngram, ngramStore[ngram])
    #fi
    return
#fed

# Record Source includes ngram, and what lines/paragraphs inc ngram
def src_ngram_add(docMetaStore, docStat, ngram, lineID, srcID):
    ngram = str(ngram)
    srcID = str(srcID)
    #lineID = str(lineID)

    if _docMetaStore_ngrams:
        # Add ngram's existence into Meta Storage
        if docMetaStore[srcID]['ngrams'] == []:
            docMetaStore[srcID]['ngrams'] = [ngram]
            if _trace: print('CREATE: docMetaStore[srcID][ngrams]:', srcID, ngram, docMetaStore[srcID]['ngrams'][-10:])
        elif ngram not in docMetaStore[srcID]['ngrams']:
            docMetaStore[srcID]['ngrams'].append(ngram)
            if _trace: print('ADD: docMetaStore[srcID][ngrams]:', srcID, ngram, docMetaStore[srcID]['ngrams'][-10:])
        #fi
    #fi

    # Add/initialize ngram and line(s) info Source Statistics
    if not docStat.has_key(srcID):
        docStat[srcID] = { ngram :[lineID] }
        if _trace: print('CREATE: docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
    elif ngram not in docStat[srcID]:
        # if ngram hasn't been initialized
        docStat[srcID][ngram] = [lineID]
        if _trace: print('CREATE: docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
    elif lineID not in docStat[srcID][ngram]:
        # if line isn't recorded as a viable match, then add to list
        docStat[srcID][ngram].append(lineID)
        if _trace: print('ADD: docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
    #fi
    return
#fed

# Close/Cleanup datastores
def close_datastores(configStore, ngramStore, docMetaStore, docStat):
    configStore.sync()
    configStore.close()

    ngramStore.sync()
    ngramStore.close()

    docMetaStore.sync()
    docMetaStore.close()

    docStat.sync()
    docStat.close()
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
