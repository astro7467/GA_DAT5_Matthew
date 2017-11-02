#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    SearchWithInference - Proof of concept for PerSea (Personal Search)
    Copyright (C) 2017  Matthew A. Snell (Astro7467@XLTech.io)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This software is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import collections
import getopt
import random
import os
import shelve
import sys

import swi_lib as swi

from tqdm import tqdm

# import numpy as np
# import tensorflow as tf


#     This system only has a simple minimum DB design
#     *** will not scale - proof of concept only ***
#     for full multi doc indexing, would need to breakdown datastores
#     to faciliate concurrency, and different document types and meta data needs.
#
#
# TODOS / Full Product Considerations:
# - Leverage Classes for all DB/Data Model Interations
# - Seperate W/R DB activity (eg index) from RO DB activity (search)
# - Search History (Learning/ngram considerations)
# - Explore ngram vectorizations
# - Split each ngram width into separate tables - better manage ngram changes
# - Consider 1st Pass index (Fast - monogram) vs background index (bigram -> Ngram)
# - line/cell/tag/paragraph weights or tags eg. <h1> > <h2>, 'word': 'definition' weighswicfg['vrate D']B backend from code (i.e. shelve/cpickle vs sqlite/mySQL/Marina/PostgresSQL)

_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace

# Min data spec version expected & used
dataSpecVer = 0.1


def word2vec():
    # Run Word2Vec on existing data (Vectorized)
    # Extract Data for inference searching

    swi.trace_log(_logSysLevel, _logStatus, 'Initialising...', context='Word2Vec')
    dbstores = swi.open_datastores()
    swicfg = swi.sys_config(dbstores)
    swi.chk_coredb_keys(dbstores, swicfg)

    wordCounts = collections.Counter()

    swi.trace_log(_logSysLevel, _logStatus,
                  'Checking/creating temporary storage...', context='Word2Vec')

    w2vdbstore = shelve.open(str(swicfg['w2vdata']),
                             flag='c', protocol=4, writeback=True)

    swi.trace_log(_logSysLevel, _logTrace,
                  'Creating empty Counts...', context='Word2Vec')
    w2vdbstore['counts'] = list()
    w2vdbstore.sync()

    swi.trace_log(_logSysLevel, _logTrace,
                  'Creating empty Dictionary...', context='Word2Vec')
    w2vdbstore['dict'] = dict()
    w2vdbstore.sync()

    swi.trace_log(_logSysLevel, _logTrace,
                  'Creating empty Reverse Dictionary...', context='Word2Vec')
    w2vdbstore['revdict'] = dict()
    w2vdbstore.sync()

    swi.trace_log(_logSysLevel, _logTrace,
                  'Creating empty Vector List...', context='Word2Vec')
    w2vdbstore['vectors'] = list()
    w2vdbstore.sync()

    swi.trace_log(_logSysLevel, _logTrace,
                  'Creating empty Vector Set List...', context='Word2Vec')
    w2vdbstore['vectorset'] = set()
    w2vdbstore.sync()

    swi.trace_log(_logSysLevel, _logInfo,
                  'Building Consolidated Vector List...', context='Word2Vec')

    for srcID in tqdm(dbstores['docmeta'], dynamic_ncols=True, position=0):
        swi.trace_log(_logSysLevel, _logTrace,
                      srcID, context='Word2Vec - Checking srcID')

        if srcID in dbstores['vectorized']:
            if len(dbstores['vectorized'][srcID]) > 0 \
                    and len(dbstores['docmeta'][srcID]['wordcount']) > 0:

                swi.trace_log(_logSysLevel, _logTrace,
                              srcID, context='Word2Vec - Adding srcID')
                w2vdbstore['vectors'].extend(dbstores['vectorized'][srcID])

                swi.trace_log(_logSysLevel, _logTrace,
                              srcID, context='Word2Vec - Updating Vector Set')
                w2vdbstore['vectorset'].update(dbstores['vectorized'][srcID])

                w2vdbstore.sync()
                swi.trace_log(_logSysLevel, _logTrace,
                              w2vdbstore['vectors'], context='Word2Vec - Extended vectors')

                wordCounts += dbstores['docmeta'][srcID]['wordcount']
                swi.trace_log(_logSysLevel, _logTrace,
                              wordCounts, context='Word2Vec - Updated wordCounts')

    swi.trace_log(_logSysLevel, _logInfo,
                  w2vdbstore['vectors'], context='Word2Vec - Consolidated Vectors')

    swi.trace_log(_logSysLevel, _logInfo,
                  'Converting Word Counts to Sorted List by Count...', context='Word2Vec')

    # convert Counter() dict to [key, value] list in list
    for word, wordcount in [(word, wordCounts[word])
                            for word
                            in sorted(wordCounts, key=wordCounts.get, reverse=True)]:
        w2vdbstore['counts'].append([word, wordcount])

    swi.trace_log(_logSysLevel, _logInfo,
                  w2vdbstore['counts'], context='Word2Vec - Consolidated Counts')

    swi.trace_log(_logSysLevel, _logInfo,
                  'Building Vector Set Dictionary & Reverse Dictionary...', context='Word2Vec')
    for word, vector in [(word, dbstores['dict'][word])
                         for word in dbstores['dict']
                         if dbstores['dict'][word] in w2vdbstore['vectorset'] ]:
        w2vdbstore['dict'][word] = vector
        w2vdbstore['revdict'][vector] = word

    swi.trace_log(_logSysLevel, _logInfo,
                  w2vdbstore['revdict'], context='Word2Vec - Reverse Dictionary')
    swi.trace_log(_logSysLevel, _logInfo,
                  'RevDict '+str(len(w2vdbstore['revdict']))+' / '+str(len(set(w2vdbstore['vectors'])))+' Unique Vectors', context='Word2Vec - RevDict vs Unique Vectors')
    swi.trace_log(_logSysLevel, _logInfo,
                  'Syncing and saving datastore...', context='Word2Vec')

    w2vdbstore.sync()
    w2vdbstore.close()

    swi.tf_word2vec(dbstores, swicfg)

    return


def system_analyse():
    # Provide some insight to the datastores

    dbstores = swi.open_datastores()
    swicfg = swi.sys_config(dbstores)
    swi.chk_coredb_keys(dbstores, swicfg)

    line = '-' * 80

    tqdm.write(line)
    tqdm.write('Config File Info')
    tqdm.write(line)
    tqdm.write('Config Record Count:', len(dbstores['config'].keys()))
    swi.trace_log(_logConfig, _logStatus, swicfg, 'Config Key & Value Pairs')

    dictVector = max(dbstores['dict'].values())
    vecVector = max(dbstores['vectors']['vectors'].keys())
    vector = swi.dictionary_vector(dbstores)
    tqdm.write('Vector Trace: Highest Dictionary / Highest Vectors / Current Counter', dictVector, '/', vecVector, '/', vector)

    tqdm.write(line)
    tqdm.write('DocMeta File Info')
    tqdm.write(line)
    tqdm.write('DocMeta Record Count:', len(dbstores['docmeta'].keys()))
    tqdm.write('Ten Random Samples:')
    keyList = list(dbstores['docmeta'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbstores['docmeta'][key], 'DocMeta ' + key)

    tqdm.write(line)
    tqdm.write('DocStat File Info')
    tqdm.write(line)
    tqdm.write('DocStat Record Count:', len(dbstores['docstat'].keys()))
    tqdm.write('Ten Random Samples:')
    keyList = list(dbstores['docstat'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbstores['docstat'][key], 'DocStat ' + key)

    tqdm.write(line)
    tqdm.write('Ngram File Info')
    tqdm.write(line)
    tqdm.write('Ngram Record Count:', len(dbstores['ngram'].keys()))
    tqdm.write('Ten Random Samples:')
    keyList = list(dbstores['ngram'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbstores['ngram'][key], 'Ngram ' + key)

    tqdm.write(line)
    tqdm.write('Sources File Info')
    tqdm.write(line)
    tqdm.write('Sources Record Count:', len(dbstores['sources'].keys()))
    tqdm.write('Ten Random Samples:')
    keyList = list(dbstores['sources'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbstores['sources'][key], 'Source ' + key)


    tqdm.write(line)
    tqdm.write('Dictionary & Vector File Info')
    tqdm.write(line)
    tqdm.write('Dictionary & Vector Key Counts (Should be Equal):', len(dbstores['dict'].keys()), '/', len(dbstores['vectors']['vectors'].keys()))
    tqdm.write('Dictionary & Vectors (inc Validation) (word->vector->word):')
    keyList = list(dbstores['dict'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        tqdm.write(key, '(word) ->', dbstores['dict'][key], '(vector) ->', dbstores['vectors']['vectors'][dbstores['dict'][key]], '(word)')
        swi.dict_parse_words(dbstores, swicfg, [key], xcheck=True)

    tqdm.write(line)
    tqdm.write('Vector & Dictionary (inc Validation) (vector->word->vector):')
    tqdm.write(line)
    keyList = list(dbstores['vectors']['vectors'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        tqdm.write(key, '(vector) ->', dbstores['vectors']['vectors'][key], '(word) ->', dbstores['dict'][dbstores['vectors']['vectors'][key]], '(vector)')
        swi.dict_parse_words(dbstores, swicfg, [dbstores['vectors']['vectors'][key]], xcheck=True)

    tqdm.write(line)
    tqdm.write('Vectorized Datastore:')
    tqdm.write(line)
    keyList = list(dbstores['vectorized'].keys())
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbstores['vectorized'][key], 'Source ' + key)

    tqdm.write(line)

    swi.close_datastores(dbstores)
    return


def search(searchText, similarity=True, openclose=True, dbstores=None, swicfg=None):
    # Search for string in index

    if openclose:
        dbstores = swi.open_datastores()
        swicfg = swi.sys_config(dbstores)
        swi.chk_coredb_keys(dbstores, swicfg)
        swi.trace_log(_logSysLevel, _logConfig,
                      swicfg, context='Search - swicfg')

    searchText = swi.normalise_text(searchText)
    words = sorted(list(set(searchText.split())))

    # Parse the words and add to a dictionary for future use
    swi.dict_parse_words(dbstores, swicfg, words)

    ngramList = swi.build_ngrams(words, swicfg['ngram'])

    swi.trace_log(_logSysLevel, _logInfo, words, context='Search - Search Words')
    swi.trace_log(_logSysLevel, _logInfo, ngramList, context='Search - ngramList')

    topSrcMatches, topSrcInfo, unseenNgrams = swi.calc_matches(dbstores, ngramList, 10)

    tqdm.write('-'*80)
    tqdm.write('topSrcMatches:')
    for index in range(0, len(topSrcMatches)):
        srcID = topSrcMatches[index]
        tqdm.write(' '.join([str(index+1).rjust(4),
                             ':',
                             dbstores['docmeta'][srcID]['cat'],
                             dbstores['docmeta'][srcID]['subcat'],
                             str(srcID),
                             str(topSrcInfo[srcID]['score']).ljust(22),
                             str(topSrcInfo[srcID]['ngrams']).rjust(4),
                             dbstores['docmeta'][srcID]['path']] ))
    tqdm.write('-'*80)

    swi.trace_log(_logSysLevel, _logStatus,
                  sorted(ngramList), context='Search - ngrams used')
    swi.trace_log(_logSysLevel, _logStatus,
                  sorted(unseenNgrams), context='Search - ngrams not used')

    # Look for Word2Vec similarity data and add the 1st to our search words list
    # and run the search again to see if we get better results
    if similarity:
        swi.trace_log(_logSysLevel, _logStatus,
                      "Initiated Similarity Extension Search", context='Search')
        withExtraWords = words[:]
        for word in words:
            if word in dbstores['similarity']:
                if len(dbstores['similarity'][word]) > 1:
                    withExtraWords.append(dbstores['similarity'][word][0])
                    swi.trace_log(_logSysLevel, _logStatus,
                                  dbstores['similarity'][word][0], context='Search - Add Similiar Word')

        # if we found additional words, then lets do a 2nd run
        if len(withExtraWords) > len(words):
            extraSearchText = ' '.join([str(word) for word in withExtraWords])
            search(extraSearchText, similarity=False, openclose=False, dbstores=dbstores, swicfg=swicfg)

    if openclose:
        swi.close_datastores(dbstores)

    return


def index_files():
    dbstores = swi.open_datastores()
    swicfg = swi.sys_config(dbstores)
    swi.chk_coredb_keys(dbstores, swicfg)

    swi.trace_log(_logSysLevel, _logConfig, swicfg, context='swicfg')

    # Index *.TXT files
    srcCat = 'FILE'
    srcSubCat = 'TXT'
    fileList = []

    for root, dirs, files in os.walk(swicfg['corpus']):
        for fileName in files:
            if fileName.endswith((".txt",".TXT",".Txt")): fileList.append(os.path.join(root, fileName))

    swi.trace_log(_logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log(_logSysLevel, _logTrace, fileList[-10:], context='Last 10 Files to scan')
    swi.parse_file_txt(dbstores, swicfg, fileList, srcCat, srcSubCat)

    # Index *.CSV files
    srcCat = 'FILE'
    srcSubCat = 'CSV'
    fileList = []

    for root, dirs, files in os.walk(swicfg['corpus']):
        for fileName in files:
            if fileName.endswith((".csv",".CSV",".Csv")): fileList.append(os.path.join(root, fileName))

    swi.trace_log(_logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log(_logSysLevel, _logTrace, fileList[-10:], context='Last 10 Files to scan')
    swi.parse_file_txt(dbstores, swicfg, fileList, srcCat, srcSubCat)

    swi.ngram_srcdoc(dbstores, swicfg)
    swi.vectorize_src(dbstores, swicfg)

    swi.close_datastores(dbstores)
    return


def validate_dict():
    # Force Scan of every Dictionary <-> Vector Pair

    line = '-' * 80
    tqdm.write(line)
    swi.trace_log(_logSysLevel, _logStatus, 'Validating Dictionary...')
    dbstores = swi.open_datastores()
    swicfg = swi.sys_config(dbstores)
    total = len(dbstores['dict'].keys())

    swi.trace_log(_logSysLevel, _logStatus, 'Checking Vector...')

    for word in tqdm(list(dbstores['dict'].keys()), desc='Words/Vectors', dynamic_ncols=True, position=0):
        swi.dict_parse_words(dbstores, swicfg, [word], xcheck=True)

    swi.trace_log(_logSysLevel, _logStatus, 'Number of keys Dict: ' + str(len(dbstores['dict'].keys())))
    swi.trace_log(_logSysLevel, _logStatus, 'Number of keys Vect: ' + str(len(dbstores['vectors']['vectors'].keys())))
    tqdm.write(line)

    swi.close_datastores(dbstores)
    return


def main(argv):
    # Parse commandline options

    helpText = 'USAGE: SearchWithInference.py [-h|--help]|[-a|--analyse|--stats]|[-i|--index]|[--stopwords|--import-stopwords|--load-stopwords]|<[-s|--search|--find] "text">'

    if len(argv) == 0:
        tqdm.write(helpText)
        sys.exit(2)

    try:
        opts, args = getopt.getopt(argv,"ahiws:",["analyse", "analysis", "analyze", "help", "index", "search=", "stats", "statistics", "w2v", "word2vect", "word2vectorize", "valdict", "stopwords", "import-stopwords", "load-stopwords"])
    except getopt.GetoptError:
        tqdm.write(helpText)
        tqdm.write('\n')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Help')
            tqdm.write(helpText)
            sys.exit()
        elif opt in ("-i", "--index"):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Index')
            index_files()
            sys.exit()
        elif opt in ("-s", "--search", '--find'):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Search')
            search(arg)
            sys.exit()
        elif opt in ("-a", "--analyse", "--analysis", "--analyze", "--stats", "--statistics"):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Statistics')
            system_analyse()
            sys.exit()
        elif opt in ("--valdict"):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Validate Dictionary <-> Vector')
            validate_dict()
            sys.exit()
        elif opt in ("-V", "--w2v", "--word2vect", "--word2vectorize"):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Word2Vec')
            word2vec()
            sys.exit()
        elif opt in ("--stopwords", "--import-stopwords", "--load-stopwords"):
            swi.trace_log(_logSysLevel, _logInfo, 'Arg - Import Stopwords')
            swi.import_stopwords()
            sys.exit()
    return


if __name__ == '__main__':
    main(sys.argv[1:])
