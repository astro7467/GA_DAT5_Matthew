#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    SearchWithInference - Proof of concept for PerSea (Personal Search)
    Copyright (C) 2017  Matthew A. Snell (Astro7467@XLTech.io)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ThisysConfig['vis dis']tributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    YousysConfig['vve rec']eived a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

"""
This system only has is a simple minimum DB design
    *** will not scale - proof of concept only ***
    for full multi doc indexing, would need to breakdown datastores
    to faciliate concurrency, and different document types and meta data needs.
"""

"""
TODOS / Full Product Considerations:
- Leverage Classes for all DB/Data Model Interations
- Seperate W/R DB activity (eg index) from RO DB activity (search)
- Search History (Learning/ngram considerations)
- Explore ngram vectorizations
- Split each ngram width into separate tables - better manage ngram changes
- Consider 1st Pass index (Fast - monogram) vs background index (bigram -> Ngram)
- line/cell/tag/paragraph weights or tags eg. <h1> > <h2>, 'word': 'definition' weighsysConfig['vrate D']B backend from code (i.e. shelve/cpickle vs sqlite/mySQL/Marina/PostgresSQL)
"""

import swi_lib as swi
import getopt
import random
import os
import sys

_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace

dataSpecVer = 0.1 # Min data spec version expected & used

# Provide some insight to the datastores
def system_analyse():
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)

    line = '-' * 80

    print(line)
    print('Config File Info')
    print(line)
    print('Config Record Count:', len(dbStores['config'].keys()))
    swi.trace_log(_logConfig, _logStatus, sysConfig, 'Config Key & Value Pairs')

    swi.chk_coredb_keys(dbStores, sysConfig)

    dictVector = max(dbStores['dict'].values())
    vecVector = max(dbStores['vectors']['vectors'].keys())
    vector = swi.dictionary_vector(dbStores)
    print('Vector Trace: Highest Dictionary / Highest Vectors / Current Counter', dictVector, '/', vecVector, '/', vector)

    print(line)
    print('DocMeta File Info')
    print(line)
    print('DocMeta Record Count:', len(dbStores['docmeta'].keys()))
    print('Ten Random Samples:')
    keyList = list( dbStores['docmeta'].keys() )
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbStores['docmeta'][key], 'DocMeta ' + key)
    #rof

    print(line)
    print('DocStat File Info')
    print(line)
    print('DocStat Record Count:', len(dbStores['docstat'].keys()))
    print('Ten Random Samples:')
    keyList = list( dbStores['docstat'].keys() )
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbStores['docstat'][key], 'DocStat ' + key)
    #rof

    print(line)
    print('Ngram File Info')
    print(line)
    print('Ngram Record Count:', len(dbStores['ngram'].keys()))
    print('Ten Random Samples:')
    keyList = list( dbStores['ngram'].keys() )
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbStores['ngram'][key], 'Ngram ' + key)
    #rof

    print(line)
    print('Sources File Info')
    print(line)
    print('Sources Record Count:', len(dbStores['sources'].keys()))
    print('Ten Random Samples:')
    keyList = list( dbStores['sources'].keys() )
    random.shuffle(keyList)
    for key in keyList[:10]:
        swi.trace_log(_logConfig, _logStatus, dbStores['sources'][key], 'Source ' + key)
    #rof

    print(line)
    print('Dictionary & Vector File Info')
    print(line)
    print('Dictionary & Vector Key Counts (Should be Equal):', len(dbStores['dict'].keys()), '/', len(dbStores['vectors']['vectors'].keys()))
    print('Dictionary & Vectors (inc Validation) (word->vector->word):')
    keyList = list( dbStores['dict'].keys() )
    random.shuffle(keyList)
    for key in keyList[:10]:
        print(key, '(word) ->', dbStores['dict'][key], '(vector) ->', dbStores['vectors']['vectors'][dbStores['dict'][key]], '(word)')
        swi.dict_parse_words(dbStores, sysConfig, [key], xcheck=True)
    #rof

    print(line)
    print('Vector & Dictionary (inc Validation) (vector->word->vector):')
    print(line)
    keyList = list( dbStores['vectors']['vectors'].keys() )
    random.shuffle(keyList)
    for key in keyList[:10]:
        print(key, '(vector) ->', dbStores['vectors']['vectors'][key], '(word) ->', dbStores['dict'][dbStores['vectors']['vectors'][key]], '(vector)')
        swi.dict_parse_words(dbStores, sysConfig, [dbStores['vectors']['vectors'][key]], xcheck=True)
    #rof

    print(line)

    swi.close_datastores(dbStores)
#fed

# Search for string in index
def search(searchText):
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)
    swi.chk_coredb_keys(dbStores, sysConfig)

    searchText = swi.normalise_text(searchText)
    words = sorted(list(set(searchText.split())))

    swi.dict_parse_words(dbStores, sysConfig, words)

    ngramList = swi.build_ngrams(words, sysConfig['ngram'])

    swi.trace_log( _logSysLevel, _logConfig, sysConfig, context='sysConfig')
    swi.trace_log( _logSysLevel, _logInfo, words, context='Search Words')
    swi.trace_log( _logSysLevel, _logInfo, ngramList, context='ngramList')

    topSrcMatches, topSrcInfo, unseenNgrams = swi.calc_matches(dbStores, ngramList, 10)

    print('topSrcMatches:')
    for index in range(0, len(topSrcMatches)):
        srcID = topSrcMatches[index]
        print(str(index+1).rjust(4), ':', dbStores['docmeta'][srcID]['cat'], dbStores['docmeta'][srcID]['subcat'], str(srcID), str(topSrcInfo[srcID]['score']).ljust(18), str(topSrcInfo[srcID]['ngrams']).rjust(4), dbStores['docmeta'][srcID]['path'])
    #rof

    swi.trace_log( _logSysLevel, _logStatus, sorted(ngramList), context='ngrams used')
    swi.trace_log( _logSysLevel, _logStatus, sorted(unseenNgrams), context='ngrams not used')

    swi.close_datastores(dbStores)
    return
#fed

def w2v_files():
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)

    swi.trace_log( _logSysLevel, _logConfig, sysConfig, context='sysConfig')

    # Index *.TXT files
    srcCat = 'FILE'
    srcSubCat = 'TXT'
    fileList = []

    for root, dirs, files in os.walk(sysConfig['corpus']):
        for file in files:
            if file.endswith((".txt",".TXT",".Txt")): fileList.append(os.path.join(root, file))
        #rof
    #rof
    swi.trace_log( _logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log( _logSysLevel, _logTrace, fileList[-10:], context='Last 10 Files to scan')
    swi.vector_file(dbStores, sysConfig, fileList, srcCat, srcSubCat)

    # Index *.CSV files
    srcCat = 'FILE'
    srcSubCat = 'CSV'

    for root, dirs, files in os.walk(sysConfig['corpus']):
        for file in files:
            if file.endswith((".csv",".CSV")): fileList.append(os.path.join(root, fileName))
        #rof
    #rof
    swi.trace_log( _logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log( _logSysLevel, _logTrace, fileList[-10:], context='Last 10 Files to scan')
    swi.vector_file(dbStores, sysConfig, fileList, srcCat, srcSubCat)

    swi.close_datastores(dbStores)
    return
#fed

def index_files():
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)

    swi.trace_log( _logSysLevel, _logConfig, sysConfig, context='sysConfig')
    swi.chk_docmeta_keys(dbStores, sysConfig)

    # Index *.TXT files
    srcCat = 'FILE'
    srcSubCat = 'TXT'
    fileList = []

    for root, dirs, files in os.walk(sysConfig['corpus']):
        for fileName in files:
            print(root, dirs, fileName)
            if fileName.endswith((".txt",".TXT",".Txt")): fileList.append(os.path.join(root, fileName))
        #rof
    #rof
    swi.trace_log( _logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log( _logSysLevel, _logTrace, fileList[-10:], context='Last 10 Files to scan')
    swi.parse_file_txt(dbStores, sysConfig, fileList, srcCat, srcSubCat)

    # Index *.CSV files
    srcCat = 'FILE'
    srcSubCat = 'CSV'
    fileList = []

    for root, dirs, files in os.walk(sysConfig['corpus']):
        for fileName in files:
            if fileName.endswith((".csv",".CSV",".Csv")): fileList.append(os.path.join(root, fileName))
        #rof
    #rof
    swi.trace_log( _logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log( _logSysLevel, _logTrace, fileList[-10:], context='Last 10 Files to scan')
    swi.parse_file_txt(dbStores, sysConfig, fileList, srcCat, srcSubCat)

    swi.ngram_srcdoc(dbStores, sysConfig)

    swi.close_datastores(dbStores)
    return
#fed

# Force Scan of every Dictionary <-> Vector Pair
def validate_dict():
    line = '-' * 80
    print(line)
    swi.trace_log( _logSysLevel, _logStatus, 'Validating Dictionary...')
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)
    count = 0
    total = len(dbStores['dict'].keys())

    swi.trace_log( _logSysLevel, _logStatus, 'Checking Vector...')
    minVector = max(dbStores['vectors']['vectors'].keys())
    vector = swi.dictionary_vector(dbStores)

    if vector <= minVector:
        oldVector = vector
        dbStores['config']['nextvector'] = minVector + 1
        dbStores['config'].sync()
        vector = swi.dictionary_vector(dbStores)
        swi.trace_log( _logSysLevel, _logStatus, {'OldVector': oldVector, 'NewVector': vector, 'MinVector': minVector}, context='Bad Next Vector Found')
    #fi

    for word in list(dbStores['dict'].keys()):
        swi.dict_parse_words(dbStores, sysConfig, [word], xcheck=True)
        if count % 100 == 0:
            swi.trace_log( _logSysLevel, _logStatus, 'Progress: ' + str(count).rjust(len(str(total))+1) + ' of ' + str(total) + ' Last Vector: ' + str(dbStores['config']['nextvector']) + ' - ' + word)
        #fi
        count += 1
    #rof
    swi.trace_log( _logSysLevel, _logStatus, 'Number of keys Dict: ' + str(len(dbStores['dict'].keys())) )
    swi.trace_log( _logSysLevel, _logStatus, 'Number of keys Vect: ' + str(len(dbStores['vectors']['vectors'].keys())) )
    print(line)
    swi.close_datastores(dbStores)
#def

# Parse commandline options
def main(argv):
    helpText = 'USAGE: SearchWithInference.py <[-h|--help]|[-a|--analyse|--stats][-i|--index]|[-s|--search|--find] text>'

    if len(argv) == 0:
        print(helpText)
        sys.exit(2)
    #fi

    try:
        opts, args = getopt.getopt(argv,"ahiVs:",["analyse", "analysis", "analyze", "help", "index", "search=", "stats", "statistics", "vector", "vectorize", "valdict","import-stopwords"])
    except getopt.GetoptError:
        print(helpText)
        print()
        sys.exit(2)
    #yrt

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Help')
            print(helpText)
            sys.exit()
        elif opt in ("-i", "--index"):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Index')
            index_files()
            sys.exit()
        elif opt in ("-s", "--search", '--find'):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Search')
            search(arg)
            sys.exit()
        elif opt in ("-a", "--analyse", "--analysis", "--analyze", "--stats", "--statistics"):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Statistics')
            system_analyse()
            sys.exit()
        elif opt in ("--valdict"):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Validate Dictionary <-> Vector')
            validate_dict()
            sys.exit()
        elif opt in ("-V", "--vector", "--vectorize"):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Vectorize')
            w2v_files()
            sys.exit()
        elif opt in ("--import-stopwords"):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Import Stopwords')
            swi.import_stopwords()
            sys.exit()
        #fi
    #rof
    return
#fed

if __name__ == '__main__':
    main(sys.argv[1:])
#fi
