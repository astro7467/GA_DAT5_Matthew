#!/usr/bin/env python2
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
import os
import sys

_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace

dataSpecVer = 0.1 # Min data spec version expected & used

#Search for string in index
def search(searchText):
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)

    searchText = swi.normalize_text(searchText)
    words = searchText.split()
    ngramList = swi.build_ngrams(words, sysConfig['ngram'])

    swi.trace_log( _logSysLevel, _logInfo, {'searchText': searchText})
    swi.trace_log( _logSysLevel, _logConfig, {'ngramWidth': sysConfig['ngram']})
    swi.trace_log( _logSysLevel, _logInfo, {'ngramList': ngramList})

    topSrcMatches, topSrcInfo, unseenNgrams = swi.calc_matches(dbStores, ngramList, 10)

    print 'topSrcMatches:'
    for index in range(0, len(topSrcMatches)):
        srcID = topSrcMatches[index]
        print str(index+1).rjust(4), ':', dbStores['docmeta'][srcID]['cat'], dbStores['docmeta'][srcID]['subcat'], str(srcID), str(topSrcInfo[srcID]['score']).rjust(14), str(topSrcInfo[srcID]['ngrams']).rjust(4), dbStores['docmeta'][srcID]['path']
    #rof

    swi.trace_log( _logSysLevel, _logStatus, "ngrams used:")
    swi.trace_log( _logSysLevel, _logStatus, sorted(ngramList))
    swi.trace_log( _logSysLevel, _logStatus, "ngrams not indexed:")
    swi.trace_log( _logSysLevel, _logStatus, sorted(unseenNgrams))

    swi.close_datastores(dbStores)
    return
#fed

def index_files():
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)

    swi.trace_log( _logSysLevel, _logConfig, sysConfig)

    # Index *.TXT files
    srcCat = 'FILE'
    srcSubCat = 'TXT'
    fileList = []

    for root, dirs, files in os.walk(sysConfig['corpus']):
        for file in files:
            if file.endswith((".txt",".TXT")): fileList.append(os.path.join(root, file))
        #rof
    #rof
    swi.trace_log( _logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log( _logSysLevel, _logTrace, ['Last 10 Files to scan'] + fileList[-10:])
    swi.index_file_txt(dbStores, sysConfig, fileList, srcCat, srcSubCat)

    # Index *.CSV files
    srcCat = 'FILE'
    srcSubCat = 'CSV'
    fileList = []

    for root, dirs, files in os.walk(sysConfig['corpus']):
        for file in files:
            if file.endswith((".csv",".CSV")): fileList.append(os.path.join(root, file))
        #rof
    #rof
    swi.trace_log( _logSysLevel, _logInfo, 'Found '+str(len(fileList))+' '+srcCat+' of '+srcSubCat+' to scan')
    swi.trace_log( _logSysLevel, _logTrace, ['Last 10 Files to scan'] + fileList[-10:])
    swi.index_file_txt(dbStores, sysConfig, fileList, srcCat, srcSubCat)

    swi.close_datastores(dbStores)
    return
#fed

# Parse commandline options
def main(argv):
    helpText = 'USAGE: SearchWithInference.py <[-h|--help]|[-i|--index]|[-s|--search|--find] text>'

    if len(argv) == 0:
        print helpText
        sys.exit(2)
    #fi

    try:
        opts, args = getopt.getopt(argv,"his:",["help","index","search="])
    except getopt.GetoptError:
        print helpText
        print
        sys.exit(2)
    #yrt

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Help')
            print helpText
            sys.exit()
        elif opt in ("-i", "--index"):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Index')
            index_files()
        elif opt in ("-s", "--search", '--find'):
            swi.trace_log( _logSysLevel, _logInfo, 'Arg - Search')
            search(arg)
        #fi
    #rof
    return
#fed

if __name__ == '__main__':
    main(sys.argv[1:])
#fi
