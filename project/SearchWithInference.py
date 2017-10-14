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

"""
    Current design is a simple minimum DB design
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
- Split each ngram width into separate tables - better manage & scale ngram changes
- Consider 1st Pass index (Fast - monogram) vs background index (bigram -> Ngram)
- line/cell/tag/paragraph weights or tags eg. <h1> > <h2>, 'word': 'definition' weighing
- Seperate DB backend from code (i.e. shelve/cpickle vs sqlite/mySQL/Marina/PostgresSQL)
"""

import swi_lib as swi
import getopt
import glob
import os
import sys

from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer

trace = True # Print out extra info as we go
dataSpecVer = 0.1 # Min data spec version expected & used

#Search for string in index
def search(searchText):
    configStore, ngramStore, docMetaStore, docStat, src2uuid = swi.open_datastores()
    uuidInstance, corpusDir, ngramWidth, StoreDataVersion = swi.sys_config(configStore)

    searchText = swi.normalize_text(searchText)
    words = searchText.split()
    calcNgrams = swi.build_ngrams(words, ngramWidth)

    if trace:
        print('INFO: Text         :', searchText)
        print('INFO: Width, ngrams:', ngramWidth, calcNgrams)
    #fi

    topSrcMatches = swi.calc_matches(ngramStore, calcNgrams, 10)

    print 'topSrcMatches:'
    for index in range(0, len(topSrcMatches)):
        srcID = topSrcMatches[index]
        print index+1, ':', srcID, docMetaStore[srcID]['cat'],docMetaStore[srcID]['subcat'],  docMetaStore[srcID]['path']
    #rof
    swi.close_datastores(configStore, ngramStore, docMetaStore, docStat)
    return
#fed

def index_files():
    configStore, ngramStore, docMetaStore, docStat, src2uuid = swi.open_datastores()
    uuidInstance, corpusDir, ngramWidth, StoreDataVersion = swi.sys_config(configStore)

    if trace: print('PROCESS: uuidInstance, corpusDir, ngramWidth, StoreDataVersion:', uuidInstance, corpusDir, ngramWidth, StoreDataVersion)

    srcCat = 'FILE'
    srcSubCat = 'TXT'

    #path = os.path.join(corpusDir, '*.txt')
    #foundfiles = glob.glob(path)
    foundFiles = []
    for root, dirs, files in os.walk(corpusDir):
        for file in files:
            if file.endswith(".txt"):
                 foundFiles.append(os.path.join(root, file))
            #fi
        #rof
    #rof

    if trace: print('PROCESS: Files:', foundFiles[:-10])

    # Generate Vectorization of ngrams and strip stop words
    vectorizer = CountVectorizer(ngram_range=(1, ngramWidth), stop_words='english')
    ngramAnalyzer = vectorizer.build_analyzer()

    # for each file, get a UID and parse
    for fileName in foundFiles:
        # Build a individual File Breakdown dictionary
        srcID = swi.uuid_source( src2uuid, '' + srcCat + ':' + srcSubCat + ':' + fileName + '', uuidInstance)
        swi.init_source(docMetaStore, docStat, srcID, fileName, srcCat, srcSubCat)

        if trace: print('PROCESS: SrcID, Filename, SrcCat, SrcSubCat:', srcID, fileName, srcCat, srcSubCat)

        if not docMetaStore[srcID]['indexed']:
            lineID = 0
            with open( fileName, mode = 'rU' ) as currFile:
                lineNgrams = ngramAnalyzer( swi.normalize_text(fileName) )

                if trace: print('PROCESS: LineID, fileName:', lineID, swi.normalize_text(fileName))

                # For each word/ngram add to master dictionary with FileID & In FileDict
                for item in lineNgrams:
                    #first Record Ngram is in File, then record which lines have the Ngram
                    swi.ngram_store_add(ngramStore, item, srcID)
                    swi.src_ngram_add(docMetaStore, docStat, item, 0, srcID)
                #rof

                # for each line get a UID and parse line
                for lineID, line in enumerate(currFile, 1):
                    normalizedText = swi.normalize_text(line)
                    if not normalizedText in [None, '', ' ']:
                        if trace: print('PROCESS: LineID, normalizedText:', lineID, normalizedText)

                        # store the lines vectorization for later analysis
                        lineNgrams = ngramAnalyzer( normalizedText )

                        # For each word/ngram add to master dictionary with FileID & In FileDict
                        for item in lineNgrams:
                            #first Record Ngram is in File, then record which lines have the Ngram
                            swi.ngram_store_add(ngramStore, item, srcID)
                            swi.src_ngram_add(docMetaStore, docStat, item, lineID, srcID)
                        #rof
                    #fi
                #rof
            #htiw
            docMetaStore[srcID]['indexed'] = True
            ngramStore.sync()
            docMetaStore.sync()
            docStat.sync()
        #fi
    #rof
    swi.close_datastores(configStore, ngramStore, docMetaStore, docStat)
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
            if trace: print('INFO: Arg - Help')
            print helpText
            sys.exit()
        elif opt in ("-i", "--index"):
            if trace: print('INFO: Arg - Index')
            index_files()
        elif opt in ("-s", "--search", '--find'):
            if trace: print('INFO: Arg - Search')
            search(arg)
        #fi
    #rof
    return
#fed

if __name__ == '__main__':
    main(sys.argv[1:])
#fi
