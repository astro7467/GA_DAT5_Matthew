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

import swi_lib as swi
import glob
import os

from sklearn.feature_extraction.text import CountVectorizer #, TfidfVectorizer

trace = True # Print out extra info as we go
dataSpecVer = 0.1 # Min data spec version expected & used

def index():
    configStore, ngramStore, docMetaStore, docStat, src2uuid = swi.open_datastores()
    uuidInstance, corpusDir, ngramWidth, StoreDataVersion = swi.sys_config(configStore)

    srcCat = 'FILE'
    srcSubCat = 'TXT'
    path = os.path.join(corpusDir, '*.txt')
    files = glob.glob(path)
    if trace: print('PROCESS: Files:', files[:-10])

    #Generate Vectorization of ngrams and strip stop words
    vectorizer = CountVectorizer(ngram_range=(1, ngramWidth), stop_words='english')
    ngramAnalyzer = vectorizer.build_analyzer()

    # for each file, get a UID and parse
    for fileName in files:
        # Build a individual File Breakdown dictionary
        srcID = swi.uuid_source( src2uuid, '' + srcCat + ':' + srcSubCat + ':' + fileName + '', uuidInstance)
        #print srcID, fileName
        swi.init_source(docMetaStore, docStat, srcID, fileName, srcCat, srcSubCat)

        if trace: print('PROCESS: SrcID, Filename, SrcCat, SrcSubCat:', srcID, fileName, srcCat, srcSubCat)
        if note docMetaStore[srcID]['indexed']:
            with open( fileName, mode = 'rU' ) as currFile:
                # for each line get a UID and parse line
                for lineID, line in enumerate(currFile):
                    if trace: print('PROCESS: LineID, line:', lineID, line[:-128])

                    #print lineID, swi.normalize_text(line)

                    # store the lines vectorization for later analysis
                    lineNgrams = ngramAnalyzer( swi.normalize_text(line) )
                    #print lineID, lineNgrams

                    # For each word/ngram add to master dictionary with FileID & In FileDict
                    for item in lineNgrams:
                        # First Record Ngram is in File, then record which lines have the Ngram
                        swi.ngram_store_add(ngramStore, item, srcID)
                        swi.src_ngram_add(docMetaStore, docStat, item, lineID, srcID)

                    # rof
                # rof
            # htiw
            docMetaStore[srcID]['indexed'] = True
            ngramStore.sync()
            docMetaStore.sync()
            docStat.sync()
        # fi
    # rof
    close_datastores(configStore, ngramStore, docMetaStore, docStat)
    return

#Search for string in index
def search(argv):
    configStore, ngramStore, docMetaStore, docStat, src2uuid = swi.open_datastores()
    uuidInstance, corpusDir, ngramWidth = swi.sys_config(configStore)
    words = argv.split()
    calcngrams = build_ngrams(words, ngramWidth)

    if trace:
        print('INFO: Text', argv)
        print('INFO: ngrams', calcngrams)

    close_datastores(configStore, ngramStore, docMetaStore, docStat)
    return

# Parse commandline options
def main(argv):
    helpText = 'SearchWithInference.py <[-h|--help]|[-i|--index]|[-s|--search|--find] text>'
   try:
      opts, args = getopt.getopt(argv,"his:",["help","index","search="])
   except getopt.GetoptError:
      print helpText
      print
      sys.exit(2)
   for opt, arg in opts:
      if opt in ('-h', '--help'):
         print helpText
         sys.exit()
     elif opt in ("-i", "--index"):
         index
     elif opt in ("-s", "--search", '--find'):
         search(arg)
    return

if __name__ == '__main__':
    main(sys.argv[1:])
