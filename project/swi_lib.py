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
import re
import shelve
import uuid

trace = True

# Normalize Text Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace html breaks with newline
    norm_text = re.sub(r'<br *\/*>', '\n', norm_text)
    # Replace non-AlphaNumeric|Newline with Space
    norm_text = re.sub(r'[^\w\n]+', ' ', norm_text)
    return norm_text

### Open Datastores and return handles
def open_datastores():
    # Config Datastore
    rootConfig = shelve.open('data/config.db', writeback=True)

    # Ngram Datastore (Core index)
    rootNgram = shelve.open('data/ngram.db', writeback=True)

    # Document Meta Data Datastore
    rootDocMeta = shelve.open('data/docmeta.db', writeback=True)

    # Document Stats (file ngrams) Datastore
    rootDocStat = shelve.open('data/docstat.db', writeback=True)

    # File Type/Path to UUID Datastore
    rootSources = shelve.open('data/sources.db', writeback=True)

    return rootConfig, rootNgram, rootDocMeta, rootDocStat, rootSources

# UUID Management
def sys_config(configStore):

    try:
        uuidInstance = configStore['uuid']
    except:
        configStore['uuid'] = uuid.uuid1()
        uuidInstance = configStore['uuid']

    try:
        corpusDir = configStore['corpus']
    except:
        configStore['corpus'] = './corpus'
        corpusDir = configStore['corpus']

    try:
        ngramWidth = configStore['ngram']
    except:
        configStore['ngram'] = 3
        ngramWidth = configStore['ngram']

    try:
        StoreDataVersion = configStore['version']
    except:
        configStore['version'] = 0.1
        StoreDataVersion = configStore['version']

    configStore.sync()

    return uuidInstance, corpusDir, ngramWidth, StoreDataVersion

# Match or Create a UUID for data sources
def uuid_source(rootSources, srcPath, Instance):
    try:
        srcUUID = rootSources[srcPath]
    except:
        rootSources[srcPath] = uuid.uuid3(Instance, srcPath)
        srcUUID = rootSources[srcPath]

    rootSources.sync()

    return srcUUID

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
            'indexed'   :False
            'verdate'   :'UNK',
            'lastidx'   :'UNK',
            'qltyscore' :0,
            'indexscore':0,
            'xrefscore' :0,
            'ngrams'    :[] }
        docMetaStore.sync()

    # Flush Anagram Line/paragrph data
    if not docStat.has_key(srcID):
        #del docStat[srcID]
        docStat[srcID] = {}
        docStat.sync()

    return

# Record/Add Source with nGram usage
def ngram_store_add(ngramStore, ngram, srcID):
    ngram = str(ngram)
    srcID = str(srcID)
    # initialize item if not already in the master dictionory
    if not ngramStore.has_key(ngram):
        ngramStore[ngram] = [srcID]
        if trace: print('CREATE: ngramStore[ngram]:', srcID, ngram, ngramStore[ngram][-10:])
    elif srcID not in ngramStore[ngram]:
        # if File isn't recorded as a viable match, then add to list
        ngramStore[ngram].append(srcID)
        if trace: print('ADD: ngramStore[ngram]:', srcID, ngram, ngramStore[ngram][-10:])
    return

# Record Source includes ngram, and what lines/paragraphs inc ngram
def src_ngram_add(docMetaStore, docStat, ngram, lineID, srcID):
    ngram = str(ngram)
    srcID = str(srcID)
    #lineID = str(lineID)

    # Add ngram's existence into Meta Storage
    if docMetaStore[srcID]['ngrams'] == []:
        docMetaStore[srcID]['ngrams'] = [ngram]
        if trace: print('CREATE: docMetaStore[srcID][ngrams]:', srcID, ngram, docMetaStore[srcID]['ngrams'][-10:])
    elif ngram not in docMetaStore[srcID]['ngrams']:
        docMetaStore[srcID]['ngrams'].append(ngram)
        if trace: print('ADD: docMetaStore[srcID][ngrams]:', srcID, ngram, docMetaStore[srcID]['ngrams'][-10:])

    # Add/initialize ngram and line(s) info Source Statistics
    if not docStat.has_key(srcID):
        docStat[srcID] = { ngram :[lineID] }
        if trace: print('CREATE: docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
    elif ngram not in docStat[srcID]:
        # if ngram hasn't been initialized
        docStat[srcID][ngram] = [lineID]
        if trace: print('CREATE: docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
    elif lineID not in docStat[srcID][ngram]:
        # if line isn't recorded as a viable match, then add to list
        docStat[srcID][ngram].append(lineID)
        if trace: print('ADD: docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
    return

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

# Build ngram list from given word list
def build_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])
