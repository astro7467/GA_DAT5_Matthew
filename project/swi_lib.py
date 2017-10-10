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
    
    configStore.sync()
    
    return uuidInstance, corpusDir, ngramWidth

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
    # Flush srcID's Meta Data Record, and set minimum values
    if docMetaStore.has_key(srcID):
        del docMetaStore[srcID]
        
    docMetaStore[srcID] = {
        'path':str(srcPath),
        'cat':str(srcCat),
        'subcat':str(srcSubCat),
        'verdate':'UNK',
        'lastidx':'UNK',
        'qltyscore':str(0),
        'indexscore':str(0),
        'xrefscore':str(0),
        'ngrams':[] }
    docMetaStore.sync()
    
    # Flush Anagram Line/paragrph data
    if docStat.has_key(srcID):
        del docStat[srcID]
        docStat.sync()

    return

# Record/Add Source with nGram usage
def ngram_store_add(ngramStore, ngram, srcID):
    ngram = str(ngram)
    srcID = str(srcID)
    # initialize item if not already in the master dictionory
    if not ngramStore.has_key(ngram):
        ngramStore[ngram] = [srcID]
    elif srcID not in ngramStore[ngram]:
        # if File isn't recorded as a viable match, then add to list
        ngramStore[ngram].append(srcID)
    if trace: print('ngramStore[ngram]:', srcID, ngram, ngramStore[ngram][-10:])
    return
    
# Record Source includes ngram, and what lines/paragraphs inc ngram
def src_ngram_add(docMetaStore, docStat, ngram, lineID, srcID):
    ngram = str(ngram)
    srcID = str(srcID)
    #lineID = str(lineID)
    
    # Add ngram's existence into Meta Storage 
    if docMetaStore[srcID]['ngrams'] == []:
         docMetaStore[srcID]['ngrams'] = [ngram]
    elif ngram not in docMetaStore[srcID]['ngrams']:
        if trace: print('docMetaStore[srcID][ngrams]:', srcID, ngram, docMetaStore[srcID]['ngrams'][-10:])
        docMetaStore[srcID]['ngrams'].append(ngram)
    
    # Add/initialize ngram and line(s) info Source Statistics
    if not docStat.has_key(srcID):
        docStat[srcID] = { ngram :[lineID] }
    # if ngram hasn't been initialized
    elif ngram not in docStat[srcID]:
        docStat[srcID][ngram] = [lineID]
    elif lineID not in docStat[srcID][ngram]:
        # if line isn't recorded as a viable match, then add to list
        if trace: print('docStat[srcID][ngram]:', srcID, ngram, docStat[srcID][ngram][-10:])
        docStat[srcID][ngram].append(lineID)
    return
