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
import ZODB, ZODB.FileStorage
import persistent

class Account(persistent.Persistent):

    def __init__(self):
        self.balance = 0.0

    def deposit(self, amount):
        self.balance += amount

    def cash(self, amount):
        assert amount < self.balance
        self.balance -= amount

### Open Datastores and return handles
def open_datastores():
   
    # Ngram Datastore (Core index)
    storageNgram = ZODB.FileStorage.FileStorage('data/ngram.fs')
    dbNgram = ZODB.DB(storageNgram)
    connectionNgram = dbNgram.open()
    rootNgram = connectionNgram.root
    
    # Document MetaData Datastore (Target datastore)
    storageDocMeta = ZODB.FileStorage.FileStorage('data/docmeta.fs')
    dbDocMeta = ZODB.DB(storageDocMeta)
    connectionDocMeta = dbDocMeta.open()
    rootDocMeta = connectionDocMeta.root

    return rootNgram, rootDocMeta

