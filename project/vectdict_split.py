#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import swi_lib as swi

_logError = 0
_logStatus = 1
_logConfig = 2
_logInfo = 3
_logTrace = 4
_logSysLevel = _logTrace

dataSpecVer = 0.1 # Min data spec version expected & used

line = '-' * 80

def bld_vect():
    print line
    print 'Building Vectors...'
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)
    print 'Building new dictionary...'
    dbStores['vector'].update(dict(zip(dbStores['vectdict']['dict'].values(), dbStores['vectdict']['dict'].keys())))
    dbStores['vectors'].sync()
    print 'Number of keys produced:', len(dbStores['vector'].keys())
    print line
    swi.close_datastores(dbStores)

def bld_dict():
    print line
    print 'Building Dictionary...'
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)
    print 'Building new dictionary...'
    dbStores['dict'].update(dict(zip(dbStores['vectdict']['dict'].keys(), dbStores['vectdict']['dict'].values())))
    dbStores['dict'].sync()
    print 'Number of keys produced:', len(dbStores['dict'].keys())
    print line
    swi.close_datastores(dbStores)

def validate_dict():
    print line
    swi.trace_log( _logSysLevel, _logStatus, 'Validating Dictionary...')
    dbStores = swi.open_datastores()
    sysConfig = swi.sys_config(dbStores)
    count = 0
    total = len(dbStores['dict'].keys())

    swi.trace_log( _logSysLevel, _logStatus, 'Checking Vector...')
    minVector = max(dbStores['vector'].keys())
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
    swi.trace_log( _logSysLevel, _logStatus, 'Number of keys Vect: ' + str(len(dbStores['vector'].keys())) )
    print line
    swi.close_datastores(dbStores)
#def

def main():
    swi.trace_log( _logSysLevel, _logStatus, 'Starting up...')
    #bld_vect()
    #bld_dict()
    validate_dict()

if __name__ == '__main__':
    main()
#fi
