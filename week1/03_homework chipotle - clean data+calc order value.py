#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 15:12:06 2017

@author: astro7467
"""

'''
 Sum the value of the orders & print those > $10
'''

# row index reference
idx_order_id = 0
idx_order_qty = 1
idx_order_item = 2
idx_order_desc = 3
idx_order_price = 4

import csv
import re
from collections import defaultdict

# specify that the delimiter is a tab character
# make 'file_nested_list' = list of rows

with open( 'data/order.tsv', mode = 'rU') as f:
    file_nested_list = [row for row in csv.reader( f, delimiter='\t')]

# separate the header and data
header = file_nested_list[0]
data = file_nested_list[1:]

# Cleanse data to a basic standard
for row in data:
    # Eliminate [ & ] from Toppings
    row[idx_order_desc] = re.sub(r'[\[\]]', '', row[idx_order_desc])
    
    # Remove NULL toppings
    if row[idx_order_desc] == 'NULL': row[idx_order_desc] = ''
    
    # Make Order ID's clean to 00000 format
    row[idx_order_id] = row[idx_order_id].strip().rjust(5,"0")

    # Make Order Item Price into a float
    row[idx_order_price] = float(re.sub(r'[^\d.]', '', row[idx_order_price]))

order_value = defaultdict(float)

#For each unique Order ID, add the line item price
for row in data:
    order_value[row[idx_order_id]] += row[idx_order_price]

print "\nInterate over the Orders, printing if only $30+"
for key, value in sorted(order_value.iteritems()):
    if value > 30: print "Order", key, ": $", value

print "\nInterate over the Orders, sorted from highest to lowest $ value, printing if only $30+"
for key, value in sorted(order_value.iteritems(), key=lambda (k,v): (v,k), reverse=True):
    if value > 30: print "Order", key, ": $", value


