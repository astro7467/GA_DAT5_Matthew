#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 12:12:11 2017

@author: astro7467
"""

def fib_value(numb):
    '''returns the nth numb in fibonaci sequence'''
    if numb <= 0 or isinstance(numb, float) : return 0
    if numb == 1 : return 1
    if numb == 2 : return 1
    return (fib_value(numb - 2) + fib_value(numb - 1))

print fib_value(12)
print fib_value(24)
print fib_value(48)
print fib_value(96)
print fib_value(192)
print fib_value(400)
print fib_value(1200)
