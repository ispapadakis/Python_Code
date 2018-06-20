#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:56:15 2017

@author: Yanni Papadakis
"""

import re
def nthNumber(s, n):
    pattern = '[^1-9]*\d+'*(n-1)+'[^1-9]*(\d+)'
    return re.match(pattern, s).group(1)
    
nthNumber("8one 003number 201numbers li-000233le number444",4) # "233"

pattern = r'((?<=Isaac) Newton){{{}}}'.format(2)
m = re.match(pattern, "Isaac NewtonIsaac Newton")
m.groups()

m = re.match('[^1-9]*\d+[^1-9]*\d+[^1-9]*\d+[^1-9]*(\d+)', "8one 003number 201numbers li-000233le number444")
m.groups()

re.match('.*o[^n]*n', "8one 003number 201numbers li-000233le number444")

re.match(r'.*(.{3})\t.*(.{3})$', "cough\tbough").group(1)
re.match(r'.*(.{3})\t.*(.{3})$', "CodeFig!ht\tWith all your might").group(1)

letter = '''Everything is fine, fine perfectly here. 
Here I have fun (fun all the day!) days. Although I miss you, 
so please please, Jane, write, write me whenever you can! Can you 
do that? That is the only (!!ONLY!!) thing I ask from you, you sunshine.'''

pattern = r'(\w+)\W+\1'
regex = re.compile(pattern,re.I)
re.findall(regex, letter)

rules = "Roll d6-3 and 4d4+3 to pick a weapon, and finish the boss with 3d7!"
pattern = r'([0-9]+)?d([0-9]+)(([-+])([0-9]+))?'
regex = re.compile(pattern)
formulas = re.findall(regex, rules)
for formula in formulas:
        rolls = int(formula[0]) if formula[0] else 1
        dieType = int(formula[1])