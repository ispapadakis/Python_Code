#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:49:07 2016

@author: Yanni Papadakis
"""
import sys
import os

def main():
    '''
    usage: grepword.py word infile1 [infile2 [... infileN]]
    '''
    
    if len(sys.argv) < 3: 
        sys.exit(main.__doc__)

    word = sys.argv[1]
    files = sys.argv[2:]
    
    files_expand = set()
    for fname in files:
        star_pos = fname.find('*')
        if star_pos > -1:
            try:
                prefix = fname[:star_pos]
                suffix = fname[star_pos+1:]
                flist = [f.lower() for f in os.listdir() 
                    if prefix == f[:star_pos] and suffix == f[-len(suffix):]]
            except all:
                sys.exit('Wildcard Format: [prefix]*[suffix]')
            files_expand = files_expand | set(flist)
        else:
            files_expand.add(fname.lower())
    
    for filename in sorted(files_expand):
        print('\nIn file {}'.format(filename))
        found = False
        for lino, line in enumerate(open(filename), start=1):
            if word in line:
                found = True
                print("{0}:{1}:{2:.40}".format(filename.upper(), lino,
                      line.strip()))
        if not found: print('Word not Found')
                     
main()        