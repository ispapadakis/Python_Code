#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:40:21 2016

@author: Yanni Papadakis
"""

import re
p = re.compile('[a-z]+')
p  #doctest: +ELLIPSIS

m = p.match("")
print(m)

m = p.match('tempo')
print(m.group(),m.span())

m = p.match('::: message')
if m is not None:
    print(m.group(),m.span())
else:
    print('No Match')
    
s = p.search('::: message')
if s is not None:
    print(s.group(),s.span())
else:
    print('No Match')
    
p = re.compile('\d+')
p.findall('12 drummers drumming, 11 pipers piping, 10 lords a-leaping')

iterator = p.finditer('12 drummers drumming, 11 ... 10 ...')
iterator  

for match in iterator:
    print(match.span())
    
    
print(re.match(r'From\s+', 'Fromage amk'))
print(re.match(r'From\s+', 'From amk Thu May 14 19:12:10 1998'))


p = re.compile('(ab)*')
print(p.match('ababababab').span())
print(p.match('abababab ab').span())
print(p.search('abababab ab').span())
re.split(p,'abababab ab')

p = re.compile('\b(ab)*')
re.split(p,'abababab ab')

p = re.compile(r'\b(ab)*')
re.split(p,'abababab ab')

s = p.search('abababab ab')
print(s.group(0))
print(s.group(1))
print(s.group(1,0))
print(s.span())
print(s.span(1))
try:
    print(s.span(2))
except IndexError as ie:
    print(ie)

print(s.groups())

p = re.compile(r'(\b\w+)\s+\1')
p.search('Paris in the the spring').group()

m = re.match("([abc])+", "abc")
print(m.group())
print(m.groups())

m = re.match("(?:[abc])+", "abc")
print(m.group())
print(m.groups())

m = re.match("(?:a(b)(c))+", "abc")
print(m.group())
print(m.groups())


p = re.compile(r'(?P<word>\b\w+\b)')
m = p.search( '(((( Lots of punctuation )))' )
m.group('word')

m.group(1)

m.groups()


p = re.compile(r'( \w+)+')
m = p.search( '(((( Lots of punctuation )))' )
m.group()

m.groups()
m.group(1)

# Split
p = re.compile(r'\W+')
p.split('This is a test, short and sweet, of split().')

p.split('This is a test, short and sweet, of split().', 3)

addr = [
'John Doe 123 Test Street Baltimore MD 99999',
'Foo Bar 123-1000 JFK Boulevard Apt. 1C New York NY 55555',
'No1 Realty 001 Park Avenue New York NY 55555-1234',
'Belle (2010) 111 Boonnies Str New York NY 50000-1234'
]

zipref = re.compile(r'(\b\d{5}-\d{4}|\b\d{5})$')

for a in addr:
    m = zipref.search(a)
    print(a[:m.start()],m.group())
    print(zipref.split(a))

addnumref = re.compile(r'\s+(\b\d{1,6}-\d{1,6}\b|\b\d{1,9}\b)\s+\w+')

for a in addr:
    m = addnumref.search(a)
    print(a[:m.start()],'|',m.group(1),'|',a[m.end(1):])


p = re.compile(r'\W+')
p2 = re.compile(r'(\W+)')
print(p.split('This... is a test.'))
#['This', 'is', 'a', 'test', '']
print(p2.split('This... is a test.'))
#['This', '... ', 'is', ' ', 'a', ' ', 'test', '.', '']


#The module-level function re.split() adds the RE to be used as the first argument, but is otherwise the same.

re.split('[\W]+', 'Words, words, words.')
#['Words', 'words', 'words', '']
re.split('([\W]+)', 'Words, words, words.')
#['Words', ', ', 'words', ', ', 'words', '.', '']
re.split('[\W]+', 'Words, words, words.', 1)
['Words', 'words, words.']

re.split(r'(\W+)', 'Words, words, words.')

p2 = re.compile(r'([\W]+)')
print(p2.split('This... is a test.'))

p = re.compile('x')
p.sub('-', 'abxd')
#'-a-b-d-'
p = re.compile(r'\Bx+')
p.sub('-', 'abxd')

p = re.compile('section{ ( [^}]* ) }', re.VERBOSE)
p.sub(r'subsection{\1}','section{First} section{second}')
#'subsection{First} subsection{second}'

#Greedy versus Non-Greedy

s = '<html><head><title>Title</title>'
len(s)
#32
print(re.match('<.*>', s).span())
#(0, 32)
print(re.match('<.*>', s).group())
#<html><head><title>Title</title>
print(re.match('<.*?>', s).group())
#<html>

