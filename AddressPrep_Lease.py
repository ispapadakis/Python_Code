# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 2016

@author: PapadakisI

Read Address File and Prep DUNS Match
Use field separators in client file
"""

import re
import sys
  
def read_city_re():
    '''
    Reads File With City Names.
    Use a single RE for 1-Word Names and
    individual REs for Names with more than 2 words.
    '''
    # More than 1 word per name
    f = open('data/CityRe.txt', 'r')
    lst = []
    for line in f:
        lst += [line.rstrip()]
    f.close()
    # Only 1 word per name
    lst += [r'[-\w`]{3,20}']
    return '|'.join(lst)
    
def read_province_re():
    '''
    Reads File with Canada Province Info.
    Constructs REs for common Province Abbreviations.
    
    Returns RE and 2-Letter Abbreviation for each Province
    '''
    d = dict()
    f = open('data/canada_province_abbr.csv', 'r')
    for line in f:
        lst = line.rstrip().split(',')
        if d.setdefault(lst[0]) is None:
            d[lst[0]] = lst[1]
        else:
            d[lst[0]] = lst[1] + '|' + d[lst[0]]
    f.close()
    lst = []
    for k in d:
        lst += ['(?P<{0}>{1})'.format(k,d[k])]
    return ('|'.join(lst), d.keys())
    
def search_for_province_info(s_str):
    '''
    Performs Search.
    Replaces non-standard abbreviations with 2-letter code.
    Assumes only one province match and uses first match encountered.     
    
    Returns Province Code and left fragment remaining.
    '''
    search_s = prov_re.search(s_str)
    if search_s is None:
        print("*** WARNING: No Province > " + s_str, file=err)
        return ('', s_str[:-1]) # Due to many lines with only 1 letter for Province
    else:
        for k in d_keys:
            if search_s.group(k) is not None:
                return (k, s_str[:search_s.start(k)])
        

def read_delimited_file_header(fh,delm='\t'):   
    header = fh.readline().rstrip().split(delm)
    print('\n {:3} {:20}'.format('No','Column'))
    for col in enumerate(header):
        print('{:>3} {:20}'.format(*col))
    return header
    
def remove_attn_field(s_str):
    '''
    Removes Attention to Fields from Company Name and Address Line
    '''
    search_s = attn_re.search(s_str)
    if search_s is not None:
        #print("*** WARNING: Attn String > " + search_s.group(), file=err)
        clean_str = s_str[:search_s.start()] + s_str[search_s.end():]
        return clean_str
    else:
        return s_str

def search_for_postal_code(s_str):
    '''
    Locates Canadian Postal Code Using Characteristic Pattern
    '''
    search_s = postal_re.search(s_str)
    if search_s is None:
        print("*** WARNING: No Postal Code > " + s_str, file=err)
        return ('',s_str)
    else:
        return (search_s.group(1), s_str[:search_s.start()])
    

def search_for_city_info(s_str):
    '''
    Locates City Name Using RE
    '''
    search_s = city_re.search(s_str)
    if search_s is None:
        print("*** WARNING: No City Info > " + s_str, file=err)
        return ('', s_str)
    else:
        return (search_s.group(2), s_str[:search_s.start()])

def read_address_line(l,e=sys.stdout):
    # Line Transformations to Improve REGEX Matching
    # Very Important
    l = l.upper()
    l = l.replace(',','')
    l = l.replace('.','')
    l = l.replace('\"','')
    l = l.replace("'",'`')
    l = l.replace('|',' ')
    
    l = city_except_re.sub(r'\1 ',l)
    
    # Initialize Variables
    res = dict()

    # Missing Address Info
    if len(l) < 10:
        print("*** WARNING: Missing Address Line > "+l, file=e)
        res['name'] = 'Missing'
        return res
    
    # Discard ATTN Field
    l = remove_attn_field(l)
        
    # Postal Code
    res['postal_code'], frag = search_for_postal_code(l)
        
    # Province
    res['province'], frag = search_for_province_info(frag)
    
    # City
    res['city'], frag = search_for_city_info(frag)

    # PO Box
    pobox_s = pobox_re.search(frag)
    if pobox_s is not None:
        res['addr'] = pobox_s.group(1)
        res['name'] = frag[:pobox_s.start()]
    else:
        # Address    
        addr_n_s = addr_n_re.search(frag)
        if addr_n_s is None:
            res['name'] = frag
            res['addr'] = ''
        else:
            res['addr_n'] = addr_n_s.group(1)
            res['name']  = frag[:addr_n_s.start()]
            res['street'] = frag[addr_n_s.end():]
            res['addr'] = res['addr_n'] + ' ' + res['street']
    return res

def get_country(s_str):
    if s_str[0] == '@':
        return (s_str[1:], 'US')
    else:
        return (s_str, 'CA')
    
# MAIN PROGRAM
    
# REGEX Search Strings
prov_re_txt, d_keys = read_province_re()
city_re_txt = read_city_re()
city_except_re_txt = r'(\bTORONTO\B|\bMONTREAL\B|\bCALGARY\B|\bOTTAWA\B|\bEDMONTON\B|\bMISSISSAUGA\B|\bWINNIPEG\B|\bVANCOUVER\B|\bBRAMPTON\B|\bHAMILTON\B|\bSURREY\B|\bLAVAL\B|\bHALIFAX\B|\bLONDON\B|\bMARKHAM\B|\bVAUGHAN\B|\bGATINEAU\B|\bLONGUEUIL\B|\bBURNABY\B|\bSASKATOON\B|\bKITCHENER\B|\bWINDSOR\B|\bREGINA\B|\bRICHMOND\B|\bOAKVILLE\B|\bBURLINGTON\B|\bSUDBURY\B|\bSHERBROOKE\B|\bOSHAWA\B|\bSAGUENAY\B|\bLÉVIS\B|\bBARRIE\B|\bABBOTSFORD\B)'

# REGEX Compilation
attn_re = re.compile(r'\b(ATTN:|ATT:|A/S |C\s*[/]?O |RE:)\s*\w+[-]?\s*(\b\w+)?\s*')
postal_re = re.compile(r'(([A-Z]\d[A-Z]\s?\d[A-Z]\d)|((\bUSA\b)?\s*[\d\s]{5,6}(-\d{1,4})?))\b\s*$')
prov_re = re.compile(' (' + prov_re_txt + ')\s*(\w{0,3})?\s*$')
city_re = re.compile('( )('+city_re_txt + r')\s*(\w{0,2})?\s*$',re.I)
pobox_re = re.compile(r'\s((P[.]?\s*O[.]?\s+BOX|\s*BOX|\s*C[.]?\s*P[.]?)\s*\d+) ')
addr_n_re = re.compile(r'\s*(\d+-\d+\b|\d+\b|\d+[ABCEWSN])\s*')
city_except_re = re.compile(city_except_re_txt, re.I)
   
# Open Raw Data File
f = open('data/XeroxLeaseClients.txt', 'r', encoding = 'utf-8')
header = read_delimited_file_header(f,'\t')
# Open Error File
err = open('data/XeroxCanada_Read_Errors.txt','w')
# Open Output File for Writing
o = open('data/XeroxLeaseCorrect.csv','w')
# Order of Columns in Output File
outfile_order = ['custno','name','addr','city','province','country','o_postal_code','sic']
# Output File Delimeter
o_dlm = ','
# Write Header Line
print(o_dlm.join(outfile_order),file=o)

i = 0
for line in f:
    #i += 1
    #if i > 300: break
    out = dict()
    try:
        line = line.rstrip().split('\t')
        bill_address = '|'.join(line[1:7])
        out = read_address_line(bill_address)
        out['custno'] = line[0]
        out['o_postal_code'], out['country'] = get_country(line[7])
        if len(line) >= 9:
            out['sic'] = line[8]
        else:
            out['sic'] = ''
        print(o_dlm.join([out[k] for k in outfile_order]), file=o)
    except IndexError:
        sys.exit(line)


f.close()
err.close()
o.close()

f = open('data/XeroxLeaseCorrect.csv')
header = read_delimited_file_header(f,',')

count_can = 0
count_usa = 0
count_oth = 0
for line in f:
    country = line.rstrip().split(',')[5]
    if country == 'US':
        count_usa += 1
    elif country == 'CA':
        count_can += 1
    else:
        count_oth += 1
        
print('Counts')
print('Canada: {0:,d} USA: {1:,d} Unknown {2:,d}'.format(count_can,count_usa,count_oth))
        
        


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
