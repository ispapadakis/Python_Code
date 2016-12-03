# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 19:02:13 2016

@author: ioannis
"""
import sys

def main():
    """Quick & dirty code to get exceptions."""
    options = process_options()
    print(options)
    no_title = options['no_title']
    print(no_title)

    return 0

def process_options(no_title=False, num_format='.5f'):
    if any([a.find('-h') == 0 for a in sys.argv[1:]]):
        usage_info()
    no_title = any([a.find('-n') == 0 for a in sys.argv[1:]])
    f_flag = [a.find('-f') == 0 for a in sys.argv[1:]]
    for v in enumerate(sys.argv[1:]):
        if f_flag[v[0]]:
            num_format = v[1][2:]
            if all([num_format.find(a) == -1 for a in ['f','d']]):
                print('--- Format Specification Error ---')
                usage_info()
            break
    return {'no_title': no_title, 'num_format': num_format}
    
def usage_info():
    print(main.__doc__)
    sys.exit()
    

main()