#! /usr/bin/env python
import sys
if len(sys.argv) > 1:
    print("hi! {}".format(' '.join(sys.argv[1:])))
sys.stdout.write("hello from Python %s\n" % (sys.version,))