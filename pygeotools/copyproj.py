#! /usr/bin/env python

import sys
from pygeotools.lib import geolib

src_fn = sys.argv[1]
dst_fn = sys.argv[2]

geolib.copyproj(src_fn, dst_fn)
