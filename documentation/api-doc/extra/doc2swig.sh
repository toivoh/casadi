#!/bin/bash

if [ ! -f extra/doxy2swig.py ]; then
  cd extra && wget http://www.aero.iitb.ac.in/~prabhu/software/code/python/doxy2swig.py && cd ..
fi

cd extra && python doxy2swigX.py ../XML/index.xml ../../../swig/doc.i && cd ..
