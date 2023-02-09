#!/bin/bash

INFO='default'
LR=0.00007
TOLERANCE=89 # use 89 pixels for 2deg tolerance or 222 pixels for 5 deg tolerance
PLOTBOOL=false # set true to plot precision-recall curve

if $PLOTBOOL
then
  python evaluation.py --lr=$LR --tolerance=$TOLERANCE --info=$INFO --plot-bool
else
  python evaluation.py --lr=$LR --tolerance=$TOLERANCE --info=$INFO
fi

