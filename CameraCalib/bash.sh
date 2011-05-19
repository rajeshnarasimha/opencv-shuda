#!/bin/bash
echo "clean..."
if [ -f Exe ]; then
    rm Exe
fi
if [ -f main.o ]; then
    rm main.o
fi
if [ -f calibrationthroughimages.o ];then
    rm calibrationthroughimages.o
fi
echo "make..."
make -j8 
echo "run..."
if [ -f Exe ]; then
    ./Exe Img/
fi
echo "end."
