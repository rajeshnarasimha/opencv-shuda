#!/bin/bash
echo "clean..."
if [ -f Exe ]; then
    rm Exe
fi
echo "make..."
make 
echo "run..."
if [ -f Exe ]; then
    ./Exe
fi
echo "
end."
