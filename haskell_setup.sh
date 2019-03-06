#!/bin/bash

git clone https://github.com/AdLucem/hasktorch.git
cd hasktorch/
git submodule update --init --recursive
cd ffi/deps/aten
mkdir build 
cd build/
cmake .. -DNO_CUDA=true -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CC_COMPILER=gcc -DCXX=g++ -DCC=gcc -Wno-dev -DCMAKE_INSTALL_PREFIX=.
make install
cd ../../../../..
ln -fs hasktorch/cabal/project.freeze-8.4.2 cabal.project.freeze
./make_cabal_local.sh
source setenv
