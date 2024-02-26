@echo off
git clone https://gitlab.com/libeigen/eigen.git External/eigen
git clone https://github.com/nothings/stb.git External/stb
git clone https://github.com/osqp/osqp.git External/osqp

cd External/osqp
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release . --target install

cd ../../..

mkdir Build
cd Build
cmake ..
make
