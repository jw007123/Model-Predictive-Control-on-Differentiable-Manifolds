#!/bin/sh

if [ ! -d "External/eigen" ]
then
	git clone https://gitlab.com/libeigen/eigen.git External/eigen
fi

if [ ! -d "External/osqp" ]
then
	git clone https://github.com/osqp/osqp.git External/osqp
	cd External/osqp
	mkdir Build
	cd Build
	cmake -G "Unix Makefiles" ..
	cmake --build . --config Release . --target install
	cd ../../..
fi

if [ ! -d "Build" ]
then
    mkdir Build
fi

cd Build
cmake ..
make
