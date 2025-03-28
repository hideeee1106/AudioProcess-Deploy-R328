#!/bin/bash

# build
BUILD_DIR=./build_r328/

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

#GCC_COMPILER=/root/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf

cd ${BUILD_DIR}
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MNN=ON\
    -DENABLE_TIME_MONITOR=OFF \
#    -DCMAKE_C_COMPILER=${GCC_COMPILER}-gcc \
#    -DCMAKE_CXX_COMPILER=${GCC_COMPILER}-g++
make -j6

make install

cd -
