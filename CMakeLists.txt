cmake_minimum_required(VERSION 3.21)
project(NKF_MNN_Deploy_R328)

set(CMAKE_CXX_STANDARD 11)

message("Build in MNN")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

# mnn infer
set(MNN_DIR /home/hideeee/CLionProjects/snoring-yolo/3rdparty/MNN/linux/include)
set(MNN_lIB_DIR /home/hideeee/CLionProjects/snoring-yolo/3rdparty/MNN/linux/lib)
include_directories(${MNN_DIR})
link_directories(${MNN_lIB_DIR})
message("MNN Lib Path:${MNN_lIB_DIR}")
message("MNN DIR Path:${MNN_DIR}")

# MNN Libraries
set(MNN_LIBRARIES MNN)

# C_API comple files
set(C_API_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/c_api/c_api.cc)

# NS compile files
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/ns/ NS_SRC_FILES)
# Set compile files
set(COMPILE_FILES  ${C_API_FILES} ${NS_SRC_FILES})


# Build shared library
set(LIB_NAME AudioProcess-lite)
add_library(${LIB_NAME} SHARED ${COMPILE_FILES})
target_link_libraries(${LIB_NAME} ${MNN_LIBRARIES})
message("build KARLMAN AEC with MNN on R328 ")

##build target
#add_executable(TestNkfAecSample sample/test_nkf_aec.cpp ${COMPILE_FILES})
#target_link_libraries(TestNkfAecSample ${MNN_LIBRARIES})
