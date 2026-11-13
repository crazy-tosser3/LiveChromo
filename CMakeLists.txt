cmake_minimum_required(VERSION 3.20)
project(real_time_segmentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

set(
        ONNXRUNTIME_ROOT
        ${CMAKE_SOURCE_DIR}/third_party/onnxruntime
        CACHE PATH
        "Path to an ONNX Runtime package (CPU, CUDA or CoreML)"
)

add_executable(segmentation main.cpp)

target_include_directories(
        segmentation
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${ONNXRUNTIME_ROOT}/include
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
        ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/coreml
)


target_link_directories(segmentation PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(
        segmentation
        PRIVATE
        ${OpenCV_LIBS}
        onnxruntime
)
set_target_properties(segmentation PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
