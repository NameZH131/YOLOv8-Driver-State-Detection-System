cmake_minimum_required(VERSION 3.22.1)

project("yolov8pose")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NCNN SDK directory (each arch has its own include/lib)
set(NCNN_DIR ${CMAKE_SOURCE_DIR}/ncnn/${ANDROID_ABI})

# OpenCV SDK directory
set(OPENCV_DIR ${CMAKE_SOURCE_DIR}/opencv)

include_directories(
    ${NCNN_DIR}/include
    ${NCNN_DIR}/include/ncnn
    ${OPENCV_DIR}/include
    ${OPENCV_DIR}/include/opencv2
    ${CMAKE_SOURCE_DIR}
)

# Add NCNN static library
add_library(ncnn STATIC IMPORTED)
set_target_properties(ncnn PROPERTIES
    IMPORTED_LOCATION ${NCNN_DIR}/lib/libncnn.a
)

# glslang static libraries (required by ncnn)
add_library(glslang STATIC IMPORTED)
set_target_properties(glslang PROPERTIES
    IMPORTED_LOCATION ${NCNN_DIR}/lib/libglslang.a
)

add_library(glslang-default-resource-limits STATIC IMPORTED)
set_target_properties(glslang-default-resource-limits PROPERTIES
    IMPORTED_LOCATION ${NCNN_DIR}/lib/libglslang-default-resource-limits.a
)

# OpenCV static libraries (opencv-mobile v29 - NDK r27 compatible)
set(OPENCV_LIB_DIR ${OPENCV_DIR}/${ANDROID_ABI})

add_library(opencv_core STATIC IMPORTED)
set_target_properties(opencv_core PROPERTIES
    IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_core.a
)

add_library(opencv_imgproc STATIC IMPORTED)
set_target_properties(opencv_imgproc PROPERTIES
    IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_imgproc.a
)

add_library(opencv_features2d STATIC IMPORTED)
set_target_properties(opencv_features2d PROPERTIES
    IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_features2d.a
)

add_library(opencv_video STATIC IMPORTED)
set_target_properties(opencv_video PROPERTIES
    IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_video.a
)

add_library(opencv_photo STATIC IMPORTED)
set_target_properties(opencv_photo PROPERTIES
    IMPORTED_LOCATION ${OPENCV_LIB_DIR}/libopencv_photo.a
)

# Vulkan support
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    add_definitions(-DUSE_VULKAN)
    include_directories(${Vulkan_INCLUDE_DIRS})
endif()

# OpenMP support
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

# Native library
add_library(
    yolov8pose
    SHARED
    yolov8pose.cpp
    native-lib.cpp
)

# Link libraries
target_link_libraries(yolov8pose PRIVATE
    ncnn
    glslang
    glslang-default-resource-limits
    opencv_imgproc
    opencv_core
    opencv_features2d
    opencv_video
    opencv_photo
    android
    log
    jnigraphics
)

if(Vulkan_FOUND)
    target_link_libraries(yolov8pose PRIVATE Vulkan::Vulkan)
endif()