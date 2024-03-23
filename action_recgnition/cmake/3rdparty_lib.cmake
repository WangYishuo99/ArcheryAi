
set(PATH_3RDPARTY "${PROJECT_SOURCE_DIR}/3rdparty")

# **spdlog***
set(SPDLOG_INCLUDE_DIR "${PATH_3RDPARTY}/spdlog/include")
message("log dir : ${SPDLOG_INCLUDE_DIR}")
set(SPDLOG_LIBRARY_DIRS "${PATH_3RDPARTY}/spdlog/lib")

# **opencv***
set(OPENCV_INCLUDE_DIR "${PATH_3RDPARTY}/opencv4.5.4/opencv/build/include")
set(OPENCV_LIBRARY_DIRS "${PATH_3RDPARTY}/opencv4.5.4/opencv/build/x64/vc15/lib")
set(OPENCV_DLL_DIRS "${OPENCV_LIBRARY_DIRS}")





#**trt**
set(TRT_DIR "${PATH_3RDPARTY}/TensorRT-8.2.1.8")  
set(TRT_INCLUDE_DIRS "${TRT_DIR}/include") 
set(TRT_LIB_DIRS "${TRT_DIR}/lib") 

#python

