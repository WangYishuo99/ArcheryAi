add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories( ${OPENCV_INCLUDE_DIR} )
link_directories(${OPENCV_LIBRARY_DIRS})

include_directories( ${SPDLOG_INCLUDE_DIR} )
link_directories(${SPDLOG_LIBRARY_DIRS})

include_directories( ${ONNXRUNTIME_INCLUDE_DIRS} )
link_directories(${ONNXRUNTIME_LIB_DIRS})

find_package(CUDAToolkit REQUIRED)

message(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${TRT_INCLUDE_DIRS})

link_directories(${CUDAToolkit_LIBRARY_DIR})
link_directories(${TRT_LIB_DIRS})

link_directories(${PLUGIN_LIBRARY_DIR})

#python
# set(MCPP "E:/action_recgnition/3rdparty/matplotlib-cpp")
# include_directories(${MCPP})
# include_directories("C:/Users/binli/anaconda3/envs/paddle/include")
# include_directories("C:/Users/binli/anaconda3/envs/paddle/Lib/site-packages/numpy/core/include")

# link_directories("C:/Users/binli/anaconda3/libS")


file(GLOB_RECURSE demo_srcs *.cpp)

source_group_by_dir(demo_srcs)
source_group_by_dir(demo_hdrs)


set(dependend_LIB
    nvinfer
    nvonnxparser
    nvinfer_plugin
    debug opencv_world454d
    optimized opencv_world454 
    debug spdlogd
    optimized spdlog 
    CUDA::cudart
)




add_executable(${PROJECT_NAME} ${demo_srcs})

target_link_libraries(${PROJECT_NAME} ${dependend_LIB})

