set(BUILD_CUDA YES CACHE BOOL "" FORCE)
set(BUILD_GIT_VERSION YES CACHE BOOL "" FORCE)
set(BUILD_TESTING YES CACHE BOOL "" FORCE)
set(TREAT_WARNINGS_AS_ERRORS NO CACHE BOOL "" FORCE)
set(THIRD_PARTY_MIRROR aliyun CACHE STRING "" FORCE)
set(PIP_INDEX_MIRROR "https://pypi.tuna.tsinghua.edu.cn/simple" CACHE STRING "" FORCE)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
set(CMAKE_GENERATOR Ninja CACHE STRING "" FORCE)
set(CUDA_NVCC_GENCODES "arch=compute_75,code=sm_75" CACHE STRING "" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda CACHE STRING "" FORCE)
set(CUDNN_ROOT_DIR /usr/local/cudnn CACHE STRING "" FORCE)
set(CMAKE_C_COMPILER_LAUNCHER ccache CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache CACHE STRING "" FORCE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld" CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER "/usr/lib64/ccache/nvcc" CACHE STRING "" FORCE)
set(CMAKE_CUDA_HOST_COMPILER "/usr/lib64/ccache/g++" CACHE STRING "")
# set(CMAKE_C_COMPILER_LAUNCHER ccache CACHE STRING "")
# set(CMAKE_CXX_COMPILER_LAUNCHER ccache CACHE STRING "")
# set(CMAKE_CUDA_COMPILER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_C_LINKER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_CXX_LINKER_LAUNCHER ccache CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "")
set(CUDNN_STATIC OFF CACHE BOOL "")
