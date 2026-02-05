cmake_minimum_required(VERSION 3.17)
project(Astra-Trainer VERSION 1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)
add_executable(astra-trainer ${SRCS})

set_target_properties(astra-trainer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_link_libraries(astra-trainer PRIVATE
    CUDA::cudart CUDA::cublas CUDA::cusparse
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(astra-trainer PRIVATE OpenMP::OpenMP_CXX)
endif()
