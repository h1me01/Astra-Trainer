cmake_minimum_required(VERSION 3.18)
project(AstraTrainer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp)
add_executable(astra_trainer ${SRCS})

target_include_directories(astra_trainer PRIVATE include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(astra_trainer PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::nvtx3
)

target_compile_options(astra_trainer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Debug>:   -G -g -O0>
        $<$<CONFIG:Release>: -O3 --use_fast_math -lineinfo>
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        $<$<CONFIG:Debug>:   -g -O0>
        $<$<CONFIG:Release>: -O3>
    >
)

set_target_properties(astra_trainer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
