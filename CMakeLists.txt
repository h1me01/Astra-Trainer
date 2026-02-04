cmake_minimum_required(VERSION 3.17)
project(Astra-Trainer VERSION 1.0 LANGUAGES CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math -funroll-loops -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G -lineinfo")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
        -Xcompiler=-ffast-math,-march=native,-mtune=native,-funroll-loops \
        --ptxas-options=-v")
endif()

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX)

if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)

add_executable(astra-trainer ${SRCS})

set_target_properties(astra-trainer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

target_include_directories(astra-trainer PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(astra-trainer PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Threads::Threads
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(astra-trainer PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET astra-trainer POST_BUILD
        COMMAND ${CMAKE_STRIP} --strip-unneeded $<TARGET_FILE:astra-trainer>
        COMMENT "Stripping symbols from release binary"
    )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
