cmake_minimum_required(VERSION 3.17)
project(Astra-Trainer VERSION 1.0 LANGUAGES CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages with proper error handling
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find OpenMP with better error handling
find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP not found. Building without OpenMP support.")
endif()

# Collect source files
file(GLOB_RECURSE SRCS src/*.cu src/*.cpp src/*.h)

# Create executable
add_executable(Astra-Trainer ${SRCS})

# Set CUDA properties
set_target_properties(Astra-Trainer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON)

# Detect CUDA architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Add CUDA include directories
target_include_directories(Astra-Trainer PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Platform-specific compiler flags
if(MSVC)
    if(OpenMP_CXX_FOUND)
        target_compile_options(Astra-Trainer PRIVATE 
            $<$<COMPILE_LANGUAGE:CXX>:/openmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /openmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        )
    else()
        target_compile_options(Astra-Trainer PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        )
    endif()
    target_link_options(Astra-Trainer PRIVATE /NODEFAULTLIB:LIBCMT)
else()
    if(OpenMP_CXX_FOUND)
        target_compile_options(Astra-Trainer PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        )
    else()
        target_compile_options(Astra-Trainer PRIVATE 
            $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
        )
    endif()
endif()

# Link libraries using modern CMake targets
target_link_libraries(Astra-Trainer PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    Threads::Threads
)

# Link OpenMP only if found
if(OpenMP_CXX_FOUND)
    target_link_libraries(Astra-Trainer PRIVATE OpenMP::OpenMP_CXX)
endif()
