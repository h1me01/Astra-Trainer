#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <stdlib.h>

using namespace std::filesystem;

#ifndef NDEBUG
#define ASSERT(expr) ((void)0)
#else
#define ASSERT(expr)                                                                                                   \
    {                                                                                                                  \
        if (!static_cast<bool>(expr)) {                                                                                \
            printf("ASSERT: %s\n", #expr);                                                                             \
            printf("    file: %s\n", __FILE__);                                                                        \
            printf("    line: %d\n", __LINE__);                                                                        \
            printf("    func: %s\n", __FUNCTION__);                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    }
#endif

#define CUDA_ASSERT(ans)                                                                                               \
    {                                                                                                                  \
        if (ans != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA_ASSERT: %s %s %d\n", cudaGetErrorString(ans), __FILE__, __LINE__);                   \
            exit(ans);                                                                                                 \
        }                                                                                                              \
    }

template <typename T>
using SPtr = std::shared_ptr<T>;

template <typename T>
using WPtr = std::weak_ptr<T>;

template <typename T>
using Ptr = std::unique_ptr<T>;

template <typename T, typename U>
auto dpc(U&& ptr) {
    return std::dynamic_pointer_cast<T>(std::forward<U>(ptr));
}

inline void error(const std::string& message) {
    std::cerr << "Error: " << message << std::endl;
    std::abort();
}

inline std::vector<std::string> files_from_paths(const std::vector<std::string>& paths) {
    std::vector<std::string> files;
    for (const auto& path : paths) {
        try {
            for (const auto& entry : recursive_directory_iterator(path)) {
                if (entry.is_regular_file())
                    files.push_back(entry.path().string());
            }
        } catch (const filesystem_error& e) {
            std::cerr << "Filesystem error in path " << path << ": " << e.what() << std::endl;
        }
    }

    return files;
}

class Logger {
  public:
    Logger() {}

    Logger(std::string path) { file = std::ofstream(path, std::ios::app); }

    void open(std::string path, bool append = false) {
        if (file.is_open())
            file.close();

        if (append)
            file = std::ofstream(path, std::ios::app);
        else
            file = std::ofstream(path);
    }

    ~Logger() {
        if (file.is_open())
            file.close();
    }

    void write(std::initializer_list<std::string> args) {
        for (auto i = args.begin(); i != args.end(); ++i) {
            if (i != args.begin())
                file << ",";
            file << *i;
        }

        file << std::endl;
        file.flush();
    }

  private:
    std::ofstream file;
};

class Timer {
  public:
    Timer() { start_point = steady_clock::now(); }

    long long elapsed_time() const {
        return std::chrono::duration_cast<format>(steady_clock::now() - start_point).count();
    }

  private:
    using format = std::chrono::seconds;
    using steady_clock = std::chrono::steady_clock;
    steady_clock::time_point start_point;
};
