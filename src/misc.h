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
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std::filesystem;

#define CHECK(expr)                                                                                                    \
    do {                                                                                                               \
        if (!static_cast<bool>(expr)) {                                                                                \
            printf("CHECK failed: %s\n", #expr);                                                                       \
            printf("    file: %s\n", __FILE__);                                                                        \
            printf("    line: %d\n", __LINE__);                                                                        \
            printf("    func: %s\n", __FUNCTION__);                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        cudaError_t result = (expr);                                                                                   \
        if (result != cudaSuccess) {                                                                                   \
            printf("CUDA error: error when calling %s\n", #expr);                                                      \
            printf("    file: %s\n", __FILE__);                                                                        \
            printf("    line: %d\n", __LINE__);                                                                        \
            printf("    error: %s\n", cudaGetErrorString(result));                                                     \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

template <typename T>
using UPtr = std::unique_ptr<T>;

template <typename T>
using SPtr = std::shared_ptr<T>;

inline std::string format_number(float num, int precision = 6) {
    return std::format("{:.{}g}", num, precision);
}

inline void error(const std::string& message) {
    std::cerr << "Error | " << message << std::endl;
    std::abort();
}

inline std::vector<std::string> files_from_folder(std::string path) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : recursive_directory_iterator(path))
            if (entry.is_regular_file())
                files.push_back(entry.path().string());
    } catch (const filesystem_error& e) {
        std::cerr << "Filesystem error in path " << path << ": " << e.what() << std::endl;
    }

    return files;
}

class Logger {
  public:
    Logger() {}
    Logger(std::string path) { file = std::ofstream(path, std::ios::app); }

    ~Logger() {
        if (file.is_open())
            file.close();
    }

    void open(std::string path, bool append = false) {
        if (file.is_open())
            file.close();
        if (append)
            file = std::ofstream(path, std::ios::app);
        else
            file = std::ofstream(path);
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
    using format = std::chrono::milliseconds;
    using steady_clock = std::chrono::steady_clock;
    steady_clock::time_point start_point;
};
