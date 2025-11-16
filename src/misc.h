#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <stdlib.h>

#define ASSERT(expr)                                                                                                   \
    {                                                                                                                  \
        if(!static_cast<bool>(expr)) {                                                                                 \
            printf("ASSERT: %s\n", #expr);                                                                             \
            printf("    file: %s\n", __FILE__);                                                                        \
            printf("    line: %d\n", __LINE__);                                                                        \
            printf("    func: %s\n", __FUNCTION__);                                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    }

#define CUDA_ASSERT(ans)                                                                                               \
    {                                                                                                                  \
        if(ans != cudaSuccess) {                                                                                       \
            fprintf(stderr, "CUDA_ASSERT: %s %s %d\n", cudaGetErrorString(ans), __FILE__, __LINE__);                   \
            exit(ans);                                                                                                 \
        }                                                                                                              \
    }

inline void error(const std::string &message) {
    std::cerr << "Error: " << message << std::endl;
    std::abort();
}

inline std::string get_activation_name(int type) {
    // clang-format off
    switch(type) 
    {
    case 0: return "Linear";
    case 1: return "ReLU";
    case 2: return "CReLU";
    case 3: return "SRelu";
    case 4: return "SCReLU";
    case 5: return "Sigmoid";
    case 6: return "Tanh";
    default: return "Unknown";
    }
    // clang-format on
}

inline std::string format_number(float num) {
    std::ostringstream oss;
    oss << num;
    return oss.str();
}

class Logger {
  public:
    Logger() {}

    Logger(std::string path) {
        file = std::ofstream(path, std::ios::app);
    }

    void open(std::string path, bool append = false) {
        if(file.is_open())
            file.close();

        if(append)
            file = std::ofstream(path, std::ios::app);
        else
            file = std::ofstream(path);
    }

    ~Logger() {
        if(file.is_open())
            file.close();
    }

    void write(std::initializer_list<std::string> args) {
        for(auto i = args.begin(); i != args.end(); ++i) {
            if(i != args.begin())
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
    void start() {
        start_point = steady_clock::now();
        end_point = steady_clock::time_point();
        prev_duration = 0;
    }

    void stop() {
        end_point = steady_clock::now();
    }

    long long elapsed_time() const {
        return std::chrono::duration_cast<ms>(end_point - start_point).count();
    }

    // returns true if the provided has elapsed since the last call
    bool is_time_reached(long long time) {
        long long elapsed = elapsed_time();
        if(elapsed - prev_duration > time) {
            prev_duration = elapsed;
            return true;
        }

        return false;
    }

  private:
    using ms = std::chrono::milliseconds;
    using steady_clock = std::chrono::steady_clock;

    steady_clock::time_point start_point, end_point;

    long long prev_duration = 0;
};
