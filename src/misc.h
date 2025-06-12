#pragma once

#include <array>
#include <chrono>
#include <cstdarg>
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

inline std::string getActivationName(int type) {
    // clang-format off
    switch(type) 
    {
    case 0: return "Linear";
    case 1: return "ReLU";
    case 2: return "CReLU";
    case 3: return "SCReLU";
    case 4: return "Sigmoid";
    default: return "Unknown";
    }
    // clang-format on
}

inline std::string formatNumber(float num) {
    std::ostringstream oss;
    oss << num;
    return oss.str();
}

inline int getBucketSize(const std::array<int, 64> &king_bucket) {
    if(king_bucket.empty())
        return 1;

    int max_value = king_bucket[0];
    for(size_t i = 1; i < king_bucket.size(); ++i)
        if(king_bucket[i] > max_value)
            max_value = king_bucket[i];

    return max_value + 1;
}

inline std::vector<std::string> fetchFilesFromPath(const std::string &path) {
    std::cout << "Loading files from folder: " << path << std::endl;

    std::vector<std::string> files;
    try {
        for(const auto &entry : std::filesystem::recursive_directory_iterator(path))
            if(entry.is_regular_file()) {
                files.push_back(entry.path().string());
                std::cout << "Added: " << entry.path() << std::endl;
            }
    } catch(const std::filesystem::filesystem_error &e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }

    if(files.empty()) {
        std::cerr << "No training data found" << std::endl;
        exit(1);
    }

    return files;
}

inline int getNextTrainingIndex(const std::string &output_path) {
    int max_index = 0;

    for(const auto &entry : std::filesystem::directory_iterator(output_path)) {
        if(!entry.is_directory())
            continue;

        std::string folder_name = entry.path().filename().string();
        if(folder_name.find("training_") == 0) {
            int index = std::stoi(folder_name.substr(9));
            max_index = std::max(max_index, index);
        }
    }

    return max_index + 1;
}

class Logger {
  public:
    Logger(std::string path) {
        file = std::ofstream(path);
    }

    ~Logger() {
        file.close();
    }

    void write(std::initializer_list<std::string> args) {
        for(auto i = args.begin(); i != args.end(); ++i) {
            if(i != args.begin())
                file << ",";
            file << "\"" << *i << "\"";
        }

        file << std::endl;
    }

  private:
    std::ofstream file;
};

class Timer {
  private:
    using ms = std::chrono::milliseconds;
    using steady_clock = std::chrono::steady_clock;

    steady_clock::time_point start_point, end_point;

    long long prev_duration = 0;

  public:
    void start() {
        start_point = steady_clock::now();
        end_point = steady_clock::time_point();
        prev_duration = 0;
    }

    void stop() {
        end_point = steady_clock::now();
    }

    long long getElapsedTime() const {
        return std::chrono::duration_cast<ms>(end_point - start_point).count();
    }

    // returns true if the provided has elapsed since the last call
    bool isTimeReached(long long time) {
        long long elapsed = getElapsedTime();
        if(elapsed - prev_duration > time) {
            prev_duration = elapsed;
            return true;
        }

        return false;
    }
};
