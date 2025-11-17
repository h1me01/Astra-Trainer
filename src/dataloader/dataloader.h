#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#ifdef _MSC_VER
#pragma warning(push, 0)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif

#include "../training_data_formats/include.h"
#include "training_data_stream.h"

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace dataloader {

inline std::function<bool(const DataEntry &)> skip_predicate = [](const DataEntry &e) {
    if(e.score == 32002) // value none
        return true;
    if(e.ply < 20)
        return true;
    if(e.isCapturingMove() || e.isInCheck())
        return true;

    auto do_wld_skip = [&]() {
        std::bernoulli_distribution distrib(1.0 - e.score_result_prob());
        auto &prng = rng::get_thread_local_rng();
        return distrib(prng);
    };

    if(do_wld_skip())
        return true;

    return false;
};

class Dataloader {
  public:
    Dataloader(                                    //
        const std::vector<std::string> &filenames, //
        int batch_size,                            //
        int concurrency,                           //
        bool cyclic)
        : m_batch_size(batch_size),                     //
          m_concurrency(concurrency), m_cyclic(cyclic), //
          m_stream(training_data::open_sfen_input_file_parallel(
              std::max(1, concurrency / 2), filenames, cyclic, skip_predicate)) //
    {
        m_stop_flag.store(false);

        auto worker = [this]() {
            std::vector<DataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load()) {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    m_stream->fill(entries, m_batch_size);
                    if(entries.empty())
                        break;
                }

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() {
                        return m_batches.size() < static_cast<std::size_t>(m_concurrency + 1) || m_stop_flag.load();
                    });

                    m_batches.emplace_back(std::move(entries));

                    lock.unlock();
                    m_batches_any.notify_one();
                }
            }

            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(1, m_concurrency - std::max(1, m_concurrency / 2));
        for(int i = 0; i < num_feature_threads; ++i) {
            m_workers.emplace_back(worker);
            m_num_workers.fetch_add(1);
        }
    }

    std::vector<DataEntry> next() {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if(!m_batches.empty()) {
            auto batch = std::move(m_batches.front());
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }

        return {};
    }

    ~Dataloader() {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for(auto &worker : m_workers)
            if(worker.joinable())
                worker.join();
    }

  private:
    int m_batch_size;
    int m_concurrency;
    bool m_cyclic;
    std::array<int, 64> m_input_bucket{};

    std::deque<std::vector<DataEntry>> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;

    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

} // namespace dataloader
