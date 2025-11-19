#pragma once

#include <deque>

#include "../training_data_formats/include.h"

namespace dataloader {

class Dataloader {
  public:
    Dataloader(int batch_size,
               int concurrency,
               std::vector<std::string> filenames,
               std::function<bool(const TrainingDataEntry &)> skip_predicate = nullptr)
        : m_batch_size(batch_size), m_concurrency(concurrency), m_filenames(filenames) {

        auto is_not_binpack = [](const std::string &f) {
            return f.size() < 8 || f.compare(f.size() - 8, 8, ".binpack") != 0;
        };

        m_filenames.erase( //
            std::remove_if(m_filenames.begin(), m_filenames.end(), is_not_binpack),
            m_filenames.end());

        m_stream = std::make_unique<CompressedTrainingDataEntryParallelReader>( //
            std::max(1, concurrency / 2),
            filenames,
            std::ios::in | std::ios::binary,
            m_cyclic,
            skip_predicate);

        m_stop_flag.store(false);
        m_num_workers.store(0);

        const int num_feature_threads = std::max(1, m_concurrency - std::max(1, m_concurrency / 2));
        for(int i = 0; i < num_feature_threads; ++i) {
            m_workers.emplace_back(&Dataloader::worker_loop, this);
            m_num_workers.fetch_add(1);
        }
    }

    ~Dataloader() {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for(auto &worker : m_workers)
            if(worker.joinable())
                worker.join();
    }

    // prevent copying
    Dataloader(const Dataloader &) = delete;
    Dataloader &operator=(const Dataloader &) = delete;

    std::vector<TrainingDataEntry> next() {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { //
            return !m_batches.empty() || m_num_workers.load() == 0;
        });

        if(!m_batches.empty()) {
            auto batch = std::move(m_batches.front());
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }

        return {};
    }

    std::vector<std::string> get_filenames() const {
        return m_filenames;
    }

  private:
    void worker_loop() {
        std::vector<TrainingDataEntry> entries;
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
    }

  private:
    int m_batch_size;
    int m_concurrency;
    bool m_cyclic = true;
    std::vector<std::string> m_filenames;

    std::deque<std::vector<TrainingDataEntry>> m_batches;

    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;

    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;

    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers;

    std::vector<std::thread> m_workers;
    std::unique_ptr<CompressedTrainingDataEntryParallelReader> m_stream;
};

} // namespace dataloader
