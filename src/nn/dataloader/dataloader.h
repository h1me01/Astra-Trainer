#pragma once

#include <deque>

#include "../../training_data_format/include.h"
#include "utils.h"

namespace nn::dataloader {

class Dataloader {
  public:
    Dataloader(
        int thread_count,
        std::vector<std::string> filenames,
        std::function<bool(const TrainingDataEntry&)> skip_predicate = nullptr
    )
        : thread_count(thread_count),
          filenames(filenames) {

        utils::validate_files(filenames);

        stream = std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(
            std::max(1, thread_count / 2), filenames, std::ios::in | std::ios::binary, cyclic, skip_predicate
        );
    }

    ~Dataloader() {
        stop_flag.store(true);
        batches_not_full.notify_all();

        for (auto& worker : workers)
            if (worker.joinable())
                worker.join();
    }

    Dataloader(const Dataloader&) = delete;
    Dataloader& operator=(const Dataloader&) = delete;

    void init(int batch_size) {
        this->batch_size = batch_size;

        stop_flag.store(false);
        num_workers.store(0);

        const int num_feature_threads = std::max(1, thread_count - std::max(1, thread_count / 2));
        for (int i = 0; i < num_feature_threads; ++i) {
            workers.emplace_back(&Dataloader::worker_loop, this);
            num_workers.fetch_add(1);
        }
    }

    std::vector<TrainingDataEntry> next() {
        std::unique_lock lock(batch_mutex);
        batches_any.wait(lock, [this]() { //
            return !batches.empty() || num_workers.load() == 0;
        });

        if (!batches.empty()) {
            auto batch = std::move(batches.front());
            batches.pop_front();

            lock.unlock();
            batches_not_full.notify_one();

            return batch;
        }

        return {};
    }

    std::vector<std::string> get_filenames() const { return filenames; }

  private:
    void worker_loop() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(batch_size);

        while (!stop_flag.load()) {
            entries.clear();

            {
                std::unique_lock lock(stream_mutex);
                stream->fill(entries, batch_size);
                if (entries.empty())
                    break;
            }

            {
                std::unique_lock lock(batch_mutex);
                batches_not_full.wait(lock, [this]() {
                    return batches.size() < static_cast<std::size_t>(thread_count + 1) || stop_flag.load();
                });

                batches.emplace_back(std::move(entries));

                lock.unlock();
                batches_any.notify_one();
            }
        }

        num_workers.fetch_sub(1);
        batches_any.notify_one();
    }

  private:
    int batch_size;
    int thread_count;
    bool cyclic = true;
    std::vector<std::string> filenames;

    std::deque<std::vector<TrainingDataEntry>> batches;

    std::mutex batch_mutex;
    std::mutex stream_mutex;

    std::condition_variable batches_not_full;
    std::condition_variable batches_any;

    std::atomic_bool stop_flag;
    std::atomic_int num_workers;

    std::vector<std::thread> workers;
    std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> stream;
};

} // namespace nn::dataloader
