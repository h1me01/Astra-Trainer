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
        : thread_count_(thread_count),
          filenames_(filenames) {

        utils::validate_files(filenames);

        stream_ = std::make_unique<binpack::CompressedTrainingDataEntryParallelReader>(
            std::max(1, thread_count / 2), filenames, std::ios::in | std::ios::binary, true, skip_predicate
        );
    }

    ~Dataloader() {
        stop_flag_.store(true);
        batches_not_full_.notify_all();

        for (auto& worker : workers_)
            if (worker.joinable())
                worker.join();
    }

    Dataloader(const Dataloader&) = delete;
    Dataloader& operator=(const Dataloader&) = delete;

    void init(int batch_size) {
        this->batch_size_ = batch_size;

        stop_flag_.store(false);
        num_workers_.store(0);

        const int num_feature_threads = std::max(1, thread_count_ - std::max(1, thread_count_ / 2));
        for (int i = 0; i < num_feature_threads; ++i) {
            workers_.emplace_back(&Dataloader::worker_loop, this);
            num_workers_.fetch_add(1);
        }
    }

    std::vector<TrainingDataEntry> next() {
        std::unique_lock lock(batch_mutex_);
        batches_any_.wait(lock, [this]() { //
            return !batches_.empty() || num_workers_.load() == 0;
        });

        if (!batches_.empty()) {
            auto batch = std::move(batches_.front());
            batches_.pop_front();

            lock.unlock();
            batches_not_full_.notify_one();

            return batch;
        }

        return {};
    }

    std::vector<std::string> get_filenames() const { return filenames_; }

  private:
    void worker_loop() {
        std::vector<TrainingDataEntry> entries;
        entries.reserve(batch_size_);

        while (!stop_flag_.load()) {
            entries.clear();

            {
                std::unique_lock lock(stream_mutex_);
                stream_->fill(entries, batch_size_);
                if (entries.empty())
                    break;
            }

            {
                std::unique_lock lock(batch_mutex_);
                batches_not_full_.wait(lock, [this]() {
                    return batches_.size() < static_cast<std::size_t>(thread_count_ + 1) || stop_flag_.load();
                });

                batches_.emplace_back(std::move(entries));

                lock.unlock();
                batches_any_.notify_one();
            }
        }

        num_workers_.fetch_sub(1);
        batches_any_.notify_one();
    }

  private:
    int batch_size_;
    int thread_count_;
    std::vector<std::string> filenames_;

    std::deque<std::vector<TrainingDataEntry>> batches_;

    std::mutex batch_mutex_;
    std::mutex stream_mutex_;

    std::condition_variable batches_not_full_;
    std::condition_variable batches_any_;

    std::atomic_bool stop_flag_;
    std::atomic_int num_workers_;

    std::vector<std::thread> workers_;
    std::unique_ptr<binpack::CompressedTrainingDataEntryParallelReader> stream_;
};

} // namespace nn::dataloader
