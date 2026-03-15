#include "trainer.h"

namespace trainer {

// Helper

void print_progress(int epoch, int batch, float loss, int pos_count, int time_ms, bool new_line = false) {
    if (new_line)
        printf("\r\033[K");

    float time_sec = time_ms / 1000.0f;

    printf(
        "\repoch/batch = %3d/%4d | loss = %1.6f | pos/sec = %7d | time = %3ds%s",
        epoch,
        batch,
        loss / batch,
        (int)round(pos_count / time_sec),
        (int)round(time_sec),
        new_line ? "\n" : ""
    );

    if (!new_line)
        fflush(stdout);
}

int epoch_from_checkpoint(const std::string checkpoint) {
    std::string name = std::filesystem::path(checkpoint).parent_path().filename().string();

    size_t dash_pos = name.find_last_of('_');
    if (dash_pos == std::string::npos) {
        std::cout << "Trainer: Could not parse epoch from checkpoint name, starting from epoch 0\n";
        return 0;
    }

    std::string epoch_str = name.substr(dash_pos + 1);
    if (epoch_str == "final") {
        std::cout << "Trainer: Loading from final checkpoint, starting new training cycle\n";
        return 0;
    }

    try {
        return std::stoi(epoch_str);
    } catch (...) {
        std::cout << "Trainer: Could not parse epoch from checkpoint name, starting from epoch 0\n";
        return 0;
    }
}

// Trainer

void Trainer::load_checkpoint(const std::string& path) {
    if (!exists(path))
        error("Trainer: Checkpoint path does not exist: " + path);

    try {
        model_.load_params(path + "/model.bin");
        optim_->load(path);
        current_epoch_ = epoch_from_checkpoint(path);

        std::cout << "Trainer: Loaded checkpoint from " << path << std::endl;
    } catch (const std::exception& e) {
        error("Trainer: Failed loading checkpoint from " + path + ": " + e.what());
    }
}

void Trainer::save_checkpoint(const std::string& path) {
    try {
        create_directories(path);
    } catch (const filesystem_error& e) {
        error("Trainer: Failed creating directory " + path + ": " + e.what());
    }

    model_.save_params(path + "/model.bin");
    model_.save_quantized_params(path + "/quantized_model.nnue");
    optim_->save(path);

    std::cout << "Trainer: Saved checkpoint to " << path << std::endl;
}

void Trainer::fit(const std::string output_path) {
    std::string training_folder = output_path.empty() ? config_.name : output_path + "/" + config_.name;

    if (!exists(training_folder)) {
        try {
            create_directory(training_folder);
        } catch (const filesystem_error& e) {
            error("Trainer: Failed creating directory " + training_folder + ": " + e.what());
        }
    } else {
        error("Trainer: Output directory already exists: " + training_folder);
    }

    Logger log;
    log.open(training_folder + "/log.txt", false);
    log.write({"epoch", "loss"});

    print_info(current_epoch_, training_folder);

    std::cout << "\n================================= Training =================================\n\n";

    for (; current_epoch_ < config_.epochs; current_epoch_++) {
        Timer timer;

        lr_sched_->step(current_epoch_);
        wdl_sched_->step(current_epoch_);

        loss_->clear();

        const int display_epoch = current_epoch_ + 1;

        for (int batch = 1; batch <= config_.batches_per_epoch; batch++) {
            auto current_data = dataloader_->next();
            prepare_batch(current_data);

            optim_->zero_grads();
            model_.forward(current_data);
            loss_->compute(model_.output(), targets_);
            model_.backward();
            optim_->step(lr_sched_->get(), current_data.size());

            if (batch % 100 == 0) {
                print_progress(display_epoch, batch, loss_->get(), config_.batch_size * batch, timer.elapsed_time());
            }
        }

        float epoch_loss = loss_->get();

        print_progress(
            display_epoch,
            config_.batches_per_epoch,
            epoch_loss,
            config_.batch_size * config_.batches_per_epoch,
            timer.elapsed_time(),
            true
        );

        log.write(
            {std::to_string(display_epoch),
             std::to_string(epoch_loss / config_.batches_per_epoch)}
        );

        if (display_epoch % std::max(config_.save_rate, 1) == 0 || display_epoch == config_.epochs)
            save_checkpoint(training_folder + "/checkpoint_" + std::to_string(display_epoch));
    }
}

void Trainer::prepare_batch(const std::vector<TrainingDataEntry>& batch) {
    model_.prepare_inputs(batch);

    for (size_t i = 0; i < batch.size(); i++) {
        float score_target = sigmoid(batch[i].score / config_.eval_scale);
        float wdl_target = (batch[i].result + 1) / 2.0f;
        targets_(i) = wdl_sched_->get() * wdl_target + (1.0f - wdl_sched_->get()) * score_target;
    }

    model_.inputs_to_dev();
    targets_.host_to_dev();
}

} // namespace trainer
