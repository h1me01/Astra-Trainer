#include "model.h"

namespace model {

// Helper

void print_progress(int epoch, int batch, float loss, int pos_count, int time_ms, bool newline = false) {
    if (newline)
        printf("\r\033[K");

    float time_sec = time_ms / 1000.0f;

    printf(
        "\repoch/batch = %3d/%4d | loss = %1.6f | pos/sec = %7d | time = %3ds%s",
        epoch,
        batch,
        loss,
        (int)round(pos_count / time_sec),
        (int)round(time_sec),
        newline ? "\n" : ""
    );

    if (!newline)
        fflush(stdout);
}

int epoch_from_checkpoint(const std::string& checkpoint_name) {
    size_t dash_pos = checkpoint_name.find_last_of('_');
    if (dash_pos == std::string::npos) {
        std::cout << "Could not parse epoch from checkpoint name, starting from epoch 0\n";
        return 0;
    }

    std::string epoch_str = checkpoint_name.substr(dash_pos + 1);
    if (epoch_str == "final") {
        std::cout << "Loading from final checkpoint, starting new training cycle\n";
        return 0;
    }

    try {
        return std::stoi(epoch_str);
    } catch (...) {
        std::cout << "Model: Could not parse epoch from checkpoint name, starting from epoch 0\n";
        return 0;
    }
}

// Model

void Model::init() {
    if (is_initialized)
        return;

    loss = get_loss();
    optim = get_optim().take();
    lr_sched = get_lr_scheduler();
    wdl_sched = get_wdl_scheduler();
    dataloader = get_dataloader();

    if (!loss || !optim || !lr_sched || !wdl_sched)
        error("Model: All components (loss, optimizer, scheduler) must be initialized!");

    targets = Array<float>(config.batch_size, true);

    nn::graph::Graph::BuildContext ctx;
    nn::graph::Graph::BuildContext::active = &ctx;
    auto* output = build();
    nn::graph::Graph::BuildContext::active = nullptr;
    graph = make_ptr<nn::graph::Graph>(output, std::move(ctx));
    network = make_ptr<nn::Network>(*graph.get());

    dataloader->init(config.batch_size);
    network->init(config.batch_size);
    optim->init(network->get_params());

    is_initialized = true;
}

void Model::print_info(int epoch, const std::string& output_path) const {
    std::cout << "\n=============================== Training Data ==============================\n\n";
    const auto& training_files = dataloader->get_filenames();
    if (training_files.empty())
        error("Model: No training data found in the specified paths!");

    for (const auto& f : training_files)
        std::cout << f << std::endl;

    std::cout << "\n=============================== Trainer Info ===============================\n\n";
    std::cout << "Model name:        " << name << std::endl;
    std::cout << "Device:            " << get_device_Info() << std::endl;
    std::cout << "Epochs:            " << config.epochs << std::endl;
    std::cout << "Batch Size:        " << config.batch_size << std::endl;
    std::cout << "Batches/Epoch:     " << config.batches_per_epoch << std::endl;
    std::cout << "Save Rate:         " << config.save_rate << std::endl;
    std::cout << "LR Scheduler:      " << lr_sched->get_info() << std::endl;
    std::cout << "WDL Scheduler:     " << wdl_sched->get_info() << std::endl;
    std::cout << "Output Path:       " << output_path << std::endl;

    if (!loaded_checkpoint.empty())
        std::cout << "Loaded Checkpoint: " << loaded_checkpoint << std::endl;
    else if (!loaded_model.empty())
        std::cout << "Loaded Model:      " << loaded_model << std::endl;

    if (epoch > 0)
        std::cout << "\nResuming from epoch " << epoch << " with learning rate " << lr_sched->get() << std::endl;
}

void Model::next_batch(const std::vector<TrainingDataEntry>& ds) {
    fill_inputs(ds);

    for (auto& input : get_inputs())
        input->get_indices().host_to_dev_async();

    targets.host_to_dev_async();
}

float Model::predict(std::string fen) {
    Position pos;
    pos.set(fen);

    std::vector<TrainingDataEntry> ds{{pos}};

    next_batch(ds);
    network->forward(ds);

    auto& output = network->get_output().get_data();
    output.dev_to_host();

    return output(0);
}

void Model::train(std::string output_path) {
    init();

    std::string training_folder;
    Logger log;
    int epoch = 0;

    if (loaded_checkpoint.empty()) {
        training_folder = output_path.empty() ? name : output_path + "/" + name;
        create_directory(training_folder);
        log.open(training_folder + "/log.txt", false);
        log.write({"epoch", "loss"});
    } else {
        training_folder = loaded_checkpoint.substr(0, loaded_checkpoint.find_last_of('/'));
        std::string checkpoint_name = loaded_checkpoint.substr(loaded_checkpoint.find_last_of('/') + 1);

        epoch = epoch_from_checkpoint(checkpoint_name);
        lr_sched->lr_from_epoch(epoch);
        log.open(training_folder + "/log.txt", true);
    }

    print_info(epoch, training_folder);

    std::cout << "\n================================= Training =================================\n\n";

    const int positions_per_epoch = config.batch_size * config.batches_per_epoch;

    for (epoch = epoch + 1; epoch <= config.epochs; epoch++) {
        Timer timer;
        loss->clear();

        lr_sched->step(epoch);
        wdl_sched->step(epoch);

        for (int batch = 1; batch <= config.batches_per_epoch; batch++) {
            auto data_entries = dataloader->next();
            next_batch(data_entries);

            optim->clear_grads();
            network->forward(data_entries);
            loss->compute(targets, network->get_output());
            network->backward();
            optim->step(lr_sched->get(), data_entries.size());

            if (batch == config.batches_per_epoch || !(batch % 100)) {
                print_progress(
                    epoch,
                    batch,
                    loss->get_loss() / (config.batch_size * batch),
                    config.batch_size * batch,
                    timer.elapsed_time()
                );
            }
        }

        float epoch_loss = loss->get_loss() / positions_per_epoch;
        print_progress(epoch, config.batches_per_epoch, epoch_loss, positions_per_epoch, timer.elapsed_time(), true);

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});
        if (epoch % std::max(config.save_rate, 1) == 0 || epoch == config.epochs)
            save_checkpoint(training_folder + "/checkpoint_" + std::to_string(epoch));
    }
}

} // namespace model
