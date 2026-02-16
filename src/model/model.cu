#include "model.h"

namespace model {

void print_progress(int epoch, int batch, float loss, int pos_per_sec, int time_sec, bool newline = false) {
    if (newline)
        printf("\r\033[K");

    printf(
        "\repoch/batch = %3d/%4d | loss = %1.6f | pos/sec = %7d | time = %3ds%s",
        epoch,
        batch,
        loss,
        pos_per_sec,
        time_sec,
        newline ? "\n" : ""
    );

    if (!newline)
        fflush(stdout);
}

void Model::init() {
    if (is_initialized)
        return;

    targets = Array<float>(config.batch_size, true);

    network = std::make_unique<nn::Network>();
    stm_input = std::make_shared<nn::Input>(32);
    nstm_input = std::make_shared<nn::Input>(32);

    dataloader = std::make_unique<Dataloader>(
        config.batch_size, config.thread_count, get_training_files(), [this](const TrainingDataEntry& e) {
            return filter_entry(e);
        }
    );

    loss = get_loss();
    optim = get_optim();
    lr_sched = get_lr_scheduler();
    wdl_sched = get_wdl_scheduler();

    if (!loss || !optim || !lr_sched)
        error("All components (loss, optimizer, scheduler) must be initialized!");

    network->set_output(build(stm_input, nstm_input));

    network->init(config.batch_size);
    stm_input->init(config.batch_size);
    nstm_input->init(config.batch_size);
    optim->init(network->get_params());

    is_initialized = true;
}

void Model::print_info(int epoch, const std::string& output_path) const {
    std::cout << "\n=============================== Training Data ==============================\n\n";
    const auto& training_files = dataloader->get_filenames();
    if (training_files.empty())
        error("No training data found in the specified paths!");

    for (const auto& f : training_files)
        std::cout << f << std::endl;

    std::cout << "\n=============================== Trainer Info ===============================\n\n";
    std::cout << "Model name:        " << name << std::endl;
    std::cout << "Epochs:            " << config.epochs << std::endl;
    std::cout << "Batch Size:        " << config.batch_size << std::endl;
    std::cout << "Batches/Epoch:     " << config.batches_per_epoch << std::endl;
    std::cout << "Save Rate:         " << config.save_rate << std::endl;
    std::cout << "Thread Count:      " << config.thread_count << std::endl;
    std::cout << "Eval Div:          " << config.eval_div << std::endl;
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

void Model::fill_inputs(std::vector<TrainingDataEntry>& ds) {
    auto& stm_features = stm_input->get_output();
    auto& nstm_features = nstm_input->get_output();

    const int max_entries = stm_input->get_size();

    for (size_t i = 0; i < ds.size(); i++) {
        const Position& pos = ds[i].pos;

        const Color stm = pos.sideToMove();
        const Square ksq_stm = pos.kingSquare(stm);
        const Square ksq_nstm = pos.kingSquare(!stm);

        const int offset = i * max_entries;

        Bitboard pieces = pos.piecesBB();

        int count = 0;
        for (auto sq : pieces) {
            Piece p = pos.pieceAt(sq);

            int idx = offset + count;
            stm_features(idx) = feature_index(p.type(), p.color(), sq, ksq_stm, stm);
            nstm_features(idx) = feature_index(p.type(), p.color(), sq, ksq_nstm, !stm);

            count++;
        }

        if (count < max_entries) {
            for (int i = count; i < max_entries; i++) {
                int idx = offset + i;
                stm_features(idx) = -1;
                nstm_features(idx) = -1;
            }
        }

        float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / config.eval_div));
        float wdl_target = (ds[i].result + 1) / 2.0f;

        targets(i) = wdl_sched->get() * score_target + (1.0f - wdl_sched->get()) * wdl_target;
    }

    stm_features.host_to_dev_async();
    nstm_features.host_to_dev_async();
    targets.host_to_dev_async();
}

float Model::predict(const std::string& fen) {
    Position pos;
    pos.set(fen);

    std::vector<TrainingDataEntry> ds{{pos}};

    fill_inputs(ds);
    network->forward(ds);

    auto& output = network->get_output().get_data();
    output.dev_to_host();

    return output(0) * config.eval_div;
}

void Model::train(const std::string& output_path) {
    init();

    std::string training_folder;
    Logger log;
    int epoch = 0;

    if (loaded_checkpoint.empty()) {
        training_folder = output_path + "/" + name;
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
        loss->reset();

        wdl_sched->step(epoch);

        for (int batch = 1; batch <= config.batches_per_epoch; batch++) {
            auto data_entries = dataloader->next();
            fill_inputs(data_entries);

            network->clear_all_grads(optim.get());
            network->forward(data_entries);
            loss->compute(targets, network->get_output());
            network->backward();
            optim->step(lr_sched->get(), data_entries.size());

            if (batch == config.batches_per_epoch || !(batch % 100)) {
                auto elapsed = timer.elapsed_time();

                print_progress(
                    epoch,
                    batch,
                    loss->get_loss() / (config.batch_size * batch),
                    round(config.batch_size * batch / (float)std::max(elapsed, 1LL)),
                    elapsed
                );
            }
        }

        float epoch_loss = loss->get_loss() / positions_per_epoch;

        auto elapsed = timer.elapsed_time();

        print_progress(
            epoch,
            config.batches_per_epoch,
            epoch_loss,
            round(positions_per_epoch / (float)std::max(elapsed, 1LL)),
            elapsed,
            true
        );

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});

        if (epoch % std::max(config.save_rate, 1) == 0 || epoch == config.epochs)
            save_checkpoint(training_folder + "/checkpoint_" + std::to_string(epoch));

        lr_sched->step(epoch);
    }
}

} // namespace model
