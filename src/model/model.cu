#include "model.h"

namespace model {

void print_progress(int epoch, int batch, float loss, int pos_per_sec, int time_sec, bool newline = false) {
    if (newline)
        printf("\r\033[K");

    printf(
        "\repoch/batch = %3d/%4d | loss = %1.8f | pos/sec = %7d | time = %3ds%s",
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
    std::cout << "Learning Rate:     " << config.lr << std::endl;
    std::cout << "Eval Div:          " << config.eval_div << std::endl;
    std::cout << "Lambda Start:      " << config.lambda_start << std::endl;
    std::cout << "Lambda End:        " << config.lambda_end << std::endl;
    std::cout << "Output Path:       " << output_path << std::endl;

    if (!loaded_checkpoint.empty())
        std::cout << "Loaded Checkpoint: " << loaded_checkpoint << std::endl;
    else if (!loaded_model.empty())
        std::cout << "Loaded Model:      " << loaded_model << std::endl;

    if (epoch > 0)
        std::cout << "\nResuming from epoch " << epoch << " with learning rate " << lr_sched->get_lr() << std::endl;
}

void Model::fill_inputs(std::vector<TrainingDataEntry>& ds, float lambda) {
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

        targets(i) = lambda * score_target + (1.0f - lambda) * wdl_target;
    }

    stm_features.host_to_dev();
    nstm_features.host_to_dev();
    targets.host_to_dev();
}

float Model::predict(const std::string& fen) {
    Position pos;
    pos.set(fen);

    std::vector<TrainingDataEntry> ds{{pos}};

    fill_inputs(ds, 0.0f);
    network->forward(ds);

    auto& output = network->get_output().get_output();
    output.dev_to_host();

    return output(0) * config.eval_div;
}

void Model::train(const std::string& output_path, const std::string& checkpoint_name) {
    init();

    std::string training_folder;
    Logger log;
    int epoch = 0;

    if (checkpoint_name.empty()) {
        training_folder = output_path + "/" + name;
        create_directory(training_folder);

        log.open(training_folder + "/log.txt", false);
        log.write({"epoch", "loss"});
    } else {
        const std::string checkpoint_path = output_path + "/" + checkpoint_name;

        if (!exists(checkpoint_path))
            error("Checkpoint path does not exist: " + checkpoint_path);

        load_params(checkpoint_path + "/model.bin");
        optim->load(checkpoint_path);

        epoch = epoch_from_checkpoint(checkpoint_name);
        lr_sched->lr_from_epoch(epoch);

        training_folder = checkpoint_path.substr(0, checkpoint_path.find_last_of('/'));
        log.open(training_folder + "/log.txt", true);

        loaded_checkpoint = checkpoint_name;
    }

    print_info(epoch, training_folder);

    std::cout << "\n================================= Training =================================\n\n";

    const int positions_per_epoch = config.batch_size * config.batches_per_epoch;

    Timer timer;
    for (epoch = epoch + 1; epoch <= config.epochs; epoch++) {
        timer.start();
        loss->reset();

        float lambda = config.lambda_start + (config.lambda_end - config.lambda_start) * (epoch / float(config.epochs));

        for (int batch = 1; batch <= config.batches_per_epoch; batch++) {
            auto data_entries = dataloader->next();
            fill_inputs(data_entries, lambda);

            network->forward(data_entries);
            loss->compute(targets, network->get_output());
            network->backward();
            optim->step(lr_sched->get_lr(), data_entries.size());
            network->clear_grads();

            timer.stop();
            auto elapsed = timer.elapsed_time();

            if (batch == config.batches_per_epoch || !(batch % 100)) {
                print_progress(
                    epoch,
                    batch,
                    loss->get_loss() / (config.batch_size * batch),
                    (int)round(1000.0f * config.batch_size * batch / elapsed),
                    (int)elapsed / 1000
                );
            }
        }

        float epoch_loss = loss->get_loss() / positions_per_epoch;
        timer.stop();
        auto elapsed = timer.elapsed_time();

        print_progress(
            epoch,
            config.batches_per_epoch,
            epoch_loss,
            (int)round(1000.0f * positions_per_epoch / elapsed),
            (int)elapsed / 1000,
            true
        );

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});

        if (epoch % std::max(config.save_rate, 1) == 0 || epoch == config.epochs)
            save_checkpoint(training_folder + "/checkpoint_" + std::to_string(epoch));

        lr_sched->step(epoch);
    }
}

} // namespace model
