#include "model.h"

namespace model {

void Model::fill_inputs(std::vector<TrainingDataEntry> &ds, float lambda) {
    auto &stm_features = stm_input->get_output();
    auto &nstm_features = nstm_input->get_output();

    const int max_entries = stm_input->get_size();

    for(size_t i = 0; i < ds.size(); i++) {
        const Position &pos = ds[i].pos;

        Square ksq_w = pos.kingSquare(Color::White);
        Square ksq_b = pos.kingSquare(Color::Black);

        bool wtm = pos.sideToMove() == Color::White;
        Bitboard pieces = pos.piecesBB();

        const int offset = i * max_entries;

        int count = 0;
        for(auto sq : pieces) {
            Piece p = pos.pieceAt(sq);
            int w_idx = feature_index(p.type(), p.color(), sq, ksq_w, Color::White);
            int b_idx = feature_index(p.type(), p.color(), sq, ksq_b, Color::Black);

            int idx = offset + count;
            stm_features(idx) = wtm ? w_idx : b_idx;
            nstm_features(idx) = wtm ? b_idx : w_idx;

            count++;
        }

        if(count < max_entries) {
            for(int i = count; i < max_entries; i++) {
                int idx = offset + i;
                stm_features(idx) = -1;
                nstm_features(idx) = -1;
            }
        }

        float score_target = 1.0f / (1.0f + expf(-float(ds[i].score) / params.eval_div));
        float wdl_target = (ds[i].result + 1) / 2.0f;

        targets(i) = lambda * score_target + (1.0f - lambda) * wdl_target;
    }

    stm_features.host_to_dev();
    nstm_features.host_to_dev();
    targets.host_to_dev();
}

void Model::train(std::string output_path, std::string checkpoint_name) {
    init();

    std::string training_folder;
    Logger log;
    int epoch;

    if(checkpoint_name.empty()) {
        std::stringstream new_folder_path;
        new_folder_path << output_path << "/" << name;
        training_folder = new_folder_path.str();

        create_directory(training_folder);

        log.open(training_folder + "/log.txt", false);
        log.write({"epoch", "loss"});

        epoch = 0;
    } else {
        const std::string checkpoint_path = output_path + "/" + checkpoint_name;

        if(!exists(checkpoint_path))
            error("Checkpoint path does not exist: " + checkpoint_path);

        load_weights(checkpoint_path + "/weights.bin");
        optim->load(checkpoint_path);

        epoch = epoch_from_checkpoint(checkpoint_name);
        lr_sched->lr_from_epoch(epoch);

        if(epoch > 0)
            std::cout << "Resuming from epoch " << epoch << " with learning rate " << lr_sched->get_lr() << std::endl;

        training_folder = checkpoint_path.substr(0, checkpoint_path.find_last_of('/'));

        log.open(training_folder + "/log.txt", true);

        loaded_checkpoint = checkpoint_name;
    }

    print_info(training_folder);

    std::cout << "\n================================= Training =================================\n\n";

    const int positions_per_epoch = params.batch_size * params.batches_per_epoch;

    Timer timer;
    for(epoch = epoch + 1; epoch <= params.epochs; epoch++) {
        timer.start();
        loss->reset();

        float lambda = params.lambda_start + (params.lambda_end - params.lambda_start) * (epoch / float(params.epochs));

        for(int batch = 1; batch <= params.batches_per_epoch; batch++) {
            auto data_entries = dataloader->next();
            fill_inputs(data_entries, lambda);

            timer.stop();
            auto elapsed = timer.elapsed_time();

            if(batch == params.batches_per_epoch || timer.is_time_reached(1000)) {
                printf("\repoch/batch = %3d/%4d | loss = %1.8f | pos/sec = %7d | time = %3ds",
                       epoch,
                       batch,
                       loss->get_loss() / (params.batch_size * batch),
                       (int) round(1000.0f * params.batch_size * batch / elapsed),
                       (int) elapsed / 1000);
                std::cout << std::flush;
            }

            network->forward(data_entries);
            loss->compute(targets, network->get_output());
            network->backward();
            optim->step(lr_sched->get_lr(), data_entries.size());
        }

        float epoch_loss = loss->get_loss() / positions_per_epoch;

        timer.stop();
        auto elapsed = timer.elapsed_time();

        printf("\r\033[K"); // clear the current line
        printf("epoch/batch = %3d/%4d | loss = %1.8f | pos/sec = %7d | time = %3ds\n",
               epoch,
               params.batches_per_epoch,
               epoch_loss,
               (int) round(1000.0f * positions_per_epoch / elapsed),
               (int) elapsed / 1000);

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});

        if(epoch % std::max(params.save_rate, 1) == 0 || epoch == params.epochs) {
            std::string suffix = epoch == params.epochs ? "final" : std::to_string(epoch);
            save_checkpoint(training_folder + "/checkpoint_" + suffix);
        }

        lr_sched->step(epoch);
    }
}

} // namespace model
