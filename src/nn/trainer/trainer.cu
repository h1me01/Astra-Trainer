#include "../utils/utils.h"
#include "trainer.h"

using namespace std::filesystem;

namespace nn {

void Trainer::save_checkpoint(const std::string &path) {
    try {
        create_directories(path);
    } catch(const filesystem_error &e) {
        error("Failed creating directory " + path + ": " + e.what());
    }

    network->save_weights(path);
    network->save_quantized_weights(path);

    if(optim != nullptr)
        optim->save(path);

    std::cout << "Saved checkpoint" << std::endl;
}

void Trainer::train(std::vector<std::string> data_path, std::string output_path, std::string checkpoint_name) {
    const std::vector<std::string> files = utils::files_from_path(data_path);

    std::string training_folder;
    Logger log;
    int epoch;

    if(checkpoint_name.empty()) {
        std::cout << "No checkpoint path provided, training from scratch\n";

        std::stringstream new_folder_path;
        new_folder_path << output_path << "/" << model->get_name();
        training_folder = new_folder_path.str();

        create_directory(training_folder);
        std::cout << "Created folder: " << training_folder << std::endl;

        log.open(training_folder + "/log.txt", false);
        log.write({"epoch", "loss"});

        epoch = 0;
    } else {
        std::cout << "Loading checkpoint from " << checkpoint_name << " ..." << std::endl;
        const std::string checkpoint_path = output_path + "/" + checkpoint_name;

        if(!exists(checkpoint_path)) {
            error("Checkpoint path does not exist: " + checkpoint_path);
        }

        load_weights(checkpoint_path + "/weights.bin");
        optim->load(checkpoint_path);
        std::cout << std::endl;

        epoch = utils::epoch_from_checkpoint(checkpoint_name);
        lr_sched->lr_from_epoch(epoch);

        if(epoch > 0)
            std::cout << "Resuming from epoch " << epoch << " with learning rate " << lr_sched->get_lr() << std::endl;

        training_folder = checkpoint_path.substr(0, checkpoint_path.find_last_of('/'));

        std::cout << "Using existing folder: " << training_folder << std::endl;
        log.open(training_folder + "/log.txt", true);

        loaded_checkpoint = checkpoint_name;
    }

    print_info(output_path);

    // init dataloader
    Dataloader dataloader(files, params.batch_size, params.thread_count, true);

    std::cout << "\n================================ Training ================================\n\n";

    const int positions_per_epoch = params.batch_size * params.batches_per_epoch;

    Timer timer;
    for(epoch = epoch + 1; epoch <= params.epochs; epoch++) {
        timer.start();
        loss->reset();

        float lambda = params.lambda_start + (params.lambda_end - params.lambda_start) * (epoch / float(params.epochs));

        for(int batch = 1; batch <= params.batches_per_epoch; batch++) {
            batch_data::data_entries = dataloader.next();
            ASSERT(batch_data::data_entries.size() == (size_t) params.batch_size || //
                   batch_data::data_entries.size() == 1);

            network->fill_inputs(batch_data::data_entries, lambda, params.eval_div);

            timer.stop();
            auto elapsed = timer.elapsed_time();

            if(batch == params.batches_per_epoch || timer.is_time_reached(1000)) {
                printf("\repoch/batch = %3d/%4d, loss = %1.8f, pos/sec = %7d, time = %3ds",
                       epoch,
                       batch,
                       loss->get_loss() / (params.batch_size * batch),
                       (int) round(1000.0f * params.batch_size * batch / elapsed),
                       (int) elapsed / 1000);
                std::cout << std::flush;
            }

            network->forward();
            loss->compute(network->get_targets(), network->get_output());
            network->backward();
            optim->step(lr_sched->get_lr(), batch_data::data_entries.size());
        }

        float epoch_loss = loss->get_loss() / positions_per_epoch;

        timer.stop();
        auto elapsed = timer.elapsed_time();

        printf("\r\033[K"); // clear the current line
        printf("epoch/batch = %3d/%4d, loss = %1.8f, pos/sec = %7d, time = %3ds\n",
               epoch,
               params.batches_per_epoch,
               epoch_loss,
               (int) round(1000.0f * positions_per_epoch / elapsed),
               (int) elapsed / 1000);

        log.write({std::to_string(epoch), std::to_string(epoch_loss)});

        if(epoch % params.save_rate == 0 || epoch == params.epochs) {
            std::string suffix = epoch == params.epochs ? "final" : std::to_string(epoch);
            save_checkpoint(training_folder + "/checkpoint-" + suffix);
        }

        lr_sched->step(epoch);
    }
}

} // namespace nn
