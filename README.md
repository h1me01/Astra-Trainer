# Astra-Trainer

Super fast CUDA-Neural-Network Trainer for my chess engine Astra.

## Features

- Supports data loading from Stockfish's binpack files
- CUDA-accelerated neural network training
- Optimized for chess engine development

## Building

### Prerequisites
- CMake
- CUDA Toolkit

### Clone Repository
```bash
git clone https://github.com/h1me01/Astra-Trainer.git
cd Astra-Trainer
```

### Build Release
```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target Astra-Trainer --config Release --parallel 4
./build-release/release/astra-trainer
```

## Credits

- [CudAD](https://github.com/Luecx/CudAD)
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch)
