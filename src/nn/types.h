#pragma once

#include <memory>

namespace nn {

class Input;
class LayerBase;
class Loss;
class Optimizer;
class FeatureTransformer;
class Affine;
class LRScheduler;

using InputPtr = std::shared_ptr<Input>;
using LayerPtr = std::shared_ptr<LayerBase>;
using AffinePtr = std::shared_ptr<Affine>;
using FeatureTransformerPtr = std::shared_ptr<FeatureTransformer>;
using LossPtr = std::shared_ptr<Loss>;
using OptimizerPtr = std::shared_ptr<Optimizer>;
using LRSchedulerPtr = std::shared_ptr<LRScheduler>;

} // namespace nn
