#pragma once

#include "../../misc.h"

namespace nn::wdl_sched {

class WDLScheduler {
  public:
    WDLScheduler(float val)
        : val(val) {}

    virtual void step(int epoch) {}

    float get() const { return val; }

    virtual std::string get_info() const = 0;

  protected:
    float val;
};

} // namespace nn::wdl_sched
