#pragma once

#include "wdl_scheduler.h"

namespace nn::wdl_sched {

struct Constant : public WDLScheduler {
    Constant(float val)
        : WDLScheduler(val) {
        if (val < 0 || val > 1)
            error("Constant WDL Scheduler: value must be between 0 and 1!");
    }

    std::string get_info() const override { return "Constant(val=" + format_number(val) + ")"; }
};

} // namespace nn::wdl_sched
