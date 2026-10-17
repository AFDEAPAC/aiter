// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "rmsnorm_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    RMSNORM_QUANT_PYBIND;
    m.def("rmsnorm_dual_ptq",
          &aiter::rmsnorm_dual_ptq,
          py::arg("out"),
          py::arg("q_out"),
          py::arg("q_scale"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("epsilon"));
}
