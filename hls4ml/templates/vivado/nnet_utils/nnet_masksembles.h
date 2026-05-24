//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_MASKSEMBLES_H_
#define NNET_MASKSEMBLES_H_

#include "ap_fixed.h"
#include "nnet_common.h"
#include <cmath>
#include <random>
#include <stdint.h>


namespace nnet {

struct masksembles_config
{
    // Internal data type definitions
    typedef int weight_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned num_masks = 4;
    static const unsigned n_filt = 1;

    // Resource reuse info
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

// *************************************************
//       Masksembles
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void masksembles(
  data_T data[CONFIG_T::n_in],
  res_T res[CONFIG_T::n_in],
  typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::n_out],
  int mask_index)
{
    #pragma HLS PIPELINE

  for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
    res[ii] = data[ii];
  }
}
}

#endif
