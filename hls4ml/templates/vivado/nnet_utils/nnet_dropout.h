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

#ifndef NNET_DROPOUT_H_
#define NNET_DROPOUT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_common.h"
#include <cmath>
#include <stdint.h>


namespace nnet {

// 32-bit Fibonacci LFSR (maximal-length polynomial x^32 + x^22 + x^2 + x + 1)
inline ap_uint<32> lfsr_next(ap_uint<32> state) {
    #pragma HLS INLINE
    ap_uint<1> b = state[31] ^ state[21] ^ state[1] ^ state[0];
    return (state << 1) | b;
}

struct dropout_config
{
    // IO size
    static const unsigned n_in = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

// *************************************************
//       Bayesian Dropout
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void dropout(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in])
{
    #pragma HLS PIPELINE

    static ap_uint<32> lfsr_state = 0xACE1u;

    // Threshold for keeping: top 16 bits of LFSR compared against keep_rate * 65536
    data_T keep_rate = 1 - CONFIG_T::drop_rate;
    ap_uint<16> threshold = (ap_uint<16>)(keep_rate * 65536);

    bool random_array[CONFIG_T::n_in];
    RandomNumberLoop: for (int i = 0; i < CONFIG_T::n_in; i++) {
        lfsr_state = lfsr_next(lfsr_state);
        ap_uint<16> rand_val = lfsr_state(31, 16);
        random_array[i] = (rand_val < threshold);
    }
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        data_T zero = {};
        data_T temp = random_array[ii] ? data[ii] : zero;
        res[ii] = temp * keep_rate;
    }
}
}

#endif
