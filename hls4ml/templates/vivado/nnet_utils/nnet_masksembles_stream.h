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

#ifndef NNET_MASKSEMBLES_STREAM_H_
#define NNET_MASKSEMBLES_STREAM_H_

#include <cmath>
#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include "nnet_stream.h"
#include "nnet_masksembles.h"

namespace nnet {

// *************************************************
//       Masksembles
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void masksembles(
  hls::stream<data_T> &data_stream,
  hls::stream<res_T> &res_stream,
  typename CONFIG_T::weight_t weights[CONFIG_T::n_in*CONFIG_T::num_masks],
  int mask_index) {

    typename data_T::value_type data[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=res complete

    DataPrepare: for(int i_in = 0; i_in < CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
        DataPack: for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data[i_in * data_T::size + i_pack] = data_pack[i_pack];
        }
    }

    MaskLoop: for (int i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        typename data_T::value_type zero = 0;
        res[i] = (weights[mask_index * CONFIG_T::n_filt + i % CONFIG_T::n_filt]) ? data[i] : zero;
    }

    ResWrite: for(unsigned i_out = 0; i_out < CONFIG_T::n_in / res_T::size; i_out++) {
        if (CONFIG_T::n_in / res_T::size > 1) {
            #pragma HLS PIPELINE
        }
        res_T res_pack;
        #pragma HLS DATA_PACK variable=res_pack
        ResPack: for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = res[i_out * res_T::size + i_pack];
        }
        res_stream.write(res_pack);
    }
}
}

#endif
