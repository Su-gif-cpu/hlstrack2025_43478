/*
 * (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#ifndef _XFCOMPRESSION_LZ_COMPRESS_HPP_
#define _XFCOMPRESSION_LZ_COMPRESS_HPP_

/**
 * @file lz_compress.hpp
 * @brief Header for modules used in LZ4 and snappy compression kernels.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "compress_utils.hpp"
#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

namespace xf {
namespace compression {

/**
 * @brief This module reads input literals from stream and updates
 * match length and offset of each literal.
 *
 * @tparam MATCH_LEN match length
 * @tparam MIN_MATCH minimum match
 * @tparam LZ_MAX_OFFSET_LIMIT maximum offset limit
 * @tparam MATCH_LEVEL match level
 * @tparam MIN_OFFSET minimum offset
 * @tparam LZ_DICT_SIZE dictionary size
 *
 * @param inStream input stream
 * @param outStream output stream
 * @param input_size input size
 */
template <int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int MATCH_LEVEL = 6,
          int MIN_OFFSET = 1,
          int LZ_DICT_SIZE = 1 << 12,
          int LEFT_BYTES = 64>
void lzCompress(hls::stream<ap_uint<8> >& inStream, hls::stream<ap_uint<32> >& outStream, uint32_t input_size) {
    const int c_dictEleWidth = (MATCH_LEN * 8 + 24);
    typedef ap_uint<MATCH_LEVEL * c_dictEleWidth> uintDictV_t;
    typedef ap_uint<c_dictEleWidth> uintDict_t;

    if (input_size == 0) return;
    
    // ========== 冷热数据混合存储策略 ==========
    const int HOT_DICT_SIZE = 366;  // 热数据大小
    const int COLD_DICT_SIZE = LZ_DICT_SIZE - HOT_DICT_SIZE;  // 冷数据大小
    
    // 热数据字典（高频访问）- 使用LUTRAM
    uintDictV_t dict_hot[HOT_DICT_SIZE];
#pragma HLS BIND_STORAGE variable = dict_hot type = RAM_2P impl = LUTRAM
    
    // 冷数据字典（低频访问）- 使用BRAM  
    uintDictV_t dict_cold[COLD_DICT_SIZE];
#pragma HLS RESOURCE variable = dict_cold core = RAM_T2P_BRAM

    uintDictV_t resetValue = 0;
    for (int i = 0; i < MATCH_LEVEL; i++) {
#pragma HLS UNROLL
        resetValue.range((i + 1) * c_dictEleWidth - 1, i * c_dictEleWidth + MATCH_LEN * 8) = -1;
    }
    
// 初始化热数据字典
dict_flush_hot:
    for (int i = 0; i < HOT_DICT_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 6
        dict_hot[i] = resetValue;
    }
    
// 初始化冷数据字典
dict_flush_cold:
    for (int i = 0; i < COLD_DICT_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 15
        dict_cold[i] = resetValue;
    }

    uint8_t present_window[MATCH_LEN];
#pragma HLS ARRAY_PARTITION variable = present_window complete
    for (uint8_t i = 1; i < MATCH_LEN; i++) {
#pragma HLS PIPELINE off
        present_window[i] = inStream.read();
    }
    
lz_compress:
    for (uint32_t i = MATCH_LEN - 1; i < input_size - LEFT_BYTES; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = dict_hot inter false
#pragma HLS dependence variable = dict_cold inter false
        uint32_t currIdx = i - MATCH_LEN + 1;
        // shift present window and load next value
        for (int m = 0; m < MATCH_LEN - 1; m++) {
#pragma HLS UNROLL
            present_window[m] = present_window[m + 1];
        }
        present_window[MATCH_LEN - 1] = inStream.read();

        // Calculate Hash Value
        uint32_t hash = 0;
        if (MIN_MATCH == 3) {
            hash = (present_window[0] << 4) ^ (present_window[1] << 3) ^ (present_window[2] << 2) ^
                   (present_window[0] << 1) ^ (present_window[1]);
        } else {
            hash = (present_window[0] << 4) ^ (present_window[1] << 3) ^ (present_window[2] << 2) ^ (present_window[3]);
        }
        // 使用bitmask确保索引在范围内
        hash = hash & (LZ_DICT_SIZE - 1);

        // ========== 冷热数据字典选择逻辑 ==========
        uintDictV_t dictReadValue;
        bool is_hot_region = (hash < HOT_DICT_SIZE);
        
        if (is_hot_region) {
            dictReadValue = dict_hot[hash];
        } else {
            dictReadValue = dict_cold[hash - HOT_DICT_SIZE];
        }

        // 优化：直接构造新条目，减少位操作层数
        uintDict_t newEntry;
        newEntry.range(7, 0) = present_window[0];
        newEntry.range(15, 8) = present_window[1];
        newEntry.range(23, 16) = present_window[2];
        newEntry.range(31, 24) = present_window[3];
        newEntry.range(39, 32) = present_window[4];
        newEntry.range(47, 40) = present_window[5];
        newEntry.range(c_dictEleWidth - 1, 48) = currIdx;
        
        // 优化：使用位拼接操作，一次性完成移位和拼接
        uintDictV_t dictWriteValue = (dictReadValue << c_dictEleWidth) | newEntry;
        
        // ========== 字典更新 - 根据区域选择 ==========
        if (is_hot_region) {
            dict_hot[hash] = dictWriteValue;
        } else {
            dict_cold[hash - HOT_DICT_SIZE] = dictWriteValue;
        }

        // Match search and Filtering - 优化版本
        uint8_t match_length = 0;
        uint32_t match_offset = 0;
        
        for (int l = 0; l < MATCH_LEVEL; l++) {
#pragma HLS UNROLL
            uintDict_t compareWith = dictReadValue.range((l + 1) * c_dictEleWidth - 1, l * c_dictEleWidth);
            uint32_t compareIdx = compareWith.range(c_dictEleWidth - 1, MATCH_LEN * 8);
            
            // 计算连续匹配长度
            uint8_t len = 0;
            for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
                uint8_t cmpByte = compareWith.range((m + 1) * 8 - 1, m * 8);
                if (present_window[m] == cmpByte) {
                    len++;
                } else {
                    break;
                }
            }
            
            // 有效性检查 - 简化为单条语句减少多路复用器
            bool is_valid = (len >= MIN_MATCH) && 
                           (currIdx > compareIdx) &&
                           ((currIdx - compareIdx - 1) < LZ_MAX_OFFSET_LIMIT) &&
                           ((currIdx - compareIdx - 1) >= MIN_OFFSET) &&
                           !(len == 3 && (currIdx - compareIdx - 1) > 4096);
            
            if (is_valid && len > match_length) {
                match_length = len;
                match_offset = currIdx - compareIdx - 1;
            }
        }
        ap_uint<32> outValue = 0;
        outValue.range(7, 0) = present_window[0];
        outValue.range(15, 8) = match_length;
        outValue.range(31, 16) = match_offset;
        outStream << outValue;
    }
    
lz_compress_leftover:
    for (int m = 1; m < MATCH_LEN; m++) {
#pragma HLS PIPELINE
        ap_uint<32> outValue = 0;
        outValue.range(7, 0) = present_window[m];
        outStream << outValue;
    }
    
lz_left_bytes:
    for (int l = 0; l < LEFT_BYTES; l++) {
#pragma HLS PIPELINE
        ap_uint<32> outValue = 0;
        outValue.range(7, 0) = inStream.read();
        outStream << outValue;
    }
}

/**
 * @brief This is stream-in-stream-out module used for lz compression. It reads input literals from stream and updates
 * match length and offset of each literal.
 *
 * @tparam MATCH_LEN match length
 * @tparam MIN_MATCH minimum match
 * @tparam LZ_MAX_OFFSET_LIMIT maximum offset limit
 * @tparam MATCH_LEVEL match level
 * @tparam MIN_OFFSET minimum offset
 * @tparam LZ_DICT_SIZE dictionary size
 *
 * @param inStream input stream
 * @param outStream output stream
 */
template <int MAX_INPUT_SIZE = 64 * 1024,
          class SIZE_DT = uint32_t,
          int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int CORE_ID = 0,
          int MATCH_LEVEL = 6,
          int MIN_OFFSET = 1,
          int LZ_DICT_SIZE = 1 << 12,
          int LEFT_BYTES = 64>
void lzCompress(hls::stream<IntVectorStream_dt<8, 1> >& inStream, hls::stream<IntVectorStream_dt<32, 1> >& outStream) {
    const uint16_t c_indxBitCnts = 24;
    const uint16_t c_fifo_depth = LEFT_BYTES + 2;
    const int c_dictEleWidth = (MATCH_LEN * 8 + c_indxBitCnts);
    typedef ap_uint<MATCH_LEVEL * c_dictEleWidth> uintDictV_t;
    typedef ap_uint<c_dictEleWidth> uintDict_t;
    const uint32_t totalDictSize = (1 << (c_indxBitCnts - 1)); // 8MB based on index 3 bytes
#ifndef AVOID_STATIC_MODE
    static bool resetDictFlag = true;
    static uint32_t relativeNumBlocks = 0;
#else
    bool resetDictFlag = true;
    uint32_t relativeNumBlocks = 0;
#endif

    // ========== 第二个模板函数也应用冷热混合策略 ==========
    const int HOT_DICT_SIZE = 366;
    const int COLD_DICT_SIZE = LZ_DICT_SIZE - HOT_DICT_SIZE;
    
    uintDictV_t dict_hot[HOT_DICT_SIZE];
#pragma HLS BIND_STORAGE variable = dict_hot type = RAM_2P impl = LUTRAM
    
    uintDictV_t dict_cold[COLD_DICT_SIZE];
#pragma HLS RESOURCE variable = dict_cold core = RAM_T2P_BRAM

    // local buffers for each block
    uint8_t present_window[MATCH_LEN];
#pragma HLS ARRAY_PARTITION variable = present_window complete
    hls::stream<uint8_t> lclBufStream("lclBufStream");
#pragma HLS STREAM variable = lclBufStream depth = c_fifo_depth
#pragma HLS BIND_STORAGE variable = lclBufStream type = fifo impl = srl

    // input register
    IntVectorStream_dt<8, 1> inVal;
    // output register
    IntVectorStream_dt<32, 1> outValue;
    // loop over blocks
    while (true) {
        uint32_t iIdx = 0;
        // once 8MB data is processed reset dictionary
        // 8MB based on index 3 bytes
        if (resetDictFlag) {
            ap_uint<MATCH_LEVEL* c_dictEleWidth> resetValue = 0;
            for (int i = 0; i < MATCH_LEVEL; i++) {
#pragma HLS UNROLL
                resetValue.range((i + 1) * c_dictEleWidth - 1, i * c_dictEleWidth + MATCH_LEN * 8) = -1;
            }
        // 初始化热数据字典
        dict_flush_hot:
            for (int i = 0; i < HOT_DICT_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 6
                dict_hot[i] = resetValue;
            }
        // 初始化冷数据字典
        dict_flush_cold:
            for (int i = 0; i < COLD_DICT_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 15
                dict_cold[i] = resetValue;
            }
            resetDictFlag = false;
            relativeNumBlocks = 0;
        } else {
            relativeNumBlocks++;
        }
        // check if end of data
        auto nextVal = inStream.read();
        if (nextVal.strobe == 0) {
            outValue.strobe = 0;
            outStream << outValue;
            break;
        }
    // fill buffer and present_window
    lz_fill_present_win:
        while (iIdx < MATCH_LEN - 1) {
#pragma HLS PIPELINE II = 1
            inVal = nextVal;
            nextVal = inStream.read();
            present_window[++iIdx] = inVal.data[0];
        }
    // assuming that, at least bytes more than LEFT_BYTES will be present at the input
    lz_fill_circular_buf:
        for (uint16_t i = 0; i < LEFT_BYTES; ++i) {
#pragma HLS PIPELINE II = 1
            inVal = nextVal;
            nextVal = inStream.read();
            lclBufStream << inVal.data[0];
        }
        // lz_compress main
        outValue.strobe = 1;

    lz_compress:
        for (; nextVal.strobe != 0; ++iIdx) {
#pragma HLS PIPELINE II = 1
#ifndef DISABLE_DEPENDENCE
#pragma HLS dependence variable = dict_hot inter false
#pragma HLS dependence variable = dict_cold inter false
#endif
            uint32_t currIdx = (iIdx + (relativeNumBlocks * MAX_INPUT_SIZE)) - MATCH_LEN + 1;
            // read from input stream into circular buffer
            auto inValue = lclBufStream.read(); // pop latest value from FIFO
            lclBufStream << nextVal.data[0];    // push latest read value to FIFO
            nextVal = inStream.read();          // read next value from input stream

            // shift present window and load next value
            for (uint8_t m = 0; m < MATCH_LEN - 1; m++) {
#pragma HLS UNROLL
                present_window[m] = present_window[m + 1];
            }

            present_window[MATCH_LEN - 1] = inValue;

            // Calculate Hash Value
            uint32_t hash = 0;
            if (MIN_MATCH == 3) {
                hash = (present_window[0] << 4) ^ (present_window[1] << 3) ^ (present_window[2] << 2) ^
                       (present_window[0] << 1) ^ (present_window[1]);
            } else {
                hash = (present_window[0] << 4) ^ (present_window[1] << 3) ^ (present_window[2] << 2) ^
                       (present_window[3]);
            }
            // 使用mask确保索引在范围内
            hash = hash & (LZ_DICT_SIZE - 1);

            // ========== 冷热数据字典选择逻辑 ==========
            uintDictV_t dictReadValue;
            bool is_hot_region = (hash < HOT_DICT_SIZE);
            
            if (is_hot_region) {
                dictReadValue = dict_hot[hash];
            } else {
                dictReadValue = dict_cold[hash - HOT_DICT_SIZE];
            }
            
            uintDictV_t dictWriteValue = dictReadValue << c_dictEleWidth;
            for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
                dictWriteValue.range((m + 1) * 8 - 1, m * 8) = present_window[m];
            }
            dictWriteValue.range(c_dictEleWidth - 1, MATCH_LEN * 8) = currIdx;
            
            // ========== 字典更新 - 根据区域选择 ==========
            if (is_hot_region) {
                dict_hot[hash] = dictWriteValue;
            } else {
                dict_cold[hash - HOT_DICT_SIZE] = dictWriteValue;
            }

            // Match search and Filtering
            // Comp dict pick
            uint8_t match_length = 0;
            uint32_t match_offset = 0;
            for (int l = 0; l < MATCH_LEVEL; l++) {
#pragma HLS UNROLL
                uint8_t len = 0;
                uintDict_t compareWith = dictReadValue.range((l + 1) * c_dictEleWidth - 1, l * c_dictEleWidth);
                uint32_t compareIdx = compareWith.range(c_dictEleWidth - 1, MATCH_LEN * 8);
                // Optimized match length calculation with early termination
                for (uint8_t m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
                    if (present_window[m] == compareWith.range((m + 1) * 8 - 1, m * 8)) {
                        len++;
                    } else {
                        break;
                    }
                }
                // Validate match conditions
                bool valid_match = (len >= MIN_MATCH) && (currIdx > compareIdx) && 
                                   ((currIdx - compareIdx) < LZ_MAX_OFFSET_LIMIT) &&
                                   ((currIdx - compareIdx - 1) >= MIN_OFFSET) &&
                                   (compareIdx >= (relativeNumBlocks * MAX_INPUT_SIZE));
                if (valid_match && (len == 3) && ((currIdx - compareIdx - 1) > 4096)) {
                    len = 0;
                } else if (!valid_match) {
                    len = 0;
                }
                if (len > match_length) {
                    match_length = len;
                    match_offset = currIdx - compareIdx - 1;
                }
            }
            outValue.data[0].range(7, 0) = present_window[0];
            outValue.data[0].range(15, 8) = match_length;
            outValue.data[0].range(31, 16) = match_offset;
            outStream << outValue;
        }

        outValue.data[0] = 0;
    lz_compress_leftover:
        for (uint8_t m = 1; m < MATCH_LEN; ++m) {
#pragma HLS PIPELINE II = 1
            outValue.data[0].range(7, 0) = present_window[m];
            outStream << outValue;
        }
    lz_left_bytes:
        for (uint16_t l = 0; l < LEFT_BYTES; ++l) {
#pragma HLS PIPELINE II = 1
            outValue.data[0].range(7, 0) = lclBufStream.read();
            outStream << outValue;
        }

        // once relativeInSize becomes 8MB set the flag to true
        resetDictFlag = ((relativeNumBlocks * MAX_INPUT_SIZE) >= (totalDictSize)) ? true : false;
        // end of block
        outValue.strobe = 0;
        outStream << outValue;
    }
}

} // namespace compression
} // namespace xf
#endif // _XFCOMPRESSION_LZ_COMPRESS_HPP_