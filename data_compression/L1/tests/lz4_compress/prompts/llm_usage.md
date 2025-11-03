# 大模型辅助使用记录

## 基本信息

- **模型名称**：DeepSeek-V3.2
- **提供方 / 访问方式**：
  - DeepSeek Official Web Platform (https://chat.deepseek.com)
- **使用日期**：2025-10-28
- **项目名称**：HLS压缩算法URAM优化方案

---

## 使用场景 1

### 主要用途
<代码重构>

### 完整 Prompt 内容
```
在实现HLS压缩算法时URAM占用资源过多，LUT占用少，可以改动#pragma HLS BIND_STORAGE variable = dict type = RAM_T2P impl = BRAM这行代码用LUT来存储吗（此处粘贴完整代码）
```

### 模型输出摘要
是的，可以通过修改 #pragma HLS BIND_STORAGE 指令来尝试使用LUT资源代替BRAM/URAM。以下是几种修改方案：
方案1：使用分布式RAM（LUTRAM）
cpp
#pragma HLS BIND_STORAGE variable=dict type=RAM_1P impl=LUTRAM
或者
cpp
#pragma HLS BIND_STORAGE variable=dict type=RAM_T2P impl=LUTRAM
方案2：完全使用LUT实现
cpp
#pragma HLS BIND_STORAGE variable=dict type=RAM_1P impl=distributed
或者
cpp
#pragma HLS BIND_STORAGE variable=dict type=RAM_T2P impl=distributed
方案3：数组分区 + LUT
如果LUT资源仍然紧张，可以结合数组分区：
cpp
#pragma HLS BIND_STORAGE variable=dict type=RAM_1P impl=distributed#pragma HLS ARRAY_PARTITION variable=dict cyclic factor=4 dim=1
方案4：优化字典结构
考虑到你的字典位宽很大（MATCH_LEVEL * c_dictEleWidth），还可以考虑以下优化：
cpp
// 将大字典拆分为多个小字典typedef ap_uint<c_dictEleWidth> uintDict_t;
uintDict_t dict[MATCH_LEVEL][LZ_DICT_SIZE];#pragma HLS BIND_STORAGE variable=dict type=RAM_1P impl=distributed#pragma HLS ARRAY_PARTITION variable=dict complete dim=1
完整修改示例
cpp
template <int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int MATCH_LEVEL = 6,
          int MIN_OFFSET = 1,
          int LZ_DICT_SIZE = 1 << 12,
          int LEFT_BYTES = 64>void lzCompress(hls::stream<ap_uint<8> >& inStream, hls::stream<ap_uint<32> >& outStream, uint32_t input_size) {
    const int c_dictEleWidth = (MATCH_LEN * 8 + 24);
    typedef ap_uint<MATCH_LEVEL * c_dictEleWidth> uintDictV_t;
    typedef ap_uint<c_dictEleWidth> uintDict_t;

    if (input_size == 0) return;
    
    // Dictionary - 使用分布式RAM（LUT实现）
    uintDictV_t dict[LZ_DICT_SIZE];#pragma HLS BIND_STORAGE variable=dict type=RAM_T2P impl=distributed
    // 或者使用: #pragma HLS BIND_STORAGE variable=dict type=RAM_T2P impl=LUTRAM
    
    uintDictV_t resetValue = 0;
    for (int i = 0; i < MATCH_LEVEL; i++) {#pragma HLS UNROLL
        resetValue.range((i + 1) * c_dictEleWidth - 1, i * c_dictEleWidth + MATCH_LEN * 8) = -1;
    }
// 初始化字典
dict_flush:
    for (int i = 0; i < LZ_DICT_SIZE; i++) {#pragma HLS PIPELINE II = 1#pragma HLS UNROLL FACTOR = 14
        dict[i] = resetValue;
    }

    uint8_t present_window[MATCH_LEN];#pragma HLS ARRAY_PARTITION variable = present_window complete
    for (uint8_t i = 1; i < MATCH_LEN; i++) {#pragma HLS PIPELINE off
        present_window[i] = inStream.read();
    }
    // ... 其余代码}

### 人工审核与采纳情况
采用RAM_1P + LUTRAM，但延迟增加，继续向大模型提问。

---

## 使用场景 2

### 主要用途
<代码重构>

### 完整 Prompt 内容
```
RAM_1P + LUTRAM是可行的，但是此时LUT占用多，BRAM占用少，没有很好地利用BRAM资源，导致延迟增加。可以把字典分为一部分用BRAM，一部分用LUT存储吗
```

### 模型输出摘要
是的，您可以将字典分割为两部分，一部分使用BRAM，另一部分使用LUTRAM，这样可以更好地平衡资源使用。以下是几种实现方案：
方案1：按地址范围分割存储
方案2：按哈希值分割存储
方案3：按匹配级别分割存储
方案4：动态混合存储（推荐）

完整实现建议
我建议从方案1开始，它实现简单且效果可预测：
确定分割比例：根据可用BRAM和LUT资源调整 BRAM_DICT_SIZE 和 LUT_DICT_SIZE
替换所有字典访问：将原来的 dict[addr] 替换为 read_dict(addr) 和 write_dict(addr, data)

性能分析：综合后检查时序和资源使用情况
分割比例建议：
如果BRAM资源充足：BRAM_DICT_SIZE = LZ_DICT_SIZE * 3 / 4
如果希望平衡：BRAM_DICT_SIZE = LZ_DICT_SIZE / 2
如果BRAM资源紧张：BRAM_DICT_SIZE = LZ_DICT_SIZE / 4

这种混合存储策略可以充分利用两种存储资源的优势，达到更好的资源平衡。

### 人工审核与采纳情况
采纳方案4：动态混合存储，继续提问。

---

## 使用场景 3

### 主要用途
<代码重构>

### 完整 Prompt 内容
```
采用冷热数据混合存储策略，其余部分代码怎么修改（附完整代码）
```

### 模型输出摘要
基于您提供的代码，我来完善冷热数据混合存储策略的实现。主要问题是您定义了dict_hot和dict_cold但没有实际使用，仍然在使用原来的dict数组。以下是完整的修改方案：
完整修改后的代码
// ========== 冷热数据混合存储策略 ==========
    const int HOT_DICT_SIZE = 512;  // 热数据大小
    const int COLD_DICT_SIZE = LZ_DICT_SIZE - HOT_DICT_SIZE;  // 冷数据大小
    
    // 热数据字典（高频访问）- 使用LUTRAM
    uintDictV_t dict_hot[HOT_DICT_SIZE];#pragma HLS BIND_STORAGE variable = dict_hot type = RAM_1P impl = LUTRAM
    
    // 冷数据字典（低频访问）- 使用BRAM  
    uintDictV_t dict_cold[COLD_DICT_SIZE];#pragma HLS RESOURCE variable = dict_cold core = RAM_T2P_BRAM

    uintDictV_t resetValue = 0;
    for (int i = 0; i < MATCH_LEVEL; i++) {#pragma HLS UNROLL
        resetValue.range((i + 1) * c_dictEleWidth - 1, i * c_dictEleWidth + MATCH_LEN * 8) = -1;
    }
    // 初始化热数据字典
dict_flush_hot:
    for (int i = 0; i < HOT_DICT_SIZE; i++) {#pragma HLS PIPELINE II = 1#pragma HLS UNROLL FACTOR = 16
        dict_hot[i] = resetValue;
    }
    // 初始化冷数据字典
dict_flush_cold:
    for (int i = 0; i < COLD_DICT_SIZE; i++) {#pragma HLS PIPELINE II = 1#pragma HLS UNROLL FACTOR = 16
        dict_cold[i] = resetValue;
    }

    uint8_t present_window[MATCH_LEN];#pragma HLS ARRAY_PARTITION variable = present_window complete
    for (uint8_t i = 1; i < MATCH_LEN; i++) {#pragma HLS PIPELINE off
        present_window[i] = inStream.read();
    }

### 人工审核与采纳情况
采纳混合存储策略，逐步调整展开因子和热数据字典大小，找到最优方案，解决了BRAM占用率过高的问题。

------

## 总结

### 整体贡献度评估

- 大模型在本项目中的总体贡献占比：约 50%
- 主要帮助领域：代码优化策略制定、HLS编译指示语法指导、资源平衡方案设计
- 人工介入与修正比例：约 50%


### 学习收获
HLS存储资源管理：深入理解了BIND_STORAGE编译指示的type和impl参数的组合使用
混合存储策略：学会了如何将大型数据结构分割到不同类型的存储资源中
冷热数据分离：掌握了基于访问频率的动态数据迁移策略
资源平衡技巧：了解了如何根据FPGA资源特性（BRAM vs LUTRAM）优化设计
性能与资源权衡：学会了在不同存储配置下平衡时序性能和资源消耗的方法
HLS编译指示进阶用法：包括RAM_1P vs RAM_T2P的区别、LUTRAM vs distributed的实现差异

---

## 附注

使用记录真实性：本记录真实反映了大模型在HLS压缩算法优化过程中的辅助作用
创新性说明：虽然大模型提供了技术方案指导，但具体的参数调优、方案选择和最终实现均由人工完成
技术验证：所有建议方案均经过实际HLS综合验证，确保技术可行性
