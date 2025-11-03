# 大模型辅助使用记录

## 基本信息

- **模型名称**：gpt-5
- **提供方 / 访问方式**：cursor
- **使用日期**：2025.10.31
- **项目名称**：cholesky算子优化

---

## 使用场景 1

### 主要用途
性能优化建议
### 完整 Prompt 内容
```
我现在需要优化Cholesky算法，我们的计算标准是T_exec = Estimated_Clock_Period × Cosim_Latency，尝试降低这两个变量即可，下面这几个是官方给出的优化策略 你先看看我现在是怎么优化的 看看还有没有什么其他优化的空间Cholesky (ARCH0)：
复数定点运算优化
循环流水线设计（关注对角线计算和非对角线更新）
数组分区减少访存冲突
平方根运算延迟优化
利用矩阵对称性减少计算量
```

### 模型输出摘要
你已经在 choleskyBasic 中实现了循环流水、数组完全分区、复数共轭乘法的累加修复，以及通过 x_sqrt/x_rsqrt 优化复数定点开方/倒数开方，这些都能显著改善 T_exec 指标。
对角线和非对角线部分都使用了 sum 缓存并配合 #pragma HLS PIPELINE，结合 L_internal 的内部缓存，能减少访存冲突和重复读取。
ARCH1/ARCH2 中还使用了倒数缓存、循环 tripcount 约束等，这些也会帮助综合器推高时序。
cholesky.hppLines 345-359
        for (int k = 0; k < j; k++) {
#pragma HLS PIPELINE II = 1
            if (LowerTriangularL == true) {
                retrieved_L = L_internal[j][k];
            } else {
                retrieved_L = L_internal[k][j];
            }
            sum[j] += hls::x_conj(retrieved_L) * retrieved_L;
        }
降低 Estimated_Clock_Period 的空间
消除冗余条件：off_diag_loop 目前遍历所有 i 并依赖 if (i > j)，可以改成 for (int i = j + 1; i < RowsColsA; ++i)，减少比较逻辑和控制依赖，帮助 II 下降。
拆分/局部变量化累加：sum[j] 在对角与非对角共享，会产生伪依赖。建议改用局部 sum_diag、sum_off（或将 sum 按第二维 dim=1 分区），并在循环末尾一次性写回，减轻多端口读写压力。
共轭复用：提前缓存 hls::x_conj(L_internal[j][k]) 和 hls::x_conj(L_internal[i][k])，避免同一周期内多次调用共轭函数，提高逻辑共享机会。
明确资源绑定：对频繁的乘法/开方操作加入 #pragma HLS BIND_OP 或 #pragma HLS RESOURCE core=Mul_LUT/SQRT，帮助工具把关键算子放到 DSP 或专用 IP，从而缩短目标时钟周期。
固定点对齐：如果目标位宽允许，考虑在 Traits 中将 PROD_T/ACCUM_T 缩减若干位（尤其复数虚部恒为零的情况），可以减少加法器/乘法器位宽，缩短组合路径。
降低 Cosim_Latency 的空间
外层循环流水化：在 col_loop 顶部尝试 #pragma HLS PIPELINE II=1，同时确保内部 sum 依赖已经打破；即便无法完全达到 II=1，也能让综合器探索跨列重叠。
部分/完全展开：对小规模矩阵（例如 3×3、4×4）可在配置阶段设置 #pragma HLS UNROLL factor=RowsColsA；即使只在 diag_loop 或 off_diag_sum_loop 内做 2/4 级展开，也能显著压缩列延迟。
数据流划分：在 col_loop 内把“对角计算”“非对角更新”拆成两个函数并 #pragma HLS DATAFLOW，通过 channel 连接，让对角值一计算完就驱动后续 off-diagonal 计算。
缓存对称元素：若输入矩阵实际只提供半三角，可在读入阶段一次性构造 A[i][j] 和 A[j][i]，主循环中只访问单一副本，避免重复共轭和条件判断。
顶层流式优化：cholesky 顶层可以把读取、分解、写回三个阶段拆分，在 hls::stream 上用 DATAFLOW 并行，提高协同仿真时的吞吐。
建议的下一步
先针对 choleskyBasic 做循环范围与累加变量的修改，观察调度报告中 II 和时钟周期的变化；必要时再尝试外层流水和数据流重构。
若目标设计只覆盖固定矩阵阶数，评估小规模展开对资源的增量，再决定是否引入。
调整后运行 HLS 综合/仿真，对比 Estimated_Clock_Period 与 Cosim_Latency 的变化，确保数值正确性保持一致。

### 人工审核与采纳情况
<!-- 说明：
采用了共轭复用：提前缓存 hls::x_conj(L_internal[j][k]) 和 hls::x_conj(L_internal[i][k])，避免同一周期内多次调用共轭函数，提高逻辑共享机会。
明确资源绑定：对频繁的乘法/开方操作加入 #pragma HLS BIND_OP 或 #pragma HLS RESOURCE core=Mul_LUT/SQRT，帮助工具把关键算子放到 DSP 或专用 IP，从而缩短目标时钟周期。
固定点对齐：如果目标位宽允许，考虑在 Traits 中将 PROD_T/ACCUM_T 缩减若干位（尤其复数虚部恒为零的情况），可以减少加法器/乘法器位宽，缩短组合路径。
消除冗余条件在进行验证后发现latency升高
-->

---

## 使用场景 2

### 主要用途
性能优化建议

### 完整 Prompt 内容
```
latency升高了 查看综合报告看看是为啥 再好好优化一下 可以大胆一点 不要他原来的框架结构 用自己的结构来
```

### 模型输出摘要
调整说明
换用了 hls::sqrtf 的快速平方根路径：新增了一组 cholesky_fast_sqrt_real、cholesky_get/set_real 辅助函数，把对角元素的开方统一映射到浮点 sqrtf，再写回到原数据类型；同时为每列缓存倒数，避免额外除法。
更新 choleskyAlt 的对角处理逻辑，让平方根与倒数计算都走新的“实数→浮点→实数”通道，并在除零时回退到 0。
保留了循环级别的完全分区和 UNROLL 优化，使得小尺寸矩阵的内层求和仍是并行的。
建议
重新跑一次 csim/csynth/cosim，重点关注 choleskyAlt_false_3_* 的延迟和 x_sqrt_* 是否已消失；如果仍有残余，可考虑再加 #pragma HLS BIND_OP 对 hls::sqrtf 指定更快的 DSqrt IP。
若精度需要更高（特别是在 ap_fixed 场景），可以把 sqrtf 迭代次数提升或改为双精度 hls::sqrt。

### 人工审核与采纳情况
采纳了此建议，latency略有下降

---

## 使用场景 3

### 主要用途
性能优化建议

### 完整 Prompt 内容
```
进行全列展开 (完全矩阵展开)
对于 RowsColsA ≤ 4 的场景，可以直接生成固定流程：
预先读取 A 到寄存器；
使用模板递归或手写展开（L00、L10、L11…）直接构建结果。
```

### 模型输出摘要
新增 choleskySmall，针对 RowsColsA ≤ 4 时采用完全展开的计算路径：所有 j/i/k 循环均 #pragma HLS UNROLL，并对 A/L/L_internal 做完全分区。这样 3×3、4×4 场景的 Cholesky 计算在编译期展开成固定流程，Latency 固定在几十个拍以内。
在 choleskyTop 中加入编译期条件：当矩阵阶数 ≤4 时优先调用 choleskySmall，否则继续走原有 ARCH0/1/2 分支，保持大规模矩阵的通用实现。
其它细节
提前把对角/倒数的辅助函数抽成通用工具（cholesky_fast_sqrt_real、cholesky_get/set_real 等），以便新路径复用，并避免重复 sqrt/div 逻辑。
未修改 kernel 层文件，测试工程保持原设置。
下一步
重新运行 csynth/cosim：kernel_cholesky_0 在 3×3 场景应出现显著更低的 Latency，同时 Slack 也会因为全展开的组合较短而保持正值。
若后续要扩展到其它固定维度，可以在 choleskySmall 中调整 RowsColsA 判断或增加特定优化。

### 人工审核与采纳情况
采纳了此建议，latency大幅度下降
## 总结

### 整体贡献度评估
<!-- 例如：
- 大模型在本项目中的总体贡献占比：约 X%
- 主要帮助领域：代码优化 / 文档撰写 / 调试分析
- 人工介入与修正比例：约 Y%
-->

### 学习收获
<!-- 通过与大模型交互，您学到了哪些新知识或优化技巧？ -->

---

## 附注

- 请确保填写真实、完整的使用记录
- 如未使用大模型辅助，请在此文件中注明"本项目未使用大模型辅助"
- 评审方将参考此记录了解项目的独立性与创新性
