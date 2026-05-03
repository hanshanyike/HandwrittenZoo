# Multi-Head Latent Attention (MLA)

## 算法简介
多头潜在注意力（Multi-Head Latent Attention, MLA）是 DeepSeek-V2/V3 的核心创新之一。它通过低秩联合压缩 Key 和 Value，并引入解耦 RoPE，将推理时的 KV Cache 压缩数十倍（如 DeepSeek-V2 中约 93.3%），同时保持甚至超越标准 MHA 的模型能力。

## 核心思想
1. **低秩 K-V 联合压缩**: 不直接缓存高维的多头 K/V，而是将隐藏状态压缩到一个低维潜在向量 $c_t^{KV}$，推理时只缓存 $c_t^{KV}$。
2. **Query 低秩压缩**: 训练时对 Q 也做低秩压缩，减少激活显存。
3. **解耦 RoPE**: 将位置信息单独放在一小部分 Q/K（$q^R, k^R$）上并施加 RoPE，而压缩后的 $k^C$ 不携带位置信息。这样 $W^{UK}$ 可在推理时吸收进 $W^Q$，避免显式展开 K。

## 数学公式
### 低秩压缩
Query 压缩（训练时省显存）：

$$
c_t^Q = W^{DQ} h_t, \quad q_t^C = W^{UQ} c_t^Q
$$

KV 联合压缩（推理缓存核心）：

$$
c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}
$$

$$
k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}
$$

### 解耦 RoPE
带位置信息的解耦部分：

$$
q_t^R = \text{RoPE}(W^{QR} c_t^Q), \quad k_t^R = \text{RoPE}(W^{KR} h_t)
$$

拼接后参与注意力：

$$
q_{t,i} = [q_{t,i}^C;\, q_{t,i}^R], \quad k_{t,i} = [k_{t,i}^C;\, k_t^R]
$$

注意力输出（V 只来自 $v^C$）：

$$
o_{t,i} = \sum_{j \le t} \text{Softmax}_j\left(\frac{q_{t,i}^T k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^C
$$

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n^2 \cdot d_{model})$，与 MHA 同阶。
- **空间复杂度（推理 KV Cache）**: $O(b \cdot n \cdot (d_c + d_h^R) \cdot l)$。
  - DeepSeek-V2 中约为 $(512 + 64) \cdot l = 576 \cdot l$ 每 token。
  - 相比 MHA 的 $2 \cdot 128 \cdot 128 \cdot l = 32768 \cdot l$，压缩约 **56 倍**。
- **与替代方案对比**:
  - MHA: 能力最强，Cache 最大。
  - GQA: Cache 中等，能力中等。
  - MQA: Cache 最小，能力可能下降。
  - MLA: Cache 接近 MQA 量级，能力优于 MHA。

## 面试高频考点
1. **MLA 为什么能大幅减少 KV Cache？**
   **答案**: 标准 MHA 缓存 $n_h$ 组 K/V；MLA 将 K/V 联合压缩到低维潜在向量 $c_t^{KV}$（如 512 维），推理时只缓存 $c_t^{KV}$ 和解耦的 $k_t^R$（如 64 维），总量从 $2 n_h d_h$ 降到 $d_c + d_h^R$。

2. **什么是“矩阵吸收”（absorption）优化？**
   **答案**: 由于 $k^C = W^{UK} c^{KV}$，注意力分数中的 $q^C (W^{UK} c^{KV})^T$ 可结合为 $(q^C W^{UK}) (c^{KV})^T$。推理时预计算 $W^{UQ} W^{UK}$ 的乘积，避免每次显式展开 K，进一步节省计算。

3. **为什么 MLA 需要“解耦 RoPE”？**
   **答案**: 若对压缩后的 $k^C$ 直接施加 RoPE，RoPE 矩阵会夹在 $W^{UK}$ 与 $c^{KV}$ 之间，导致 $W^{UK}$ 无法被吸收。解耦 RoPE 将位置信息放到独立的 $k^R$ 上，$k^C$ 保持位置无关，从而保证吸收优化的正确性。

4. **MLA 的 KV Cache 与 GQA 的对比？**
   **答案**: DeepSeek-V2 的 MLA Cache 约等价于 2.25 个分组的 GQA，但模型能力优于标准 MHA。这是 MLA 的核心优势：用极小的 Cache 获得极强的表达能力。

## 代码解析
- `w_dq`, `w_uq`: Query 的低秩下投影与上投影。
- `w_dkv`, `w_uk`, `w_uv`: KV 联合压缩的下投影，以及 K/V 的上投影。
- `w_qr`, `w_kr`: 解耦 RoPE 的 Q/K 投影。
- `_apply_rope`: 简化的旋转位置编码实现，对张量最后两维做交替旋转。
- `forward`:
  1. 分别计算 $q^C, q^R, c^{KV}, k^C, v^C, k^R$。
  2. 对 $q^R, k^R$ 应用 RoPE。
  3. 拼接 $[q^C; q^R]$ 和 $[k^C; k^R]$ 计算注意力。
  4. V 只使用 $v^C$，最终经 `w_o` 投影输出。

## 参考资料
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2025)
- [Multi-Head Latent Attention 详解](https://blog.csdn.net/zyctimes/article/details/159170189)
- [Notes on MLA](https://newfacade.github.io/notes-on-llm/techniques/mla.html)
