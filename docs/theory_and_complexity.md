# STE 理论、收敛界与复杂度分析

本文档给出本项目噪声感知 STE 的理论口径。结论用于解释算法设计与限定可宣称范围，不把有限样本实验外推为对任意网络和噪声分布的普适保证。

## 1. 问题定义与符号

令噪声训练目标为

$$
F(\theta)=\mathbb{E}_{(x,y),\zeta}
\left[\ell\left(f_{\zeta}(x;\theta),y\right)\right],
\tag{1}
$$

其中 $\theta$ 为网络参数，$\zeta$ 表示编程、漂移、保持、温度、串扰、供电、输出读噪声和量化等随机或非光滑非理想因素。第 $t$ 步使用 surrogate gradient $\hat g_t$ 更新

$$
\theta_{t+1}=\theta_t-\eta\hat g_t.
\tag{2}
$$

| 符号 | 含义 |
| --- | --- |
| $F, F_*$ | 噪声训练目标及其下界 |
| $L$ | $F$ 的梯度 Lipschitz 常数 |
| $\hat g_t$ | STE 给出的随机 surrogate gradient |
| $b_t$ | $\hat g_t$ 相对 $\nabla F(\theta_t)$ 的条件偏差 |
| $\sigma_g^2$ | surrogate gradient 的条件方差上界 |
| $D_t$ | 在线 profile 产生的逐通道对角缩放 |
| $d_{\min}$ | 梯度缩放下限，本项目在线方案取 0.5 |
| $R,I,O$ | 矩阵乘的行数、输入维度和输出维度 |
| $K$ | 独立物理读出次数 |

## 2. STE 的偏差与方差

量化和随机采样使真实路径不可导或梯度方差过大。框架令 forward 使用严格噪声算子，backward 使用理想矩阵乘或饱和感知 surrogate。条件于当前参数，定义

$$
\mathbb{E}_t[\hat g_t]=\nabla F(\theta_t)+b_t,
\qquad
\mathbb{E}_t\!\left[\left\|\hat g_t-
\mathbb{E}_t[\hat g_t]\right\|^2\right]\leq\sigma_g^2.
\tag{3}
$$

均方梯度估计误差可分解为

$$
\mathbb{E}_t\!\left[\left\|\hat g_t-\nabla F(\theta_t)\right\|^2\right]
=\|b_t\|^2+
\mathbb{E}_t\!\left[\left\|\hat g_t-
\mathbb{E}_t[\hat g_t]\right\|^2\right].
\tag{4}
$$

式 (4) 解释了本项目为何同时估计系统偏差和随机方差。仅增加读出次数主要降低独立随机项，不能消除由非线性饱和、量化阈值和尺度失配引起的系统偏差。

对通道 $c$ 的 paired clean/noisy 输出 $y_c,\tilde y_c$，令 $e_c=\tilde y_c-y_c$。在线 profile 用信号投影分离两部分：

$$
\rho_{b,c}^2=
\frac{\langle y_c,e_c\rangle^2}{\langle y_c,y_c\rangle^2},
\qquad
\rho_{v,c}^2=
\frac{\|e_c\|^2-
\langle y_c,e_c\rangle^2/\|y_c\|^2}{\|y_c\|^2}.
\tag{5}
$$

实现中对式 (5) 使用 batch/空间维均值、数值下限和 EMA，并设置

$$
d_c=\operatorname{clip}\!\left(
\left(1+\lambda_v\rho_{v,c}^2+
\lambda_b\rho_{b,c}^2\right)^{-1/2},
d_{\min},1\right),
\qquad \hat g_t=D_tg_t^{\mathrm{STE}}.
\tag{6}
$$

式 (6) 不是把物理噪声调小，而是对 backward 的可信度做有界校正；forward 仍使用原始 `noise_scale=1.0`。

## 3. 有界梯度缩放

因为 $d_c\in[d_{\min},1]$，对任意向量 $g$ 有

$$
d_{\min}\|g\|_2\leq\|D_tg\|_2\leq\|g\|_2.
\tag{7}
$$

上界抑制高方差通道，下界避免强噪声通道被完全屏蔽。若未缩放估计器的偏差上界为 $B_0$，且 $\|\nabla F(\theta_t)\|\leq G$，则缩放后偏差可保守界为

$$
\left\|\mathbb{E}_t[D_tg_t^{\mathrm{STE}}]-
\nabla F(\theta_t)\right\|
\leq B_0+(1-d_{\min})G.
\tag{8}
$$

式 (8) 表明缩放不能无限增强：过低的 floor 虽可减少方差，也会增加优化偏差。本项目因此使用 0.5 的在线 floor，并以多 seed 配对统计评价方法，而不依据单 seed 峰值作结论。

## 4. 非凸 SGD 收敛界

作如下标准假设：

1. $F$ 下界为 $F_*$，且为 $L$-smooth。
2. 条件偏差满足 $\|b_t\|\leq B$。
3. 条件方差满足式 (3)，且迭代期望存在。
4. 使用固定步长 $0<\eta\leq 1/(4L)$。

由 smoothness、Young 不等式和
$\mathbb{E}_t\|\hat g_t\|^2\leq
2\|\nabla F(\theta_t)\|^2+2B^2+\sigma_g^2$，可得

$$
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\|\nabla F(\theta_t)\|^2
\leq
\frac{2(F(\theta_0)-F_*)}{\eta T}
+2(1+L\eta)B^2+L\eta\sigma_g^2.
\tag{9}
$$

当估计器无偏，即 $B=0$，选择 $\eta=\mathcal{O}(T^{-1/2})$ 可恢复非凸随机优化常见的 $\mathcal{O}(T^{-1/2})$ 平均驻点界。对有偏 STE，算法收敛到由 $B^2$ 控制的邻域。在线 bias/variance profile、饱和感知修正和激活预条件的目标，分别是降低式 (9) 中的 $B$、$\sigma_g^2$，或避免 forward 进入导致二者增大的强饱和区域。

该界不直接保证分类精度，也不证明当前在线估计器必然减小 $B$。尤其是 $D_t$ 与当前 noisy forward 使用同一 batch 时存在相关性；要得到完全严格的自适应估计器证明，还需额外的鞅差或稳定性条件。因此，式 (9) 是设计依据和性能上限解释，算法收益仍以配对多 seed 实验为准。

## 5. 时间与空间复杂度

对矩阵乘 $[R,I]\times[I,O]$，主计算量为 $\Theta(RIO)$。卷积可取

$$
R=NH_oW_o,\quad
I=C_{in}k_hk_w/G,\quad
O=C_{out}/G,
\tag{10}
$$

并对 $G$ 个 group 求和，因此与标准 grouped convolution 具有相同的渐近 MAC 复杂度。

| 路径 | 每层训练时间 | 主要额外训练内存 | 推理额外成本 |
| --- | --- | --- | --- |
| digital clean | $\Theta(RIO)$ | 标准 autograd 激活 | 无 |
| noisy + plain STE | $\Theta(RIO)$，另有随机采样和逐元素非理想项 | noisy/ideal 中间量，约 $\mathcal{O}(RI+IO+RO)$ | noisy forward 本身 |
| exact $K$-read STE | $\Theta(KRIO)$ | 顺序实现仍需保存各 read 的 backward 状态，最坏随 $K$ 增长 | $K$ 次物理读出 |
| moment-matched $K$-read | $\Theta(RIO)$ | 与单读同阶 | 仅作训练近似，不用于正式推理 |
| online profile STE | 两次 forward + 一次 backward，仍为 $\Theta(RIO)$ | paired clean 输出 $\mathcal{O}(RO)$；持久通道状态 $\mathcal{O}(O)$ | 无，profile 只在训练启用 |

`NoisyConv2d` 的完整 unfold workspace 为
$\mathcal{O}(NH_oW_oC_{in}k_hk_w)$。若每次只处理 $h_c$ 个输出行，峰值 workspace 降为

$$
\mathcal{O}(Nh_cW_oC_{in}k_hk_w),
\tag{11}
$$

计算量不变。shared-read 实现让同一次物理读出的权重侧随机状态跨 chunk 复用，因此 $h_c$ 只控制内存，不再隐式改变物理噪声采样次数。

## 6. RTX 4090 实测效率

运行命令：

```bash
PYTHONPATH=src python scripts/benchmark_efficiency.py \
  --batch-size 8 \
  --warmup-steps 10 \
  --steps 50
```

环境为 RTX 4090、PyTorch 2.6.0、CUDA 12.6，模型为 CIFAR stem ResNet18，输入为固定合成 batch。每个计时步均包含 `zero_grad`、forward、交叉熵、backward 和 SGD update；在线模式还完整包含 paired clean forward、逐通道 profile reduction 与 noisy forward。

| 路径 | step time | 相对 clean | throughput | 增量峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| clean | 4.87 ms | 1.00x | 1643.94 images/s | 35.50 MiB |
| plain STE | 25.71 ms | 5.28x | 311.12 images/s | 278.35 MiB |
| online profile STE | 47.96 ms | 9.86x | 166.80 images/s | 296.45 MiB |

在线 profile 相对 plain STE 的时间倍率为 1.87x，额外增量峰值显存为 18.10 MiB。该结果说明其主要成本来自第二次 forward，而逐通道统计本身只带来较小显存增量。数据见 `runs/efficiency_benchmark.csv`，环境与参数见 `runs/efficiency_benchmark_metadata.json`，统计图见 `docs/figures/efficiency_benchmark.png`。

这是一项训练步骤 microbenchmark，不是端到端数据加载吞吐，也不是独立多次运行的置信区间。报告中的效率结论仅限于该固定硬件、软件版本、模型、batch 和输入尺寸。
