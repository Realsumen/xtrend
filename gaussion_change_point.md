# CHANGE POINT DETECTION USING GAUSSIAN PROCESSES

## 高斯过程变点分割 Gaussian Process Change Point Segmentation

### **数据准备**

价格序列 $p(i)_{1:T}$,

变点检测 (CPD) 回看窗口 LBW $l_{\text{lbw}}$, 文中设置为21

变点检测评价 (CPD) 阈值 $\nu$, min. 用来判断优化值是否为变点

片段最大长度 $l_{\max}$ 限制每个片段的最大长度，确保在长时间段内能够检测到潜在的变点，即使市场状态在此期间可能发生变化。

片段最大长度和变点评价阈值配对为：$\{(21, 0.9), (63, 0.95)\}$

片段最小长度 $l_{\min}$, 限制每个片段的最小长度。 确保检测到的变点所形成的片段具有足够的长度，避免将非常短的片段误认为是有效的市场状态或模式。文中设置为5

### 结果

价格序列片段 $\{ p(i)_{t0:t1} \}_{(t0,t1) \in R}$，每个是一个价格序列

### 具体流程

1. **初始化**: $t \leftarrow T$, $t1 \leftarrow T$, regimes $R \leftarrow \emptyset$;
2. **while** $t \geq 0$ **do**
   1. 使用 Matérn 3/2 kernel 在 $p_{-l_{\text{lbw}}:t}$ 拟合高斯过程 (GP)，并计算 marginal likelihood,$L_M$;
   2. 使用变点核 Change-point kernel 在 $p_{-l_{\text{lbw}}:t}$ 拟合变点高斯过程， 并 marginal likelihood, $L_C$ 以及优化后的变点位置 hyperparameter $t_{\text{CPD}}$;
   3. **if** $\frac{L_C}{L_M + L_C} \geq \nu$ **then**
      1. $t0 \leftarrow \lceil t_{\text{CPD}} \rceil$;
      2. **if** $t1 - t0 \geq l_{\min}$ **then**
         1. $R \leftarrow R \cup \{(t0, t1)\}$; // 把窗口添加到集合中
      3. **end if**
      4. $t \leftarrow \lfloor t_{\text{CPD}} \rfloor - 1$;  // 把窗口移动到变点之前
      5. $t1 \leftarrow t$;
   4. **else**
      1. $t \leftarrow t - 1$;
      2. **if** $t1 - t > l_{\max}$ **then**
         1. $t \leftarrow t1 - l_{\max}$;
      3. **end if**
      4. **if** $t1 - t = l_{\max}$ **then**
         1. $R \leftarrow R \cup \{(t, t1)\}$;
         2. $t1 \leftarrow t$;
      5. **end if**
   5. **end if**
3. **end while**

**注**：一个典型的选择可能是 Ornstein–Uhlenbeck 过程，即 Matérn 核 1/2 核。为了与参考工作 [4] 一致，我们使用 Matérn 3/2 核。

## 变点检测原理

change_point_detection,pdf 详细描述了如何使用高斯过程（Gaussian Processes, GP）进行变点检测。以下是该方法的要点解读：

### 背景与数据处理

1. **回归问题**：经典的单变量回归问题形式为 $y(x) = f(x) + \epsilon$，其中 $\epsilon$ 是添加噪声。目标是评估函数 $f$ 和条件概率分布 $p(y|x)$。

2. **时间序列数据**：对于某资产 $i$，数据为时间 $T$ 的收盘价序列 $p_t^{(i)}$。由于金融时间序列通常在均值上非平稳，我们使用算术收益率 $r_t^{(i)}$ 进行分析。

3. **标准化**：为了消除线性趋势并保持一致性，将时间窗口内的收益率标准化，使均值为零，方差为单位。

### 高斯过程回归

1. **高斯过程**：GP回归是一种概率性、非参数的方法，适用于机器学习和时间序列分析。其通过核函数 $k(\cdot)$ 指定 GP，并由一组超参数参数化。尽管 GP 通常使用平稳核函数，但在处理非平稳时间序列时也表现良好。

2. **噪声处理**：假设输出具有噪声方差 $\sigma_n$，高斯过程模型为
   $r_t^{(i)} = f(t) + \epsilon_t$，其中 $f \sim GP(0, k)$，而 $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$

### 核函数选择

   **Matérn 3/2 核函数**：对于噪声较大且不光滑的金融数据，Matérn 3/2 核函数是一种良好的选择。其定义为 $k(x, x') = \sigma_h^2 \left(1 + \frac{\sqrt{3}|x - x'|}{\rho}\right) \exp\left(-\frac{\sqrt{3}|x - x'|}{\rho}\right)$。在代码实现中，可以直接使用 `gpflow.kernels.Matern32()`

### 变点检测

1. **变点模型**：假设时间序列在变点 $c$ 处发生剧变，变点之前的观测值对之后的观测值无信息作用。变点窗口需预先设定，并假设其包含单一变点。使用不同的协方差函数描述变点前后的两个区域，定义为：
   $$
   k_R(x, x') =\begin{cases}k_1(x, x') & x, x' < c \\ k_2(x, x') & x, x' \geq c \\0 & \text{otherwise}\end{cases}
   $$
2. **变点核函数的逼近**：为避免直接优化大量的 GP 参数，通过 sigmoid 函数 $\sigma(x) = \frac{1}{1 + \exp(-s(x - c))}$ 逼近协方差的剧变，从而得到变点核函数：
   $$
   k_C(x, x') = k_1(x, x') \sigma(x) \sigma(x') + k_2(x, x') (1 - \sigma(x)) (1 - \sigma(x'))
   $$

    文中的核函数可以变点核函数通过 `gpflow.kernel.change_point` 构建。

### 优化与应用

1. **参数优化**：通过最大化边际似然估计（Type II maximum likelihood）优化 GP 参数，使用 GPflow 框架和 L-BFGS-B 算法。

2. **变点评分**：通过引入变点核函数的超参数，计算负对数边际似然 **nlml** (Negative Log Marginal Likelihood) 的减少量，衡量数据的非平衡程度。变点评分 $\nu_t^{(i)}$ 计算如下：
   $$
   \nu_t^{(i)} = \frac{1}{1 + \exp\left(-(\text{nlml}_{\xi_C} - \text{nlml}_{\xi_M})\right)}
   $$
   其中 $\text{nlml}_C$ 和 $\text{nlml}_M$ 分别为包含和不包含变点核函数的负对数边际似然。，表达式如下

   $$
   \text{nlml}_{\xi_C} = \min_{\xi_C} \left( \frac{1}{2} \hat{r}^T V^{-1} \hat{r} + \frac{1}{2} \log |V| + \frac{l + 1}{2} \log 2\pi \right)
   $$
   $\xi$ 表示高斯回归的超参数。$V$ 是协方差矩阵，计算方法是
   $$V = K + \sigma_n^2I$$
   其中 $K$ 是核计算出的协方差矩阵。$\sigma_n^2$ 是噪声方差。$I$ 是单位矩阵。

此方法通过高斯过程回归和特定的核函数选择，有效检测金融时间序列中的变点，并评估其对市场变化的影响。
