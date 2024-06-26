# CHANGE POINT DETECTION USING GAUSSIAN PROCESSES

change_point_detection,pdf 详细描述了如何使用高斯过程（Gaussian Processes, GP）进行变点检测。以下是该方法的要点解读：

## 变点检测原理

### 背景与数据处理

1. **回归问题**：经典的单变量回归问题形式为 $y(x) = f(x) + \epsilon$，其中 $\epsilon$ 是添加噪声。目标是评估函数 $f$ 和条件概率分布 $p(y|x)$。

2. **时间序列数据**：对于某资产 $i$，数据为时间 $T$ 的收盘价序列 $p_t^{(i)}$。由于金融时间序列通常在均值上非平稳，我们使用算术收益率 $r_t^{(i)}$ 进行分析。

3. **标准化**：为了消除线性趋势并保持一致性，将时间窗口内的收益率标准化，使均值为零，方差为单位。

### 高斯过程回归

1. **高斯过程**：GP回归是一种概率性、非参数的方法，适用于机器学习和时间序列分析。其通过核函数 $k(\cdot)$ 指定 GP，并由一组超参数参数化。尽管 GP 通常使用平稳核函数，但在处理非平稳时间序列时也表现良好。

2. **噪声处理**：假设输出具有噪声方差 $\sigma_n$，高斯过程模型为
   $r_t^{(i)} = f(t) + \epsilon_t$，其中 $f \sim GP(0, k)$，而 $\epsilon \sim \mathcal{N}(0, \sigma_n^2)$

### 核函数选择

1. **Matérn 3/2 核函数**：对于噪声较大且不光滑的金融数据，Matérn 3/2 核函数是一种良好的选择。其定义为 $k(x, x') = \sigma_h^2 \left(1 + \frac{\sqrt{3}|x - x'|}{\rho}\right) \exp\left(-\frac{\sqrt{3}|x - x'|}{\rho}\right)$。

### 变点检测

1. **变点模型**：假设时间序列在变点 $c$ 处发生剧变，变点之前的观测值对之后的观测值无信息作用。变点窗口需预先设定，并假设其包含单一变点。使用不同的协方差函数描述变点前后的两个区域，定义为：
   $$
   k_R(x, x') =\begin{cases}k_1(x, x') & x, x' < c \\ k_2(x, x') & x, x' \geq c \\0 & \text{otherwise}\end{cases}
   $$
2. **变点核函数的逼近**：为避免直接优化大量的 GP 参数，通过 sigmoid 函数 $\sigma(x) = \frac{1}{1 + \exp(-s(x - c))}$ 逼近协方差的剧变，从而得到变点核函数：
   $$
   k_C(x, x') = k_1(x, x') \sigma(x) \sigma(x') + k_2(x, x') (1 - \sigma(x)) (1 - \sigma(x'))
   $$

### 优化与应用

1. **参数优化**：通过最大化边际似然估计（Type II maximum likelihood）优化 GP 参数，使用 GPflow 框架和 L-BFGS-B 算法。

2. **变点评分**：通过引入变点核函数的超参数，计算负对数边际似然的减少量，衡量数据的非平衡程度。变点评分 $\gamma_t^{(i)}$ 计算如下：
   $$
   \gamma_t^{(i)} = \frac{1}{1 + \exp\left(-(\text{nlml}_C - \text{nlml}_M)\right)}
   $$
   其中 $\text{nlml}_C$ 和 $\text{nlml}_M$ 分别为包含和不包含变点核函数的负对数边际似然。，表达式如下

   $$
   \text{nlml}_{\xi_C} = \min_{\xi_C} \left( \frac{1}{2} \hat{r}^T V^{-1} \hat{r} + \frac{1}{2} \log |V| + \frac{l + 1}{2} \log 2\pi \right)
   $$
   $\xi$ 表示高斯回归的超参数。$V$ 是协方差矩阵，计算方法是
   $$V = K + \sigma_n^2I$$
   其中 $K$ 是核计算出的协方差矩阵。$\sigma_n^2$ 是噪声方差。$I$ 是单位矩阵。

此方法通过高斯过程回归和特定的核函数选择，有效检测金融时间序列中的变点，并评估其对市场变化的影响。

## GPflow 自定义Kernal

### 基础概念

GPflow是一个基于TensorFlow的高斯过程（Gaussian Process）库，主要用于构建和优化高斯过程模型。它提供了灵活且强大的工具，用于处理机器学习中的非参数贝叶斯方法。高斯过程是一种用于回归和分类任务的统计模型，可以提供预测的置信度估计。

Kernel（核函数）：核函数是高斯过程的关键组件，它定义了数据点之间的相似性。不同的核函数可以捕捉不同的信号特征。常见的核函数包括RBF核（径向基核）、Matern核、线性核等。核函数的作用是计算两个输入数据点之间的相似度，从而影响高斯过程的协方差矩阵。

Model（模型）：模型是高斯过程的具体实现，它将核函数与数据结合起来，进行训练和预测。在GPflow中，常见的模型包括gpflow.models.GPR（高斯过程回归）和gpflow.models.SVGP（稀疏变分高斯过程）。模型利用核函数计算训练数据的协方差矩阵，然后通过最大化对数似然或变分下界等方法进行参数优化。

### 自定义核用来Chang point Detection

接下来，我们通过继承基类`gpflow.kernels.Kernel`来创建一个新的布朗运动核类，并实现以下三个函数：

1. **\__init__**：构造函数。在这个简单的例子中，构造函数不需要参数（尽管可以方便地传入初始值，例如方差）。它必须用适当的参数调用父类的构造函数。布朗运动只在一维空间中定义，为了简化假设活跃维度为[0]。

    我们使用了`Parameter`类来添加一个参数。使用这个类可以在计算核函数时使用该参数，并且该参数会自动被识别用于优化（或MCMC）。在这里，方差参数初始化为1，并被约束为正值。

2. **K**：这是实现核函数的地方。它接收两个参数，X和X2。按照惯例，我们使第二个参数可选（默认为None）。

    在K函数内部，所有的计算必须使用TensorFlow。在这里我们使用了`tf.minimum`。当GPflow执行K函数时，X和X2将是TensorFlow张量，而参数如`self.variance`也表现得像TensorFlow张量。

3. **K_diag**：这个便捷函数允许GPflow在预测时节省内存。它只是K函数的对角线，在X2为None的情况下。它必须返回一个一维向量，所以我们使用TensorFlow的reshape命令。

以下是具体实现代码：

```python
class Brownian(gpflow.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.variance = gpflow.Parameter(1.0, transform=positive())
        
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return self.variance * tf.minimum(X, tf.transpose(X2))
    
    def K_diag(self, X):
        return tf.reshape(self.variance * X, [-1])
```

这个代码创建了一个新的布朗运动核类，具有一个可优化的方差参数，并且实现了核矩阵的计算函数和对角线计算函数。这样，我们就可以在GPflow中使用这个新的核函数来构建高斯过程模型了。

