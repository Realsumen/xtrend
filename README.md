# x-trend 预测网络

![模型结构](assets/README/image.png)

## Inputs

$s$：资产的类别信息，用于提供时间序列的类别信息

$\xi$: 资产的波动率加权回报率序列

$x$: 一个趋势序列

![回报序列](assets/README/image-2.png)
![x特征序列](assets/README/image-1.png)

## 输出

***We want to model the NEXT-DAY VOLATILITY scaled return.***

