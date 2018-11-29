# transfer learning

这个博客主要是因为最近看了几篇关于无监督迁移学习在行人重识别领域的论文，发现隔了几天，自己对论文就忘记得差不多了，所以对论文的关键内容做个简单记录。
参考链接：
[Transfer Learning](https://github.com/layumi/DukeMTMC-reID_evaluation/blob/master/State-of-the-art/README.md)

____

## ARN

[Adaptation and Re-Identification Network: An Unsupervised Deep Transfer Learning Approach to Person Re-Identification](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf)
Yu-Jhe Li, Fu-En Yang, Yen-Cheng Liu, Yu-Ying Yeh, Xiaofei Du, and Yu-Chiang Frank Wang, CVPR 2018 Workshop

这篇论文主要分离了数据集的特有特征和行人特征。
作者是台湾人，没有公布代码。有其他人复现了[代码](https://github.com/huanghoujing/ARN)，但是效果很差。
我下一步也会尝试复现一下。

### 网络架构

![ARN的网络架构](./pic/ARN.png)
根据作者的描述，

* $E_I$是resnet50的前四个layer,输入是3X256X256,输出$X^s$是2048X7X7
* $E_T,E_C,E_S$,是相同的网络架构，来自FCN的三层，通过查阅FCN的网络设置，初步猜想是FCN的conv6，conv7，conv8，相应的Decoder暂时按照反卷积来设置。这一部分还需要参考FCN的网络设置。
* $E_T,E_C,E_S$ conv6:7X7X2048,relu6,drop6(0.5),conv7:1X1X2048,relu6,drop6(0.5),conv8:1X1X2048,至于conv6,7的bn和conv8的bn，relu要不要，还需要实验的验证
* 在FCN中，逆卷积的使用方式是 deconv(k=64, s=32, p=0)+crop(offset=19)，参考资料:[FCN学习:Semantic Segmentation](https://zhuanlan.zhihu.com/p/22976342?utm_source=tuicool&utm_medium=referral),[经典网络复现系列（一）：FCN](https://blog.csdn.net/zlrai5895/article/details/80473814)
* 反卷积的时候一般都是k=2n, s=n,
* 参考FCN和pytorch的入门与实践第六章的生成器，我们的Decoder使用deconv(k=1,s=1), deconv(k=1, s=1), deconv(k=7, s=1)
* encoder和decoder都使用bn和relu
* 分类层有dropout
* 学习率，$E_I=10^{-7}, E_T E_C E_S D_C = 10^{-3}, C_S = 2*10^{-3}  $，并且在前几个epoch只更新$E_I$
* 优化器：SGD

### 损失函数

分类损失
$$L_{class}=-\sum_{i=1}^{N_s}y_i^s.log\hat{y}_i^s \tag {1}$$

对比损失
$$L_{ctrs}=\sum_{i,j}{\lambda}(e_{c,i}^s-e_{c,j}^s)^2+ ({1-\lambda}) [max(0, m-(e_{c,i}^s-e_{c,j}^s))]^2 \tag {2}$$

重构误差
$$ L_{rec} = \sum_{i=1}^{N_s} ||X_i^s-\hat{X_i^s}||^2 + \sum_{i=1}^{N_t} ||X_i^t-\hat{X_i^t}||^2 \tag 3 $$

差别损失
$$ L_{diff} = || {H_c^s}^T H_p^s ||_F^2 + || {H_c^t}^T H_p^t ||_F^2 \tag 4 $$

总损失
$$ L_{total} = L_{class} + \alpha*L_{ctrs} + \beta*L_{rec} + \gamma*L_{diff}    \tag 5$$

$$ \alpha=0.01, \beta= 2.0, \gamma=1500 $$

### 模块分析

三个模块:

1. **$ L_{rec} $**
2. **$ L_{class} $和$ L_{ctrs} $**
3. **$ E_T$和$E_S$**

#### 半监督$ L_{rec} $

这里不是很懂这个重构误差损失函数的作用，下面的这个解释也不行。重构损失是半监督损失函数。暂时理解成重构损失保证在获取特征的过程中尽可能减少信息损失。或者说，类似PCA，保留主成分，这个主成分只能保证尽可能地把样本分开。至于这个主成分是否重要，是否有利于分类，不得而知。
参考链接：[深度学习中的“重构”](https://blog.csdn.net/hijack00/article/details/52238549)
作者在这里提示，当只有重构损失函数的时候，应该保持$E_I$不更新，只更新$E_C$.

S: Market, T: Duke; S: Duke, T: Market
| method | rank-1 | mAP | rank-1 | mAP |
|:-:|:-:|:-:|:-:|:-:|
| $L_{rec}$ | 44.5 | 20.3 | 31.2 | 18.4 |

#### 监督$ L_{rec} $, $ L_{class} $和$ L_{ctrs} $

半监督和监督
监督损失使得共享空间捕获到行人语义信息。

S: Market, T: Duke; S: Duke, T: Market
| method | rank-1 | mAP | rank-1 | mAP |
|:-:|:-:|:-:|:-:|:-:|
| w/o $ L_{class} $, $ L_{ctrs} $ | 52.2 | 23.7 | 36.7 | 19.6 |
| w $ L_{class} $, $ L_{ctrs} $ | 70.3 | 39.4 | 60.2 | 33.4 |
| $L_{rec}$ | 44.5 | 20.3 | 31.2 | 18.4 |
| $L_{rec}$, $ L_{class} $和$ L_{ctrs} $ | 60.5 | 28.7 | 48.4 | 26.8 |

#### 无监督$ L_{rec} $, $ E_T $和$ E_S $

特有特征的提取是为了去除共享空间的噪声。
假设共享空间存在，且特有特征空间存在，如果没有特有特征的提取，那么得到的行人特征或多或少地都会包含特征空间的基向量。
当然，这里也隐含了一些假设，共享空间和特有空间一定是线性无关的。空间的基向量是2048维。

S: Market, T: Duke; S: Duke, T: Market
| method | rank-1 | mAP | rank-1 | mAP |
|:-:|:-:|:-:|:-:|:-:|
| w/o  $ E_T $, $ E_S $ | 60.5 | 28.7 | 48.4 | 26.8 |
| w $ L_{class} $, $ L_{ctrs} $ | 70.3 | 39.4 | 60.2 | 33.4 |
| $L_{rec}$ | 44.5 | 20.3 | 31.2 | 18.4 |
| $ L_{rec} $, $ E_T $和$ E_S $ | 52.2 | 23.7 | 36.7 | 19.6 |

____

## HHL

[Generalizing A Person Retrieval Model Hetero- and Homogeneously](https://github.com/zhunzhong07/zhunzhong07.github.io/blob/master/paper/HHL.pdf)
Zhun Zhong, Liang Zheng, Shaozi Li, Yi Yang, ECCV 2018

code: <https://github.com/zhunzhong07/HHL>
web: <http://zhunzhong.site/paper/HHL.pdf>
中文: <http://www.cnblogs.com/Thinker-pcw/p/9787440.html>
