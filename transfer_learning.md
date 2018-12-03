# transfer learning

这个博客主要是因为最近看了几篇关于无监督迁移学习在行人重识别领域的论文，发现隔了几天，自己对论文就忘记得差不多了，所以对论文的关键内容做个简单记录。

参考链接: [Transfer Learning](https://github.com/layumi/DukeMTMC-reID_evaluation/blob/master/State-of-the-art/README.md)

因为在某些情况下，图片或者公式无法正常显示，所以，我基本会同步到我的博客
<https://tjjtjjtjj.github.io/2018/11/29/person-reid-transfer-learning/#more>
____

## ARN

[Adaptation and Re-Identification Network: An Unsupervised Deep Transfer Learning Approach to Person Re-Identification](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w6/Li_Adaptation_and_Re-Identification_CVPR_2018_paper.pdf)

Yu-Jhe Li, Fu-En Yang, Yen-Cheng Liu, Yu-Ying Yeh, Xiaofei Du, and Yu-Chiang Frank Wang, CVPR 2018 Workshop

这篇论文主要分离了数据集的特有特征和行人特征，从而使不同数据集的行人特征投射到统一特征空间中。

作者是台湾人，没有公布代码。有其他人复现了[代码](https://github.com/huanghoujing/ARN)，但是效果很差。

我下一步也会尝试复现一下。

### 网络架构

![ARN的网络架构](./pic/ARN/ARN.png)

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
$$ L_{total} = L_{class} + \alpha L_{ctrs} + \beta L_{rec} + \gamma L_{diff} \tag {5} $$

其中
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

| method    |rank-1| mAP  |rank-1| mAP  |
|:---------:|:----:|:----:|:----:|:----:|
| $L_{rec}$ | 44.5 | 20.3 | 31.2 | 18.4 |

#### 监督$ L_{rec} $, $ L_{class} $和$ L_{ctrs} $

半监督和监督

监督损失使得共享空间捕获到行人语义信息。

S: Market, T: Duke; S: Duke, T: Market

| method                                 |rank-1| mAP  |rank-1| mAP  |
|:-:                                     |:-:   |:-:   |:-:   |:-:   |
| w/o $ L_{class} $, $ L_{ctrs} $        | 52.2 | 23.7 | 36.7 | 19.6 |
| w $ L_{class} $, $ L_{ctrs} $          | 70.3 | 39.4 | 60.2 | 33.4 |
| $L_{rec}$                              | 44.5 | 20.3 | 31.2 | 18.4 |
| $L_{rec}$, $ L_{class} $和$ L_{ctrs} $ | 60.5 | 28.7 | 48.4 | 26.8 |

#### 无监督$ L_{rec} $, $ E_T $和$ E_S $

特有特征的提取是为了去除共享空间的噪声。

假设共享空间存在，且特有特征空间存在，如果没有特有特征的提取，那么得到的行人特征或多或少地都会包含特征空间的基向量。

当然，这里也隐含了一些假设，共享空间和特有空间一定是线性无关的。空间的基向量是2048维。

S: Market, T: Duke; S: Duke, T: Market

| method                        |rank-1| mAP  |rank-1| mAP  |
|:-:                            |:-:   |:-:   |:-:   |:-:   |
| w/o  $ E_T $, $ E_S $         | 60.5 | 28.7 | 48.4 | 26.8 |
| w $ L_{class} $, $ L_{ctrs} $ | 70.3 | 39.4 | 60.2 | 33.4 |
| $L_{rec}$                     | 44.5 | 20.3 | 31.2 | 18.4 |
| $ L_{rec} $, $ E_T $和$ E_S $ | 52.2 | 23.7 | 36.7 | 19.6 |

____

## HHL

[Generalizing A Person Retrieval Model Hetero- and Homogeneously](https://github.com/zhunzhong07/zhunzhong07.github.io/blob/master/paper/HHL.pdf)

Zhun Zhong, Liang Zheng, Shaozi Li, Yi Yang, ECCV 2018

code: <https://github.com/zhunzhong07/HHL>

web: <http://zhunzhong.site/paper/HHL.pdf>

中文: <http://www.cnblogs.com/Thinker-pcw/p/9787440.html>

preson-reid中主要面临的问题：

1. 数据集之间的差异
2. 数据集内部摄像头的差异

解决方法：

1. 相机差异：利用StarGAN进行风格转化
2. 数据集差异：将源域/目标域图片视为负匹配

数据集之间的三元组损失有把不同数据集的行人特征映射到同一特征空间的效果。

### 网络架构

![HHL的网络架构](./pic/HHL/HHL.png)

网络的简要介绍

* CNN是resnet50，网络包括两个分支，一个计算源数据集的分类损失，一个计算相似度学习的triplet损失。
* FC-2014的组成：linear(2048，1024)-->bn(1024)-->relu-->dropout(0.5),相当于一个embedding。
* FC-#ID是linear(1024,751), FC-128是linear(1024, 128), 两个分支的具体情况是：
* * x1-->linear(2048, 1024)-->x2-->bn(1024)-->x3-->relu-->x4-->dropout(0.5)-->x5-->linear(102, 751)-->x6
* * x1-->linear(2048, 1024)-->x2-->bn(1024)-->x3-->relu-->x4-->linear(1024, 128)
* 网络的triplet损失是Batch Hard Triplet Loss
* 网络的输入设置：在每一个batch中，对于分类损失，source domain随机选取batchsize=128张图片，对于triplet损失，source domain随机选取8个人的共batchsize=64张图片，其中连续的8张图片属于同一个人，target domain随机选取batchsize=16个人的共16X9=144张图片，假设这16个人都是不同的人。实验发现，当source domain的分类损失的图片比较少的时候，无法实现预期效果，其他情况下没有测试。当batchsize是这样的配比时，可以达到作者的效果。理由未知．
* starGAN是离线训练
* 学习率设置：base：$10^{-1}$，其他：$10^{-2}$，并且每过40个epoch，学习率阶梯性地乘以0.1.一共训练60个epoch就可以达到预期效果，这部分设置和PCB很类似。不知道是经验还是怎么。
* 关于StarGAN待自己复现之后再做进一步解释，现在只复现过StyleGAN。
* triplet损失的margin=0.3

### 损失函数

分类损失
$$L_{cross}=-\sum_{i=1}^{N_s}y_i^s.log\hat{y}_i^s$$

triplet损失
$$L_T=\sum_{x_a, x_p, x_n}[D_{x_a, x_p}+m-D_{x_a, x_n}]_+$$

相机不变性的triplet损失

目标域中一张原始图片作为anchor，StarGAN图片为positive，其他图片为negative
$$L_C=L_T(({x_t^i})_{i=1}^{n_t}\bigcup(x_{t*}^i)_{i=1}^{n_t^*})$$

域不变性的triplet损失

源域中一张图片为anchor，同一id的其他图片作为positive，目标域的任一图片为negative
$$L_D=L_T((x_s^i)_{i=1}^{n_s}\bigcup(x_t^i)_{i=1}^{n_t})$$

相机不变性和域不变性的triplet损失

是将相机不变性和域不变性合为一体，源域的positive不变，negative为源域的其他图片和目标域的图片，目标域的positive不变，negative为源域的图片和目标域的其他行人图片

$$L_{CD}=L_T((x_s^i)_{i=1}^{n_s}\bigcup(x_t^i)_{i=1}^{n_t}\bigcup{x_{t*}^i\}_{i=1}^{n_t^*)}$$

总损失：
$$L_{HHL}=L_{cross}+\beta*L_{CD}$$

其中：
$$\beta=0.5$$

### 模块分析

1. **starGAN**
2. **sample方法**

#### starGAN

在源数据集上训练，在目标数据集上测试不同图像增强方法下的图片距离，通过表格可以得出，预训练的模型对于目标数据集的随机翻转等等有很好的鲁棒性，但是，对于不同摄像头的同一个人，其距离还是很大。因此，利用StarGAN和相机不变性的triplet损失来减少由于摄像头带来的偏差。

| Source | Target | Random Crop | Random Flip | CamStyle Transfer |
|:------:|:------:|:-----------:|:-----------:|:-----------------:|
|Duke    |Market  | 0.049       | 0.034       |0.485              |
|Market  |Duke    | 0.059       | 0.044       |0.614              |

#### sample方法

对于目标域的取样方法，对比了三种方法的性能，分别是随机取样、聚类取样、有监督取样，通过下图可以看出，这三种方法的性能是一样的，最后，作者给的代码是随机取样。

![sample](./pic/HHL/HHL2.png)

### 实验设置

#### Camera style transfer model：StarGAN

使用StarGAN进行对于摄像头风格转化。

* 2 conv + 6 residual + 2 transposed
* input 128X64
* Adam $\beta_1=0.5, \beta_2=0.999$
* 数据初始化:随机翻转和随机裁剪
* 学习率：前100个epoch为0.0001，后100个epoch线性衰减到0

#### Re-ID model training

* 设置可以参考Zhong, Z., Zheng, L., Zheng, Z., Li, S., Yang, Y.: Camera style adaptation for person re-identification
* input 256*128
* 数据初始化：随机裁剪和随机翻转
* dropout=0.5
* 学习率：新增的层：0.1，base：0.01，每隔40个epoch乘以0.1
* mini-batch：源域上对于IDE为128，对于tripletloss是64.目标域上对于triplet loss是16.
* epoch=60
* 测试：2048-dim计算欧式距离

### 超参数设置

* triplet loss的权重$\beta$
* 一个batch中目标域上$n_t$的个数

#### 参数的设置$\beta$

![$\beta$参数的设置](./pic/HHL/HHL3.png)

$\beta$应该设置成0.4-0.8

#### 参数的设置$n_t$

![$n_t$参数的设置](./pic/HHL/HHL4.png)

$n_t$在当前设置(源域上对于IDE为128，对于tripletloss是64)下，应该$n_t>16$

通过上述参数的设置，结合自己实验时的错误，不妨这么理解，在固定mini-batch=128的情况下

* 首先引入源域的triplet\_loss，并调整batch和$\beta$，使效果达到最优，,batch的选取2倍数的等间隔，$\beta$可以取等间隔，最后batch=64，即128/2=64，$\beta$则可以先固定成某个值.
* 然后引入目标域的triplet\_loss，并且要先考虑只有目标域的性能，再考虑结合的性能，每次都需要重新考虑$\beta$和batch的大小
* 这么一想，这篇论文做的实验还是很多的。

### 实验结果

![实验结果](./pic/HHL/HHL5.png)

通过结果我们看出来，其实提升的效果主要来源于$L_C$，说明预训练的模型对于目标域不同摄像头的图片鲁棒性很差。

是否说明预训练的模型只学习到了源数据集的跨摄像头的不变行人特征，而对于目标域的摄像头下的不同风格很敏感，而对目标域的同一摄像头下的行人特征很鲁棒。

$L_T$的提升效果很小是否可以说明目标数据集与源数据集的行人特征空间本身就已经很好地重合了，假如tripl\_loss真得具有将不同数据集的行人特征映射到同一特征空间的效果的话。

通过这篇论文，我们能学到的东西很多，比如对比实验，参数设置实验，想法验证实验等等。

### 补充：triplet\_loss

发现triplet_loss很厉害的样子，不妨看看是个什么情况。

参考链接：
[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)

[Re-ID with Triplet Loss](http://www.itkeyword.com/doc/2025902251705572502/re-id-with-triplet-loss)

[In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)

[code](https://github.com/VisualComputingInstitute/triplet-reid)

[Triplet Loss and Online Triplet Mining in TensorFlow](https://omoindrot.github.io/triplet-loss)这个博客讲述了triplet\_loss的起源、发展和具体使用的几种形式。最后的结论是应该使用在线的batch hard策略。

[Re-ID with Triplet Loss](http://www.itkeyword.com/doc/2025902251705572502/re-id-with-triplet-loss)这篇博客则逻辑性地介绍了各种triplet\_loss的变体。最后的结论是batch hard+soft margin效果更好。

也有提及到，triplet\_loss总是不如分类损失强。