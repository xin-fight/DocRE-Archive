# Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling

**ATLOP — 关系抽取：解决 - 多实体，多label**

| <font style="background: Aquamarine">adaptive thresholding - adaptive-threshold loss</font> | <font style="background: Aquamarine"> localized context pooling</font> | <font style="background: Aquamarine">contrastive learning</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |



在本文中，我们提出了**自适应阈值化**和**局部上下文池化**两种新的技术来解决 ==**多标签和多实体问题**==

> 自适应阈值处理用可学习的实体相关阈值代替了先前工作中用于多标签分类的全局阈值
>
> 本地化的上下文池直接将注意力从预先训练好的语言模型转移到定位有利于决定关系的相关上下文上



> group bilinear
>
> 按组划分后发现：矩阵w在每一组中进行成，代码中是 - 先每一组的向量成，再将每组融合，之后再进行矩阵w的乘
>
> 
>
> 一个长句子被划分为两个短句，分别进行注意力，那之后的结果如何得到
>
> ```
> 之前的output2，mask2都是要将被分成两个句子合并时，免去重复区域的累加带来的影响
> !!!但是对于att只是简单的相加，然后归一化，有一下问题：
>     设：前512个单词和后512个单词之间重复单词A; 前512个单词和后512个单词中不在A中的分别为B，C
>     1. B中的单词和C中的单词之间的attention没法计算
>     2. A中的单词之间的attention在计算att1 + att2时被重复计算了
> ```

## 问题描述

> **multi-entity - 每个实体在不同的实体对中都有相同的表示，这可能会带来来自无关上下文的噪声** 
>
> **multi-label - 阈值并不是对于所有实例而言都是最佳的**



解决**multi-entity**问题提出的方案：

 a document graph

> 构造一个具有依赖性结构、启发式结构或结构化注意的文档图。所构建的图连接了在文档中相距很远的实体，从而减轻了基于RNN的编码器在获取远程信息时不足

transformer-based models

> 可以隐式地建模远程依赖关系，目前还不清楚图结构是否仍然在BERT等预先训练的语言模型上有帮助
>
> 也可应用预先训练过的语言模型而不引入图结构的方法，它们简单地对实体标记的嵌入进行平均，以获得实体嵌入，并将它们输入分类器，以获得关系标签。==**然而，每个实体在不同的实体对中都有相同的表示，这可能会带来来自无关上下文的噪声。**== 



问题1：<font color="red">所有实体对使用相同的实体嵌入</font> — **localized context pooling technique：==它使用与 当前实体对相关的附加上下文 增强了实体的嵌入，让嵌入与实体对有关==**

> 直接将注意力头从预先训练过的语言模型中转移出来，以获得实体级的注意。然后，**对于一对实体中的两个实体，我们通过乘法来合并它们的注意力，以找到对它们和这两个实体都很重要的上下文。** 
>
> **localized context pooling** - 将预先训练好的注意力转移到实体对的相关上下文，以获得更好的实体表示。

<hr>

解决**multi-label**问题提出的方案 — 简化为一个二分类问题

> 在训练之后，**对类别概率应用全局阈值以获得关系标签**。这种方法涉及启发式阈值调整，并且**当来自开发数据的调整的阈值对于所有实例可能不是最佳的时，会引入决策错误**
>
> 关系的数量是不同的（多标签问题），而且模型可能不是全局校准的，所以**相同的概率对所有实体对来说并不意味着都是一样的**。



问题2：<font color="red">阈值对于所有实例可能不是最佳的</font> — **adaptive thresholding technique：它用一个可学习的阈值类替换了全局阈值，通过adaptive-threshold loss学习**

> 这是一种基于秩的损失，它将正类的对数推到阈值以上，并在模型训练中将负类的对数推到以下。在测试时，我们返回比阈值类更高的类作为预测标签，如果该类不存在，则返回NA。这种技术消除了阈值调优的需要，也使阈值可调节到不同的实体对，从而导致更好的结果
>
> **adaptive-threshold loss** - 它可以学习一个**依赖于实体对**的自适应阈值，并减少了使用全局阈值引起的决策错误。



## 基础模型 - 基于BERT

**Encoder**

**entity marker technique**

> 我们通过在**提及的开始和结尾**插入一个特殊的符号**“*”**来标记实体提到的位置 — 它改编于**实体标记技术（entity marker technique）**
>
> 文档由编码器进行一次编码，所有实体对的分类都基于相同的上下文嵌入。==**我们将提及开始的“*”作为提及嵌入**==

对于某个实体的所有提及，我们应用 **logsumexp pooling**，一个最大池的平滑版本，得到实体嵌入$h_{e_i}$

> **logsumexp pooling** - 这种汇集积累了来自文档中提及的信号。它在实验中比平均池化具有更好的性能。
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529160818384.png" alt="image-20230529160818384" style="zoom:50%;" />
>
> 它特别适用于处理数值稳定性问题和处理大量或小量的值。
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529155845083.png" alt="image-20230529155845083" style="zoom:67%;" />

**Binary Classififier**

**group bilinear**:为了减少双线性分类器中的参数数量 — 将嵌入维度分割成k个等大小的组，并在组内应用双线性	

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529161512378.png" alt="image-20230529161512378" style="zoom:67%;" />
>
> 我们使用二进制交叉熵损失来进行训练。
>
> 在推理过程中，我们调整一个**全局阈值θ**，使开发集上的评估指标（RE的F1分数）最大化，如果P（r|es，eo）>θ，则返回r作为关联关系，如果不存在关系，则返回NA。



## 模型改进

**Adaptive Thresholding**  - 将全局阈值换成自适应阈值 - `losses.py`

> 对比学习
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529170057596.png" alt="image-20230529170057596" style="zoom:67%;" />
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529172506000.png" alt="image-20230529172506000" style="zoom:67%;" />
>
> **这个阈值类学习一个与实体相关的阈值**。它是全局阈值的替代品，因此消除了在开发集上调整阈值的需要
>
> <font style="background: Aquamarine">**这个自适应阈值是 与实体相关的，实体不同，对应的阈值不同**</font>
>
> > 如果一个实体对被正确分类，则正类的对数应高于阈值，而负类的对数应低于阈值。
>
> > **<font style="background: Aquamarine">第一部分：包括积极类和TH类。由于可能存在多个正类，总损失被计算为 所有正类上的类别交叉熵损失的和；L1推动所有正类的对数都高于TH类。如果没有阳性标签，则不使用它</font>**
> >
> > **<font style="background: Aquamarine">第二部分L2涉及到负类和阈值类。它是 一个类别交叉熵损失，TH类是真正的标签。它将负类的对数拉到低于TH类。</font>**

<hr>

**Localized Context Pooling** - 增强实体嵌入 - `model.py - get_hrt()`

> 其中我们使用与两个实体相关的附加局部上下文嵌入来增强实体对的嵌入。
>
> 我们首先将“*”符号作为提及级注意，然后将同一实体的注意平均，获得实体级注意AE $i∈R^{H×l}$，表示第i个实体对所有标记的注意。
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529195543352.png" alt="image-20230529195543352" style="zoom:67%;" />
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529195607214.png" alt="image-20230529195607214" style="zoom:67%;" />
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230613222406435.png" alt="image-20230613222406435" style="zoom: 67%;" />
>
> > $a^{(s,o)}=q^{(s,o)}/1^Tq^{(s,o)}$
> >
> >
> > 根据您提供的公式 $a^{(s,o)} = \frac{{q^{(s,o)}}}{{1^Tq^{(s,o)}}}$，可以解释如下：
> >
> > 假设 $a^{(s,o)}$、$q^{(s,o)}$ 和 $1$ 都是向量。这个公式表示计算向量 $q^{(s,o)}$ 除以 $1^Tq^{(s,o)}$ 的结果，并将结果赋值给向量 $a^{(s,o)}$。
> >
> > 其中，$1^T$ 表示向量 $1$ 的转置。在这里，它用于计算向量 $1$ 和向量 $q^{(s,o)}$ 的内积，即 $1^Tq^{(s,o)}$。内积是将两个向量逐元素相乘并求和的运算。
> >
> > 然后，向量 $q^{(s,o)}$ 被除以 $1^Tq^{(s,o)}$，得到的结果赋值给向量 $a^{(s,o)}$。
> >
> > 总结起来，**这个公式的目的是将向量 $q^{(s,o)}$ 归一化，使得它的和等于 $1$**，并将归一化结果存储在向量 $a^{(s,o)}$ 中。



## 模型结果

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529211156029.png" alt="image-20230529211156029" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529211938681.png" alt="image-20230529211938681" style="zoom:67%;" />

## 实验结论

预先训练过的语言模型可以捕获实体之间的长距离依赖关系，而不需要显式地使用图结构

发现实体标记(entity markers)的改进很小（在def F1中为0.24%），但仍然在模型中使用该技术，因为它使提及嵌入和提及级的推导注意更容易。

在训练后调整每类阈值会导致对开发集的严重过拟合。而我们的自适应阈值技术在训练中学习阈值，这可以推广到测试集。

Localized Context Pooling技术可以捕获实体对的相关上下文，从而缓解了多实体问题。

> 当实体的数量大于5时，我们的Localized Context Pooling可以获得更好的结果。当实体数量的增加时，改进就会变得更加显著。
