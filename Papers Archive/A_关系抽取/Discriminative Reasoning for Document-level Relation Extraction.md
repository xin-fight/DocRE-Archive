# Discriminative Reasoning for Document-level Relation Extraction

| <font style="background: Aquamarine">discriminative reasoning framework</font> | <font style="background: Aquamarine"> explicitly model the reasoning paths 显示建模推理路径</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |



> 如何对推理路径中的三个任务进行识别？



在本文中，我们提出了一种新的**判别推理框架来明确地考虑不同的推理类型**。我们使用元路径策略来提取不同推理类型的推理路径。在此基础上，我们提出了一个判别推理网络（DRN），其中我们同时使用异构图上下文和文档级上下文来表示不同的推理路径



提出了一个新的 判别推理框架（ a novel discriminative reasoning framework）来明确地 ==**建模推理技能之间的路径**==。

> 因此，设计了一个判别推理网络，**基于所构造的图和向量化的文档上下文来估计 不同推理路径 	的关系概率分布，从而识别它们之间的关系**

## 问题描述

问题1：先进的DocRE模型通常使用通用的**多跳卷积网络** 隐式地对这些推理技能进行建模，而**没有明确地考虑直觉推理技能**，这可能会阻碍DocRE的进一步改进。

>  直觉推理技能：pattern recognition； common-sense reasoning； logical reasoning； coreference reasoning
>
> **intra-sentence reasoning** (including **pattern recognition** and **common-sense reasoning**), **logical reasoning**, and **coreference reasoning**



解决：

我们提出了一种新的**discriminative reasoning framework**来==**明确地建模这些 推理技能**==的推理处理

> 推理技能有：如句内推理（包括模式识别和常识推理）、逻辑推理和共引用推理。

**discriminative reasoning network**

> 该网络作用：==基于所构造的异构图和向量化的原始文档 **对推理路径进行编码**，从而通过分类器识别出两个实体之间的关系==



## 2. Discriminative Reasoning Framework

**显式地建模不同的推理技能**，以识别输入文档中每个实体对之间的关系。

该架构包含了==三个部分==： **defifinition of reasoning paths, modeling reasoning discriminatively, and multi-reasoning based relation classifification.**

### 2.1 ==Defifinition of Reasoning Path推理路径的定义==

原来的四种推理技能（Yao et al.，2019）被进一步细化为**三种推理技能：句内推理、逻辑推理和共引用推理**

> 与Xu等人的工作中所定义的meta-path相比，在所定义的推理路径中没有任何实体。
>
> ==**推理路径不适用实体的原因：**==
>
> > i)原因路径**更注重提及和引用的句子**
> >
> > ii)**实体通常包含在提及中**；
> >
> > iii)使路径推理的**建模更加简单**。
>

**Intra-sentence reasoning path**

> $$PI_{ij}=m_{i}^{s_{1}} \circ  s_{1}\circ  m_{j}^{s_{1}}$$ （“◦”表示从ei到ej的推理路径上的一个推理步骤。）

**Logical reasoning path**

> $$PL_{ij}=m_{i}^{s_{1}}\circ s_{1}\circ m_{l}^{s_{1}}\circ m_{l}^{s_{2}}\circ s_{2}\circ m_{j}^{s_{2}}$$； bridge entity $e_l$ 

**Coreference reasoning path**

> $$PC_{ij}=m_{i}^{s_{1}}\circ s_{1}\circ s_{2}\circ m_{j}^{s_{2}}$$
>
> 参考词是指两个实体ei和ej中的一个，它们与另一个实体出现在同一句子中。==我们简化了条件，并假设当**实体 出现在不同的句子中时，存在一个共引用推理路径**==

### 2.2 Modeling Reasoning Discriminatively用推理路径分别对推理建模

基于定义的推理路径，我们**将DocRE问题分解为 三个推理子任务：句内推理（IR）、逻辑推理（LR）和共指推理（CR）**。

**Modeling Intra-Sentence Reasoning** - 句子内推理被建模以识别该实体对之间的关

> $$R _ { P I } ( r ) = P ( r | e _ { i j } , P I _ { i j } , D )$$ 

**Modeling Logical Reasoning** - 逻辑推理被建模以识别该实体对之间的关系

> $$ R _ { P L } ( r ) = P ( r | e _ { i } , e _ { j } , P L _ { i j } , D )$$ 
>
> **但是桥接实体$e_l$分别与头尾实体出现，因此逻辑推理进一步形式化如下**：
>
> $$R _ { P L } ( r ) = P ( r | e _ { i } , e _ { j } , e _ { l } , P I _ { i l } \circ P I _ { lj } , D )$$

**Modeling Coreference Reasoning** - 将共引用推理建模，以识别该实体对之间的关系

> $$R _ { P C } ( r ) = P ( r | e _ { i } , e _ { j } , P C _ { i j } , D )$$

### 2.3 Multi-reasoning Based Relation Classifification基于多重推理的关系分类

在DocRE任务中，一个实体通常涉及到**依赖于不同推理类型的多种关系**，因此，==**一个实体对之间的关系可以由多种推理类型的推理**==而不是单一的推理类型来推理。

<font color='red'>**基于所提出的三个推理子任务，将一个实体对之间的关系推理视为一个多推理分类问题**</font>

> 对于一个推理类型，在两个实体之间通常有多个推理路径:
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230625161056870.png" alt="image-20230625161056870" style="zoom: 67%;" />
>
> 注意，**所有的实体对都至少有一个推理路径来自三个定义的推理子任务中的一个推理路径**
>
> $K$是一种推理技能的**推理路径数**，某个推理子任务的推理路径**数量大于K时，我们选择前K个推理路径**；否则，我们使用实际的推理路径。
>
> > 经过实验发现K=3时效果相对较好；之后**随着超参数K的持续增加，在开发和测试集上的F1分数开始下降**
> >
> > * 一方面，原因可能是由于**过多的推理路径所提供的推理信息被重复**，甚至在剩余的9.60%的推理路径中出现噪声。
> > * K=3使得DRN在开发和测试集上获得最高的F1分数。

## 3. Discriminative Reasoning Network

**建模三个已定义的推理子任务，以识别文档中两个实体之间的关系**

根据Zeng等人和Zhou等人的工作，我们使用**两种上下文表示（异构图上下文表示和文档级上下文表示）**在Eq中对不同的推理路径进行区别性建模，2.2节公式。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230625170505497.png" alt="image-20230625170505497" style="zoom: 50%;" />

### 3.1 Heterogeneous Graph Context Representation - ==提及节点 和 句子节点异构图的上下文表示(HGCRep) $g_n$==

形式上，我们将每个**单词的嵌入we**与其**实体类型的嵌入wt**以及**共引用嵌入wc**作为**单词b的表示**[we：wt：wc]。

得到单词的表示后，让这些单词表示序列依次输入到BiLSTM来 **对文档进行向量化** 得到 ==文档输入向量$D$==



与Zeng等人的工作(**GAIN**包含**mention node** and **document node**)类似，==**我们构造一个包含 句子节点(sentence node) 和 提及节点(mention node) 的异构图。**==

> a heterogeneous graph 包含了 four kinds of edges：
>
> * **sentence-sentence edge**：所有句子直接被链接
>
> * **sentence-mention edge**：句子节点和位于句子中的提及节点
>
> * **mention-mention edge**：在同一句子中提到的所有节点
>
> * **co-reference edge**：所有提及的引用同一实体的节点

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230625200108561.png" alt="image-20230625200108561" style="zoom:50%;" />

**之后利用GCN对异构图进行编码，并在此基础上学习 <font color='red'>异构图的上下文表示（heterogeneous graph context representation，HGCRep) </font>**

> The **HGCRep** of each mention node and sentence node $g_n$ — ==**每个 提及节点 和 句子节点 的异构图的上下文表示**==
>
> > **提及节点 和 句子节点 的异构图的上下文表示** $$g _ { n } = [ v _ { n } : p _ { n } ^ { 1 } : p _ { n } ^ { 2 } : \cdots : p _ { n } ^ { l - 1 } ]$$
> >
> > * $v_n$ is the initial representation of the *n*-th node
> > * $ p _ { n } ^ { 1 } : p _ { n } ^ { 2 } : \cdots : p _ { n } ^ { l - 1 }$ 与图卷积有关
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230625195811544.png" alt="image-20230625195811544" style="zoom:50%;" />
>
> > <font color='red'> **a heterogeneous graph representation异构图表示为$G=\{g_1,...g_N\}$**</font>

### 3.2 Document-level Context Representation - ==提及和句子 文档级上下文表示(DLCRep) $c_n$==

在DocRE任务中，这些**推理技能严重依赖于原始文档上下文信息，而不是异构图上下文信息**

我们使用**自注意机制**（Vaswani et al.，2017）来学习一个基于向量化的输入文档D的 **<font color='red'>文档级上下文表示（document-level context representation DLCRep）</font>**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230625202315774.png" alt="image-20230625202315774" style="zoom:50%;" />

> * {K，V}是指**使用线性层 <u>从向量化的输入文档$D$ 转换而来</u> key矩阵和value矩阵**。
>
> 得到单词的表示后，让这些单词表示序列依次**输入到BiLSTM来 对文档进行向量化** 得到 ==**文档输入向量$$D = \{ H ^ { 1 } , \quad H ^ { 2 } , \cdots , \quad H ^ { N } \}$$**==
>
> > 对于该文档第n个句子嵌入：$$H^n = ( h _ { 1 } ^ { n } , h _ { 2 } ^ { n } , \cdots , h _ { J n } ^ { n } )$$；其中 $h^j_i$表示文档中 **第j句的第i个词的隐藏表示**
>
> ==**$c_{s1}$表示句s1和的DLCReps；$c_{m_{j}^{s_{2}}}$表示提及$m_{j}^{s_{2}}$的DLCReps**==
>
> > 为了简单起见，我们使用 ==**提及或句子中的 head word 的隐藏状态来表示它们，以便于得到$C_{s1}或者C_{m^{s2}_j}$**==

### 3.3 Modeling of Reasoning Paths - ==将定义的**推理路径** 编码建模为相应的 **推理表示**==

我们使用**连接操作对推理路径上的推理步骤进行建模**，从而将第 2.1节中定义的**推理路径** 编码建模为相应的推理表示

<font color='red'> **使用提及或者句子的HGCRep以及DLCRep构建 reasoning representation 推理表示：**</font>

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230625210648937.png" alt="image-20230625210648937" style="zoom:67%;" />

> $m^{s_k}_i$表明在句子$s_k$中出现的关于$e_i$的提及	
>
> <hr>第2.1节中定义的推理路径：
>**Intra-sentence reasoning path**
>
> > $$PI_{ij}=m_{i}^{s_{1}} \circ  s_{1}\circ  m_{j}^{s_{1}}$$
>
> **Logical reasoning path**
>
> > $$PL_{ij}=m_{i}^{s_{1}}\circ s_{1}\circ m_{l}^{s_{1}}\circ m_{l}^{s_{2}}\circ s_{2}\circ m_{j}^{s_{2}}$$； bridge entity $e_l$ 
>
> **Coreference reasoning path**
>
> > $$PC_{ij}=m_{i}^{s_{1}}\circ s_{1}\circ s_{2}\circ m_{j}^{s_{2}}$$

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230626155441481.png" alt="image-20230626155441481" style="zoom: 67%;" />



## 4. Experiments

对于一个实体对，可能有多个关系，因此**需要一个阈值θ 用于控制在提取的关系事实的数量**。

**特别是，预测结果根据他们的置信度进行排序，并通过开发集上的F1得分从上到下遍历这个列表，选取最大F1对应的得分值为阈值*θ*.**

> ==一旦训练了一个模型，我们得到每个triple example (subject,object,relation)的置信度分数（公式12）。我们根据预测结果的置信度对预测结果进行排序，并通过开发集上的F1得分从上到下遍历该列表，**选取最大F1对应的得分值作为阈值θ**。==



通过实验：

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230626175719816.png" alt="image-20230626175719816" style="zoom: 67%;" />

> 验证了**区分性建模推理**（modeling reasoning discriminatively）比原始的通用神经网络方法更有利于DocRE。

消融实验：

以往的工作没有使用区分推理框架，而是平均提及表示（HGCRep或DLCRep）来得到实体表示，并连接两个实体表示来对关系进行分类，我们将其表示为统一模型。

> 首先，无论使用什么推理上下文，DocRE模型都受益于我们的鉴别推理框架(discriminative reasoning framework)
>
> 其次，DLCRep和HGCRep在捕获推理路径上的节点信息方面都起着重要的作用。



统计得知：**我们定义的三种推理技能可以完全覆盖所有的实体对，而不管这些实体对是否有关系**

**显式地建模推理类型可以有效地推进DocRE**
