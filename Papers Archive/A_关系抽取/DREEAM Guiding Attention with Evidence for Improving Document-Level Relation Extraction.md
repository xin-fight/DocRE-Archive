# DREEAM Guiding Attention with Evidence for Improving Document-Level Relation Extraction

| <font style="background: Aquamarine">memory-efficient method</font> | <font style="background: Aquamarine">Evidence-guided Attention Mechanism</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| **<font style="background: Aquamarine">relation-agnostic ER与关系无关的证据抽取</font>** | **<font style="background: Aquamarine">weakly-supervised setting - 采用了教师模型预测的证据作为监督信号，以实现对未标记数据的ER自训练</font>** |



## 问题描述

DocRE中的证据检索（ER）面临着两个主要问题

问题1： ==**high memory consumption**==

> 之前解决证据检索问题时，会引入额外的网络，增加了内存消耗；并且**为了计算每个实体对的每个句子的证据分数，模块必须遍历所有（实体对、句子）组合**
>
> **我们不是引入外部ER模块，而是直接指导DocRE系统关注证据**
>
> To reduce the memory consumption, we propose **D**ocument level **R**elation **E**xtraction with **E**vidence-guided **A**ttention **M**echanism (==DREEAM==)
>
> 采用证据信息作为监督信号，从而==**引导DocRE系统 注意模块 对证据分配较高的权重**==
>
> * 通过直接引导注意力，将证据信息纳入基于变压器的DocRE系统中
> * 具体来说，我们**监督实体对特定的局部上下文嵌入的计算**。局部上下文嵌入是基于编码器中所有标记嵌入之间注意力的加权和，==**被训练为为证据分配更高的权重，否则则分配更低的权重。**== 



问题2： ==**limited availability of annotations**==

> 我们提出了一种自我训练策略，让DREEAM**从大量数据上自动生成的证据中学习ER**
>
> * teacher model：对人工标注的数据进行训练，从远端监督的数据中检索silver evidence
>
> * student model：从silver evidence中学习ER。进一步在人工标注数据上进一步微调，以完善其知识



## 模型改进

### ATLOP

**获取token embeddings and cross-token dependencies**：虽然原来的ATLOP只采用了最后一层，==**但这项工作采用了最后三层的平均值**==

> 试点实验表明，使用最后3层比只使用最后一层产生更好的性能。

**每个token**对于实体对$(e_s,e_o)$的重要性，记为$q_{(s,o)}		$—  ==$$q ^ { ( s , o ) } = \frac { a _ { s } o a _ { 0 } } { a _ { s }^T a _ { 0 } }$$==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230828211823239.png" alt="image-20230828211823239" style="zoom: 33%;" />

### DREEAM

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230805115036491.png" alt="image-20230805115036491" style="zoom: 80%;" />

> 图a：DREEAM用于监督和自训练，**与不同的监控信号共享相同的体系结构**
>
> * 利用Attention计算的得到每个token对实体对的重要性
>
>   **teacher model**：先将其转化成句子级别的重要性，之后在**人工标注数据集**上对**每个具有有效关系的实体对**计算损失
>
>   **student model**：直接**使用token级别**的证据分布，让模型产生的分布和teacher model分布接近
>
> 图b：表明teacher model与student model自训练流程
>
> * 类似Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation（KD-DocRE）

#### Teacher model - 用人类证据 指导 句子级重要性，生成 关注证据的局部上下文嵌入$c^{(s,o)}$ 

==记录**对于实体对$(e_s,e_o)$，每个token的重要性**$q_{(s,o)}	$==：<font style="background: Aquamarine">**利用Attentions计算得到**</font>，而我们**需要获取 每个句子对于这个 实体对 的重要性**

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231030141616255.png" alt="image-20231030141616255" style="zoom:33%;" />



<hr>



上述公式可以生成一个嵌入，但我们**想用一个证据分布来指导$q_{(s,o)}	$**，以帮助生成一个 <font style="background: Aquamarine">以证据为中心的局部上下文嵌入 $c^{(s,o)}$</font> 

* <font style="background: Aquamarine">**句子级别的重要性    $p^{(s,o)}$** （$p^{(s,o)}\in R^{| \chi _{D}|}$）：</font> 对于 **句子i** - <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230815111200006.png" alt="image-20230815111200006" style="zoom:33%;" /> 将一个句子中的==每个**token**的权重相加== **得到该句子的重要性**

* 我们进一步使用 由gold evidence计算出的 human-annotated evidence distribution 监督$p^{(s,o)}$ 

  > 首先设置一个==**对每个有效的关系标签 - binary vector $v^{(s,r,o)}$**，维度是句子的总数 （$ v^{(s,r,o)} \in R^{| \chi _{D}|}$）==：如果**第$i$个句子是**$(e_s,r,e_o)$的证据句，那么将==$v_i^{(s,r,o)}$设置为1，否则设置为0==
  >
  > 然后我们边缘化所有有效的关系，并==规范化边缘化向量得到 $v^{(s,o)}$== — 根据数据集可得
  >
  > > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230828202926560.png" alt="image-20230828202926560" style="zoom: 50%;" />
  > >
  > > 方程式9背后原理：==在关系分类器之前的模块并不明确地知道具体的关系类型==。因此，我们**引导编码器内的注意力模块 生成 与==关系无关的== token级别依赖关系** —— **<font style="background: Lime">relation-agnostic ER与关系无关的证据抽取</font>** 
  > >
  > > * <font style="background: Aquamarine">想要引导注意力模块关注对与该实体对重要的句子，而**关注的重要的句子是 ==和实体对相关== 而==非 与某个具体关系相关==**</font>
  > >
  > > * <font style="background: Aquamarine">our method can only retrieve relation-agnostic evidence.（关系无关的证据）</font>
  > >
  > >   > 与**SAIS**不同，==DREEAM不能为每个关系标签指定证据句==。
  > >   >
  > >   > 因此，**当一个实体对拥有多个关系时，无论关系类型如何，DREEAM检索出来为相同的证据，即使证据对某些关系是正确的，但对其他关系则不是**。



**Loss Function**

**目的**：==用**人类证据$v^{(s,o)}$** 来指导 **句子级别的重要性 $p^{(s,o)}$**== 生成 **关注证据的局部上下文嵌入(an evidence focused localized context embedding ) $c^{(s,o)}$** 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230828210411660.png" alt="image-20230828210411660" style="zoom: 50%;" />

* 该损失函数可以统一 **人类证据$v^{(s,o)}$**  和 **句子级别的重要性 $p^{(s,o)}$**  ，以便利用 ==句子重要性$p^{(s,o)}$== 通过 ==token重要性$q_{(s,o)}$== 产生更好的 **局部上下文嵌入** $c^{(s,o)}$
* $v^{s,o}$ 和 $p^{s,o}$ 分布由 证据句 和 模型的attention得到，图中矩阵横坐标表示实体对$s,o$，纵坐标表示文档中的句子，即对于实体对，让**证据句 指导 attention**



#### Student Model - 从teacher中学习token级别的证据分布

学生模型训练如图2中b所示，与教师模型类似，学生模型的监督包括两部分： **RE binary cross entropy loss** and an **ER self-training loss**.

* 对于ER训练：首先，我们让教师模型在==**远程监督数据**==上进行推断，从而为每个实体对$(e_s,e_o)$产生一个**教师模型 关于token的证据分布 $\hat{q}^{(s,o)}$**。接下来，我们使用类似于公式10的kl-散度损失来训练学生模型的ER



**Loss Function**

仍然使用KL散度，训练学生模型让 **对于实体对$(e_s,e_o)$，学生模型 token的证据分布 **$q^{(s,o)} $   和   **教师模型关于token 的证据分布 $\hat{q}^{(s,o)}$ ** 之间的**距离尽量缩小**

> 当教师模型训练好时，其**注意模块 对证据分配较高的权重**，==因此**该模型就可以得到关于token的证据分布**，也就是知道了那些token对实体对重要== 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230828225506472.png" alt="image-20230828225506472" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230829105758115.png" alt="image-20230829105758115" style="zoom: 50%;" />

在对远端监督数据进行训练后，**利用人工注释的数据进一步细化学生模型，利用可靠的监督信号来完善其关于DocRE和ER的知识**

#### Teacher / Student model之间loss区别

Teacher - $L_{ER}^{gold}$ 和 Student  - $L_{ER}^{silver}$的区别：

* $L_{ER}^{gold}$：是**句子级**别的，$L_{ER}^{silver}$：是**token级**别的

  > 在人工标注的数据上，从句子级别的注释中获取token级别的证据分布是复杂的。然而，在远监督数据上，可以很容易地从教师模型的预测中获得。
  >
  > 因此，我们采用token级证据分布，从微观的角度为ER自我训练提供监督

* $L_{ER}^{gold}$：仅在**具有有效关系的实体对上计算**，$L_{ER}^{silver}$：是对文档中的**所有实体对进行计算**

  > 如此设计是由于student会在远程距离监督数据集上进行预训练，而**远距离监督数据上的关系标签的可靠性低**，需要重新对每个实体对进行计算



#### Inference - inference-stage fusion

我们应用**adaptive thresholding法获得RE预测**，选择得分高于阈值类的**关系**作为预测。

对于ER任务，我们采用==**静态阈值法(static thresholding)**==，选择重要性高于预定义阈值的句子作为**证据**。

我们进一步结合了==**inference-stage fusion**== strategy（Xie et al. (2022). **EIDER**论文中提出）

RE模型在**原始文档**获得的一组关系预测结果$S_{h,t,r}^{(O)}$，在**pseudo document**获得的结果为$S_{h,t,r}^{(E)}$，最后，我们通过一个混合层聚合两组预测来融合结果：

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230829151837031.png" alt="image-20230829151837031" style="zoom: 33%;" />

## 结果

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230829161542315.png" alt="image-20230829161542315" style="zoom: 33%;" />

**ER的self-training提高了RE**

我们的ER自我训练方法成功地从没有证据注释的关系远监督数据中获取证据知识。

用证据指导注意力有利于改善RE。

ER自我训练比ER微调更重要

> 一方面，我们观察到，**在大量数据上禁用ER自我训练会导致证据知识的巨大损失**，而这些知识无法通过对更小的证据注释数据集进行微调来恢复。
>
> 另一方面，我们可以得出结论，DREEAM**在没有任何证据注释的情况下成功地从数据中检索了证据知识**，证明了我们的ER自我训练策略的有效性。



相比之下，DREEAM将注意力权重的总和作为证据分数，因此既没有引入新的可训练参数，也没有引入昂贵的矩阵计算。因此，我们看到DREEAM比其竞争对手更有memory-efficient能力。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230829203638074.png" alt="image-20230829203638074" style="zoom: 50%;" />
