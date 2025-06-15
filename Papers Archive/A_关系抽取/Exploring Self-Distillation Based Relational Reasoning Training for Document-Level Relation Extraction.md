# Exploring Self-Distillation Based Relational Reasoning Training for Document-Level Relation Extraction

| <font style="background: Aquamarine">Reasoning Multi-head Self-attention (R-MSA)</font> | <font style="background: Aquamarine">Self-distillation training framework</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |



> **DocRE-SD**可以发现定义了一些推理模式，这个推理1是进行了特征拼接



## 问题描述

问题1：

之前的工作通常**只考虑一种推理模式**，其中对关系三元组的覆盖范围是有限的；

><font color='red'>*reasoning multi* *head self-attention* unit.</font> - 利用**四个注意头分别建模四种常见的推理模式**



问题2：

他们没有**显示地模拟关系推理**的过程

> <font color='red'> self-distillation training framework - contains two branches sharing parameters</font>
>
> *  first branch: 文档中**随机mask一些实体对的特征向量**，然后利用其他相关实体对的特征信息，训练我们的推理模块来推断它们之间的关系；不过测试时不需要随机mask，会造成<font color='red'> input gap</font>
> * second branch：减少input gap，进行不mask操作的常规监督训练，利用 Kullback-Leibler divergence loss来最小化两个分支预测之间的差异



## 模型改进

### Reasoning Multi-head Self-attention (R-MSA) unit - 建模四种推理模式

**Reasoning Multi-head Self-attention (R-MSA) unit**

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706174246581.png" alt="image-20230706174246581" style="zoom:33%;" />
>
> > c)部分有四个头，分别==建模四种常见的推理模式==；
> >
> > $M^{(l)}_{s,o}$是第$l$层 实体对$(e_s, e_o)$ 对应的 **entity pair feature matrix**；		
> >
> > $$F _ { i } ^ { ( l , 1 ) } = W _ { d } [ M _ { s , i } ^ { ( l ) } ; M _ { i , o } ^ { ( l ) }] + b _ { d , }, i = \{ 1 , 2 , \cdots , N \}$$ （公式3），**实体e 经过中间实体i 到达实体o**
> >
> > 通过**注意机制** 得到 对实体对$(e_s, e_o)$ 的 ==**R-MSA 某个注意力头的输出向量** $M^{(l，1)}_{s,o}$== （以第一个注意力头为例）
> >
> > > Q是实体对$e_{s,o}$对应的特征$Q=M _ { s , o } ^ { ( l ) }$；	$$K = V = [ M _ { s , o } ^ { ( l ) } ; F _ { 1 } ^ { ( l , 1 ) } ; \cdots ; F _ { N } ^ { ( l , 1 ) } ]$$ ；
> > >
> > > Reasoning Module利用 **其他相关实体对 的特征信息** — Reasoning Module多头注意力中的K V是 **实体对$e_{s,o}$对应的特征**与 **实体对==对应 行 与 列 上全部特征==** 
> >
> > $M^{(l，1)}_{s,o}$ 第$l$层R-MSA 第1个注意力头的输出向量；										最后汇总所有注意力头的输出，得到第$l$层 ==**R-MSA的输出**$$\widetilde{M} _ { s , o } ^ { ( l ) }$$== 
>
> <hr>
>
> * **encoder**：按照**ATLOP**流程进行训练，然后构造了一个**<font color='red'>entity pair feature matrix $M$</font>**，以方便推理模块的计算
>
>   > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706194920899.png" alt="image-20230706194920899" style="zoom: 33%;" />
>   >
>   > * ==$F_{s,o}$== 是将 上下文增强后 **实体对表示** 经过FFN得到
>   >
>   > * 最后将文档中的 **所有实体对特征向量** 合并成一个==**entity pair feature matrix** $M^{0}=[F_{s,o}]_{N*N}$== （N为实体的个数；第s行，第o列对应的值 表示 实体对$e_{s,o}$对应的特征）
>
> 
>
> * **Reasoning Module**: 
>
>   > 图2中b)所示，每个 Each reasoning layer contains **four components**: 	a *reasoning multi-head self-attention* (**R-MSA**) unit, 	a **FFN** unit, 	and two **layer normalization** sublayers.
>   >
>   > * R-MSA：利用 ==**四个注意头 分别建模四种常见的推理模式**==，对于每个头来说 计算过程如下：（以<font color='red'>**第一个注意力头为例**</font>，共有四个头）
>   >
>   >   > 是传统的多头自我注意的一种变体，==利用**四个注意头分别建模四种常见的推理模式**==（见表1）— 可以**识别出$76.5\%$的三元组**
>   > >
>   >   > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706173312500.png" alt="image-20230706173312500" style="zoom:33%;" />
>   >
>   > * **R-MSA操作流程：**
>   >
>   >   0. 以第一个头为例：**第一个头负责建模 推理模型：==$[(e_s, r_1, e_i), (e_i, r_2, e_o)] => (e_s, r_3, e_o)$==** ；其他的头负责的推理模型图表1所示
>   >
>   > 
>   >
>   >   1. 在第$(l+1)$reasoning layer而言，先将 **实体对$(e_s, e_o)$对应的 entity pair feature matrix $M^{l}$ 中的==第$s$行 和 第$o$列 所有的 特征向量 连接==**， 然后用线性层 **缩减维度** 
>   >
>   >      > $$F _ { i } ^ { ( l , 1 ) } = W _ { d } [ M _ { s , i } ^ { ( l ) } ; M _ { i , o } ^ { ( l ) }] + b _ { d , }, i = \{ 1 , 2 , \cdots , N \}$$  — N为实体的个数，即==认为其他的实体都可以成为桥接实体$e_i$==
>   >      >
>   >      > > 上标$l$表明第$l+1$层，$1$表示这是第一个注意力头
>   >
>   > 
>   >
>   >   2. 通过**注意机制** 得到 对实体对$(e_s, e_o)$ 的 ==**R-MSA 某个注意力头的输出向量** $M^{(l，1)}_{s,o}$== （以第一个注意力头为例）
>   >
>   >      > **Q**是实体对$e_{s,o}$对应的特征$M _ { s , o } ^ { ( l ) }$ ;	 **K V**是 **实体对$e_{s,o}$对应的特征**与**实体对 $(e_s, e_i), (e_i, e_o)$组成的特征，即考虑了 头尾实体 与 所有的桥接实体特征**； $$K = V = [ M _ { s , o } ^ { ( l ) } ; F _ { 1 } ^ { ( l , 1 ) } ; \cdots ; F _ { N } ^ { ( l , 1 ) } ]$$ ；
>   >
>   >      最后**==汇总==**所有注意力头的输出，得到==**R-MSA的输出**$$\widetilde{M} _ { s , o } ^ { ( l ) }$$== 
>   >      
>   >      > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706205959887.png" alt="image-20230706205959887" style="zoom:30%;" />
>   >      >
>   >      > Reasoning Module利用 **其他相关实体对 的特征信息** — Reasoning Module多头注意力中的K V是 **实体对$e_{s,o}$对应的特征** + **实体对==对应 行$M _ { s , i } ^ { ( l ) }$ 与 列$M _ { i , o } ^ { ( l ) }$ 上全部特征==** 
>   >      >
>   >      > 
>   >      >
>   >      > 对于每个头，通过注意力机制得到$M^{(l，1)}_{s,o}$，经过**汇总多头后**，得到==**R-MSA的输出**$$\widetilde{M} _ { s , o } ^ { ( l ) }$$== 
>   >
>   > 
>   >
>   >   3. 通过将R-MSA的输出$$\widetilde{M} _ { s , o } ^ { ( l ) }$$组合成一个大矩阵$\widetilde{M}^{(l)}$，计算得到最后一个reasoning layer 更具表达能力的 **feature matrix $M^{(L)}$**，并通过其进行预测
>   >
>   >      > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706210701178.png" alt="image-20230706210701178" style="zoom:33%;" />
>
> 
>
> * 之前的研究主要考虑第一种模式，我们的推理模式更全面，对关系三元组的覆盖范围更高



### Self-distillation training framework - 显示地模拟关系推理

在文档中**随机屏蔽一些实体对特征向量**，再利用其他相关实体对的特征信息，训练我们的推理模块来推断这些关系。通过这样做，我们可以==**明确地建模关系推理的过程，这可以为我们的模型提供更明确的推理监督信号**==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706191947216.png" alt="image-20230706191947216" style="zoom: 33%;" />

**first branch:** 

> 文档中==**随机mask一些 实体对 的特征向量**== 得到新的特征向量$\widehat{M}^{0}$，将新特征输入到**Reasoning Module**，利用 **其他相关实体对 的特征信息**，**预测the relation probability distributions of all entity pairs$$\{ \widehat{p} ( r | e _ { s } , e _ { o } ) \}$$**
>
> > Reasoning Module利用 **其他相关实体对 的特征信息** — Reasoning Module多头注意力中的K V是 **实体对$e_{s,o}$对应的特征**与 **实体对==对应 行 与 列 上全部特征==**  $$K = V = [ M _ { s , o } ^ { ( l ) } ; F _ { 1 } ^ { ( l , 1 ) } ; \cdots ; F _ { N } ^ { ( l , 1 ) } ]$$
>
> 
>
> ==**通过这样做，我们 明确地建模了关系推理的过程，这可以为推理模块提供更明确的推理监督信号。**==
>
> 
>
> 不过测试时不需要随机mask，会造成<font color='red'> **input gap**</font>



**second branch：**

> 为了减少<font color='red'> **input gap**</font>，进行<font color='red'>不mask操作的常规监督训练</font>，利用 <font color='red'>Kullback-Leibler divergence loss</font> 来**最小化两个分支预测之间的差异**
>
> > 直接将原实体对的 feature matrix $M^0$ 输入到Reasoning Module得到$$\{ {p} ( r | e _ { s } , e _ { o } ) \}$$ 
> >
> > **引入KL divergence loss最小化${p} ( r | e _ { s } , e _ { o } )$ 与 $\widehat{p} ( r | e _ { s } , e _ { o } )$ 之间的差距**
> >
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706221002575.png" alt="image-20230706221002575" style="zoom: 50%;" />
>
> 
>
> 这样我们将训练中first branch中学习到的 推理能力 转移到 测试场景中



**Training objective**

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706221349076.png" alt="image-20230706221349076" style="zoom:33%;" />
>
> 前两项损失 如图3所示用于监督两个branch的关系预测
>
> 为了解决**多标签**和**不平衡的标签分布**问题，我们采用==adaptive thresholding loss（ATLOP）==来建模我们前两项分类损失$L_c$与$\widehat{L}_C$
>
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706222029169.png" alt="image-20230706222029169" style="zoom:33%;" />



**Curriculum Learning strategy** - ==选中实体对的方式==

> 在first branch中，以一种 ==**简单到困难的方式 动态地选择**==(就是慢慢增大mask率) 被mask实体对
>
> 我们先以 **$\gamma_t$的mask率 均匀抽样** 实体对，来被mask
>
> > 直观地看，**更高的mask率可能会使模型训练更加分散**。因此，为了更好地训练我们的模型，我们==开始用一个较小的mask率进行训练，并将其线性增加到最大的mask率**$\gamma_{max}$**==：$\gamma_{t}=min(\gamma_{max}, t/T)$，其中t是当前的训练步数，T是最大的训练步数



## 结果

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230707142319183.png" alt="image-20230707142319183" style="zoom:33%;" />

1. 请注意，ATLOP-BERT本质上是我们模型的一个变体，其中我们的reasoning module and self-distillation training framework被删除了。

   > 而我们的性能比ATLOP好 — **reasoning module and self-distillation training framework可以提高任务性能**

   我们注意到，基于GNN的模型**比利用 实体对 之间的依赖性进行关系推理的模型获得的结果更差**，如DocuNet-BERT和KD-BERT

   > 表明 **==实体对之间的依赖关系== 比 实体之间或提及之间的依赖关系 对关系推理更有用。** 

2. 从Intra-F1以及Infer-F1提升的效果可以看到，我们模型在 Intra-*F*1只有微小提升，但在Inter-*F*1提升较大

   > 我们的模型的优点在于**提取跨句子关系**，其中大部分需要关系推理的帮助。

3. 分别使用 vanilla knowledge distillation (Hinton et al. 2015) and R-Drop (Wu et al. 2021a),对我们的模型进行训练。与这两种变体相比，我们的模型仍然取得了更好的性能

   >我们的self-distillation training framework 可以更有效地刺激我们的模型的推理能力。

4. **Infer-Ac**度量了在数据集中符合**这四种推理模式的关系三元组的预测精度**。在这个指标上，我们的模型也明显超过了以前的基线

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706173312500.png" alt="image-20230706173312500" style="zoom:33%;" />	