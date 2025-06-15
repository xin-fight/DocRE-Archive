# Improving Long Tailed Document-Level Relation Extraction via Easy Relation Augmentation and Contrastive Learning

| <font style="background: Aquamarine">long-tailed distribution problem</font> | <font style="background: Aquamarine">**Easy Relation Augmentation(ERA) mechanism**</font> | <font style="background: Aquamarine">**MoCo-DocRE**</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------------: |



> 从损失函数上看，我感觉好像这篇文章没有考虑到多标签的问题，就是简单的把**实体对存在的关系 的 relation representation 之间的相似度越来越高**
>
> > 但在4.1节中：在DocRE设置下，我们认为**语义相似的样本应该是具有相同关系r的实体对，包括原始对和ERA的增广对**。然而，只有少数实体对在一个文档中具有相同的关系，特别是对于尾关系r。
> >
> > * 按照道理来说这个语义相似的样本有相同的关系 — 那就应该按照对应的r进行区分，对于**某个r**来说构造不同的正负样本
> > * 而不应该像公式中：实体对存在的关系整体作为正样本



论文中使用的增强是ERA: **增加尾部关系频率,增强实体对的表示**：

> **<font color='red'>认为对于 上下文的轻微扰动 不应该影响关系预测</font>**
>
> ==由于对于上下文的轻微扰动不应该影响关系预测，因此我们对**合并的上下文表示(pooled context representation) $c_{h,t}$** 加入了人为扰动（**在attention score**$A_{h,t}$ **上乘以 *random mask $p$***（$$A _ { h , t } ^ { \prime } = p * A _ { h , t }$$ — **p的每个维度都在$\{0,1\}$**中且满足伯努利分布）），得到了**perturbed pooled context representation** $c_{h,t}^{’}$==
>
> ==**total tripe representation set** $$\mathcal{T}$$==： $$\mathcal{T} _ { orig } = \{ ( e _ { h }, c _ {h, t }, e _ { t }  ) | e _ { h}\in\xi, e_t \in \xi \}$$; 		$$\mathcal{T} _ { a u g } = \{ ( e _ { h }, c _ { i , h, t }^{’ }, e _ { t }  ) | e _ { h}\in\xi,r\in R^{aug}, e_t \in \xi \}$$ ( Easy Relation Augmentation(ERA)模块产生)
>
> > $\mathcal{T} _ { a u g }$中α是一个用于控制ERA操作数量的超参数：**<font color='red'>α distinct pertubed context representations</font>** $$\{ c _ { i , h , t }^{’} \} _ { i = 1 } ^ { | \alpha | }$$
>
> <hr>
> 
>
> **关系预测**: 
>
> 与ATLOP一样，**localized context embedding**（ to fuse the pooled context representation $c_{h,t}$ with $e_h$）以及**adaptive thresholding loss** 利用==**total tripe representation set**== $$\mathcal{T}$$进行
>
> 
>
> **MoCo-DocRE** — **contrastive learning(CL) framework for <font color='red'>unifying the augmented relation representations</font> and <font color='red'>improving the robustness of learned relation representations</font>**（统一 增强后的关系表示，提高对比学习到的 关系表示 的鲁棒性）:
>
> For a triple representation$( e _ { h }, c _ {h, t }, e _ { t }  ) \in \mathcal{T}$ 经与ATLOP相同的步骤得到**localized context embedding**，利用MLP得到了**final relation representation**:
>
> > **Anchor relation encoding**: 	==<font color='red'>$x = r e l u ( W _ { 2 } ( W _ { 1 } [ h : t ] + b _ { 1 } ) + b _ { 2 } )$</font>== — $d_r$ is the dimension of final relation representation $x_{h,t}$.
> >
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231030205221526.png" alt="image-20231030205221526" style="zoom:33%;" />
>



> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230704170010848.png" alt="image-20230704170010848" style="zoom: 67%;" />
>
> 为了**保持 关系表示队列$Q_r$ 中关系表示的一致性**，我们还在对比学习中使用==**动量更新模型 将 对比学习中的正样本和负样本 进行编码**==；使用  ==**INFONCE loss**== 进行对比模型框架的训练（目标是**最大化正样本对的相似度得分，同时最小化负样本对的相似度得分**）
>
> > 原始模型$\mathcal{M}$，动量更新后的模型$\mathcal{M}^{’}$ 
> >
> > > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706103249419.png" alt="image-20230706103249419" style="zoom: 25%;" />
> >
> > 将文档输入到模型$\mathcal{M}^{’}$中会得到 **relation representation $x^{'}$** (关系的特征向量)
> >
> > 然后，==如果（eh，et）**<font color='red'>存在关系r</font>**，那么 (关系的特征向量) $x^{’}$将被**<font color='red'>推到对应关系$r$的 relation representation queue $Q_r$</font>**中==
> >
> > 最终，我们可以**从relation representation queue中得到 x 的一组 正、负关系表示的集合**：$$P = U _ { r \in P _ { h , t} }  Q _ { r }$$，$$N = U _ { r \in N _ { h , t} }Q _ { r }$$
> >
> > > ==通过取集合 $P_{h,t}$ 中的所有关系 r 对应的关系表示队列 $Q_r$ 的并集，得到了集合 P==
> > >
> > > 即：**实体对中存在的关系的relation representation之间相互为正例**，实体对中不存在的关系的relation representation作为负例
> >
> > 使用**<font color='red'>INFONCE loss</font>**作为对比损失函数 - ==**目标是 最大化正样本对的相似度得分，同时 最小化负样本对的相似度得分**==，让模型可以**统一 增强后的关系表示，提高对比学习到的 关系表示 的鲁棒性** 
> >
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230705123758359.png" alt="image-20230705123758359" style="zoom:67%;" />
> >
> > > For anchor relation representation **x**,
> > >
> > > * 让其 和 **实体对存在的关系** 的 relation representation 之间的相似度越来越高
> > > * 让其 和 **实体对不存在关系** 的 relation representation 之间的相似度越来越低





## 问题描述

**问题：**

DocRE面临的不可忽视的独特挑战是**<font color='red'>长尾分布(long-tailed distribution)</font>**。

> **数据增强(Data augmentation)** 是解决长尾问题的一种常用策略
>
> > 但是如果利用一般的数据增强：文本随机删除或替换（Wei和Zou，2019）等，==头部关系也会得到扩充，这可能**导致头部关系的过拟合**==



通过之前的工作可以得到：

> 适当的对比学习框架可以**鼓励模型学习更健壮的任务无关 或与 任务相关的表征**
>
> 当样本数量较少时，比较适合使用对比学习，而**长尾关系分布的问题本质上是缺乏某些关系类型的训练样本**。



## 模型改进

提出了**Easy Relation Augmentation(ERA)**：**<font color='red'>对关系表示(relation representations)增强 而不是 对文本增强</font>**

> 因此可以在不对 长文档进行其他编码操作 的情况下增强尾关系，这**使得计算效率，也可以提高尾关系的性能**。

**基于ERA的对比学习框架(contrastive learning framework) — ERACL**：先利用 **远端监督的数据**进行预训练，之后再进行微调。



## 3. Easy Relation Augmentation — ==ERA对关系表示进行增强==

**Encoding modules**：==对每个实体对进行两个层面的编码 （与ATLOP相同）== 

> **contextualized entity representation** and **pooled context representation** via self-attention mechanism 



**Easy Relation Augmentation(ERA) mechanism**

> 通过在pooled context representation上==应用一个随机**掩码**来增强实体对表示==。
>
> 提出的**ERA不需要任何额外的 关系编码和文档编码**，就可以对尾部关系进行增强，<font color='red'>使得计算效率很高，同时也很有效</font>



### 3.2 Document Encoding

使用预训练Transformer对文档进行编码，**在每个提及前后加入*** （与ATLOP相似，都是用了实体标记技术）

模块输出：**所有单词上下文表示H $R^{l*d}$**和最后一个层**基本多头自注意力A $R^{l*l*h}$**

### 3.3 Relation Encoding

该模块的目标是：通过==聚合**contextualized entity representation**和**pooled context representation**对每个**实体进行两个层面的编码**==：

> ==与ATLOP操作相同==: 
>
> **contextualized entity representation $e_h$**:  
>
> * 先得到**提及前的"*"**的上下文表示作为 contextualized mention representation $m_{hj}$
> * 然后使用logsumexp pooling获得实体$e_h$的嵌入
>
> **pooled context representation** $c_{h,t}$ 
>
> * <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230703172820835.png" alt="image-20230703172820835" style="zoom:50%;" />
>
> 最后形成了**triple represention:** $$T _ { h , t } = ( e _ { h , c } , c _ { h , t } , e _ { t } )$$
>
> * 包含了关系预测的所有信息，并形成了我们的**ERA和对比学习框架的基础**

### 3.4 Relation Representation Augmentation - 通过上下文扰动==增加尾部关系频率,增强实体对的表示==

**triple represention:** $$T _ { h , t } = ( e _ { h , c } , c _ { h , t } , e _ { t } )$$

所有实体对的三重表示集 ==$$T _ {orig}$$==，==**<font color='red'>手动选择</font> 需要扩充的关系集**$R^{aug}$==

回想一下，**合并的上下文表示(pooled context representation) $c_{h,t}$** 编码了关系推理的唯一的上下文信息，而**==对上下文的轻微扰动不应该影响关系预测==。**



==即：由于对于上下文的轻微扰动不应该影响关系预测，因此我们对**合并的上下文表示(pooled context representation) $c_{h,t}$** 加入了人为扰动==

**<font color='red'>perturbed pooled context representation（受扰动的合并上下文表示）</font>**:

> 基于这种直觉，我们**在$c_{h,t}$上添加了一个小的扰动**：==$$A _ { h , t } ^ { \prime } = p * A _ { h , t }$$== 
>
> * **在attention score**$A_{h,t}$（$R^{l*1}$）**上乘以 *random mask $p$***（$R^{l*1}$，**p的每个维度都在$\{0,1\}$**中且满足伯努利分布）
>
> * **可以解释为随机过滤掉一些上下文信息**，因为有的的注意力分数被设置为0
> * 扰动程度通过 伯努利分布的参数 进行调节
>
> <hr>
>
> **perturbed pooled context representation**: ==$$c _ { h , t } ^ { \prime } = H ^ { T } \cdot \frac { A _ { h , t } ^ { \prime } } { 1 ^ { T } \cdot A _ { h , t } }$$==
>
>
> * **对于$R^{aug}$中关系r的 所有实体对（eh，et）**，我们应用前面的步骤==**通过使用α个不同的random mask，得到α个不同的扰动的上下文表示**$$\{ c _ { i , h , t } ^ { \prime } \} _ { i = 1 } ^ { | \alpha | }$$==；
>
>   其中**α是一个用于控制ERA操作数量的超参数。**
>
> * 通过其可以得到augmented triple representation set  ==$$T _ {aug}$$==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230703215521158.png" alt="image-20230703215521158" style="zoom:67%;" />

通过公式8得到了**<font color='red'>total tripe representation set $T$</font>**，用于**关系预测**和**对比学习框架**

### 3.5 Relation Prediction - 与ATLOP同

==与ATLOP相同==，使用了**localized context embedding**（ to fuse the pooled context representation $c_{h,t}$ with $e_h$）以及**adaptive thresholding loss** 

对于某个triple representation$$( e _ { h } , c _ { h t } , e _ { t } ) ∈ T$$，我们先让**实体嵌入融合pooled context representation** $c_{ht}$ 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230703221025386.png" alt="image-20230703221025386" style="zoom:50%;" />

**基于对比学习损失函数 —— ATL：**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230704162543003.png" alt="image-20230704162543003" style="zoom:67%;" />



## 4 ==Contrastive Learning== for relation pre-training

提出对比学习架构是为了：**<font color='red'>统一 增强后的关系表示，提高学习后的 关系表示 的鲁棒性</font>**

这个框架是为了在DocRED的**远程监督(distantly-supervised)**数据集上使用CL框架进行预训练；考虑到在表示学习阶段之后，模型将在人工标注的数据集上进行微调，因此远程监督数据集中的噪声是可以接受和修正的。

<hr>

在DocRE设置下，我们认为**具有相同关系 r 的实体对应该是语义上相似的样本**，包括原始实体对和通过ERA进行扩充的实体对。

然而，==**在一个文档中只有很少的实体对具有相同的关系，特别是对于尾部关系 r 来说。**==

> 可以通过增加bs进行接近，但会导致GPU memory需求增加
>
> 使用MOCO framework (He et al., 2020) to the DocRE setting, named **<font color='red'>MoCo-DocRE</font>**
>
> * 该框架使用了对比学习，通过**保持一个关系表示队列$Q_r$**，$Q_r$保存了 **每个关系**r∈R的前一个小批次的**q个关系表示**。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230704170010848.png" alt="image-20230704170010848" style="zoom:67%;" />

### 4.2 Anchor relation encoding $x$ - 融合增强后的实体嵌入

先进行之前的**文档编码** 和 **Easy Relation Augmentation(ERA)**  （3.2和3.4节）即得到**<font color='red'>所有实体对的triple representation集T</font>**。

接下来与ATLOP相同的方法**融合triple representation**（公式9、10）得到 **增强后的实体嵌入**

最后使用MLP**将增强后的实体嵌入融合 得到** ==**Anchor relation encoding** $$x = r e l u ( W _ { 2 } ( W _ { 1 } [ h : t ] + b _ { 1 } ) + b _ { 2 } )$$==

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230704204725927.png" alt="image-20230704204725927" style="zoom:67%;" />
>
> **<font color='red'>$d_r$是final relation representation $x_{h,t}$的维数</font>**

### 4.3 MoCo-DocRE

为了**保持$Q_r$中关系表示的一致性**，我们还在对比学习中使用==**动量更新模型 将 对比学习中的正样本和负样本 进行编码**==

**原始模型M通过反向传播进行更新，动量更新后的模型M'更新公式为**：==$$M ^ { \prime } = m \cdot M ^ { \prime } + ( 1 - m ) \cdot M$$==

接下来，我们**将文档 D 输入到$M ^ { \prime } $，得到 ** **Anchor relation encoding**==$x ^ { \prime } $==，过程与获取锚定关系表示 x 的过程相同==（公式13）==

然后，我们**根据它们的 关系标签**将 {x' | (eh, ch, t, et) ∈ T’} 推送到 |R| 个关系表示队列中

如果**关系 r 在 (eh, et) 之间存在，则 x' 将被推送到 $Q_r$ 中**。

最后可以**从队列中得到x的正、负关系表示的集合** — 最终，==P 是所有正样本关系表示的集合，N 是所有负样本关系表示的集合==

并使用  ==INFONCE loss== 进行对比模型框架的训练（目标是**最大化正样本对的相似度得分，同时最小化负样本对的相似度得分**）

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230705025634857.png" alt="image-20230705025634857" style="zoom:67%;" />

