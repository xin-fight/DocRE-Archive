# Document-level Relation Extraction via Subgraph Reasoning

| <font style="background: Aquamarine">subgraph reasoning</font> | <font style="background: Aquamarine">reasoning paths 显示建模推理路径</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |



目前大多数研究构建文档级图，然后关注图的整体结构或图中目标实体对之间的路径。

在本文中，我们提出了一种新的**子图推理（SGR）框架**来进行文档级关系提取

> SGR结合了基于图的模型和基于路径的模型的优点，**将目标实体对之间的各种路径集成到一个更简单的子图结构**中来执行关系推理。
>
> 由我们设计的**启发式策略生成的路径明确地建模了必要的推理技能**，并大致覆盖了每个关系实例的支持性句子。

## 问题描述

==**several major challenges**==

> 一般来说，**不能仅仅根据一个句子**来确定实体对之间的关系
>
> 一个实体可以在许多句子中被提到，因此**来自相应提及的信息必须被适当地聚合**，以更好地代表这个实体
>
> 许多关系**需要丰富的推理技能**（即模式识别、逻辑推理、共引用推理和常识推理）

<hr>

最近的研究构建了一个基于句法依赖性、启发式等因素的文档级图，然后使用图神经网络进行多跳推理，以获得有意义的实体表示[Zhang等人，2018（Graph convolution over pruned dependency trees improves relation extraction）；Nan等人，2020( Reasoning with latent structure refnement for document-level relation extraction)；Zeng等人，2020(**Double graph based reasoning for document-level relation extraction**)]。

**问题**


> 但是，这些方法只考虑了整体的图结构，因此它们可能**会忽略围绕目标实体对的局部上下文信息**。
>
> 此外，由于**过度平滑问题**，这些方法可能无法建模长距离实体之间的相互作用



除了	基于图的模型外，一些**基于路径的模型**试图**提取目标实体对之间的路径**，并保留足够的信息来预测关系

下面方法==**明确地**==考虑了**推理技能(pattern recognition, logical reasoning, co-reference reasoning, and common-sense reasoning)**，可以==**缓解长距离实体交互建模的问题**==，但它们单独处理每个路径，==**并不是所有的实体对都可以通过一条路径连接**==

> * Xu等人，2021(Discriminative Reasoning for Document-level Relation Extraction (DRN)；
> * Huang等人，2021(Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction)





## 模型改进

 **SubGraph Reasoning (SGR)** framework

> SGR结合了基于图和基于路径的模型的好处，**将各种路径集成到一个更简单的子图结构中，以便同时执行各种推理**

**building a heterogeneous graph**:  

> entity nodes, mention nodes, and sentence nodes	

**effective strategy to generate reasoning paths**：

> 考虑到文档级RE所需的推理技能，我们启发式地设计了一个简单但非常有效的策略来生成推理路径。这些**路径不仅直观地模拟了所有潜在的推理技能，而且在实践中还大致覆盖了带注释的支持句子**
>
> **<font color='red'>路径生成策略确保了所有的实体对都可以通过一条路径进行连接</font>**

我们根据之前**生成的路径提取了目标实体对周围的一个子图**，并首先应用了一个R-GCN，通过这种方式，==**该模型可以专注于最关键的实体、提及物和句子，并在各种推理路径上执行联合推理**==



## 2 Methodology

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230628211148933.png" alt="image-20230628211148933" style="zoom:67%;" />

单词特征$x_i$组成：word embedding and entity type embedding.

> $$\{ g _ { 1 } , g _ { 2 } , \cdots , g _ { d } \} = E n c o d e r ( \{ x _ { 1 } , x _ { 2 } , \cdots , x _ { d }  \} )$$

### 2.3 Subgraph Reasoning

#### Document Graph Construction

**building a heterogeneous graph**:  

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629011417829.png" alt="image-20230629011417829" style="zoom: 67%;" />
>
> <font color='red'>节点类型：</font>**Entity nodes, Mention nodes, and sentence nodes. **
>
> **<font color='red'>节点嵌入的初始化如下:</font>**
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230628205920432.png" alt="image-20230628205920432" style="zoom: 67%;" />
>
> > $$\{ g _ { 1 } , g _ { 2 } , \cdots , g _ { d } \} = E n c o d e r ( \{ x _ { 1 } , x _ { 2 } , \cdots , x _ { d }  \} )$$ 
>
> <hr>
>
> <font color='red'>边类型edge type t：</font>
>
> > **Mention-Entity Edge**：如果**提及引用实体**，我们将**提及节点连接到实体节点**，**以便对提及的共同引用建模**。
> >
> > **Mention-Sentence Edge**：在一个句子中*提及的同现可能表明一种关系*。因此，**如果提及位于句子中，我们在提及节点和句子节点之间添加一条边**
> >
> > **Sentence-Sentence Edge**：只在**文档中对应的句子相邻的两个句子节点之间添加边**，以保持顺序信息。



#### Reasoning Path Generation

**1.假设**

我们的方法旨在==**通过关注 相关的 实体、提及和句子，而不是整个文档来预测关系**==。

> 为此，我们**假设：包含图中 *所有节点的子集* 的两个目标实体之间的路径提供了足够的信息来确定关系**



**2.引入两种推理路径**

根据[Xu *et al.*, 2021]—Discriminative reasoning for document-level relation extraction，我们引入以下**推理路径**，它们**连接两个目标实体，并 显式地 对我们前面讨论的四种推理技能进行建模**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230628212623441.png" alt="image-20230628212623441" style="zoom:67%;" />

> > “E”, “M”, and “S” are entity node, mention node, and sentence node, respectively, “-” denotes edge
>
> **Intra-sentence Reasoning Path:** 两个**实体之间的关系在同一个句子中共存**，以<font color='red'>**E-M-S-M-E**</font>的形式表示的路径
>
> * **句内推理路径 模拟了 两种类型的推理：==模式识别(pattern recognition)和常识性推理(common-sense reasoning)==**
>
> 
>
> **Inter-sentence Reasoning Path:** 对于**两个实体，其提及不在同一句子**，我们使用句间推理路径来建模它们之间的关系。
>
> * 可以认为它是**为句子内推理路径添加了额外的桥梁（对多个句子添加桥梁）**，如上图3所示
>
> * 根据不同类型的桥梁，引入了==**逻辑推理路径(logical reasoning path)**==和==**共指推理路径(co-reference reasoning path)**==
>
>   > **logical reasoning path**：有桥接实体，表示为<font color='red'>**E-M-S-M-E(桥接实体)-M-S-M-E**</font>形式的路径，通过分别**对包含头部、桥部和尾部实体的句子**进行**推理**来建模逻辑推理
>   >
>   > **Co-reference Reasoning Path**：**相邻句子中**两个实体之间的关系大多是通过参考来建立的；可以以<font color='red'>**E-M-S-S-M-E**</font>的形式来表示为一条路径；通过**对分别包含头实体和尾实体的 两个相邻句子 进行推理**来建模共引用推理
>
>   当然，**两种类型的推理路径只是句子间推理路径的特殊情况**，它们**只包含一个桥梁**。



**3.减少路径数量** 

**通过推理路径生成，所有实体对都可以通过至少一条路径连接**，==这些路径能够捕获实体对之间的语义和结构相关性==。

但问题是，即使我们在元路径方案下生成路径，**<font color='red'>一个实体对仍然会有大量的路径，但只有少数路径是必要的</font>**

> 因此，我们只简单地通过**限制bridge的数量来限制路径的规模，从而生成更少但更好的路径来消除不相关路径中的噪声**：
>
> > 具体来说，给定一个实体对，==我们首先在它们之间搜索 **不需要桥梁的句内推理路径**（0个桥梁），如果找不到路径，则转而搜索 **没有重复的跨句子推理路径**（1个桥梁），**逐渐放宽对桥梁数量的限制，允许更多的重复**，直到至少找到一条路径为止==



**4.验证启发式路径生成策略的有效性**，我们将生成的路径与DocRED开发集上的带注释的支持句子进行了比较。

> 结果表明，75.2%的关系实例的**支持句子被生成路径中包含的句子所覆盖**，53.5%的**支持句子完全相同**。
>
> 我们发现**每个实体对之间的路径平均包含1.8个句子**，这意味着文档中多达80%的句子可以被删除。因此，我们的策略所生成的推理路径对于关系推理是足够的和非冗余的



<hr>

####  Subgraph Extraction

**一个实体对之间的多个关系，对应于多个推理路径**

之前的工作[Huang等人，2021( **Three sentences are all you need**: Local path enhanced document relation extraction)；Xu等人，2021(Discriminative Reasoning for Document-level Relation Extraction (**DRN**)]**倾向于使用每条路径来独立地预测关系，然后汇总结果**。

相比之下，**我们在图G中的实体对周围提取一个封闭的子图G‘来整合不同的路径**；具体地说，<font color='red'>**子图G‘是G的一个诱导子图**，由在路径上 **至少出现一次的节点 和 这些节点中G中的所有边组成**</font>

> 例子：**两个目标实体不出现在同一个句子，但可以由一个桥实体或桥梁句子进行联系**，我们生成如下**两个没有重复 句子间推理路径**（逻辑推理路径和共指推理路径），然后使用这些路径提取子图。
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629000328795.png" alt="image-20230629000328795" style="zoom:50%;" />



#### Subgraph Encoding

1.在这一阶段，我们首先引入**一个超级节点（super node z）**，并将**z连接到子图中的目标实体节点**。将**超级节点嵌入$h_z$初始化为目标实体节点嵌入的最大池化**。

2.对子图使用*L*-layer stacked **R-GCN**，它在**每个层中分别对不同类型的边应用消息传递**。给定第l层的节点u，u的近邻的聚合被定义为：

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629005445785.png" alt="image-20230629005445785" style="zoom:67%;" />
>
> **Edge Type:** Mention-Entity Edge, Mention-Sentence Edge, Sentence-Sentence Edge
>
> 特别地，我们将**L设置为2**，使==**超级节点能够聚合 来自*目标实体节点*及其下属 *提及节点* 的信息，并使 *句子节点* 能够聚合来自子图中所有实体和提及节点的信息**==

​	

### 2.4 Classifcation

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629013605028.png" alt="image-20230629013605028" style="zoom:50%;" />

为了计算**实体对之间关系的概率（eh，et）**，我们同时==使用了**来自全局文档编码器 和 局部子图的信息**==

> 1.使用目标实体对的 ==**全局实体感知信息**==$$h _ { c } = W _ { c } ( [ h _ { e n } ; h _ { e t } ] )$$ 
>
> 2.超级节点嵌入$h_z^L$，提供目标实体对的 ==**本地实体感知信息**==
>
> 3.所有学习到的句子节点嵌入的最大池化，它提供了**==目标实体对的 本地上下文信息==**
>
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629014411687-1687978071198-1.png" alt="image-20230629005445785" style="zoom:67%;" />
> >
> > $h_{s_i}^{(L)}$: 句子节点的编码
>
> 4.==**实体嵌入的距离$E_d(d_{ht})$**==，其中$d_{ht}$是文档中**目标实体对的 首次提及 之间的相对距离**，$E_d(·)$为相对距离嵌入层
>
> 5.然后通过MLP传递连接的表示法:
>
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629030951537.png" alt="image-20230629030951537" style="zoom:80%;" />
> >
> > 融合了：全局实体信息，本地实体信息，本地上下文信息和实体距离信息
>
> 6.损失函数BSE
>
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629031726000.png" alt="image-20230629030951537" style="zoom:80%;" />



## 结果

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230629104737216.png" alt="image-20230629104737216" style="zoom:67%;" />

> 这些结果表明，我们提出的**模型结合了 基于路径的模型 和 基于图的模型的好处**，通过利用**设计良好的 ==推理路径 和 子图结构 来学习更具表现力的特征==**。
>
> 证明了**子图推理 提高了 句子间关系实例的性能，而句子内关系实例也从中受益。**
>
> <hr>
>
> 具有子图推理的模型始终优于没有子图推理的模型，这是因为**显式地合并图结构打破了序列建模的局限性**。此外，==**好的子图结构进一步使模型更加 关注目标实体对和相关实体，从而减少了 长距离不相关实体的注入。**==
>
> 当平均提及数较小时，模型表现不佳，这表明**单一提及所携带的信息是相当有限的**，使得关系更难预测；当**平均提及数很大**时，并不是所有的提及都是预测关系所必需的，并且**不加选择地聚合信息可能会引入不相关提及的噪声**。
>
> **子图结构使模型更关注与 目标实体对相关 的提及**，因此具有子图推理的模型随着平均提及数的增加始终保持相对较高的性能，特别是当平均提及数≥4时。



## 实验

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230714204510575.png" alt="image-20230714204510575" style="zoom: 50%;" />

test

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230714213416149.png" alt="image-20230629030951537" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230717141437629.png" alt="image-20230629030951537" style="zoom:80%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230717141456429.png" alt="image-20230717141456429" style="zoom:50%;" />
