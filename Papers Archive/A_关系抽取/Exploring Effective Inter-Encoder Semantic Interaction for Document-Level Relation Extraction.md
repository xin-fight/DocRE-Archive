# Exploring Effective Inter-Encoder Semantic Interaction for Document-Level Relation Extraction



| **<font style="background: Aquamarine">Graph-Transformer Network（GTN）</font>** | **<font style="background: Aquamarine"> Global and Local Contexts of Nodes</font>** | **<font style="background: Aquamarine">Non-entity Clue Information</font>** |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **<font style="background: Aquamarine">Guide the PLM encoder  using the Structural Information of GTN</font>** | **<font style="background: Aquamarine">Knowledge Distillation</font>** |                                                              |



## 问题描述

**==基于图的方法缺点：==**

1. **文档编码器 和 图编码器之间 交互是单向且不充分的**

   > 在编码文档图时，基于图的方法通常 只考虑实体，而忽略了可能为关系推理提供关键线索的**==非实体词==** —— 这说明从文档编码器到图编码器（PLM → GNN）的**语义转换是不够的** 
   >
   > 此外，这些方法不会直接将 **图编码器的==结构信息== 传递给 文档编码器（GNN→PLM）**，导致文档编码器不能直接从图编码器中获益

2. **图编码器通常 无法捕获文档 图中节点的全局上下文**

   > 只通过聚合其相邻节点的信息来更新节点表示。然而，这种方法只关注捕获所考虑节点的本地上下文，而忽略其全局上下文



## Graph-Transformer Network (GTN)

**<font style="background: Aquamarine">Graph-Transformer Network (GTN)</font> —— 文档编码器与图编码器之间的双向语义交互**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230913175132093.png" alt="image-20230913175132093" style="zoom:50%;" />

> 先使用PLM encoder编码文档，获得token级别上下文：[cls]作为document的上下文$h_D$，* 作为提及的上下文$h(m_j)$
>
> 再构造a ==**Heterogeneous Document Graph (HDG)**==
>
> * **三种节点**：mention node; entity node; document node
>
> * **三种边：**
>
>   > **intra-entity edge**：连接 提及和对应的实体，聚合不同的提及表示 生产 **==更好的实体表示==**
>   >
>   > **intra-sentence edge**：连接出现在同一个句子中的 提及和实体节点，==**建模不同实体之间的交互**==
>   >
>   > **document edge**：所有的提及和实体节点都和document node相连，可以有效的==**改善远距离实体之间的语义交互**==
>
> **使用Graph-Transformer Network (GTN)对图进行编码**

<hr>



**<font style="background: Aquamarine">Graph-Transformer Network (GTN)两个子层：</font>**


1. **图注意子层（the Graph-Attention Sublayer - GAM）**：同时建模文档图中 **==节点的全局和局部上下文的==**

   > **有四个注意力头——同时让GTN建模 局部 和 全局上下文信息**
   >
   > * **前三个注意力头**——从 邻居节点中 捕获被考虑的节点的 **局部上下文**
   > * **最后一个头**——以从所有其他节点中捕获该节点的 **全局上下文**。

2. **交叉注意子层（the Cross-Attention Sublayer - CAM）**：使GTN能够从 文档编码器中 **==捕获非实体的线索信息==**（non-entity clue information），从而增强模型的推理能力

<hr>

引入**<font style="background: Aquamarine">两个辅助训练任务</font>（two auxiliary training tasks）**增强文档编码器 与 GTN之间 的**双向语义交互**

1. **图节点重构（the graph node reconstruction）**可以有效地训练我们的交叉注意子层，==以增强从 文档编码器 到 GTN 的语义转换==

   > 为了实现图节点的重建，我们首先**Mask**了HDG中一些节点的特征向量，然后**训练GTN重建这些节点的原始特征**。这样，我们就可以有效地训练GTN通过==**交叉注意子层（the cross-attention sublayer）**==从PLM编码器中获得更多的线索信息

2. **具有结构感知能力的对抗性知识蒸馏（the structure-aware adversarial knowledge distillation， SA-KD）** 有效地 ==将GTN的结构信息 传输到 文档编码器中==

   > SA-KD利用 GTN的结构信息 指导PLM编码：
   >
   > * 使用了一个**鉴别器（ discriminator）**来 区分由PLM编码器 和 GTN生成的节点表示。
   > * 将PLM (student) 视为**生成器（generator）**，被训练来生成遵循 GTN (teacher) 分布的节点表示 
   >
   > 通过上述**对discriminator and the generator的交替优化**，我们**将GTN的结构信息传递到PLM编码器中。**
   >
   > ==相对传统的knowledge distillation==，我们的discriminator更宽容，不需要PLM编码器和GTN来输出完全相同的节点表示。通过这样做，我们可以有效地防止模型训练的崩溃。



### GTN子层1. Graph-Attention Sublayer — ==建模HDG(异构图)中的局部和全局信息==

将所有HDG中的node进行编码，并放入==**feature matrix $F^{(0)}\in R^{N×d}$ (N为node的个数)**==；==并**对每种类型的边**建立**adjacency matrix $E_k \in R^{N×N}$**==

<hr>

有四个注意力头——**同时让GTN 建模 局部和全局上下文信息** 

**前三个注意力头** 分别建模 heterogeneous document graph **(HDG)** 中三种类型的边 —— 关注邻居节点，==**用于捕获HDG中节点的局部上下文**==

> **对于每种类型的边**，使用**邻接矩阵$E_i$** 作为 相应的注意头部中的**注意力mask矩阵**，让其**关注邻居节点**
>
> * 图中的**intra-entity edge；intra-sentence edge；document edge**
>
> 对第$l+1$层的第$i$个attention head计算公式为：
>
> * **公式中的QKV分别用： $F^{(l)}W$表示** 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230913224306248.png" alt="image-20230913224306248" style="zoom:40%;" />

**第四个注意力头 建模==捕获HDG中节点的全局上下文==**，它允许每个节点注意到所有其他节点，所以**不使用mask操作**

为了**预防HDG的第四个注意力头丢失结构信息**，我们在每个节点上添加了**实体嵌入$emb_e$ 和 句子嵌入$emb_s$**

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230913225406662.png" alt="image-20230913225406662" style="zoom:40%;" />

### GTN子层2. Cross-Attention Sublayer — ==捕获PLM全局和局部的线索信息==

**通过这个子层，我们期望GTN可以从PLM编码器中提取线索信息，以提高模型的推理能力。**

在交叉注意子层中有**==两种类型的注意头==**

1. **global attention head**： HDG中的节点可以考虑文档中的**所有单词**来捕获==全局非实体线索信息==
2. **local attention head**：通过掩码操作，每个节点只能关注其所在的**句子中的单词**，以便获取==局部非实体线索信息==



<hr>

**==基于GTN最后的输出$F^{(L)}$==，利用双线性分类器（bilinear classifer）来预测实体对之间的关系：**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914000446286.png" alt="image-20230914000446286" style="zoom:60%;" />

> **$F^{(L)}[e_s]$和$F^{(L)}[e_o]$ 是 实体$e_s$和$e_o$的特征向量**



## Model Training

**two auxiliary tasks** —— 为了增强了PLM编码器与GTN之间的**双向语义交互**

> **the graph node reconstruction**
>
> **Structure-aware Adversarial Knowledge Distillation(SA-KD)**



**所以最后的损失函数为：**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914003156306.png" alt="image-20230914003156306" style="zoom:50%;" />

> relation classifcation loss $L_R$, 	graph node reconstruction loss $L_N$,	 SA-KD loss $L_A$



#### 1. Relation Classifcation Loss $L_R$

为了缓解不平衡的关系分布问题，我们使用了**ATL**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914003335977.png" alt="image-20230914003335977" style="zoom: 67%;" />

#### 辅助任务1. Node Reconstruction Loss $L_N$ - ==语义转换==

**为了改进了从PLM编码器到GTN的语义转换（semantic transition），引入了节点重构**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914003821336.png" alt="image-20230914003821336" style="zoom:67%;" />

> 在HDG中随机**mask一些节点**
>
> 训练GTN，重构这些节点的特征$F^{(L)}[v]$为最初的特征$F^{(0)}[v]$ （$F^{(0)}[v]$是**PLM**提取，$F^{(L)}[v]$是经过了HDG和**GAM后的特征**）
>
> ==为了恢复掩蔽节点的原始特征==，GTN 必须利用**其交叉注意子层（CAM）**，让其从PLM编码器中捕获更多的线索信息。



#### 辅助任务2. Structure-aware Adversarial Knowledge Distillation(SA-KD) $L_A$ - ==PLM学习结构==

**SA-KD的目的是将GTN中的结构信息传输到PLM编码器中。**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914005430369.png" alt="image-20230914005430369" style="zoom:85%;" />

从PLM encoder第$l$层的输出中构造node feature matrix $\hat{F}^{(l)}$，之后利用MLP discriminator $\varsigma $ 区分所考虑的节点是来自 PLM编码器的输出$\hat{F}^{(l)}$ 还是 来自GTN的输出$F^{(L)}$ 

==相对传统的knowledge distillation==，我们的discriminator更宽容，==不需要PLM编码器和GTN来输出完全相同的节点表示==。通过这样做，我们可以有效地防止模型训练的崩溃。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914010304975.png" alt="image-20230914010304975" style="zoom:67%;" />

> **将PLM编码器视为生成器（generator），并训练其生成符合GTN分布的节点表示，从而使鉴别器（discriminator ）无法区分**。
>
> 基于上述思路，可以指导PLM编码器 使用 GTN的结构信息 学习更多具有表达性的实体表示
>
> 利用==多个不同的鉴别器 将GTN中的结构信息 提取到 PLM编码器的多个中间层{l}中==



在模型训练过程中，PLM encoder和GTN训练最小化总损失$L$，鉴别器训练最大化$L_A$。

> **鉴别器是为了区分，因此$L_A$越大，说明分类效果越好**
>
> PLM生产器是为了让 PLM和GTN生产的特征没有区分，**因此$L_A$越小越好**
