# Double Graph Based Reasoning for Document-level Relation Extraction

| <font style="background: Aquamarine">heterogeneous mention-level graph (hMG)</font> | <font style="background: Aquamarine">Inference Network - path reasoning mechanism 推理路径</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |



**G**raph **A**ggregation-and **I**nference **N**etwork (GAIN) 

> 采用双图设计，以更好地应对文档级RE任务
>
> 1. 首先构建了一个**异构提及级别图（heterogeneous mention-level graph (hMG)）**，来对文档中<font color='red'>**建模不同提及之间的复杂交互，提供具有文档感知功能的提及表示**</font>。
>
>    > hMG包含两种类型的节点，即：**提及节点 和 文档节点**，
>    >
>    > hMG包含三种类型的边： 即：**intra-entity edge, inter-entity edge and document edge**, 来捕获文档中实体的上下文信息。
>
>    我们在hMG上应用图卷积网络**Graph Convolutional Network** (Kipf and Welling, 2017) ，以==**获得每个提及的文档感知表示**==。
>
>    然后，通过**合并在hMG中引用同一实体的提及**来构造	**实体级图（EG）**
>
> 
>
> 2. 基于**实体级图（entity-level graph, EG）**，我们提出了一种新的 **<font color='red'>路径推理机制 (path reasoning mechanism) 来推断实体之间的关系。</font>**
>
>    > ==该模型将更加关注有用的路径。有了这个模块，一个实体可以通过融合其提及的信息来表示，这些信息通常分布在多个句子中。==
>    >
>    > 这种推理机制允许我们的模型来**推断出实体之间的多跳关系**。



## 3. Graph Aggregation and Inference Network (GAIN)

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230620211508749.png" alt="image-20230620211508749" style="zoom: 50%;" />

###  3.1 Encoding Module

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230620214044749.png" alt="image-20230620214044749" style="zoom:67%;" />

> ==利用词嵌入，实体类型嵌入，共指嵌入（类似：DocRED）==

### 3.2 Mention-level Graph Aggregation Module

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624103424248.png" alt="image-20230624103424248" style="zoom:50%;" />

> **<font color='red'>heterogeneous mention-level graph (hMG)</font>** has two different kinds of nodes: **mention node** and **document node**
>
> > mention node，表示对一个实体的一个特定提及
> >
> > **<font color='red'>document node</font>**，旨在为整个文档信息建模：这个节点可以作为一个轴，**与不同的提及进行交互，从而减少文档中它们之间的长距离**
> >
> > * **它帮助GAIN聚合文档信息，并作为一个支点，促进不同提及之间的信息交换，特别是文档中彼此远离的信息交换**
>
> hMG的去除导致Inter-F1的显著下降，这**表明我们的hMG确实有助于提及之间的交互，特别是那些分布在长距离依赖的不同句子中**
>
> <hr>
>
>  three types of edges: 
>
> * **Intra-Entity Edge**:引用**同一实体的提及完全连接**。这样，就可以建模对同一实体的不同提及之间的交互。
> * **Inter-Entity Edge**:如果**不同的实体两个提及同时出现在一个句子中，则它们与一个实体间的边相连**。通过这种方式，实体之间的交互可以通过它们被提到的同时出现来建模。
> * **Document Edge**:**所有提及都通过文档边缘连接到文档节点**。编码器通过这样的连接，文档节点可以处理所有的提及，并启用文档和提及之间的交互。此外，两个提及节点之间的距离最多为两个，而文档节点作为一个枢轴。因此，可以==**更好地建模长距离依赖关系**==。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624105452657.png" alt="image-20230624105452657" style="zoom:67%;" />

> 使用GCN聚合来自邻居的特征，也就是说**将原始原始编码层得到的特征进行聚合**
>
> 不同的GCN层表示不同抽象层次的特征，因此为了覆盖所有层次的特征，我们**将每个层的隐藏状态连接起来形成节点（提及）n的最终表示$m_u$**
>
> **实体的表示**是聚合了对于的所有提及：$$e _ { i } = \frac { 1 } { N } \sum _ { n } m _ { n }$$

### 3.3  Entity-level Graph Inference Module

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624110044307.png" alt="image-20230624110044307" style="zoom:50%;" />

> 忽略上一模块的文档节点
>
> 将边向量化：我们**合并所有连接相同两个实体提及的实体间边**，从而得到EG中的边。
>
> > $$e _ { i j } = σ ( W _ { q } [ e _ {i}, e_j ] + b _ { q } )$$
>
> **头实体与尾实体的路径path information **：头实体eh和尾实体et通过实体eo之间的第i条路径表示为（两跳路径时示例）
>
> > $$p _ { h , t } ^ { i } = [ e _ { h o } ; e _ { o t ; } ; e _ { t o} ; e _ { o h } ]$$ 
>
> **引入注意机制**：使用实体对（eh，et）作为查询，来融合eh和et之间不同路径的信息。==**该模型将更加关注有用的路径**==
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624110724718.png" alt="image-20230624110724718" style="zoom: 50%;" />
>
> ​	

有了这个模块，**一个实体可以通过融合其提及的信息来表示**，这些信息通常分布在多个句子中。

此外，**潜在的推理线索是 由实体之间的不同路径 建模的。然后可以将它们与注意机制相结合，以便我们考虑潜在的逻辑推理链来预测关系**

### 3.4  Classifification Module

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624111317642.png" alt="image-20230624111317642" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624111433310.png" alt="image-20230624111433310" style="zoom:50%;" />

$I_{h,t}$来自三个方面

> 除了头尾实体的向量信息，还有通过之前的论文来使用对比操作加强特征表示
>
> 利用文档节点帮助**聚合跨句子信息，并提供具有文档感知的表示**；
>
> 综合推理路径信息$p_{h,t}$

使用二分类交叉熵损失进行训练

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230624111946399.png" alt="image-20230624111946399" style="zoom:50%;" />