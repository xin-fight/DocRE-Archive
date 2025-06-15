# Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction

| <font style="background: Aquamarine">Non-entity Clue</font>：定义的实体结构的一种 — 句子内非实体词（intraNE） | <font style="background: Aquamarine">structure reasoning</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |



**SSAN** (Structured Self-Attention Network)的一个**关键特征**是：**将实体结构先验表述为注意偏差**

**<font style="background: Aquamarine">开发的方法适用于各种基于转换器的预训练语言模型，让其包含结构依赖关系。</font>**



实体表现出一定的**<font style="background: Aquamarine">结构</font>**，特别是==**共指依赖关系**==。我们将这种**<font style="background: Aquamarine">实体结构 表述为 提及对之间 的独特依赖关系</font>**  

> **实体结构**：描述了实体实例在文本上的分布以及它们之间的依赖关系。
>
> * **Co-occurrence structure** 共现结构
>
>   > 对于共现结构，我们将文档**分割成句子**，并将它们视为展示提及之间互动的最小单位。
>   >
>   > 因此，"True" 或 "False" 的分布标明了**句内互动（依赖于本地上下文）**和**句间互动（需要跨句子推理）**。
>   >
>   > 我们分别用 =="**intra**" 和 "**inter**"== 表示它们。
>
> * **Coreference structure** 共指结构
>
>   > 对于共引用结构，**True表示两个提及指的是同一个实体**，因此应该一起调查和推理，而**False表示一对独特的实体**，可能在特定的谓词下相关。
>   >
>   > 我们分别将它们表示为=="**coref**" 和 "**relate**"== 的。
>
> * ==**intraNE**== - **实体提及** 与 其**句内 非实体**（NE）词之间的依赖关系。
>
> * ==**NA**== - 对于其他**句子间的非实体词**，我们假设**不存在关键的依赖性**，并将其归类为NA
>
> 
>
> ==**以实体为中心的邻接矩阵 $S=\{s_{i,j}\}$ 来表示结构**==，设输入的token sequence $x=(x_1, ... , x_n)$
>
> * $s_{i,j}$表示从$x_i$ 到 $x_j$的依赖关系，取值为 ==***{intra+coref*, *inter+coref*, *intra+relate*, *inter+relate*, *intraNE*, *NA}***==



==目的==：将这些**结构融入到自主意力中**，并遍及整个编码阶段

==做法==：我们在每个自我注意构建块中设计了<font style="background: Aquamarine">**两个可选的转换模块（alternative transformation modules）**</font>，以产生注意偏差，从而**自适应地规范其注意流**

## 问题描述

图可以很好的对实体的结构进行建模，但是：由于==**编码网络和图网络之间的 异质性(heterogeneity)**==，将 上下文推理阶段 和 结构推理阶段 隔离起来，这意味着**上下文表示首先不能从结构引导中获益**



但**结构性的依赖关系 应该包含在 编码网络和整个系统中**，步骤：

1. 我们首先在一个统一的框架下**制定上述实体结构**，在那里我们**定义了 两者之间交互的 各种提及依赖关系**
2. **SSAN** (Structured Self-Attention Network) —— 配备了一种新的自我注意机制的扩展来**建模依赖**



## 整体结构

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917210010552.png" alt="image-20230917210010552" style="zoom:40%;" />

**整体结构**被表述为一个==**以实体为中心的邻接矩阵 $S=\{s_{i,j}\}$ **==，其中包含所有来自有限依赖集的元素，设输入的==token sequence $x=(x_1, ... , x_n)$==

* ==$s_{i,j}$表示从$x_i$ 到 $x_j$的依赖关系，取值为 ***{intra+coref*, *inter+coref*, *intra+relate*, *inter+relate*, *intraNE*, *NA}***==

* 为了实际的实现，我们**将依赖 从提及级 扩展到 token级** 

  如果**提及由多个subwords组成**（如E3），我们相应的为每个subwords分配依赖关系，而一个提及的subwords之间关系为 intra+coref

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917203703575.png" alt="image-20230917203703575" style="zoom: 43%;" />

在得到**调节的注意分数**$\hat{e}^l_{i,j}$后，应用softmax操作，并对值向量进行聚合

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917204249460.png" alt="image-20230917204249460" style="zoom:43%;" />

> 平行计算**unstructured attention score** 和 **structured attentive bias**，然后将它们聚集在一起，以引导最终的自我注意流。
>
> **转换模块( transformation module)**调节从xi到xj的注意力流。因此，该模型**可从对 结构性依赖关系的指导 中受益**

### Transformation Module

$transformatin(q^l_i,k^l_j,s_{i,j})$  

对每个输入的结构信息$S$中的每一个元素$s_{i,j}$参数化，并设计了<font style="background: Aquamarine">**两个可选的转换模块（alternative transformation modules）**</font>，以**产生注意偏差**

#### 1. Biaffifine Transformation

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917220554388.png" alt="image-20230917220554388" style="zoom:50%;" />

对第$l$层中的依赖$s_{i,j}$而言：

> 将每个依赖项参数化为可训练的矩阵$A_{l,s_{i,j}}$，将query和key向量投射到一个单一维度的bias中
>
> $b_{l,s_{i,j}}$ 直接**建模 独立于其上下文的 每个依赖项的先验偏差**

#### 2. Decomposed Linear Transformation

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917221639865.png" alt="image-20230917221639865" style="zoom:50%;" />

**公式中三项分布代表**：

1. 以查询标记表示为条件的偏差

2. 以关键标记表示为条件的偏差

3. 先验偏差。

<hr>
**结构化自注意（structured self-attention）**的整体计算公式为：

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917221820415.png" alt="image-20230917221820415" style="zoom:40%;" />

> 由于这些transformation layers根据上下文**自适应地建模结构依赖关系**，因此我们在不同的层 或 不同的注意力头之间 **不共享它们**



##  SSAN for Relation Extraction

在编码阶段之后，我们通过**平均池化（average pooling）**为每个**目标实体构造一个固定的维表示**$e_i$；并使用**交叉熵**作为损失函数

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917223438498.png" alt="image-20230917223438498" style="zoom:50%;" />

## 实验细节

与DocRED一样，将entity type，entity coreference添加到嵌入中。

在最终分类之前，==还**将每个实体对的 实体相对距离嵌入 连接起来**==



==新引入的**变换层（transformation layers）与 预训练的参数之间 仍然存在分布差距**，从而在一定程度上阻碍了SSAN的改进==，为了缓解这种差距：

> 利用DocRED数据集中的distantly supervised data进行预训练，然后再在 annotated training set 上进行微调
>
> **最后的性能大大提升**





实验结果：

> 在大多数结果中，**Biaffifine 比Decomp带来了更大的性能提高**，说明Biaffifine可以更好的建模structural dependencies
