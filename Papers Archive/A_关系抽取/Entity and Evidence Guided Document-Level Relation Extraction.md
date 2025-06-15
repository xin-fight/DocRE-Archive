# **Entity and Evidence Guided Document-Level Relation Extraction**

| <font style="background: Aquamarine">evidence enhanced framework</font> | <font style="background: Aquamarine">jointly train</font> | <font style="background: Aquamarine">evidence-guided attentions</font> | <font style="background: Aquamarine"> entity-guided sequences</font> |
| :----------------------------------------------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |



<font color='red'>这篇文章其实证明了： We show that evidence prediction is an important task that helps RE models perform better. 证据预测对RE任务十分有效</font>



> 首先，我们建议通过**使用注意概率作为证据预测的额外特征**，指导预先训练的LM的注意机制关注相关上下文
>
> 此外，我们没有将整个文档输入到预先训练好的lm中以获得实体表示，而是**将文档文本与头实体连接起来**，以帮助lm集中于文档中与头实体更相关的部分



**contributions**

模型有效地结合了来自证据预测的信息来指导预先训练的LM编码器，提高了关系提取和证据预测的性能。

> 我们建议为预训练的语言模型**生成多个新的实体引导的输入**:对于每个文档，我们**将每个实体与文档连接起来**，并将其作为输入序列馈送给预训练的LM编码器
>
> 我们建议使用预先训练的LM编码器的**内部注意概率作为证据预测的附加特征**。
>
> 我们的E2GRE联合训练框架接受了实体和证据的指导

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619161109996.png" alt="image-20230619161109996" style="zoom: 50%;" />

##  1. ==Entity-Guided Input Sequences-构造输入，获取头部实体h和第k个尾实体第i个关系概率==

输入序列为： “[CLS]”+ *H* + “[SEP]” + *D* +“[SEP]”（the fifirst mention of a head entity, denoted by H）

> 由于BERT的最大输入长度是512，对于任何长度超过512的输入长度，我们在输入上使用滑动窗口方法并将其分成两个块（DocRED不超过1024）：第一个块是最多512标记的输入序列；第二个块是具有偏移的输入序列，这样偏移+ 512到达序列的末端。这显示为**“[CLS]”+H+“[SEP]”+D[偏移量： end] +“[SEP]”**。
>
> ==**我们通过平均模型中重叠标记的嵌入和BERT注意概率，将这两个输入块合并。**==

我们的框架预测了**每个训练输入的Ne−1个不同的关系集，对应于Ne−1个不同的头/尾实体对**

从BERT输出中提取**头部实体嵌入和一组尾部实体嵌入**。

==**头部实体h和第k个尾实体$t_k(1<=k<=N_e-1)$之间的第$i(1<=i<=N_r)$个关系的概率**==

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619154912811.png" alt="image-20230619154912811" style="zoom:50%;" />

**最后，我们用多标签交叉熵损失来微调BERT。**

在**推理**过程中，我们将**来自同一文档的每个实体引导的输入序列的Ne−1预测关系进行分组**，以获得一个文档的最终预测集。

## 2. Evidence Guided Relation Extraction

###  2.1 Evidence Prediction-利用句子嵌入和关系嵌入学习证据句预测

**关于给定的第i个关系ri，第j句sj 为证据句的概率:**

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619162036103.png" alt="image-20230619162036103" style="zoom: 67%;" />
>
> 需要注意的是，**在训练阶段，我们在等式 2.中使用了真关系的嵌入；在测试/推理阶段，我们使用了由关系提取模型所预测的关系的嵌入**
>
> **给定头实体和第i个关系后：**
>
> * 计算某个句子$s_j$是证据句的概率 — 这种计算==仅仅考虑了句子$s_j$的编码以及关系$r_i$的编码==
> * 计算损失时，也**仅仅考虑了句子$s_j$的编码以及关系$r_i$的编码**



### *2.2 Baseline* Joint Training-初步联合训练

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619201937488.png" alt="image-20230619201937488" style="zoom:67%;" />

<hr>

### 2.3 ==Guiding BERT Attention with Evidence Prediction - 利用注意力改进证据句预测==

**BERT在以后的BERT层中学习共指和其他语义信息**

对于每一对头h和尾tk，我们引入了使用从**最后l个内部BERT层中提取的内部注意概率来进行证据预测**

模型中==***Attention Extraction***==模块：

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619205604741.png" alt="image-20230619205604741" style="zoom: 67%;" />

> 给定一对头h和尾tk，我们提取头尾标记对应的注意概率，以帮助关系提取
>
> 利用上述公式得到一个**注意概率（attention probability）张量**为：**$A˜_k∈R^{l×N_h×L×L}$**
>
> **在给定的头尾实体对（h，tk）下，通过attention probability$A˜_k$计算每个 ==句子的注意力概率表示==**
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619211507086.png" alt="image-20230619211507086" style="zoom:67%;" />
>
> > 1. 对每层**attention probability$A˜_k$**在注意力头所在的维度上应用**最大池化**，然后将多层的结果进行**平均池化**得到**$A˜_s∈R^{L×L}$**
> > 2. 根据文档中的开始和结束位置从头和尾实体令牌中提取注意力概率张量。我们对**头 和 尾 实体嵌入的所有标记的注意概率进行平均$A˜_{sk}∈R^{L}$**
> > 3. 在文档中**平均句子中每个标记的注意力**来从$A˜_{sk}$生成**句子表示**，以获得$a_{sk}∈R^{N_s}$
>
> $\star$ **通过attention probabilities $a_{sk}$以及==sentence embeddings $F^i_k$==计算某句子为证据句的概率**
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230619214432311.png" alt="image-20230619214432311" style="zoom:67%;" />

### 2.4 Joint Training with Evidence Guided Attention Probabilities

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230620103738442.png" alt="image-20230620103738442" style="zoom:67%;" />

## Result

尽管我们在关系提取方面没有最先进的性能，但我们是第一篇证明通过适当的RE和证据预测的联合训练，我们可以有效地提高两者的性能的论文。

这是由于来自预先训练的LM的**证据引导注意概率，这有助于从文档中提取相关的上下文**。这些相关上下文进一步有利于关系提取



**实体导向序列**倾向于在**RE和证据预测的任务中有助于更高的精度**；==实体引导的序列通过引导模型来关注正确的实体，从而使其在信息提取中更加精确。==

证据导向的注意往往有助于recall；这些注意有助于提供更多的指导来定位相关的上下文，从而增加对RE和证据预测的recall。



实验证明：**用于证据预测的注意概率为关系提取提供了有效的指导**
