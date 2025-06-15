# 关系抽取论文

所读论文：

> [^DocRED]:  2019 **DocRED: A Large-Scale Document-Level Relation Extraction Dataset**
>
> > https://github.com/thunlp/DocRED
>
> [^ATLOP]:  2021 **Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling （ATLOP）**
>
> >  https://github.com/wzhouad/ATLOP
>
> [^EIDER]: 2022 **EIDER: Empowering Document-level Relation Extraction with Effificient Evidence Extraction and Inference-stage Fusion**
>
> > https://github.com/Veronicium/Eider
>
> [^3]: 2021 **Entity and Evidence Guided Document-Level Relation Extraction**
>
> [^GAIN]: 2020 **Double Graph Based Reasoning for Document-level Relation Extraction**  (**GAIN**)
>
> [^DRN]: 2021 **Discriminative Reasoning for Document-level Relation Extraction** (**DRN**)
>
> > https://github.com/xwjim/DRN
>
> [^Paths]: 2021 **Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction**
>
> > https://github.com/AndrewZhe/Three-Sentences-Are-All-You-Need
>
> [^SSAN]: 2021 **Entity Structure Within and Throughout: Modeling Mention Dependencies for Document-Level Relation Extraction （SSAN）**
>
> > https://github.com/BenfengXu/SSAN
>
> [^Dense-CCNet]: 2022 **A Densely Connected Criss-Cross Attention Network for Document-level Relation Extraction (Dense-CCNet)**
> [^SGR]: 2022 **Document-level Relation Extraction via Subgraph Reasoning** (**SGR**)
>
> > https://github.com/Crysta1ovo/SGR
>
> [^RSMAN]: 2022 **Relation-Specific Attentions over Entity Mentions for Enhanced Document-Level Relation Extraction (RSMAN)**
>
> >https://github.com/FDUyjx/RSMAN
>
> [^ERA]: 2022 **Improving Long Tailed Document-Level Relation Extraction via Easy Relation Augmentation and Contrastive Learning (ERA)**
>
> [^Doc-SD]: 2023 **Exploring Self-Distillation Based Relational Reasoning Training for Document-Level Relation Extraction**
>
> [^KMGRE]: 2022 **Key Mention Pairs Guided Document-Level Relation Extraction (KMGRE)**
>
> [^KD-DocRE]: 2022 **Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation (KD-DocRE)**
>
> [^CGM2IR]: 2023 **Document-level Relation Extraction with Context Guided Mention Integration and Inter-pair Reasoning(CGM2IR)**
> [^DREEAM]: 2023 **DREEAM: Guiding Attention with Evidence for Improving Document-Level Relation Extraction(DREEAM)**
> [^PEMSCL]: 2023 **Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction(PEMSCL)**
> [^DocRE-BSI]: 2023 **Exploring Effective Inter-Encoder Semantic Interaction for Document-Level Relation Extraction(DocRE-BSI)**
>
> [^LACE]: 2023 **Document-Level Relation Extraction with Relation Correlation Enhancement(LACE)**
>
> [^AA]: 2023 **Anaphor Assisted Document-Level Relation Extraction(AA)**
>
> [^PRiSM]: 2023 **PRiSM: Enhancing Low-Resource Document-Level Relation Extraction** **with Relation-Aware Score Calibration (PRiSM)**
>
> [^Rethinking Document-Level Relation Extraction: A Reality Check]: 2023 **Rethinking Document-Level Relation Extraction: A Reality Check**
>
> [^Adaptive Hinge Balance Loss (HingeABL)]: 2024 **Adaptive Hinge Balance Loss for Document-Level Relation Extraction**
>
> 







在输入中包含这些不相关的句子有时可能会给模型带来噪声，而且有害大于有益。



## 1. 模型的区别

> * Xu等人，2021(Discriminative Reasoning for Document-level Relation Extraction (DRN)[^DRN]；
> * Huang等人，2021(Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction[^Paths])
> * Peng等人，2022(Document-level Relation Extraction via Subgraph Reasoning(SGR))[^SGR]

**1.如何产生路径**

> 除了基于图的模型外，一些**基于路径的模型（DRN，Three Sentences）**试图**提取目标实体对之间的路径**，并保留足够的信息来预测关系
>
> 下面方法==**明确地**==考虑了**推理技能(pattern recognition, logical reasoning, co-reference reasoning, and common-sense reasoning)**，==可以缓解长距离实体交互建模的问题==，但它们单独处理每个路径，==**并不是所有的实体对都可以通过一条路径连接**== — 来源SGR[^SGR]

**2.对于路径使用**

> 之前的工作[Huang等人，2021；Xu等人，2021]倾向于**使用每条路径来独立地预测关系，然后汇总结果**。
>
> SGR：在图G中的实体对周围**提取一个封闭的子图G‘来整合不同的路径**。 — 来源SGR[^SGR]



<hr>

<table>
<thead>
  <tr>
    <th>模型</th>
    <th>相同点</th>
    <th>不同点</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>GAIN[^GAIN]</td>
    <td rowspan="2">使用了Reasoning Paths - 路径推理</td>
      <td>使用<b>实体图之间的路径信息</b>，向量化Entity-level Graph(EG)图中的边，<br>
          并 <b>利用边向量 和 注意力 建模头实体到尾实体之间的路径 </b><br>
          <font color="red"><b>没有显示的建立某种类型的推理技能，只是使用了头尾向量之间的路径</b></font>
      </td>
  </tr>
  <tr>
    <td>DRN[^DRN]</td>
      <td><b>显式地建模不同的推理技能</b>，以识别输入文档中每个实体对之间的关系。<br>
          将DocRE问题 <b>显示分解为三个推理子任务：句内推理（IR:(pattern recognition and common-sense reasoning)）、逻辑推理（LR）和共指推理（CR）</b><br>
          最后使用提及或者句子的HGCRep以及DLCRep构建 reasoning representation 推理表示
      </td>
  </tr>
</tbody>
<table>

<hr>





## 2. 计算

### 2.1 单词特征组成

> **实体类型嵌入**：通过将使用嵌入矩阵分配给单词的实体类型（例如，PER、LOC、ORG）映射到一个向量中来获得的
>
> **共引用嵌入**：与**同一实体对应的提及被具有相同的实体id**，这由其在文档中首次出现的顺序决定。实体id被映射为作为共引用嵌入的向量中。

单词嵌入 + 实体嵌入 + 共指嵌入

* DocRED: 对于每个单词，输入编码器的特征是其**GloVe单词嵌入**（Pennintonetal.，2014）、**实体类型嵌入**和**共指嵌入**的连接。[^DocRED]

* GAIN: 继Yao等人（2019）之后，对于D中的每个单词wi，我们首先将**其单词嵌入与实体类型嵌入和共指嵌入**连接起来[^GAIN]

* DRN：形式上，我们将每个**单词的嵌入we**与其**实体类型的嵌入wt**以及**共指嵌入wc**作为**单词b的表示**[we：wt：wc]。[^DRN]

单词嵌入 + 实体嵌入

* SGR：word embedding and entity type embedding[^SGR]

如果文档长度超过了PrLM的最大位置，则该文档将被编码为多个重叠块，并对重叠块的上下文化嵌入进行平均



### 2.2 提及，实体，句子的表示

#### 2.2.0 mention预操作 - 实体标记技术[^ATLOP][^ERA]

**entity marker technique**

> 我们通过在**提及的开始和结尾**插入一个特殊的符号**“*”**来标记实体提到的位置 — 它改编于**实体标记技术（entity marker technique）**
>
> > 参考论文：(Zhang et al., 2017) Position aware attention and supervised data improve slot fifilling
>
> 文档由编码器进行一次编码，所有实体对的分类都基于相同的上下文嵌入。==**我们将提及开始的“*”作为提及嵌入**==



#### 2.2.1 mention / sentence - pooling表示

average pooling；max pooling；logsumexp pooling 



提及表示mi是通过对单词进行max-pooling operationon the words得到的，实体是提及的log-sum-exp[^LACE]

<hr>

#### 2.2.2 mention / sentence - 使用head word

Here, inspired by relation learning (Baldini Soares et al., 2019), we **use the hidden state of the head word** in one mention or one sentence to denote them for simplicity.[^DRN]

> 为了简化表示，**我们使用一个提及或一个句子中head word的隐藏状态来代表它们**。具体操作如下：
>
> 1. 确定头词：在提及或句子中确定头词，即主要掌握语法结构或承载最重要含义的单词。
> 2. 获取隐藏状态：利用神经网络或语言模型处理提及或句子，提取头词的隐藏状态。隐藏状态表示了该词的语义和句法特征的编码信息。
> 3. 实体表示：将获取的隐藏状态作为相应实体的表示或符号。通过使用头词的隐藏状态，而不是显式提及整个提及或句子，作者简化了对实体的引用。
>
> 通过使用头词的隐藏状态，作者实现了在研究背景下更简洁、计算效率更高的实体表示和引用方式。
>
> <hr>
>
> 选取头词的方法通常根据特定的语法规则、句法分析或领域知识来进行。以下是一些常用的方法：
>
> 1. 语法规则：根据语法规则，头词通常是一个短语中的核心成分或在句子中承担主要功能的单词。例如，在英语中，动词通常是句子的头词。
> 2. 句法分析：使用句法分析技术（如依存句法分析或成分句法分析），可以自动识别头词。这些技术可以通过分析词与词之间的关系来确定主谓关系、修饰关系等，从而找到头词。
> 3. 领域知识：对于特定领域的文本，领域专家可能具有关于头词的先验知识。他们可以基于对该领域的理解和经验来选择头词。
>
> 需要注意的是，头词的选择可能会有一定的主观性和语境依赖性。在具体应用中，可以根据任务需求和领域特点来选择合适的头词选取方法。



#### 2.2.3 实体表示 - LogSumExp pooling[^ATLOP][^EIDER][^ERA]

==a LogSumExp pooling==：将多个Embedding融合为一个Enbedding — **由提及计算实体Embedding，由单词计算句子Embedding**

对于某个实体的所有提及，我们应用 **logsumexp pooling**，一个最大池的平滑版本，得到实体嵌入$h_{e_i}$

> **logsumexp pooling** - 这种汇集积累了来自文档中提及的信号。它在实验中比平均池化具有更好的性能。
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529160818384-1687086064756-2.png" alt="image-20230529160818384" style="zoom:50%;" />
>
> 它特别适用于处理数值稳定性问题和处理大量或小量的值。
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230529155845083-1687086064755-1.png" alt="image-20230529155845083" style="zoom:67%;" />
>
> ```
> for start, end in e:
>     if start + offset < c:
>         # In case the entity mention is truncated due to limited max seq length. 
>         e_emb.append(sequence_output[i, start + offset])
>         e_att.append(attention[i, :, start + offset])
> 
> if len(e_emb) > 0:
>     m_num = len(e_emb)
>     e_emb = torch.stack(e_emb, dim=0)
>     e_emb = torch.logsumexp(e_emb, dim=0)
>     
>     e_att = torch.stack(e_att, dim=0)
>     e_att = e_att.mean(0)  # (h, seq_len) take average on att anyway..
> ```

#### 2.2.4 实体表示 - 融合

##### Type

Han and Wang [2020] utilize a **document-level entity mask method with** **type information** (**DEMMT**) - 英文综述

##### RSMAN[^RSMAN] - 利用关系嵌入

**生成固定实体表示 的简单池化操作 可能会混淆不同提及的语义**，因此==当实体涉及**多个有效关系**时，会降低关系分类的性能==

> **<font color='red'>所以需要实体的表示 特定于关系</font>** 

引入了注意力机制，利用**提及级别的特征来生成灵活的实体表示，以适应不同候选关系**。最后得到 ==**实体$e_i$特定于关系$r$的表示**==而非 固定的表示

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230703164439951.png" alt="image-20230703164439951" style="zoom: 50%;" />

##### CGM2IR[^CGM2IR] - 利用上下文嵌入

在获得实体对的上下文特征后，我们使用它们作为查询，并执行**交叉注意力**，从头或尾实体的提及嵌入中汇集与实体对相关的实体表示。

> 利用实体对的上下文嵌入c作为queries，提及嵌入作为KV得到最后的实体嵌入有实体对上下文->实体嵌入每个实体的表示都不固定

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231121023107976.png" alt="image-20231121023107976" style="zoom: 67%;" />

### 2.3 预测多label

关系抽取是多标签任务，需要模型输出多个标签：

* **对每个关系进行判断是否是该实体对的标签**$$P ( r | e _ { s } , e _ { o } )=sigmoid(...)$$（二分类），**测试时直接使用该模型对每个关系进行判断**

  > 使用binary cross-entropy（BCE），让模型对符合的关系置信度变大，二分类损失函数[^GAIN]
  >
  > 对每个实体e，通过提及和关系r直接的重要程度，得到针对关系r的实体嵌入$e^r$，通过二分类得到该关系r是否是实体的关系[^RSMAN]

  

* 使用计算$$P ( r | e _ { s } , e _ { o } )$$得到概率，并使用一个**阈值**，**大于该阈值**则说明该关系为实体对的标签

  > 使用**对比学习**构造损失函数[^ATLOP]
  >
  > 使用binary cross-entropy，**让模型对符合的关系的置信度变大**，之后在development set上进行根据预测结果的置信度对预测结果进行排序，并通过开发集上的F1得分从上到下遍历该列表，选取 **最大F1对应的得分值 作为阈值θ**[^DRN]

### 2.4 计算权重

1.利用softmax，将 计算的**两个向量的相似度 归一化** [^RSMAN]

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230724111553685.png" alt="image-20230724111553685" style="zoom:50%;" />

2.使用**注意力（Attention）**构建查询向量Q，键值对K，V [^KD-DocRE]

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230724111748851.png" alt="image-20230724111748851" style="zoom:55%;" />

### 2.5 计算Thresholding 损失

#### 2.5.1 ATL

**Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling**[^ATLOP]

>  **Adaptive Thresholding Loss**(**ATL**) 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230919015655860.png" alt="image-20230919015655860" style="zoom: 50%;" />
>
> 正类高于TH，负类低于TH



#### 2.5.2 AFL- ==Focal loss==

**Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation**[^KD-DocRE]

> **Adaptive Focal Loss** (**AFL**) as an enhancement to **Adaptive Thresholding Loss**(**ATL**) for ==**long-tail classes**==
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902151139113.png" alt="image-20230902151139113" style="zoom:33%;" />
>
> 对于positive class，我们让正例的logit **分别** 与阈值类TH的logit进行排序
>
> 我们的损失是为了更多地**关注low-confifidence classes**，从而可以对 长尾类(long-tail classes) 进行更好的优化。
>
> <hr>
>
> 这与原始的ATL不同，在原始的ATL中，用softmax函数将**所有的正类都一起排序**。



#### 2.5.3 pairwise moving-threshold loss

**Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction**[^PEMSCL]

> **pairwise moving-threshold loss** —— **实现了 部分顺序的相对阈值**
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902160634987.png" alt="image-20230902160634987" style="zoom: 30%;" />
>
> ==正关系 **没有相互比较，它们的相对排名也没有建模**（负关系也直接也没有相互比较）==，因为我们**只寻找真实的关系集，而 不关心 它们的相对真实性程度**
>
> <hr>
>
> 虽然之前的工作**ATLOP,KD-DocRE**采用类似的阈值机制，但**他们学习了 所有关系（或一组关系）与阈值类的总顺序**，==这不可避免地减少了每个关系概率与阈值的概率之间的差异（因为总概率和为1）==
>
> 但是建模关系之间的多余排序，**浪费了有限的概率质量**(总价值1.0)，**不利于多标签问题**，并且**不可避免地减小了每个关系的概率与阈值的概率之差**，如图所示，当多个关系进行比较时，相对于单独关系比较，之间的差距会被减小
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902173540308.png" alt="image-20230902173540308" style="zoom: 25%;" />



### 2.6 特殊损失函数

#### 2.6.1 Kullback Leibler (KL) Divergence loss -- 最小化两个向量的统计距离

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231029220526982.png" alt="image-20231029220526982" style="zoom:50%;" />

> DREEAM[^DREEAM]
>
> 参考：KL散度理解 - https://zhuanlan.zhihu.com/p/39682125



## 3. Tips in papers

**EIDER**[^EIDER]

> 假设**句间对的瓶颈是定位相关上下文，而相关上下文往往遍布整个文档**。EIDER在训练中学习捕捉重要的句子，并在推理中更多地关注这些重要的句子。
>
> GAIN的Inter F1比ATLOP高0.70，ATLOP的Intra F1比GAIN高0.16，说明文档级图在多跳推理中可能是有效的
>
> EIDER不涉及显式的多跳推理模块，它在Inter F1中仍然明显优于基于图的模型。



**Entity and Evidence Guided Document-Level Relation Extraction**[^3]

> 在初步实验中，我们发现，目前的==**模型在对黄金证据句子进行训练和评估时，才能在DocRED上实现约87%的RE F1**==



 What does BERT look at? an analysis of BERT’s attention

>  ==BERT learns co-reference and other semantic information in later BERT layers==[^3]
>
> BERT（Bidirectional Encoder Representations from Transformers）是一种流行的预	训练语言模型，通过利用基于Transformer的架构学习上下文化的单词表示。**尽管BERT的初始层主要捕捉基本的句法和词级信息，但是后续的BERT层可以学习更复杂的语义信息，包括共指（co-reference**）。
>
> 共指消解是指识别文本中指代同一实体的提及的任务。BERT学习共指的能力归功于其多层架构，使其能够捕捉跨不同上下文中的单词和实体之间的依赖关系和关系。
>
> 当BERT通过多个层的自注意力和前馈神经网络处理输入文本时，它逐渐融合来自前面和后面单词的上下文信息。这种**上下文信息帮助BERT捕捉不仅是局部词义，还包括文本中更广泛的语义关系和依赖，如共指。**
>
> 通过利用后续层学习到的分层和上下文化的表示，BERT能够有效地捕捉共指和其他语义信息，使其成为各种自然语言理解任务的强大模型，包括实体链接、共指消解和文本理解等。



**GAIN**[^GAIN]

> 设立文档节点，**让所有提及都通过Document Edge连接到文档节点**。编码器通过这样的连接，文档节点可以处理所有的提及，并启用文档和提及之间的交互。此外，**两个提及节点之间的距离最多为两个，而文档节点作为一个枢轴**。因此，可以==**更好地建模长距离依赖关系**==。



**Doc-SD**[^Doc-SD]

> 实验结果中，基于GNN的模型 比利用 **实体对 之间的依赖性进行关系推理的模型获得的结果更差**，如DocuNet-BERT和KD-BERT
>
> 表明 **==实体对之间的依赖关系== 比 实体之间或提及之间的依赖关系 对关系推理更有用。** 



**KMGRE**[^KMGRE]

> **Transformer模型（Vaswani等，2017）可以看作是一个全连接的图神经网络**
>
> <hr>
>
> 上面这句话根据Transformer模型（Vaswani等，2017）的结构和特点得出的。Transformer模型是一种用于自然语言处理任务的架构，其主要由自注意力机制和前馈神经网络组成。
>
> 在Transformer模型中，自注意力机制允许模型在编码和解码过程中**捕捉输入序列中不同位置之间的依赖关系**。这可以被视为对图数据中节点之间连接的建模，而节点之间的连接可以被看作是一个全连接的图。



**KMGRE**[^KMGRE]

> 直接建模**提及级关系**更为合理。
>
> 但是在实验过程中发现：**直接将实体对的标签分配给关键提及对可能会导致大量的错误标记**
>
> * 文章中 — **融合提及级别的预测**可以避免 在多标签情况下 由于错误标记的提及对 而给模型带来错误的引导。







## 4. Evaluation Metrics - 用在**Analysis&Discussion**

> **Analysis & Discussion** 
>
> <font color='red'>**写论文思路：进行实验分析时，如果取出网络某个模块，某个性能下降，说明该模型有利于提高该性能，最后可以按照具体样本结果示例进行分析**</font>
>
> Intra-F1/Inter-F1 scores；Infer-F1 scores[^GAIN] [^Doc-SD]
>
> Entity Distance；Average Number of Entity Mentions[^SGR]



 Following prior studies (Yao et al., 2019) - DocRED: A large-scale document-level relation extraction dataset.

 ==**relation extraction**==

> **F1**分数
>
> **Ign F1**测量不包括训练和开发/测试集所共享的关系的F1分数
>
> 
>
> **Intra F1**测量在共现(**句内**)实体对上的性能 [^GAIN]
>
> **Inter F1**评估没有**专有名词提及共现**的**句间**实体对 [^GAIN]

==**多条推理能力**== [^GAIN]

>**Infer-F1 scores** 只考虑在关系推理过程中所涉及的关系—当存在$e_h-r_1->e_o-r_2->e_t$是否有$e_h-r_3->e_t$我们考虑黄金关系事实r1、r2和r3

==**evidence extraction**==

>  F1 score (denoted as **Evi F1**)
>
> **PosEvi F1**仅测量积极实体对（即非na关系）的证据的f1得分。

**从==实体距离== 的角度来检验模型的性能**

> **Entity Distance**：根据实体距离来检验模型的性能，它被转换为文档中**目标实体对的第一个提及之间的相对距离**，并报告了F1分数；
>
> 为了检测某个模型是否能够更好的建模远距离实体

**从==多提及中聚合信息==的能力**

> **Average Number of Entity Mentions**：我们根据目标 **实体对的平均提及数** 来检查模型的性能。
>
> 为了检测某个模型是否能够更好的**建模多提及**的实体

**下图自定义四种Reasoning Pattern上的预测精度**

> **Infer-Ac**度量了在数据集中符合**这四种推理模式的关系三元组的预测精度**。在这个指标上，我们的模型也明显超过了以前的基线
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230706173312500-1688711877587-1.png" alt="image-20230706173312500" style="zoom:33%;" />

## 5. Dateset

**Dataset Statistics**[^EIDER]

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230621113332386.png" alt="image-20230621113332386" style="zoom:67%;" />

**A Densely Connected Criss-Cross Attention Network for Document-level Relation Extraction**[^Dense-CCNet]

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231008222741870.png" alt="image-20231008222741870" style="zoom: 33%;" />



**Rethinking Document-Level Relation Extraction: A Reality Check**[^Rethinking Document-Level Relation Extraction: A Reality Check]  Entity mention statistics on three datasets

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231204183229572.png" alt="image-20231204183229572" style="zoom:50%;" />



### 5.1 DocRED[^DocRED] / Re-DocRED / Revisit-DocRED

> DocRED，CDR，GDA三种数据集的比较在5.2节后



**DocRED does not have documents longer than 1024**

 a large scare human-annotated DocRE dataset containing 5053 documents from Wikipedia and Wikidata. It consists of 5053 documents, 132375 entities, and 56354 relations mined from Wikipedia articles

> 我们在DocRED（Yao等人，2019）上评估了我们的模型，这是一个由维基百科和维基百科构建的大规模人类注释数据集。DocRED共有96种关系类型，132,275个实体和56,354个关系事实。**DocRED中的文献==平均包含约8句话==，只能从多个句子中提取的关系事实超过40.7%。此外，61.1%的关系实例需要各种推理技能，如逻辑推理**（Yao et al.，2019）。我们遵循数据集的标准分割，3053个文档用于培训，1000个用于开发，1000个用于测试。
>
> 
>
> 1. 数据集DocRED，其中包含约7%的多关系实体对[^LACE]
>
> 2. the **7** most frequent relations account for **55%** of the total relation triples；  about **97%** of all entity pairs have the **NA label**.[^PRiSM]



**Reasoning Types**

> 我们从开发和测试集中随机抽取300个文档，其中包含3,820个关系实例，并手动分析提取这些关系所需的推理类型。表2显示了我们的数据集中主要推理类型的统计数据。从关于推理类型的统计数据中，我们有以下观察结果：
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230628105634066.png" alt="image-20230628105634066" style="zoom: 50%;" />
>
> (1)大部分关系实例（61.1%）需要推理识别，通过简单**模式识别**只能提取38.9%，这说明推理对于文档级RE是必不可少的。
>
> (2)在与推理相关的实例中，大多数（26.6%）需要**逻辑推理**，其中两个相关实体之间的关系是由一个桥梁实体间接建立的。逻辑推理要求RE系统能够建模多个实体之间的交互。
>
> (3)有相当数量的关系实例（17.6%）需要**共引用推理**，其中必须首先执行*共引用解析*，以识别丰富上下文中的目标实体
>
> (4)同样比例的关系实例（16.6%）必须基于**常识推理**进行识别，其中读者需要将文档中的关系事实与常识结合起来，以完成关系识别。
>
> 总之，DocRED需要丰富的推理技能来综合文档中的所有信息。

**Inter-Sentence Relation Instances**

> 句子间的关系的实例。我们发现**每个关系实例平均与1.6个支持句子相关**，其中**46.4%的关系实例与多个支持句子相关**。此外，详细的分析显示，**40.7%的关系事实只能从多个句子中提取出来**，说明DocRED是一个很好的文档级RE的基准。我们还可以得出结论，对多个句子的阅读、综合和推理能力对于文档级的RE是至关重要的。

**distantly supervised data** 

> 它利用一个精细的BERT模型来识别实体，并将它们链接到Wikidata。然后通过远程监督获得关系标签，按规模生成**101873个文档实例**



<hr>

其他版本

> 由于原始的DocRED有大量的假阴性样本，我们对它的**两个重新注释版本**进行了实验，即**Revisit-DocRED**（Huang等人，2022）和**Re-DocRED**（Tan等人，2022）[^KMGRE]
>
> 我们建议使用[Re-DocRED](https://arxiv.org/abs/2205.12696)数据集来完成此任务。该数据集是原始DocRED数据集的修订版本，解决了DocRED中的漏报问题。与 DocRED 相比，在 Re-DocRED 上训练和评估的模型获得了大约 13 F1。Re-DocRED 数据集可以在以下网址下载： [ https://github.com/tonytan48/Re-DocRED](https://github.com/tonytan48/Re-DocRED)。Re-DocRED 的排行榜托管在 Paperswithcode 上：https://paperswithcode.com/sota/relation-extraction-on-redocred。[^KD-DocRE]
>
> <hr>
>
> **[Revisit-DocRED](https://github.com/AndrewZhe/Revisit-DocRED)**				**[Re-DocRED](https://github.com/tonytan48/Re-DocRED)** 
>
> KMGRE[^KMGRE] 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230708182402517.png" alt="image-20230708182402517" style="zoom:25%;" />
>
> 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230914161734102.png" alt="image-20230914161734102" style="zoom: 33%;" />
>
> > 与DocRED和REDocRED不同，**Revisit-DocRED的训练集包含了大量的假阴性样本**，但它的测试集却没有。 [^DocRE-BSI]
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20231113165310879.png" alt="image-20231113165310879" style="zoom: 50%;" />
>
> > **Re-DocRed数据集统计，Anaphors是代词** [^AA]
>
> <hr>
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230829205220171.png" alt="image-20230829205220171" style="zoom:50%;" />
>
> > 值得注意的是，Re-DocRED引入了新的关系三元组，但没有提供准确的证据句子 —— DocRED和Re-DocRED训练集中关系三元组的统计：rel.表示关系三元组，**rel.w/o evi.代表没有证据句的关系三元组**[^DREEAM]
>
> <hr>
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20240317110002756.png" alt="image-20240317110002756" style="zoom:50%;" />
>
> > Re-DocRED[^Adaptive Hinge Balance Loss (HingeABL)]



### 5.2 CDR / GDA 

CDR（Li et al.，2016）**只有一种关系和两种实体**，这可能不是关系推理的理想测试平台。

> CDR（Li等人，2016年）。它是一个生物医学数据集，由**1500**个PubMed摘要组成，这些摘要被平均分为三组，用于训练、开发和测试。在这个数据集上，该模型有望预测化学实体和疾病实体之间的**二元关系**。



GDA(Wu et al. 2019)中超过85%的关系是句子内的关系，**二分类任务**

> 该数据集是一个大规模的生物医学数据集，通过远程监督的方法由MEDLINE abstracts构建。
>
> GDA包含**29,192**个文档作为训练集，**1,000**个文档作为测试集。它只包含化学实体和疾病实体之间的**一个目标关系**，即化学诱导疾病。我们遵循Tang等人（2020年）的研究，将培训集分为两部分，即23,353份培训文档，以及设置5,839份开发文档。



<hr>

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230917223635166.png" alt="image-20230917223635166" style="zoom:50%;" />

> 所有**三个数据集**平均每个文档包含超过24次提及，每个句子平均包含大约3次提及。这些统计数据进一步证明了文档级关系提取任务中实体结构的复杂性[^SSAN]



### 5.3 HacRED

调用模型[^ERA]

Qiao Cheng,    HacRED: A large scale relation extraction dataset toward hard cases in practical applications.

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230705222558474.png" alt="image-20230705222558474" style="zoom:33%;" />



### 5.4 DWIE

调用模型[^KMGRE]

> 链接：https://github.com/klimzaporojets/DWIE
>
> **DWIE** (Zaporojets et al., 2021) 是一个实体为中心的多任务数据集，包含602/98/99文档，分别用于训练、验证和测试。在DWIE数据集中，平均**每个实体对包含3.97个提及对**。大约**26%的表达关系的实体对有一个以上的关系标签。**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230708182315140.png" alt="image-20230708182315140" style="zoom: 15%;" />



## 6. Effificiency Comparison

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230618105638342.png" alt="image-20230618105638342" style="zoom:67%;" />

> 参考论文[^EIDER]

 

## 7. PLM

**BERT**

> BERT是最早发现Transformer在大规模语料库上预训练语言模型取得成功的工作之一。具体而言，BERT使用Masked Language Model和Next Sentence Prediction在BooksCorpus和Wikipedia上进行预训练。BERT有两种配置，分别是Base和Large，其中Base包含12个自注意力层，Large包含24个自注意力层。它可以轻松地在各种下游任务上进行微调，产生具有竞争力的基准性能。

**RoBERTa**

> RoBERTa是BERT的优化版本，它移除了Next Sentence Prediction任务，并采用了更大规模的文本语料库以及更多的训练步骤。目前，RoBERTa是优秀的预训练语言模型之一，在各种下游自然语言处理任务中表现优于BERT。

**SciBERT** 

> SciBERT采用与BERT相同的模型架构，但是是在科学文本上进行训练的。它**在一系列科学领域任务中表现出明显的优势**。在本文中，我们提供了在两个生物医学领域数据集上使用SciBERT初始化的SSAN模型。
