# SAIS: Supervising and Augmenting Intermediate Steps for Document-Level Relation Extraction

| <font style="background: Aquamarine">explicitly  capture relevant contexts and entity types</font> | <font style="background: Aquamarine">evidence-based data augmentation</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| **<font style="background: Aquamarine">ensemble inference</font>** | **<font style="background: Aquamarine">Four intermediate steps</font>** |



## 问题描述

问题：对**相关的上下**文和**实体类型**进行编码更具挑战性

之前的方法都是==隐式(implicitly)==的学习

我们提出了<font style="background: Aquamarine"> 显示（explicitly）</font>的学习方法，通过**监督**和**增强RE的中间步骤**（**S**upervising and **A**ugmenting **I**ntermediate **S**teps (SAIS) ）来捕获相关的**上下文**和**实体类型**（**capture textual contexts**和**entity type information**）

* 提出的SAIS也可以更加准确的 检索相应的**支持句**
* SAIS基于**证据的数据增强**（evidence-based data augmentation）和**集成推理**（ensemble inference）进一步提高了性能，同时降低了计算成本



<font style="background: Aquamarine"> 从带有标注实体提及的文档 到 最后的输出，**在推理过程中涉及到  四个中间步骤**</font>

1. **Coreference Resolution (CR): **识别文本中的不同提及（mentions）是否指代同一个实体
2. **Entity Typing (ET):** 头尾实体的类型信息 可以过滤 不可能的关系
3. **Pooled(汇总) Evidence Retrieval (PER): **  PER来区分有和没有有效支持句子的实体对；
4. **Fine-grained(细粒度) Evidence Retrieval (FER):**  FER旨在输出更具解释性的证据，涉及到**对于实体对 仅仅有效关系的 证据句**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315035.png" alt="image-20230904112154350" style="zoom: 50%;" />

> <font color='red'>如果某个实体对有多个关系，FER可以**获取到每个关系的证据句**</font>
>
> <font style="background: Aquamarine">**CR、PER和FER捕获文本上下文；ET保存实体类型信息**</font>
>
> <hr>
>
> **先用上下文嵌入与句子嵌入的得到句子sn的重要性**
>
> **之后用头尾，关系，上下文嵌入得到句子sn重要性**



<font style="background: Aquamarine">通过精心设计的任务，明确地**监督四个中间步骤(CR, ET, PER, FER)的模型的输出**</font>，显式地教模型捕获**capture textual contexts**和**entity type information**

> 通过使用**pseudo documents** or **attention masks**来增强特定的中间步骤
>
> 只有当模型**不确定其原始预测**时，才应用这两种==**evidence-based data augmentation**==和==**ensemble inference**==
>
> * 将集成学习 (ensemble learning) 仅应用于**relation triplets**的不确定子集，从而节省了计算代价。

<hr>

==和**EIDER**比较==

> EIDER是提取对该实体对重要的句子作为证据句，并将 原始文档 和 提取的证据的预测结果结合；FER提取的是每个实体对中，每个关系的证据句
>
> 即：**FER 可以更加精确，可解释的 检索证据**

## 1. Supervising Intermediate Steps ($SAIS^O_{All}$）

在四个中间步骤中**显式地**监督模型的输出

### 1.1 Document Encoding

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315036.png" alt="image-20230904151545413" style="zoom: 40%;" />

在每个**句子**的开头和结尾插入一个分类器标记**“[CLS]”**和一个分隔符标记**“[SEP]”**；每个提及之前加入**“*”**

> 让**[CLS]**和**“*”**分别表示为每个**句子** 或 **提及 的表示**

为每个句子交替使用不同的**段标记（segment token indices）**

> 段标记通常用于将不同句子或段落在输入中区分开来。在这种情况下，句子之间的段标记是交替的，以表示它们属于不同的句子或段落。这有助于模型更好地理解文本中的结构和语境，尤其是在处理多个句子的情况下。

最后得到**token embeddings H** 和 **cross-token attention A**

###  1.2 Coreference Resolution (CR) - $l_d^{CR}$

**利用CR解析指向同一个实体的提及**

通过**group bilinear** layer计算==**mi和mj是否指代同一实体的概率**==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315037.png" alt="image-20230904154228174" style="zoom: 40%;" />



由于大多数 **提及对** 并不属于同一个实体，所以**该任务是==类别不平衡==的二分类任务**，我们在二值交叉熵的基础上**采用 <font style="background: Aquamarine">focal loss</font> 来减轻这种极端的类不平衡**：

> ==focal loss==：对正负样本的损失进行加权，使模型更**关注那些更难分类的样本，从而提高对困难样本的识别能力。**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315038.png" alt="image-20230904160805742" style="zoom:40%;" />

### 1.3 Entity Typing (ET) - $l_d^{ET}$

**类型信息可以用来过滤掉不可能出现的关系**，因此==可以通过实体的嵌入对实体类型进行预测，让嵌入中保有类型信息==

1. 通过logsumexp pooling获取实体的嵌入

2. 对于某个实体e来说，既可以作为头实体，又作为尾实体，因此**某个实体e分为： 头嵌入$e_h^{'}$  ，尾嵌入$e_t^{'}$**

   该模块的任务是想要 ==让某实体的 头嵌入$e_h^{'}$  ，尾嵌入$e_t^{'}$ 保留实体类型信息==

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315039.png" alt="image-20230904161833167" style="zoom:40%;" />

3. 对于实体e，无论其作为头实体还是尾实体，其**头嵌入 和 尾嵌入 都应该保留关于e的类型信息**，因此**由实体e的两种嵌入计算实体类型**

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315040.png" alt="image-20230904164707121" style="zoom:40%;" />

### 1.4 Pooled Evidence Retrieval (PER) - $l_d^{PER}$

通过PER明确地**引导PLM中对 每个实体对的 证据句子的 注意力**，继而==找到对实体对$(e_h,e_t)$重要的证据句集合$V_{h,t}$==

1. 对于实体对$(e_h, e_t)$，计算其**上下文嵌入(context embedding)**

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315041.png" alt="image-20230904170411611" style="zoom:40%;" />

2. 利用group bilinear layer，通过context embedding $c_{h,t}$ 和 句子嵌入$s$ 得到**某个句子是否属于pooled supporting evidence** $V_{h,t}$

   但是大多数实体对之间没有关系和证据句，因此**会有严重的<font style="background: Aquamarine">类别失衡（class imbalance）</font>问题**，因此仍使用采用 <font style="background: Aquamarine">**focal loss**</font> 来减轻这种类别不平衡

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315042.png" alt="image-20230904171206885" style="zoom:40%;" />

   > 按照EIDER来说，如果==**$s_n$是实体对的证据句，那么$s_n$中的token应该和关系预测相关，且对上下文$c_{h,t}$有更大的贡献**==
   >
   > 我们在上下文嵌入$c_{h,t}$和嵌入$s_n$的之间使用双线性函数来测量句子sn的重要性
   
   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315043.png" alt="image-20230904171714667" style="zoom:40%;" />

### 1.5 Fine-grained Evidence Retrieval (FER)  - $l_d^{FER}$

**<font style="background: Aquamarine">只涉及到拥有有效关系的实体对</font>**，所以不存在类别失衡的问题，因此没有用到**focal loss**



> PER为实体A和实体B之间的关系确定了同一组证据
>
> FER为**<font style="background: Aquamarine">实体对之间 对每个关系都确定一个的证据集</font>**，并输出更多可解释的结果。

虽然通过PER获得了对于实体对$(e_h,e_t)$重要的句子，但是为了**模型的可解释性**，显式的让模型去==进一步提取 **存在关系的那些实体对$(e_h,e_t,r)$而言 重要的句子**==

1. 对于三元组$(e_h, e_t, r)$，先获取**三元组的嵌入triplet embedding** $I_{h,t,r}$，该嵌入**融合了头 尾实体的嵌入 和 关系嵌入**

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315044.png" alt="image-20230904192515246" style="zoom:40%;" />

2. 利用group bilinear layer，通过triplet embedding $I_{h,t,r}$ 和 句子嵌入$s$ 得到 **某个句子是否属于fine-grained evidence set $V_{h,t,r}$**

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315045.png" alt="image-20230904192904580" style="zoom:40%;" />

3. 由于==FER只涉及那些**拥有有效关系的实体对**==，所以相对PER而言，类别==不平衡问题不太严重==

   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315046.png" alt="image-20230904194305873" style="zoom:40%;" />

### 1.6 Relation Extraction (RE) - $l_d^{RE}$

PER中获取了实体对$(e_h,e_t)$的**相关上下文嵌入**$c_{h,t}$，ET中获取了**对于某个实体e而言**，通过线性层得到其对应的 **头嵌入$e_h^{'}$  ，尾嵌入$e_t^{'}$**

对于**某个实体e**，其既可以作为头实体，也可作为尾实体，根据之前上下文嵌入和头尾嵌入 得到其==作为头实体和尾实体的最终嵌入==**头嵌入$e_h^{''}$  ，尾嵌入$e_t^{''}$**：

* 我们使用上下文嵌入$c_{h,t}$，让其经过线性层得到 **作为头部实体时相关的上下文嵌入 $c_h^{'}$** 以及 **作为尾部实体相关的上下文嵌入 $c_t^{'}$**

* 存有实体类型信息的 作为头实体和尾实体的  **头嵌入$e_h^{'}$  ，尾嵌入$e_t^{'}$**

* 得到最终嵌入后就可以计算<font color='red'>**三元组$(g,t,r)$的置信度**</font> 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315047.png" alt="image-20230904201347326" style="zoom:40%;" />

计算实体对为关系r的概率，并使用Adapitive threshold loss

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315048.png" alt="image-20230904210058828" style="zoom:40%;" />

<hr>

通过**最小化多任务的学习目标**来整合所有的任务:

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315049.png" alt="image-20230904210400247" style="zoom:40%;" />

> Task 属于 {CR, ET, PER, FER}；$n^{Task}$属于平衡任务的超参数

<hr>

==**在推理(Inference)过程中**== 

> 我们要预测三元组$(e_h,e_t,r)$是否有效：检查RE阶段是否满足 $L_{h,t,r}^{RE}>L_{h,t,TH}^{RE}$
>
> 我们要预测某个句子是否是三元组$(e_h,e_t,r)$的证据句：检查**PER阶段**是否满足 $P_{h,t,r,s}^{FER}> α^{FER}$，$α^{FER}$是阈值

<hr>

 the pipeline $SAIS_{All}^O$ 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309042315050.png" alt="image-20230904212753977" style="zoom: 40%;" />



##  2. Augmenting Intermediate Steps - dev/test对于难以识别的实体对用进行增强

**对于那些 难预测的实体对**，利用从FER中提取的证据句来增强$SAIS_{All}^O$ 的中间步骤 来**进一步加强RE** 

### 2.1  When to Augment Intermediate Steps - 不确定集合$U$ 和 模型置信度 $L_{h,t,r}^o=L_{h,t,r}^{RE}-L_{h,t,TH}^{RE}$

==由于三元组$(e_h, e_t, r)$个数非常多，用证据对每个实体对在预测上增强，在计算上是非常困难的==

所以我们使用<font style="background: Aquamarine">**selective prediction**</font>，**选出那些模型不确定的那些三元组集合 $U$**，==**抛弃不确定的那些三元组的关系预测**==，rejection rate为 $θ\%$（不为超参数）

> 计算对于三元组$(e_h, e_t, r)$的**置信度**：$L_{h,t,r}^o=L_{h,t,r}^{RE}-L_{h,t,TH}^{RE}$
>
> **不确定集合$U$** 由 $θ\%$（rejection rate） <font color='red'>**绝对置信度$|L_{h,t,r}^o|$较低的  三元组组成**</font>
>
> 我们**拒绝**了对$(e_h, e_t, r)∈U$的关系预测，并应用**基于证据的数据增强（2.2）来提高性能**

<hr>

==确定 rejection rate $θ\%$== 

三元组集合$U$中保存的是模型不确定的那些三元组，所以：当**rejection rate $θ\%$增大时**，**不在 集合$U$的 那些三元组的不准确率会降低**

> 因为rate变大，不在集合$U$的那些三元组 大概率就是模型可以确定对应关系的



一方面，我们希望的**关系预测的不准确性有所降低**；另一方面，我们想要一个**较低的拒绝率**，这样在一个小的拒绝集上增加数据就会产生很小的计算成本

上述要求需要对Rate进行平衡

> 最小化（Risk²+Rejection Rate²）后θ%≈4.6%
>
> 而实践中通过限制集合U的个数得到θ%=1.5%



###  2.2 How to Augment Intermediate Steps  置信度 ($SAIS^D_{All} ： L_{h,t,r}^D$） ($SAIS^M_{All} ： L_{h,t,r}^M$）

对哪些模型预测**置信度$L_{h,t,r}^o$低**的实体对，==**设计以下两种类型的 基于证据的数据增强**==

**Pseudo Document-based ($SAIS^D_{All}$）**

> 将满足<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202309061944392.png" alt="image-20230906150457448" style="zoom:40%;" />的句子构建一个pseudo document，并将其送入模型得到 三元组的confidence 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230906153137271.png" alt="image-20230906153137271" style="zoom:50%;" />

**Attention Mask-based ($SAIS^M_{All}$）**

> 通过FER得到mask，并修改了context embedding
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230906153045971.png" alt="image-20230906153045971" style="zoom: 50%;" />

<hr>

**对于三元组，将得到的三种置信度组合起来并得到最终的损失：**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230906154101917.png" alt="image-20230906154101917" style="zoom:50%;" />

> 我们在==develop set==上计算loss，并通过loss得到公式18中**训练后的blending parameter $t_r$**
>
> 公式中的$L$为**不同情况下的置信度**
>
> 对于**不确定集合$U$中**的那些三元组，通过检查**其混合置信度是否为正（$P^B_{h,t,r}>0$）**，判断实体对之间**是否存在该关系**
>
> * 与0比较是因为置信度提出是通过$L_{h,t,r}^{RE}-L_{h,t,TH}^{RE}$比较得到的，已经自带了关系r和TH的比较



<font style="background: Aquamarine">**当模型对其原始预测不确定时，我们通过Augment Intermediate Steps提升了模型性能；并在模型具有自信时不使用增强，节省计算成本时**</font>



## 总的训练和测试步骤

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230906161143866.png" alt="image-20230906161143866" style="zoom: 43%;" />

## 结果

在没有任何补充任务的情况下，SAIS的RE性能与ATLOP相当

当只允许一个互补任务时，**PER是最有效的单一任务，其次是ET**

> 虽然FER在功能上与PER相似，但由于FER只涉及具有有效关系的实体对的小子集，因此FER单独带来的性能增益是有限的。

当任务联合时，**PER和ET联合**，结合了文本上下文和实体类型信息，提供了最重要的改进者

排除CR任务后F1下降的程度最小，表明 **CR 任务关于捕捉上下文的监督信号在一定程度上可以由 PER 和 FER 任务覆盖。**

$SAIS^D_{All}$ 作为一个**hard filter**，通过直接去除预测的非证据句， 而 $SAIS^M_{All}$ 是通过**基于注意力掩码的数据增强，更温和的提取上下文**

> $SAIS^D_{All}$l获得**更高的精度**，而 $SAIS^M_{All}$获得**更高的召回率**。