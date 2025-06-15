# EIDER: Empowering Document-level Relation Extraction with Efficient Evidence Extraction and Inference-stage Fusion

| <font style="background: Aquamarine">evidence enhanced framework</font> | <font style="background: Aquamarine">silver evidence labels</font> | <font style="background: Aquamarine"> adaptive-thresholding loss </font> | <font style="background: Aquamarine"> jointly train</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------------: |

​	

典型的DocRE方法盲目地将完整的文档作为输入，而文档中的一个句子子集，作为证据，通常足以让人类预测一个实体对的关系。

提出了==**evidence enhanced framework**==：通过有效地提取证据和有效地融合所提取的证据进行推理，增强了DocRE的能力

> 我们首先用一个**轻量级的证据提取模型（lightweight evidence extraction model）**来联合训练一个RE模型
>
> 我们进一步设计了一个简单而有效的推理过程，**对提取的证据和完整的文档进行RE预测**，然后通过一个**混合层融合预测（a blending layer）**。	

* 这使得模型可以**专注于重要的句子，同时仍然可以访问文档中的完整信息**
* **在输入中包含这些不相关的句子有时可能会给模型带来噪声，而且有害大于有益。**



它能有效地提取证据，并有效地利用所提取的证据来改进DocRE

在 **训练** 过程中，我们通过==使用**多任务学习联合提取关系和证据**来增强DocRE==



关于证据提取存在**两个主要挑战**:

>第一个挑战是由于训练一个额外的任务而造成的**内存和运行时开销**
>
>> 相比之下，EIDER使用了一个**更简单的证据提取模型**，它可以适用于一个GPU，并且只需要95 min的运行时。
>
>第二个挑战是，人类**注释的证据句子成本很高，而且严重依赖它们**限制了模型的适用性
>
>> 因此，我们设计了几个启发式规则，在证据注释不可用的情况下构建**银标签(silver labels)**。我们观察到，当使用我们的银色标签(silver labels)进行训练时，EIDER仍然能提高RE性能，有时甚至与使用黄金标签相当。



对于**利用 推理 中的证据来进一步增强DocRE**

> 一种简单的方法是**直接用提取的证据替换原始文档** - 论文：Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction
>
> > 然而，由于没有系统能够完美地提取证据，仅仅依赖提取的句子可能会遗漏重要信息，在某些情况下损害模型的性能
> >
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230607014434642.png" alt="image-20230607014434642" style="zoom:50%;" />
>
> 为了避免信息丢失，我们**我们通过一个融合层(a blending layer) 将原始文档的预测结果 和 提取的证据 进行融合**
>
> > 这样，==**EIDER更注意提取的重要句子，同时仍然可以访问文档中的所有信息**==。实证分析表明，去除任何一个源都会导致性能退化。
>
> **EIDER在 句间实体对 上最为重要，这表明利用证据在对多个句子进行推理时尤其有效**



**Contributions**.

> 1. 我们提出了一种**有效的联合关系 和 证据提取模型**，允许这两个任务在不严重依赖证据注释的情况下相互增强。
> 2. 我们设计了一个**简单有效的DocRE 推理过程**，增强提取的证据，使更关注重要的句子，没有信息损失。
> 3. 我们证明了我们的证据增强框架在三个DocRE数据集上优于最先进的方法。

<hr>

在本研究中，我们提出了一个证据增强的EIDER框架，它通过**联合 关系 和 证据提取 和 提取的证据 融合**来改进DocRE。

> 在训练中，RE和证据提取模型为彼此提供额外的训练信号，并**相互增强**。该联合模型在时间和记忆上都是有效的，并且**不严重依赖于人类对证据的注释**。
>
> 在推理过程中，将对**原始文档和提取的证据的预测结果相结合**，鼓励模型**在关注重要的句子的同时减少信息损失**。



## 3 Methodology

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230612231120183.png" alt="image-20230612231120183" style="zoom: 80%;" />

EIDER框架的图示如图所示。**在训练中，我们使用多任务学习共同提取关系和证据**，其中两个任务有自己的分类器并共享基础编码器(第3.1节)。**在推理中，我们使用混合层融合原始文档和提取的证据的预测**(第3.2节)。

在证据注释不可用的情况下，我们还提供了几个启发式规则来构建**银色证据标签(silver evidence labels)**作为替代方案



### 3.1 Joint **Relation** and **Evidence** Extraction

在我们的框架中，我们使用多任务学习的证据提取模型**联合训练**关系提取模型。

**实体编码：** — ==ATLOP==

> 用\*放在提及前后，用前面的*表示该提及，对该实体所有提及使用**LogSumExp pooling**计算实体编码$e_i$
>
> 为了捕获与**每个实体对相关的上下文**，我们基于预先训练的编码器的**我们首先计算他们的上下文感知表示（zh，zt）通过结合他们的实体嵌入**计算其==**上下文嵌入$c_h$**==
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230620164826108.png" alt="image-20230620164826108" style="zoom: 50%;" />
>
> > 而$A∈R^L$是他对文件中所有标记的关注，通过平均他的提及级关注来获得



**Relation Classifier 关系分类:** - **Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling**

> 预测一个实体对之间的关系我们首先通过结合他们的==**实体嵌入(entity embeddings)**$(e_h, e_t)$==和他们的==**上下文嵌入(context embedding)** $c_{h,t}$== 来计算他们的==**上下文感知表示(context-aware representations)$(z_h，z_t)$**==
>
> 然后利用一个**双线性函数（a bilinear function）**来计算eh和et之间存在的关系的可能性
>
> 由于模型对不同的实体对可能有不同的置信度，我们应用自适应阈值损失（Zhou et al.，2021），学习一个虚拟关系类TH作为每个实体对的==**动态阈值**==
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230613222810116.png" alt="image-20230613222810116" style="zoom:60%;" />
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230613222833372.png" alt="image-20230613222833372" style="zoom: 60%;" />



**Evidence Classififier 证据分类:**

>  to obtain **sentence embedding $s_n$**，we apply a **LogSumExp pooling** over all the tokens in $s_n$
>
> 如果$s_n$是（eh，et）的证据句，则$s_n$中的token将与关系预测相关，并且对$c_{h，t}$的贡献更大，我们使用上下文嵌入$c_{h、t}$和句子嵌入$s_n$之间的双线性函数来**衡量句子$s_n$对 实体对（eh，et）的重要性**
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230613232159389.png" alt="image-20230613232159389" style="zoom:67%;" />

**Effificiency Considerations:**

> EIDER 在记忆和训练时间上都显著提高，有两个原因:
>
> > 之前的 E2GRE对每个 (entity, entity, sentence, relation) 元组进行证据预测，这需要昂贵的计算，特别是当|R|很大时
> >
> > 相比之下，我们观察到**大多数实体对只有一组跨关系的证据，因此对每个实体对只预测一组证据**，减少了计算量
>
> > E2GRE将具有r = NA的实体对的证据标签视为一个空集。然而，这些**实体对可能仍然涉及到超出预定义的关系集R之外的一些关系**
> >
> > 我们只在具有**至少一个非na关系的实体对上训练证据提取模型**（这只占所有实体对的一小部分（例如，在DocRED中为2.97%））



### 3.2 Fusion of Evidence in Inference

在==推理==中，我们使用**混合层（a blending layer）**融合了对原始文档的预测和提取的证据

> 我们**将原始文档 和 提取的证据的预测结果结合**起来，若证据注释不可用，这些结果可以由我们**证据分类器(第3.1节)学习**，也可以由我们的**启发式规则(第3.3节)**构建

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230615155910371.png" alt="image-20230615155910371" style="zoom:67%;" />

RE模型在原始文档获得的一组关系预测结果$S_{h,t,r}^{(O)}$，在pseudo document获得的结果为$S_{h,t,r}^{(E)}$，最后，我们通过一个混合层聚合两组预测来融合结果：

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230829151837031.png" alt="image-20230829151837031" style="zoom: 33%;" />





### 3.3 Heuristic Evidence Label Construction - 银标签（silver labels）

在==无法获得人类证据注释的情况下==，我们设计了一套启发式规则来自动构建==**银标签（silver labels）**==进行证据提取

> 然后，我们在银标签上训练我们的联合模型，并**直接使用银标签作为伪文档进行==推理==**。

**Co-occur**共同出现

> 如果头实体和尾实体同时出现在同一个句子中，我们使用它们**同时出现的所有句子作为证据**

**Coref**头尾共指提及同时出现

> 如果专有名词提到的**头和尾实体没有同时出现，但它们的共指提及共现**，我们**使用它们共指提及共同出现的 句子作为证据**
>
> **（共指提及：coreferential mentions，共指提及指的是在文本中指向同一个实体的不同提及）**
>
> 在实践中，我们直接应用预先训练的**共指消解模型**(HOI Xu and Choi, 2020)，而不需要对我们的数据集进行微调。

**Bridge**交接实体的共指提及和头尾共现

> 如果前两个条件不满足，但存在第三个桥接实体，**桥接实体的共指提及 与 头和尾同时出现**，我们以所有的桥与头或尾共存的句子作为证据。
>
> 如果有多个桥接实体，我们选择频率最高的一个；虽然这一规则可以很容易地扩展到多个桥，但我们**根据经验观察，捕获一个桥已经导致令人满意的结果**。



<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230617164946635.png" alt="image-20230617164946635" style="zoom:50%;" />

==列出了每个规则所涵盖的关系的数量和百分比 - 我们可以看到，这三个类别涵盖了开发集中88%以上的关系。==



## Main Results

> ==EIDER==是**在gold labels下训练** 以及 **利用在推理中由我们的模型提取的证据**。
>
> ==Eider（Rule）==在 **由规则构建的银证据标签** 上训练，并在推理中利用它们。

**Relation Extraction Results**.

>我们假设==**句间对的瓶颈是定位相关上下文，而相关上下文往往遍布整个文档。**==
>
>EIDER在训练中**学会捕捉重要的句子**，在推理中更专注于这些重要的句子。
>
>
>
>GAIN的Inter F1比ATLOP高0.70，ATLOP的Intra F1比GAIN高0.16，说明文档级图在多跳推理中可能是有效的
>
>EIDER不涉及显式的多跳推理模块，**它在Inter F1中仍然明显优于基于图的模型**。
>
>
>
>对DocRED和CDR的改进远远大于对GDA的改进。我们假设这是因为GDA中超过85%的关系是句子内的，即使是单一的RE模型关注这些句子也微不足道。

**Evidence Extraction Results**

> 三个启发式规则，已经捕获了积极实体对的大部分证据。
>
> 分别训练RE模型和证据提取模型（表示为nojont）会导致性能的急剧下降。
>
> **由于关系分类器和证据分类器共享相同的基编码器，丢弃关系分类器会导致基编码器训练不足，影响性能。**



**消融实验：**

> **我们的推理过程对于证据可能不是连续的句子间对是有效的。**
>
> 性能的急剧下降表明**混合层可以成功地学习一个结合预测结果的动态阈值**
>
> **在将提取的证据馈送给re模型之前，我们进一步基于地面真实证据对RE模型进行微调(表示为FinetuneOnEvi)，但是性能没有提高**，这可能是因为证据和原始文档中的编码实体表示已经高度相似。



# 实验

```
bash scripts/train_bert.sh eider test hoi
```

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230717204734852.png" alt="image-20230717204734852" style="zoom: 33%;" />
