# 自然语言生成综述

按照输入信息的类型划分，自然语言生成可以分为三类：

> 文本到文本生成：文本到文本生成又可划分为机器翻译、摘要生成、文本简化、文本复述等
>
> 数据到文本生成：基于数值数据生成 BI（Business Intelligence）报告、医疗诊断报告等
>
> 图像到文本生成：新闻图像生成标题、通过医学影像生成病理报告、儿童教育中看图讲故事等



ACL、EMNLP、NACAL、CoNLL、 ICLR和 AAAI



NLG的体系结构可分为

> 传统的管道模型
>
> 基于神经网络的端到端（End-to-End，End2End）模型



# Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling

> 参考：https://blog.csdn.net/weixin_62321421/article/details/123337597

**adaptive thresholding** and **localized context pooling**, to solve the multi-label and multi-entity problems.

>对于文档级的RE，一个文档包含多个实体对，我们需要同时对它们之间的关系进行分类
>
>它要求RE模型识别并关注于文档中具有相关上下文的部分
>
>此外，一个实体对可以在与文档级 RE 的不同关系相关联的文档中多次出现，而句子级 RE 的每个实体对一个关系。
>
> This **multi-entity (multiple entity pairs to classify in a document)** and **multi-label (multiple relation types for a particular entity pair)** properties of document-level relation extraction make it harder than its sentence-level counterpart



> 该模型==解决了文档级RE中的多标签和多实体问题==。
>
> To tackle the **multi-entity** problem:
>
> **在本文中，我们提出了一种局部上下文池化技术localized context pooling (LOP)，而不是引入图结构。**
>
> 该技术==**解决了对所有实体对使用相同的实体嵌入的问题。**==
>
> **使用与当前实体对相关的附加上下文增强了实体的嵌入。**
>
> 我们没有从头开始训练一个新的上下文注意层，而是直接从预先训练的语言模型中转移注意力头来获得实体级别的注意力。
>
> 然后，对于一对实体中的两个实体，我们通过乘法来合并它们的注意力，以找到对他们俩都很重要的上下文。
>
> （**提出了本地化上下文池，将预先训练好的注意力转移到实体对的相关上下文，以获得更好的实体表示。**）
>
> 
>
> For the **multi-label** problem:
>
> **提出了自适应阈值化技术，它用一个可学习的阈值类代替全局阈值。**
>
> 阈值类是通过我们的自适应阈值损失学习的，这是一种基于秩的损失，**在模型训练时，将正类的对数推到阈值以上，并将模型训练中的负类的对数推到阈值以下**
>
> 在测试时，我们返回具有比阈值类更高的 logits 的类作为预测标签，或者如果此类不存在则返回 NA
>
> 这种技术消除了对阈值调整的需要，并且还使阈值可针对不同的实体对进行调整，从而获得更好的结果
>
> （**我们提出了自适应阈值损失，可学习一个依赖于实体对的自适应阈值，并减少使用全局阈值引起的决策错误。**）



**mention 就是自然文本中表达实体(entity)的语言片段。**

> 我们将把文本引用的实例称为对象或抽象提及，它们可以是命名的（如John Mayor），名词性的（如总统）或代词的（如she，it）
>
> 一个实体是指一个概念实体的所有提及
>
> 例如，在句子中：约翰·史密斯总统说他没有评论。有两个提及：约翰·史密斯和他（按照出现的顺序，他们的级别被命名和代词），但有一个实体，由集合{John Smith，he}组成。



## Enhanced BERT Baseline

**Encoder**

> 1. 通过在提及的开始和结尾插入一个特殊的符号“*”来标记实体提到的位置。它改编于实体标记技术
>
> 2. 将文档输入一个预先训练好的语言模型，以获得==**上下文嵌入 H**==
>
>    文档由编码器进行一次编码，所有实体对的分类都基于相同的上下文嵌入。**将提及开始时的“*”嵌入作为提及嵌入**
>
> 3. 选取带有*的实体，对于它所有的**提及**使用==logsumexp pooling== , a smooth version of max pooling 的方法来获得**实体嵌入**
>
> > 参考：
> >
> > The Log-Sum-Exp Trick：https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
> >
> > [CSDN - 一文弄懂LogSumExp技巧](https://helloai.blog.csdn.net/article/details/121869249?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-121869249-blog-81949957.pc_relevant_3mothn_strategy_and_data_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-6-121869249-blog-81949957.pc_relevant_3mothn_strategy_and_data_recovery&utm_relevant_index=7):LogSumExp(LSE)技巧，主要解决计算Softmax或CrossEntropy时出现的上溢或下溢问题



**Binary Classififier**

> 1. 我们使用线性层将实体映射到隐藏状态 z，然后是非线性激活。
>
> 2. 然后通过双线性函数和s型激活计算关系r的概率。
>
> 实体对包含关系的概率估计，采用的是双线性分类器模型，**每个实体的表示representation在不同的实体对中是相同的**。为了减少参数量，切分为k个等份，然后组之间使用双线性模型。
>
> > **训练**：使用二值交叉熵损失。
> >
> > **推理过程中**，我们调整一个全局阈值θ，使开发集上的评估指标（RE的F1分数）最大化
> >
> > * 如果P（r|，eo）>θ，则返回r作为关联关系，如果不存在关系，则返回NA。



## Adaptive Thresholding

由于阈值既没有封闭形式的解也不是可微分的，因此**确定阈值的常用做法是枚举 (0, 1) 范围内的多个值并选择最大化评估指标的值（RE 的 F1 分数）**

该模型可能对不同的实体对或类有不同的置信度所以一个全局阈值可能不满足要求

关系的数量是不同的（多标签问题），而且模型可能不是全局校准的，所以相同的概率对所有实体对来说并不意味着都是一样的。

为了便于解释，我们将实体对T =（es，eo）的标签划分为两个子集：正类PT和负类NT，它们的定义如下：

>> 正类PT⊆R是T中实体之间存在的关系
>>
>> 负类NT⊆R是指实体之间不存在的关系
>
>==如果一个实体对被正确分类，则正类的对数应高于阈值，而负类的对数应低于阈值==
>
>> 这里我们引入了一个**阈值类TH**，它以与其他类相同的方式自动学习(参见等式(5)).
>
>在测试时，我们将 logits 高于 TH 类的类作为正类返回，如果此类类不存在，则返回 NA。 
>
>==这个阈值类学习一个与实体相关的阈值。它是全局阈值的替代品，因此消除了在开发集上调整阈值的需要==
>
>
>
>为了学习新的模型，需要一个考虑TH类的特殊损失函数。我们设计了**基于标准分类交叉熵损失(standard                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 categorical cross**
>
>**entropy loss)**的自适应阈值损失。损失函数可分为两部分
>
>> 第一部分：包括积极类和TH类。
>>
>> 由于可能存在多个正类，总损失被计算为 **所有正类上的类别交叉熵损失的和**
>>
>> **L1推动所有正类的对数都高于TH类**。如果没有阳性标签，则不使用它
>>
>> 第二部分：涉及到负类和阈值类
>>
>> 它是 **一个类别交叉熵损失**，TH类是真正的标签。
>>
>> **它将负类的对数拉到低于TH类**。	



## Localized Context Pooling

 **logsumexp pooling** 

> 其在整个文档中累积对一个实体的所有提及的嵌入，并**为该实体生成一个嵌入**。
>
> 这种**池化积累了来自文档中提及的信号**。它在实验中比平均池化具有更好的性能。



然后将来自该文档级全局池化（document-level global pooling）的实体嵌入用于所有实体对的分类

> **但是，对于一个实体对，这些实体的某些上下文可能并不相关**    
>
> ==因此，最好有一个本地化的表示，它只关注文档中的相关上下文，这对决定这个实体对的关系很有用==                                                                                                                                                                                                                                                                                                                                                                                         

所以：提出了==本地化上下文池(**localized context pooling**)==，其中我们使用与两个实体相关的附加局部上下文嵌入来增强实体对的嵌入。

> 由于我们使用预先训练的基于**Transformer**作为编码器，它已经通过多头自注意学习了令牌级的依赖关系（Vaswani et al. 2017）
>
> 我们考虑**直接使用他们的注意头进行局部上下文池化**。

multi-head attention matrix $A_{ijk}$表示第 i个注意头中 从标记 j到标记 k的注意力

我们首先将“*”符号作为提及级的注意，然后把相同实体的提及上的注意力进行平均以获得实体级别的注意力**$A^E_i\in R^{H \times l}$**（**从第i个实体到所有tokens的注意**）

然后给定一个实体对（es，eo），我们通过**乘以对es和eo的实体级注意力**来定位对它们都重要的局部上下文，并得到局部上下文嵌入$c^{(s,0)}$ 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202211122311529.png" alt="image-20221111092905194" style="zoom:60%;" />

> H是Encoder中的上下文嵌入
>
> 通过修改等式(3)和等式(4)中的原始线性层，**将局部上下文嵌入融合到全局池化实体嵌入中，==获得不同实体对有不同的实体表示==**

做法:

> 1.获得预训练模型的注意力
>
> 2.对同一实体的提及的注意力进行平均，以获得实体级别的注意力
>
> 3.再将头实体和尾实体的注意力相乘，得到实体对的注意力
>
> 4.再通过实体对的注意力来获得对该实体对很重要的局部上下文表示
>
> 5.将获得的局部上下文表示添加到不同的全连接层中获得头实体和尾实体的隐藏状态



## Experiments

**Datasets**

> **DocRED（Yao et al 2019）**是一个用于文档级RE的大规模众包数据集。它是由维基百科的文章构建的。DocRED由3053个文档组成，用于训练。对于表达关系的实体对，大约7%的实体对有一个以上的关系标签
>
> **CDR（Li et al 2016）**是一个生物医学领域的人类注释数据集。它由500个文档组成，用于训练。其任务是预测化学和疾病概念之间的二元相互作用。
>
> **GDA（Wu et al 2019b）**是生物医学领域的一个大规模数据集。它由29192篇文章组成，用于训练。其任务是预测基因和疾病概念之间的二元相互作用。我们遵循Christopoulou, Miwa, and Ananiadou（2019）的做法，将训练集分成80/20，作为训练集和开发集。

**Main Results**

> **Sequence-based Models**：这些模型使用CNN（古德费勒等人2016年）和双向LSTM（舒斯特尔和帕利瓦尔尔1997年）神经结构**对整个文档进行编码**，然后获得实体嵌入，并预测具有双线性函数的每个实体对的关系。
>
> **Graph-based Models**：这些模型**通过学习文档的潜在图结构来构造文档图**，并使用图卷积网络进行推理（Kipf和Welling 2017）。我们包括了两个最先进的基于图的模型，AGGCN（Guo，Zhang，和Lu，2019年）和LSR（Nan等人，2020年），以进行比较。AGGCN的结果来自于Nan等人（2020年）的重新实施。
>
> **Transformer-based Models**：这些模型直接**使用预先训练过的语言模型适应文档级RE，而不使用图结构**。它们可以进一步分为**管道模型**（BERT-TS（Wang等人2019a））、**层次模型**（HIN-BERT（Tang等人2020a））和**预训练方法**(CorefBERT和CorefRoBERTa (Ye等人2020）。我们还包括了BERT基线（Wang et al. 2019a）和我们重新实施的BERT基线进行比较。

**Ablation Study**

> 自适应阈值化 和 本地化上下文池 对模型性能同样重要，当从ATLOP中删除时，dev F1分数分别下降了0.89%和0.97%	
>
> 自适应阈值化仅在使用自适应阈值化损失优化模型时有效
>
> 我们发现实体标记(entity markers)的改进很小（devf1为0.24%），但仍然在模型中使用该技术，因为它使提及嵌入和提及级注意更容易。	
>
> 
>
> 经过实验：
>
> 自适应阈值化和局部上下文池化都是有效的
>
>  Logsumexp pooling and group bilinear给基线带来了显著的增益。

**Analysis of Thresholding**

> 全局阈值不考虑模型对不同类或实例的置信度的变化，因此产生次优性能。
>
> 实验表明，在训练后调整**每类阈值（ per-class thresholding）**会导致对开发集的严重过拟合。而我们的**自适应阈值技术在训练中学习阈值，这可以推广到测试集。**

**Analysis of Context Pooling - mitigates the multi-entity issue**

> 当实体的数量大于5时，我们的本地化上下文池可以获得更好的结果。当实体数量的增加时，改进就会变得更加显著
>
> ==本地化的上下文池技术可以捕获实体对的相关上下文，从而缓解了多实体的问题。==



## Related Work

document graphs

> 该文档图提供了一种统一的提取实体对特征的方法
>
> 后来的工作通过改进神经架构（Peng et al 2017；V erga, Strubell, and McCallum 2018；Song et al 2018；Jia, Wong, and Poon 2019；Gupta et al 2019）或增加更多类型的边缘（Christopoulou, Miwa, and Ananiadou 2019；Nan et al 2020）来扩展这一想法。
>
>  In particular, Christopoulou, Miwa, and Ananiadou (2019)构造具有不同粒度（句子、提及、实体）的节点，将它们与启发式生成的边连接起来，并通过一个面向边的模型推断出关系
>
>  Nan et al. (2020)将文档图作为一个潜在的变量来处理，并通过结构化的注意来诱导它这项工作还提出了一种细化机制，可以从整个文档中实现多跳信息聚合。



<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202211141128826.png" alt="image-20221114112819629" style="zoom: 80%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/202211141136215.png" alt="image-20221114113632410" style="zoom:50%;" />



# 预训练模型梳理整合（BERT、ALBERT、XLNet详解）

> [最火的几个全网络预训练模型梳理整合（BERT、ALBERT、XLNet详解）_Reza.的博客-CSDN博客_albert预训练](https://blog.csdn.net/weixin_43301333/article/details/104861975)
