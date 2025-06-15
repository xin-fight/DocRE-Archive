# LEARNING COMMENT GENERATION BY LEVERAGING USER-GENERATED DATA

## 1. INTRODUCTION

现有的开放域评论生成模型很难训练，它们会产生重复和无趣的响应。

这个问题是：由于一篇文章的**多重multiple和矛盾contradictory的反应**，以及**检索方法的死板rigidity**。

> 评论并不是每个评论都是相关的，而且它们经常包含**不恰当的内容**
>
> 以前的生成模型存在模式崩溃问题，即模型产生的样本**多样性极低**。评论生成任务中，我们**对文章和所有有巨大差异的评论进行训练时，我们也会遇到同样的问题**。
>
> 另一方面，信息检索（IR）方法可以从真实用户中选择评论，但这种方法是不可扩展的。





we propose a combined approach to **retrieval and generation** methods.

>  attentive scorer：通过 利用用户生成的数据 来检索**信息丰富的和相关的**评论。 - 
>
> sequence-to-sequence model with copy mechanism：将这些评论和文章一起作为输入，并利用了copy mechanism	

 we propose a framework to learn comment generation by **leveraging user-generated data such as upvote count**

> 建立a neural classififier对**评论进行评分，以缓解模型崩坏的问题**
>
>  we use **a pointer-generator network [5] to learn and copy essential words** from the articles

在推理时，我们将所有未知tokens 替换成 最高的注意力权重的 源tokens ，并阻止重复tokens。



## 2. RELATED WORK

先前的响应生成工作集中在两种主要的方法上：信息检索和生成模型。

> **Information retrieval:** 
>
> [6]提出了一种基于神经的方法，通过计算技术讨论论坛中信息的期望值来澄清clarify问题。
>
> 在问答过程中，采用**TF-IDF分数去除不相关的候选答案，以减少搜索空间**[7]。
>
> 在自动评论评分器中探索了一个CNN编码器，以取代一个标准的基于IR的相似性评分器[1]。
>
> 
>
> **Generative model:**
>
> [8]引入了一种基于**门控注意神经的生成模型**，通过选择新闻上下文来**解决上下文相关性的问题**
>
> [3]研究了不同features集的影响，以衡量评论在在线论坛上的说服力。
>
> [9]表明，争论性评论表示对于[10]提供的数据集来识别建设性评论是有用的。
>
> 摘要任务中，[5]提出了具有复制机制的指针生成器网络，即从源文章中复制单词来生成摘要。与我们的工作类似，该机制会触发模型来获取文章中的一些重要关键字，并使用它来生成相关的注释。
>
> [11] [12] [13]集成了复制机制与多跳内存网络，有效地利用知识库在面向任务的对话系统中生成响应。