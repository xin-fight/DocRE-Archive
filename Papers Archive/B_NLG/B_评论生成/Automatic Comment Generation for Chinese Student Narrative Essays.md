# Automatic Comment Generation for **Chinese Student Narrative Essays**

**问题描述：**

>用自然语言生成一个流畅的评论来描述给定学生论文中 **特定片段** 的优点或缺点



**该任务的挑战主要在于以下三个方面**：

>(1)捕捉文章的语言特征，从措辞、修辞方法（例如，在例子中的“明喻”）到话语结构。
>
>(2)生成连贯的评论来正确地反映文章的优缺点（例如，该部分在示例中没有使用成语）。
>
>(3)产生信息丰富和多样化的评论，因为一般的评论，如“它是好的”，并不为学生提供任何有用的指导，而且它也被期望为不同的论文产生不同的评论。



 **Related Work**：

> **Automatic Essay Scoring**：以往的研究主要集中在对**整体文章质量评分 或 特定特征的文章评分（trait-specific essay scoring）**。
>
> * ==Holistic essay scoring== 目的是为这篇文章分配一个总分。
>
> * ==Trait-specific essay scoring== 目的是为一篇文章的不同特征分配不同的分数，如论文的清晰度、风格和叙事质量。
>
> **Essay Assessment Systems**
>
> **Planning-based Generation**：它首先预测了一个作为计划的中间表示，然后根据计划生成完整的文本。





Chinese dataset

> which contains 22,399 essay-comment pairs.



we propose ==a **planning** based generation model==：首先计划一系列**涉及特定写作技巧的 关键字**，然后将这些关键字扩展为一个完整的评论

> 一方面：**planning**帮助**在文章和潜在的写作技巧之间 建立一个明确的连接**，这==缓解了退化的问题==（模型专注于预测通用的元素，如“更生动的”，并倾向于产生通用的评论） 
>
> 另一方面：对**中间关键词进行直接控制**，==提高生成评论的正确性和信息性==
>
> 
>
> 为了提高所生成的评论的正确性和信息量:
>
> * 训练一个纠错模块（BERT）来过滤掉不正确的关键字
> * 从源文章中识别出细粒度的结构化特征，以丰富关键词。



此外，我们使用启发式技术或预先训练好的分类器，从成语、谚语、引用、描述性和修辞学方法等方面**从源文章中识别出结构化特征**。

然后，我们将这些**特征与预测的关键词结合**起来。

为了控制生成评论的类型，我们在关键字和评论之前插入一个**二进制控制代码**（0/1来描述优缺点）

在评论生成阶段，我们在训练过程中**将噪声注入ground-truth关键词**（Tan等人，2021），以**缓解规划引入的暴露偏差问题**（exposure bias issue）（Ranzata等人，2016）。



a **planning** based generation model

>  ==**a planning-based model**==: first **plans an out-of-order sequence of keywords** and then **organizes them into a complete comment**
>
>  
>
>  **Keywords Filtering and Adding**：在生成的关键词中存在两个主要问题
>
>  > * (1)一些关键词所反映的写作技能在源文章中没有使用（例如，图2中的“echo”），这使得评论生成器很难生成正确的评论。
>  >
>  > * (2)生成的关键词不足以覆盖所使用的写作技能（如图2中的环境描述），从而降低了生成的评论的信息量。
>  >
>  > 因此，我们采用**纠错模块**和**特征识别模块**对关键字进行修改，提高生成的评论的正确性和信息量。
>  >
>  > 1. ==**an error correction module**==: **filters out incorrect keywords** using a fine-tuned BERT classifier			
>  >
>  > > 利用对比学习训练
>  >
>  > 
>  >
>  > 2. ==**a feature recognition module**==: to recognize **fine-grained structured features** such as idioms, descriptive and rhetorical methods from the source essay *X* to **enrich the keywords.**   
>  >
>  > > 并且为了插入从特征识别模块中获得的新的关键词，而不必担心插入位置，我们对提取的关键词进行随机洗牌
>  > >
>  > > 使用纠错模块过滤不正确的关键字后，关键字序列可能会遗漏源文章中的一些重要特征。因此要识别输入中细粒度的特征
>  > >
>  > > 对于习语、谚语和引语，我们直接**与现成的语料库逐字匹配。**
>
>  
>
>  ==**in the comment generation stage**==, we perturb the input keywords by **inserting a random word** to **alleviate the exposure bias problem.**
>
>  > 将优化后的关键字序列与原始文章一起输入注释生成器
>
>  ==**a binary control code**== 为了控制生成评论的类型，我们在**生成关键字和评论之前**插入一个**二进制控制代码**（0/1来描述优缺点）





参考资料：

[【深度学习笔记】Seq2Seq中的曝光偏差(exposure bias)现象](http://www.sniper97.cn/index.php/note/deep-learning/note-deep-learning/4265/)

> 出现exposure bias的主要原因是==由于训练阶段采用的是Teacher Forcing，而才测评或者生成时，我们基本都采用贪心或者beam search的方法，这就造成了训练和测评上的一些gap。==在训练阶段，模型总是能获取上一步的正确答案。
>
> 这就像我们在上学时，老师带你做题很容易就做出来，因为老师每一步在引导你时，上一步都是正确答案，而到了自己做题时，由于没有办法确保上一步的正确性，因此就可能会出现一些错误。
>
> 解决：
>
> 1. beam search
>
> 2. 替换oracle words：输入模型时，**将一定概率的token随机替换成当前句子中的其他token**，然后在计算loss时，依然使用真实句子与预测结果进行loss计算，从而**让模型具备一定的”纠错“能力**，因此模型也就更容易的会在推理阶段发现之前的”错误生成“。
>
> 3. TeaForN：EMNLP 2020 [TeaForN: Teacher-Forcing with N-grams](https://arxiv.org/abs/2010.03494)，主要思路是依然是解决前瞻性不足的问题，强制模型在解码的t时刻，强制思考t+n时的输出，**相当于在训练流程中引入beam search的思路。**
>
>    <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230327145508032.png" alt="image-20230327145508032" style="zoom: 67%;" />
>
> 相关解决论文：
>
> **ERNIE-GEN**
>
> Progressive generation of long text with pretrained language models.

