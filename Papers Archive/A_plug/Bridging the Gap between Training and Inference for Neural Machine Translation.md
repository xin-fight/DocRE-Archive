# Bridging the Gap between Training and Inference for Neural Machine Translation

| <font style="background: Aquamarine">*Overcorrection Recovery* (*OR*)</font> | <font style="background: Aquamarine">exposure bias</font> | <font style="background: Aquamarine">Gumbel noise</font> |
| :----------------------------------------------------------: | :-------------------------------------------------------: | :------------------------------------------------------: |

​	 

## 问题描述

**问题1**：***overcorrection***

词级训练需要在生成的序列和地面真实序列之间的严格匹配，从而导致**对不同但合理的翻译的过度校正overcorrection**。



**问题2**：在训练时，地面真实词被用作上下文，而在推理时，整个序列是由结果模型自己生成的，因此由模型生成的前面的单词被作为上下文输入。

**后果2：**

因此，在训练和推理时的预测词来自不同的分布，**即来自数据分布，而不是模型分布**。这种差异被称为==**暴露偏差*exposure bias***==（Ranzato et al.，2015），导致了训练和推理之间的差距。



直观地说，为了解决这个问题，应该训练模型在与推理相同的条件下进行推理

>  Inspired by DATA AS DEMONSTRATOR (DAD) (Venkatraman et al., 2015), feeding as context both ground truth words and the predicted words during training can be a solution.

**整体思路：**为了减轻训练和推理之间的差异，在预测一个单词时，我们用**抽样方案提供地面真实词或先前的预测词作为上下文**。



**问题3**：NMT模型通常会优化**交叉熵损失**，这需要在词级上进行严格的成对匹配 — 一旦模型生成了一个偏离地面真实序列的单词，交叉熵损失将立即纠正错误，并将剩余的生成拉回地面真实序列。

一个句子通常有多个合理的翻译，即使模型产生了一个与地面真理不同的单词，也不能说模型犯了错误。





在本文中，我们提出了一种方法来弥补训练和推理之间的差距，并提高NMT的过校正恢复能力

> 我们的方法首先从模型预测中得到 **predicted words**(referred to as***oracle* words)**，然后**从oracle单词和基础真值单词中采样作为上下文**。
>
> 同时，不仅使用**逐词贪婪搜索**，而且**使用句子级评估(例如BLEU)来 选择oracle单词**，这在交叉熵的成对匹配限制下允许更大的灵活性
>
> 在训练开始时，模型以更大的概率选择背景真实词作为上下文。随着模型逐渐收敛，甲骨文字被更多地选择为上下文。==**这样，训练过程从完全引导的方案向较少引导的方案改变。**==
>
> 
>
> ==在这种机制下，模型有机会学习**处理推理中出现的错误，并有能力从替代翻译的过度纠正中恢复过来**。==



为了解决这些问题，我们不仅从地面真实序列中采样，还从模型的预测序列中采样，其中预测序列具有句子级最优



我们的方法的主要框架（如图1所示）是**以一定的概率将地面真实词或以前的预测词(*oracle* words)，作为上下文输入**。

通过训练模型时去处理在测试阶段内出现的情况，这可能会减少训练和推理之间的差距。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230516153031753.png" alt="image-20230516153031753" style="zoom: 67%;" />

**为了预测第j个目标词yj，我们的方法涉及到以下步骤：**

> 1. 选择第i-1步的oracle word (Section **Oracle Word Selection**)
>
> 2. 以p的概率从地面真词$y^*_{j-1}$中取样，或以1-p的概率从神谕词$y^{oracle}_{j-1}$中取样。(**Sampling with Decay**)
>
> 3. 将采样词作为$y_{j−1}$，并将等式(6)和(7)中的$y^*_{j-1}$替换为$y_{j−1}$
>
>    <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230516154908765.png" alt="image-20230516154908765" style="zoom: 67%;" />

​	

**选择Oracle words：**

选择Oracle word 应该是一个类似于地面真理的词或一个同义词

* **two methods to select the oracle words**

>  *Word-level Oracle* (called WO)：可以使用**单词级的贪婪搜索**来输出每一步的oracle
>
> >**对于第j-1步decoding step**：选择oracle words最简单的方法是 — 直接从方程(9)绘制的**单词分布$P_{j−1}$中挑选出概率最高的单词**
> >
> >==(得到结果：将模型预测词作为上下文可以减轻暴露偏差)==
> >
> ><img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230516161202276.png" alt="image-20230516161202276" style="zoom: 50%;" />
> >
> >通过引入Gumbel-Max技术（(Gumbel, 1954; Maddison et al., 2014），获得更健壮的单词级oracle 
> >
> ><img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230516162148886.png" alt="image-20230516162148886" style="zoom:50%;" />
>
> <hr>
>
> *Sentence-level Oracle* (denoted as SO)：用**波束搜索扩大搜索空间**，然后用句子级别的度量（BLEU，GLEU，ROUGE）对候选翻译进行重新排序来进一步优化oracle
>
> > 句子级Oracle的采用是为了**让句子级指标所要求的n-gram匹配的翻译更加灵活。**
> >
> > 我们首先对每个batch中的所有句子进行光束搜索，假设光束大小为k，并得到k-最佳候选翻译；在波束搜索过程中，我们还可以应用Gumbel noise。
> >
> > 然后，我们通过使用地面真实序列**计算其BLEU得分来评估每个翻译**，并使用BLEU得分最高的翻译作为*oracle sentence*
> >
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230516164745962.png" alt="image-20230516164745962" style="zoom:50%;" />
> >
> > ==**遇到的问题：**==**由于模型在每一步都从地面真词和句子级Oracle中采样，因此两个序列应该有相同的单词数**
> >
> > > 但是，我们不能用朴素的波束搜索解码算法来保证这一点。基于上述问题，我们引入了 **force decoding**，以确保两个序列的长度相同。
> >
> > ==**解决办法 — Force decoding**==：
> >
> > > 因此，在光束搜索中，一旦一个候选翻译倾向于以比$|y^*|$短或长的EOS结束，我们将强迫它生成$|y^*|$词：
> > >
> > > 1. 如果候选翻译在第j步得到一个单词分布$P_j$，其中j小于$|y^*|$且EOS是Pj中的第一个单词，那么我们**选择Pj中的第二个单词作为该候选翻译的第j个单词**。
> > > 2. 如果候选翻译在$|y^*|+1$步得到一个单词分布$P_{|y^∗|+1}$，其中EOS不是$P_{|y^∗|+1}$中的第一个单词，那么我们选择EOS作为这个候选翻译的第$|y^*|+1$单词。

​	

句子级的oracle提供了一个与地面真实序列匹配的ngram选项，**因此本质上具有从替代上下文的过度校正中恢复的能力。**

<hr>

**sampling mechanism — Sampling with Decay**

在我们的方法中，我们采用了一个抽样机制来随机选择地面真实字$y^*_{j-1}$或Oracle word $y^{oracle}_{j-1}$作为$y_{j-1}$。

> ==**在训练开始时，由于模型没有经过很好的训练，使用$y^{oracle}_{j-1}$作为$y_{j-1}$经常会导致非常缓慢的收敛，甚至被困在局部最优中**==
>
> ==**在训练结束时，如果上下文$y_{j-1}$仍然大概率选择地面真理词$y^*_{j-1}$，则模型没有完全暴露于它在推断时必须面对的情况，因此不能知道在推断的情况下如何行动。**==



在这个意义上，从 ground truth word中选择的概率p不能是固定的，而是必须随着训练的进行而逐渐降低。

>==在开始时，p=1，这意味着模型完全基于地面真实单词来训练。随着模型逐渐收敛，模型更频繁地从Oracle word中进行选择。==
>
>$\therefore$ 我们定义了一个具有衰减函数的p，它依赖于 training epochs e的指数（从0开始）
>
><img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230516190821544.png" alt="image-20230516190821544" style="zoom:50%;" />

通过上述方法选择$y_{j-1}$后，我们可以根据式(6)、(7)、(8)和(9)得到$y_{j}$的单词分布。



**尝试的另一个方向是句子级训练，认为句子级度量，例如BLEU，为生成带来了一定程度的灵活性，因此对于减轻暴露偏差问题更鲁棒。**



**实验结果得到结论：**

> 1. word-level oracle的性能提升：将预测词作为上下文可以减轻暴露偏差
>
>    sentence-level oracle性能继续提升：句子级oracle在BLEU方面优于单词级oracle
>
>    **我们推测，这种优势可能来自于单词生成的灵活性，可以减轻过度修正（overcorrection）的问题**	
>
> 2. 通过在单词级和句子级预言性单词的生成过程中加入**Gumbel noise**能够提升性能
>
>    这表明Gumbel噪声可以帮助选择每个甲骨文词，这与我们的主张一致，即**Gumbel-Max提供了一种有效和稳健的方式来从分类分布中抽样**。
>
> 3. 将oracle采样和Gumbel噪声相结合，**导致收敛速度较慢，在验证集上出现最佳结果后，训练损失不会持续减少。**
>
>    ==oracle采样和噪声可以避免过拟合，尽管它需要更长的时间来收敛==
>
> 4. 加入了噪声，获得了高BLEU
>
>    在没有任何正则化（不加入噪声）的情况下，**重复使用自己的结果会导致过拟合和快速收敛**。从这个意义上说，**模型得益于句子级采样和Gumbel noise。**
>
> 5. 交叉熵损失要求预测的序列与地面真实序列完全相同，这对于长句子来说更难实现，**而我们的句子级oracle可以帮助从这种过度校正中恢复**
>
> 6. 验证这些改进是否主要是通过解决曝光偏差问题而获得的而进行了实验
>
>    通过对 由我们的模型产生的预测分布中的概率大于由基线模型产生的概率的基本真实单词 进行计数，发现占总单词数的65.06%而验证

​	