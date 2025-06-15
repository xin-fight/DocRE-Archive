# Generating Diversified Comments via Reader-Aware Topic Modeling and Saliency Detection
问题：

大多数的评论生成方法只关注**突出性信息的探测**（saliency information extraction），而评论所隐含的读者意识因素则被忽略了

评论不仅暗示了新闻文章中不同的重要信息，而且还传达了独特的读者特征 - reader-aware factors



评论由两个突出的特点

1)读者通常会注意新闻的部分内容，这意味着**并非所有的内容信息都是突出的和重要的**（salient and important）。

2)不同的读者通常对不同的话题感兴趣，即使是同一话题，他们也可能持有**不同的观点，这使得评论多样化，信息量大**（diverse and informative）。

**直观上看，这些读者感知因素作为多样性的根本原因，在多元化评论生成任务中应与突出性信息探测共同考虑**



为了解决这个问题，我们提出了一个**统一的读者感知的话题建模 和 突出信息saliency检测框架**（a unified reader-aware topic modeling and saliency information detection framework），以提高生成的评论的质量。

* ==**a unified reader-aware topic modeling**==：设计了a Variational Generative Clustering algorithm（VGC，一个变分生成的聚类算法）用于潜在语义学习(latent semantic learning)和读者评论的话题挖掘（topic mining）

  读者感知的主题建模( reader-aware topic modeling)的**目标是从评论中进行读者感知的潜在因素挖掘。**

  **所获得的潜在因素表示**可以解释为新闻主题、用户兴趣或写作风格。为方便起见，我们将它们统称为**Topic.**

* ==**a saliency information detection framework**==：引入了Bernoulli distribution estimating on news content to select saliency information

  建立了一个显著性检测组件来 **对新闻内容进行伯努利分布估计**

  引入**Gumbel-soutmax**来解决不可微(non-differentiable)采样操作问题

* 将两者合并到**decoder**中 以控制模型生成多样化的、信息丰富的评论



We conduct extensive experiments on **three datasets in different languages**:

>  NetEase News (Chinese) (Zheng et al. 2017), 
>
> Tencent News (Chinese) (Qin et al. 2018), 
>
> Yahoo! News (English) (Yang et al. 2019).
>
> automatic evaluation and human evaluation



==将新闻标题和文本组合==

> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230325011750402.png" alt="image-20230629030951537" style="zoom:80%;" />
>
> $h^e_i$：BiLSTM对第i个词的隐藏层向量
>
> $h^{te}$​：BiLSTM对标题的隐藏层向量



**The Backbone Framework**

> 序列到序列的注意力机制框架
>
> encoder: BiLSTM
>
> decoder：LSTM（decoder初始化：$h ^ { e } = [ 前向h | x | ; 后向h _ { 0 } |$） 和  attention mechanism.
>
> 
>
> **Reader-Aware Topic Modeling**：生成模型，we employ **Bag-of-Words feature vector** **y** to represent each comment sentence
>
> > 训练结束后，能获得**K个读者感知的主题表示向量**，这些具有读者感知的主题可以用来控制生成的评论的主题多样性。
>
> 
>
> **Reader-Aware Saliency Detection**：读者感知显著性检测组件旨在从新闻文章中 ==选择最重要和对读者感兴趣的信息==
>
> > 对**每个词进行伯努利分布估计**，表示每个内容词是否重要。
> >
> > 1. BiLSTM encoder对整个文本编码，最后两个方向的隐藏层向量 被用来作为**标题表示 title representation**$h^{te}$
> >
> > 2. 最后一层使用two-layer MLP预测**每个内容词xi的被选择概率**
> >
> >    概率βi决定了单词被选择的概率（显著性），并被用于参数化一个伯努利分布。通过从伯努利分布中抽样，可以得到每个单词的一个二进制门
> >
> >    ==（伯努利分布**采样操作不可微**）== -> 我们应用Gumbel-Suftmax分布作为每个**单词选择门**的伯努利分布的替代物，在所有选择门添加一个l1范数项，达到选择更少的单词的目的
>
> 
>
> **Diversifified Comment Generation**：
>
> > 使用MLP来选择topic被选择的概率





我们将新闻标题、新闻机构和评论的最大长度分别限制在30,600和50个。超过最大长度的部分将被截断。

结论表明：

> 新闻体对新闻评论生成很重要