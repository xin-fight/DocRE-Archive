# Get To The Point: Summarization with Pointer-Generator Networks



| <font style="background: Aquamarine">copy mechanism</font> | <font style="background: Aquamarine">coverage mechanism</font> |
| :----------------------------------------------------: | :----------------------------------------------------------: |



参考官方博客：

>http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html



Neural sequence-to-sequence models have two shortcomings:

> **它们很容易不准确地复制事实细节** - 而复制是摘要的基本操作
>
> > 序列到序列模型 使得从源文本中复制一个单词W非常困难：
> >
> > * 如果w是一个罕见的单词，在训练中很少出现，因此嵌入了糟糕的单词（即，它被完全无关的单词聚集），那么从网络的角度来看，w是与许多其他单词没有区别的，因此不可能繁殖；
> > * 即使W具有好词嵌入，网络仍然难以复制该单词，例如，RNN摘要系统在尝试重建原始单词经常用另一个名字 替换 一个名字
>
> 
>
> **它们倾向于重复自己。**
>
> > Repetition may be caused by the **decoder’s  over-reliance on the decoder input** (i.e. previous summary word)，这可以从一个重复的单词通常会引发无休止的重复循环这一事实中看出



我们提出了两种正交的方式增强了标准的序列到序列的注意模型。

> we use a **hybrid pointer-generator network** that can **copy** words from the source text via *pointing*：这有助于信息的准确再现，同时保留通过生成器产生新单词的能力
>
> >  ==facilitates copying words== which improves accuracy and handling of OOV words, while retaining the ability to *generate* new words.
>
>  we use ***coverage*** to keep track of what has been summarized, which discourages repetition.
>
> > ==a novel variant of the *coverage vector*== which we use to track and control coverage of the source document. We show that **coverage is remarkably effective for eliminating repetition.**



之前提出的RNN：

> 虽然这些系统很有前途，但它们表现出不受欢迎的行为，如**不准确地再现事实细节**（inaccurately reproducing factual details），**无法处理词汇外（OOV）单词**，以及**重复它们自己**（见图1）。



**Pointer-generator network** - copy

是一个序列到序列模型，使用Bahdanau等人（2015）的**软注意分布**(soft attention distribution)**来生成由输入序列的元素组成的输出序列** (copying words)

> (i) 我们计算显式开关概率pgen，而Gu等人通过共享的softmax函数诱导竞争。(ii) 我们循环利用注意力分布作为拷贝分布，但Gu等人使用了两个独立的分布
>
> (iii) **当一个单词在源文本中多次出现时，我们将注意力分布的所有相应部分的概率质量相加**，而苗和布伦索姆则没有。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230413210015210.png" alt="image-20230413210015210" style="zoom: 67%;" />

*encoder hidden states $h_i$*；*decoder state $s_t$* 

> $\begin{aligned}
> e_i^t &=v^T \tanh \left(W_h h_i+W_s s_t+w_c c_i^t+b_{\text {attn }}\right) \\
> a^t & =\operatorname{softmax}\left(e^t\right)
> \end{aligned}$
>
> > **$c^t$表示coverage vector**
>
> $h_t^*=\sum_i a_i^t h_i$
>
> > context vector $h_t^*$ :
>
> $p_{\text {gen }}=\sigma\left(w_{h^*}^T h_t^*+w_s^T s_t+w_x^T x_t+b_{\mathrm{ptr}}\right)$
>
> $P(w)=p_{\text {gen }} P_{\text {vocab }}(w)+\left(1-p_{\text {gen }}\right) \sum_{i: w_i=w} a_i^t$
>
> > $$P_{vocab}=softmax(V^{\prime}(V \left[ s_{t},h_{t}^{*}\right] +b)+b^{\prime})$$
> >
> > if *w* is an out-of-vocabulary (OOV) word, then $P_{vocab}(w)$ is zero; w not appear in the source document, $\sum_{i: w_i=w} a_i^t=0$
> >
> > 使用了软注意分布(soft attention distribution) — $\sum_{i: w_i=w} a_i^t$ 来从输入序列中复制，而不是直接决定某个词
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230413211212148.png" alt="image-20230413211212148" style="zoom: 50%;" />



**Coverage mechanism** - eliminating repetition

 The idea is that we **use the attention distribution to keep track of what’s been covered** so far, and **penalize the network for attending to same parts again**.

*coverage vector $c^t$* 

> $c^t=\sum_{t^{\prime}=0}^{t-1} a^{t^{\prime}}$
>
> > $c^t$是 之前所有解码器时间步 的注意力分布的总和
> >
> > $c^t$是源文档单词上的（非标准化）分布，**代表了迄今为止这些词从注意机制获得的覆盖程度**。$c^0$是一个零向量，因为在第一个时间步长中，没有覆盖任何源文档。
> >
> > the **coverage** of a particular **source word** 等于 到目前为止它已经受到的关注量
>
> 
>
> $\begin{aligned}
> e_i^t &=v^T \tanh \left(W_h h_i+W_s s_t+w_c c_i^t+b_{\text {attn }}\right) \\
> a^t & =\operatorname{softmax}\left(e^t\right)
> \end{aligned}$
>
> > **确保注意力机制当前的决定（选择下一个关注的）是通过其先前的决定（总结在 $c^t$ 中）的提醒来通知的。** - 帮助注意力机制做决定，而非在其做了决定后再干预
> >
> > This ensures that the attention mechanism’s current decision (choosing where to attend next) is informed by a reminder of its previous decisions (summarized in *c* *t* ). 
> >
> > **这应该使注意机制更容易避免重复注意相同的位置，从而避免产生重复的文本**
> >
> > 这样做的目：在模型进行当前time step进行attention计算的时候，==**告诉它之前它已经关注过的token，希望避免出现连续attention到某几个token上的情况**==
>
> 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230414150623359.png" alt="image-20230414150623359" style="zoom:55%;" />
>
> > 总结不应该需要统一的覆盖范围，我们**只惩罚每个注意分布和覆盖范围之间的重叠**部分——防止重复注意
> >
> > $c_i^t$ 是$\sum_{t^{\prime}=0}^{t-1} a^{t^{\prime}}$中的第i项的值
> >
> > $a^t$：是由第t步的解码状态$s_t$，和之前所有的编码状态($h_i$)，以及之前所有计算的注意力总和$c^t=\sum_{t^{\prime}=0}^{t-1} a^{t^{\prime}}$计算而来
> >
> > $min(a_i^t,c_i^t)$表明：**比较当前注意力$a^t$和之前注意力总和$c^t$的每一项**，选择最小的作为loss
> >
> > > 如果==当前的注意力和之前的注意力和关注的词==不一样时，计算后loss会变小；如果关注的是一样的，那么loss就会比较大（loss<=1）

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230414160950122.png" alt="image-20230414160950122" style="zoom:50%;" />