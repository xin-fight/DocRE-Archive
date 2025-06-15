# Key Mention Pairs Guided Document-Level Relation Extraction

| <font style="background: Aquamarine">(MIL)Multiple Instance Learning</font> | <font style="background: Aquamarine"> multi-mention problem</font> | <font style="background: Aquamarine">multi-label problem</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |



## 问题描述

问题1：**<font color='red'>multi-mention problem - 在预测实体之间的关系时，不应平等地对待不同的提及，否则导致次优性能。</font>**

> 我们将DocRE任务重新定义为一个**<font color='red'>多实例学习（Multiple Instance Learn ing (MIL) problem）问题</font>**
>
> > 应该是将实体对认为是MIL中的bag，而某一个提及对是MIL中的instance，一个bag包含多个instance；当某个instance有某种关系时，bag就会有某个关系，当所有instance都没有关系时，才能说明bag没有关系
>
> DocRE多提及特性（ **multi-mention property** ）**<font color='red'>很难在 实体级别 上直接建立到关系映射的上下文</font>**。
>
> > **multi-mention property**: 关系可以通过实体对的不同的提及对来推断出来，同时，也有几个提及对不表达任何关系
>
> 因此，**平等对待所有提及忽略了不同提及上下文的差异，可能引入不相关信息来误导模型训练。**

> 提出了 ==**mention-level relation extractor**== 和 ==**key instance classififier**== 并且相互增强
>
> * 两者使用EM算法相互增强：
>   1. **relation extractor**计算得到Key Instance pseudo labels，表明 某mention pair是否重要，帮助key instance classifier训练
>      * 某个提及对关系$c$预测结果$p _ { \theta } ( y _ { c } | m _ { i } , m _ { j } )$ $\geqslant$ 提及对对关系$c$预测结果的平均值$\bar{p} _ { \theta } ( y _ { c } | e _ { h } , e _ { t } )$，则认为 提及对 很重要
>   2. **key instance classifier**提取出关键提及帮助relation extractor training训练
>      * 让==**Key Instance predictions模块 得到的key instance 提及对**== 与  ==**Relation Extractor认为重要的 提及对**== <font color='red'>**一致**</font>
>
> 
>
> * **relation extractor**：存在==**multi-label problem**==，可能很难区分**每个提及对在多标签情况下表达的关系类型**，使生成提及级关系的pseudo labels很困难
> * **key instance classififier**：得到==**key mention pairs**==，使得KMGRE可以有效地过滤出不表达任何关系的提及对，以减少冗余信息的影响。



问题2：**<font color='red'>multi-label problem</font>**

> 对 **mention-level relation extractor** 进行优化 — **<font color='red'>通过融合key mention pairs对应mention-level预测结果 生成 entity-level relation预测结果</font>**
>
> * 有了**实体级的logit**后，就可以使用**ATLOP的adaptive threshold loss**得到最后的 muti-label
>
> 它可以避免多标签情况下错误标签提及对导致的对模型的错误指导



## 模型改进 - KMGRE

直接建模**mention-level relation**，包含两个模块，并且使用EM算法让两个模块相互增强：

> **mention-level Relation Extractor**
>
> > 和ATLOP处理方式一样计算上下文嵌入，最后得到**提及对的表示**$$x ^ { ( i , j ) } = [ h _ { m _ { i } } ; h _ { mj } ; c ^ { ( i , j ) } ]$$ 
> >
> > 计算 ==**提及对关于关系$c$的概率**==$$p _ { \theta } ( y _ { c } | m _ { i } , m _ { j } ) = \sigma ( w _ { c } x ^ { ( i , j ) } + b _ { c } )$$
> >
> > <hr>
> >
> > 但是我们数据中只有实体对的标签，直接用于对提及对预测标签更新不合理
>
> **Key Instance Classififier**：得到==**key mention pairs**==
>
> > 和ATLOP处理方式一样计算上下文嵌入，最后得到**提及对的表示**$$x ^ { ( i , j )^\prime } = [ h^\prime _ { m _ { i } } ; h^\prime _ { mj } ; c ^ { ( i , j )^\prime } ]$$ 
> >
> > 计算 ==**提及对是一个key instance的概率**==$$p _ { \omega } ( z _ { (i，j) } | m _ { i } , m _ { j } ) = \sigma ( w _ { k } x ^ { ( i , j )^\prime } + b _ { c } )$$  		 
> >
> > * $$z_{ (i，j) } \in \{0,1\}$$ 为每个mention pair设计一个二进制变量，表明它**是否对实体对的关系标签负责**
>
> 两者使用EM算法相互增强：
>
> * 先对Relation Extractor 以及 Key Instance classififier进行多轮训练，**然后再使用EM算法**
>
> * ==E步更新参数$\omega$；W步更新$\theta$==
>
>   **E-step**：
>
>   > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230709015329361.png" alt="image-20230709015329361" style="zoom:55%;" />
>   >
>   > **E步目的**：Relation Extractor计算的 ==**提及对 是关系c概率**==  $p _ { \theta } ( y _ { c } | m _ { i } , m _ { j } )$ 以及 与gold relation labels $y$ 一起==**生成Key Instance pseudo labels $\widehat{z}$**==
>   >
>   > * E步损失函数是 ==更新**Key Instance predictions模块**的参数$\omega$== ，让**Key Instance predictions模块 得到的key instance 提及对** 与  **Relation Extractor认为重要的 提及对** <font color='red'>**一致**</font>
>   >
>   > <hr>
>   >
>   > 公式中**阈值 $\bar{p}_{\theta}$**：对该实体对内 所有的提及对 对关系$c$预测结果$p _ { \theta } ( y _ { c } | m _ { i } , m _ { j } )$ 进行**平均**；
>   >
>   > 而**公式9**计算的是==Relation Extractor根据 提及对对关系$c$的概率 所认为重要的提及对== ，**<font color='red'>某个提及对关系$c$预测结果$p _ { \theta } ( y _ { c } | m _ { i } , m _ { j } )$ $\geqslant$ 提及对对关系$c$预测结果的平均值$\bar{p} _ { \theta } ( y _ { c } | e _ { h } , e _ { t } )$，则认为 提及对 很重要</font>，即 ==$\widehat{z}_{(i,j)} = 1$==**
>   >
>   > <hr>
>   >
>   > 损失函数使用的是==**Binary Focal Loss $\mathcal{L}_w$**== — 二分类问题中处理 类别不平衡和困难样本
>   >
>   > > Binary Focal Loss**对正负样本的损失进行加权**，其中==困难样本（分类概率接近0.5）的损失权重更高，而容易样本（分类概率接近0或1）的损失权重较低==。这样可以使模型更**关注那些更难分类的样本，从而提高对困难样本的识别能力。**
>   >
>   > 
>   >
>   > **损失函数作用**：让==**Key Instance predictions模块 得到的key instance 提及对**== 与  ==**Relation Extractor认为重要的 提及对**== <font color='red'>**一致**</font> 
>   >
>   > > 即：$$p _ { \omega } ( z _ { (i，j) } | m _ { i } , m _ { j } )$$ 与 $\widehat{z}_{(i,j)}$ 接近
>
>   <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230708223927498.png" alt="image-20230708223927498" style="zoom: 40%;" />
>
>   **M-step**：缓解**<font color='red'>multi-label problem</font>**
>
>   > **M步目的**：==将key mention pairs对应的mention-level Relation Extractor 的结果进行**融合**，获得实体级关系预测==，并通过实体级关系提取损失==更新**Relation Extractor**参数θ==
>   >
>   > > <font color='red'>Relation Extractor已经计算 提及对 预测结果，不直接使用是因为</font> — 数据集中只有实体对的标签，而如果直接对 Relation Extractor计算的 提及对概率  $p _ { \theta } ( y _ { c } | m _ { i } , m _ { j } )$ 训练的话，会与直觉向违背
>   > >
>   > > ​	$\because$ 训练过程需要知道提及对对应的真实label，但如果实体对有多个label时，将label分配给key mention pairs时，不知道该分配哪个，因此会严重误导了训练Relation Extractor过程
>   > >
>   > > ​	$\therefore$ 我们选择融合关键 提及对 的嵌入，得到实体级的预测，以此训练Relation Extractor
>   >
>   > M步损失函数中，由于计算过程 **用到了Relation Extractor得到的嵌入$x ^ { ( i , j ) }$** ，所以会 ==更新**Relation Extractor模块**的参数$\theta$== 
>   >
>   > <hr>
>   >
>   > Key Instance predictions计算 ==**提及对的 key instances的概率**== $$p _ { \omega } ( z _ { (i，j) } | m _ { i } , m _ { j } )$$ 
>   >
>   > 再通过该实体对下 各个提及对的 **key instances的概率** 计算 ==阈值$\tilde{p}_w(z)$==
>   >
>   > > 若提及对的 key instances的概率 $\geqslant$ 阈值，则提及对 为**正例**，<font color='red'>**则该 提及对 在Relation Extractor得到的嵌入$x ^ { ( i , j ) }$  会用于融合， ==融合后得到 实体级的logit==**</font> (公式16)
>   >
>   > ==有了**实体级的logit**后，就可以使用**ATLOP的adaptive threshold loss**得到最后的 muti-label==
>   >
>   > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230709022005251.png" alt="image-20230709022005251" style="zoom:55%;" />
>   >
>   > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230709023457003.png" alt="image-20230709023457003" style="zoom:55%;" />



<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230709014825516.png" alt="image-20230709014825516" style="zoom: 67%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230709014853699.png" alt="image-20230709014853699" style="zoom:70%;" />