# Towards Integration of Discriminability and Robustness for Document-Level Relation Extraction

| <font style="background: Aquamarine">Pairwise Moving-Threshold Loss with Entropy Minimization</font> | <font style="background: Aquamarine">Supervised Contrastive Learning for Multi-Labels and Long-Tailed Relations</font> | <font style="background: Aquamarine">Negative Label Sampling</font> | <font style="background: Aquamarine">Two New Data Regimes</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |



## 问题以及解决思路

目标是实现对DocRE问题的**可鉴别性**和**鲁棒性**的更好的集成

1. 为赋予 <font style="background: Aquamarine">**概率输出 和 内部表示 的高可辨别性**</font>（**high discriminability** to both **probabilistic outputs** and **internal representations**），我们首先设计了一个**有效的损失函数**

   > 我们创新性地为具有挑战性的==**多标签**和**长尾学习**（multi-label and long-tailed learning  problems）==问题定制==**熵最小化**（entropy minimization）==和==**监督对比学习**（supervised contrastive learning）==。

2. 为了改善<font style="background: Aquamarine">**标签误差**</font>（**label errors**）的影响，我们采用了一种新的==**负标签抽样策略**==，以加强模型的鲁棒性。

3. 引入了两种新的<font style="background: Aquamarine">**数据机制**</font>（**data regimes**）来模拟更现实的场景，并评估我们的抽样策略



### 提高了内部嵌入和概率输出的可辨别性 设计loss

之前**学习自适应阈值**的方法，以更好地分离正关系和负关系，但这些方法都有一些==问题==：

> ATLOP，KD-DocRE: 在所有关系之间强制学习一个总顺序，从而导致**多余的比较和减少它们之间的差异**
>
> ATLOP: 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902151032454.png" alt="image-20230902151032454" style="zoom:33%;" />
>
> KD-DocRE: 
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902151139113.png" alt="image-20230902151139113" style="zoom:33%;" />
>
> <hr>
>
> NCRL—**Margin regularization**：如果它们在解决标签失衡问题时的 平均边际 低于阈值，则**不当地对所有预定义的 正实体对 标签进行惩罚**
>
> <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20240319105016564.png" alt="image-20240319105016564" style="zoom:50%;" />

相比之下，我们提出了一种学习<font style="background: Aquamarine">**部分顺序（*partial order*）**</font>的方法来**设计损失函数**，将所有的正关系 **单独排序** 在一个阈值以上，而这个阈值又排在所有的负关系之上。



### 良好的可鉴别性和鲁棒性的 设计了 负标签采样

之前的方法都没有考虑到**内部表示的可辨别性**，以及模型**对注释错误的鲁棒性**

我们设计了两种新的<font style="background: Aquamarine">**数据机制**</font>和一种新的<font style="background: Aquamarine">**负标签采样策略**</font>，即使在不完整的注释下，也能提供持续的强性能。





## 模型

### Encoder

ATLOP

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902152201093.png" alt="image-20230902152201093" style="zoom: 33%;" />

###  创新1：Pairwise Moving-Threshold Loss with Entropy Minimization ==$L_1$==

与ATLOP中的ATL损失思路类似：**鼓励每个正关系的预测得分高于NA类，并鼓励NA类的得分高于负关系**

为了区分对于实体对的正关系和负关系，我们进行了如下的定义：

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902154508304.png" alt="image-20230902154508304" style="zoom: 33%;" />

之后将实体对的 关系r 和 Na类的logits记为$f_r$ 和 $f_η$ —— $f_r$ 和 $f_η$都**表示未归一化的预测分数（对数）**



#### 1.1 Pairwise moving-threshold loss ==$P_{h,t}^n, L_{pmt}^{h,t}$==

我们引入了一种<font style="background: Aquamarine">**部分顺序（*partial order*）**</font>：

> ==正关系 **没有相互比较，它们的相对排名也没有建模**（负关系之间也没有相互比较）==，因为我们**只寻找真实的关系集，而 不关心 它们的相对真实性程度**



定义<font style="background: Aquamarine">**pairwise moving-threshold loss** </font> $L_{pmt}^{h,t}$ —— **实现了 部分顺序的相对阈值**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902160634987.png" alt="image-20230902160634987" style="zoom: 50%;" />

> 和ATLOP一样，**一个实体对对应一个阈值**$f_η$，当r为正关系时，fr > fη，当r为负关系时，fη > fr。
>
> <hr>
>
> > 虽然之前的工作**ATLOP,KD-DocRE**采用类似的阈值机制，但**他们学习了所有关系（或一组关系）和阈值类的总顺序**，==这不可避免地减少了每个关系概率与阈值的概率之间的差异（因为总概率和为1）==
> >
> > 但是建模关系之间的多余排序，**浪费了有限的概率质量**(总价值1.0)，**不利于多标签问题**，并且**不可避免地减小了每个关系的概率与阈值的概率之差**，如图所示，当多个关系进行比较时，相对于单独关系比较，之间的差距会被减小
> >
> > <img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902173540308.png" alt="image-20230902173540308" style="zoom: 33%;" />

#### 1.2 entropy minimization - 关系概率 和 阈值概率 差距变大 ==$H_{h,t}(r), L_{em}^{h,t}$==

为了==减少一个关系是正是负的不确定性，从而使其值很容易被识别==，采用<font style="background: Aquamarine">**熵最小化（entropy minimization）**</font>的原理**让熵进入损失函数**

* **熵最小化**可以**让每个 关系的概率 与 阈值的概率 之间的差异变大**，便于更好区分（即：关系概率和阈值概率差距大，损失越小）
* 对于关系r预测时，**增加 预测其为r的概率$P^r_{h,t}(r)$ 和 预测其为Na的概率为$P^n_{h,t}(r)$** 之间的

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902174229687.png" alt="image-20230902174229687" style="zoom:50%;" />

> <font style="background: Aquamarine">信息熵随着两者之间的绝对差值 增加而减小</font>，由于**Pr + Pn = 1（公式3）**，所以想要熵变小，Pr与Pn之间的差距要大
>
> 所以，==**将熵纳入损失函数将有助于强调所有关系的 成对Pr和Pη之间的差异，从而更容易从阈值NA中区分正（或负）关系。**==



<font style="background: Aquamarine">**pairwise moving threshold loss** with **entropy minimization**</font> 

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902182949383.png" alt="image-20230902182949383" style="zoom:50%;" />

$L_{pmt}^{h,t}$不足：

> 当**负关系的 数量远远超过 正关系的数量**时（Nh,t的总和可能会压倒Ph,t的总和，毕竟没有关系的实体对占大多数）这可能导致在优化过程中，==为了最小化$L_{pmt}^{h,t}$，需要将fη 设置为 远高于 每个负关系fr。**这种情况下，模型可能会过于关注负关系，而不够关注正关系**==

$L_{em}^{h,t}$弥补了不足：

> 公式6中Lem提供了一种有原则的方法来“平衡”所有关系r的概率fr和η的概率fη之间的巨大差
>
> 因为将熵最小化加入损失后，==为了使损失越小，会让**关系概率 和 阈值概率 差距变大**==，以平衡 **正负关系的概率 和 阈值概率**



### 创新2：Supervised Contrastive Learning for Multi-Labels and Long-Tailed Relations ==$L_2$==

#### 2.1 Supervised Contrastive Learning $L_{scl}^{h,t}$

公式7只关注了概率输出的差距，但也要认识到<font style="background: Aquamarine">**不同关系的 实体对 的嵌入的差异**</font>

即：==**拉近 相同关系的实体对 的嵌入，推远 不同关系的实体对 的嵌入**==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902185203452.png" alt="image-20230902185203452" style="zoom: 67%;" />

对于某个实体对$(h,t)$：

> *B* 表示一个batch中包含的实体对
>
> $S_{h,t}$ 是该bs中 与实体对$(h,t)$ **有至少一个相同关系的** 那些实体对，注意该集合不包含$(h,t)$；
>
> * 公式中 该集合$S_{h,t}$中的一个元素$p$ 被认为是positive example
> * 而negative examples则为：$B /\ (S_{h,t} ∪ {(h, t)})$ 

为了最小化$L_{scl}^{h,t}$，我们让实体对$(h,t)$的嵌入 ==**与正例的嵌入相互接近**（最大化分子），**远离负例的嵌入**（最小化分母）==

#### 2.2 解决long-tail问题 $L_{lt}^{h,t}$

<font style="background: Aquamarine">但是公式8不适用于数据集中的**long-tail**问题</font>

* 因为对于一个只有长尾关系的那些实体对，可能在该**bs中无法找到 与其 关系相同的正例**，==即公式8中的$S_{h,t}$为**空集**==



所以我们修改损失函数，**加入$L_{lt}^{h,t}$：**

* 对于因**长尾问题而无法找到正例**的那些实体对，我们==**只需要让其嵌入 远离 同一批的其他实体对的嵌入$x_d$**==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902190839885.png" alt="image-20230902190839885" style="zoom: 40%;" />

> $L_2$含义：==**若实体对$(h,t)$在bs中有正例，让其接近正例嵌入，远离负例嵌入；		如果没有正例，则远离该bs中其他所有实体对的嵌入**==
>
> * $B_p$ 是bs中 **存在关系的实体对**；即**$L_2$不考虑那些关系为Na的实体对**，因为最小化关系为NA的两个实体对之间的嵌入距离是没有意义的



<hr>



**pairwise moving threshold loss** with **entropy minimization** 合并 **Supervised Contrastive Learning** for Multi-Labels and Long-Tailed Relations

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902192100518.png" alt="image-20230902192100518" style="zoom:33%;" />

### 创新3：Negative Label Sampling ==$L^{'}$== - 解决假阴性问题

DocRE基准测试存在<font style="background: Aquamarine">严重的**假阴性问题**（severe false-negative problem）</font>，这意味着之前**被标记为NA类的不少实体对应该至少有一个关系标签**



我们提出了一种新的**负标签采样策略**

> 在计算损失函数时，只==**对每个具有NA标签的实体对采样一小部分 负关系**==。我们==**假设**这些假阴性例子的真实关系标签不在采样的样本中，因此我们可以避免在损失函数中将正确的标签作为负关系==



<font style="background: Aquamarine">思路：</font>

* 对于某个关系为Na的实体对，我们不知道其真实的关系是否不为Na，但==可以通过人为假设和采样**知道其哪些关系一定是错误的**==

* 所以我们直接**让这些错误的关系 的概率变小，并其让他们和Na之间的差距变大**



<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902194236000.png" alt="image-20230902194236000" style="zoom:50%;" />

> $B_N$ 是该bs中 没有关系的那些实体对，即**关系为Na的实体对**
>
> $N_{h,t}^{'}$ 是对$B_N$中的每个实体对，**均匀采样的一个负关系的子集**，是==关系的集合== —— 上面已人为假设：该被标记为Na的实体对的 真实关系 不在该子集中
>
> 公式12解释：对于关系为Na的实体对，抽取的某个关系r —— 根据==假设得到这些**负关系都是错误**的关系==
>
> * $P_{h,t}^n(r)$ 是实体对$(h,t)$的关系是 抽取关系r或Na情况下，为**Na的概率**
> * $H_{h,t}(r)$是实体对$(h,t)$的关系是 抽取关系r或Na情况下，**关系是抽取关系r概率 与 为Na的概率 之间差距变大的熵**

$L^{'}$值要小，需要让关系为Na的那些实体对 为**Na的概率变大（即为负关系的可能性变小），且让抽取出的负关系与Na概率差距变大**

<hr>

### ==总的损失函数==

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230902201101894.png" alt="image-20230902201101894" style="zoom:50%;" />

> $B_p$ 是bs中 **存在关系的实体对** 
>
> 结合**pairwise moving threshold loss** with **entropy minimization** $L_{pmt}^{h,t}+L_{em}^{h,t}$，**Supervised Contrastive Learning** for Multi-Labels and Long-Tailed Relations $L_2$， 和 **Negative Label Sampling** $L^{'}$ 得到最后的loss公式
>
> * ==$L_1^{NA}$是创新1中$L_1$修改版==，**让那些关系为Na的实体对使用Negative Label Sampling（创新3）进行计算，其他的不为Na的实体对则使用原始$L_1$计算**
> * $L_2$是**在实体对嵌入上进行运算的，即其并不会因为关系的改变而改变**



## Two New Data Regimes - 模拟真实环境

为了在一个更真实的实验环境中评估模型，<font style="background: Aquamarine">仔细测试模型对噪声数据的弹性</font>



==**提出了两种机制（Two New Data Regimes）**==：**OOG**-DocRE and **OGG**-DocRE，这两种机制都反映了真实场景中的训练数据是嘈杂的，而人工工作只能花费在清理一个相对较小的验证/测试集上。

> O 表示数据是unclean, noisy DocRED dataset
>
> G 表示数据是Gold labels in the new, cleaned Re-DocRED dataset
>
>  “OOG” and “OGG” 分别表示 训练，验证，测试 下数据集的状态
>
> 请注意，在这两种机制中，都没有使用来自Re-DocRed的清理训练集。



## 实验

entropy minimization $L_{em}^{h,t}$ 和 Supervised Contrastive Learning for Multi-Labels and Long-Tailed Relations $L_2$ 去掉后，尾部关系的F1下降的多

* **这表明这两个分量对长尾关系都是有用的，并强调了l2的有效性，其中一部分是为了迎合长尾关系而设计的**

损失函数仅仅从ATLOP中的ATL变成Pairwise moving-threshold loss $L_{pmt}^{h,t}$ 后，仍然比基线ATLOP性能好

* **这反映了我们的成对移动阈值损失(Pairwise moving-threshold loss) 的有效性。**

在有很多实体对被错误的标记为Na的环境下，使用negative label sampling loss后，在被清理过的数据集下模型效果的最优

* **这证明了我们的负标签采样策略在对抗被标记为NA的实体对中存在的噪声方面的有效性**



在我们的模型中，关系和阈值标签NA之间的logit差异远远大于ATLOP模型。**这表明我们的模型能够学习最终概率分数的更差异化分布。**

对于那些ATLOP预测错误的实体对，我们的模型**不仅正确地预测了其正确的标签，而且最大化了预测分数的可辨别性**。