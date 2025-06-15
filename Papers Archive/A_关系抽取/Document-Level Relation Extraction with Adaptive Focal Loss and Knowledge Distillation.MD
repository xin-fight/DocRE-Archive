# Document-Level Relation Extraction with Adaptive Focal Loss and Knowledge Distillation

| <font style="background: Aquamarine">Axial Attention-Enhanced Entity Pair Representation</font> | <font style="background: Aquamarine">Adaptive Focal Loss</font> | <font style="background: Aquamarine">Knowledge Distillation</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |



<img src="C:/Users/69014/Desktop/Github%E5%9B%BE%E5%BA%8A/local_image/image-20230723195741096.png" alt="image-20230723195741096" style="zoom:50%;" />

> **Entity Representation**：与ATLOP操作相同，利用了**contextual pooling method** 
>
> **Entity Pair Representation**：($z$是融合了上下文的实体嵌入，最后得到实体对嵌入$g^{(s,o)}$) 
>
> > <img src="C:\Users\69014\Desktop\Github图床\local_image\image-20240306165626143.png" alt="image-20240306165626143" style="zoom: 33%;" />
> >
> > <img src="C:/Users/69014/Desktop/Github%E5%9B%BE%E5%BA%8A/local_image/image-20230723232917158.png" alt="image-20230723232917158" style="zoom:50%;" />



## 1.学习实体对之间的相互依赖关系，**提高双跳关系推理**的性能

> ==**Axial Attention-Enhanced Entity Pair Representation **== $r_w^{(s,o)}$ 
>
> 我们建议使用**双跳注意来编码每个实体对（es，eo）表示的轴向相邻信息，而不是只使用头和尾嵌入来进行关系分类**。
>
> 我们的动机是：为了关注==**两跳关系**==三元组的一跳邻居。   
>
> > 使用two-hop attention来编码每个实体对（es，eo）表示的轴向相邻信息 即：**(*es, eo*) → (*es, ep*) and (*ep, eo*)** 
>
> <img src="C:/Users/69014/Desktop/Github%E5%9B%BE%E5%BA%8A/local_image/image-20230724013213569.png" alt="image-20230724013213569" style="zoom:60%;" />



## 2.不平衡的标签分布问题

> **==Adaptive Focal Loss==** 解决了标签分布不平衡的问题
>
> <font color='red'>损失函数鼓励**长尾类(long-tail classes)对总体损失的贡献更大**</font>
>
> > the output logits for all relations:
> >
> > <img src="C:/Users/69014/Desktop/Github%E5%9B%BE%E5%BA%8A/local_image/image-20230724013915445-1690133972986-1.png" alt="image-20230724013915445" style="zoom: 50%;" />
> >
> > **Adaptive Focal Loss** (**AFL**) as an enhancement to **Adaptive Thresholding Loss**(**ATL**) for ==**long-tail classes**==
> >
> > <img src="C:/Users/69014/Desktop/Github%E5%9B%BE%E5%BA%8A/local_image/image-20230724014121004.png" alt="image-20230724014121004" style="zoom:50%;" />
> >
> > > 由于正标签的分布是高度不平衡的，我们<font color='red'>利用**focal loss（Lin et al.，2017）来平衡正类的对数**</font> 
> > >
> > > **我们的损失是为了更多地关注low-confifidence classes**。==如果$P(r_i)$较低，则相关类的损失贡献将较高==，从而可以对长尾类进行更好的优化。
> >
> > 
> >
> > > **<font style="background: Aquamarine">这与原始的ATL不同，在原始的ATL中，所有的positive logits都与一个softmax函数一起排序</font>**
> > >
> > > **<font style="background: Aquamarine">而AFL是positive ri的logit分别与阈值类TH的logit进行排序（直接看ATLOP中的公式和公式9即可得到结果）</font>**



## 3.克服**human annotated data 与distantly supervised data之间的差异**

> 远距离监督适应的关键挑战(the distant supervision adaptation)是：克服远监督数据与人类标注数据的概率分布的差异
>
> ==**knowledge distillation**==
>
> >首先用少量的**人类注释数据**来训练一个teacher model。
> >
> >然后teacher model将被用来对大量的远程监督数据进行预测。生成的预测被用作预训练我们的student model的**软标签(soft labels**)。
> >
> >最后对预先训练好的student model进行了进一步的微调，同时进行了两个信号的训练
> >
> >1. 第一个信号是来自**远距离监督数据的hard labels的监督**
> >2. 第二个信号是来自**预测的soft labels**。
> >
> ><img src="C:/Users/69014/Desktop/Github%E5%9B%BE%E5%BA%8A/local_image/image-20230724015542932-1690134949984-3.png" alt="image-20230724015542932" style="zoom: 67%;" />









