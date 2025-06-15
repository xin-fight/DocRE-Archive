# Coherent Comment Generation for Chinese Articles with a Graph-to-Sequence Model

将 图 应用到 生成任务上

> 报告视频：https://vimeo.com/385264795



## 问题描述

评论生成有几个重要的挑战：

* **新闻文章可能很长**，这使得经典的序列到序列模型难以处理。相反，虽然**标题是一个非常重要的信息资源，但它可能太短，无法提供足够的信息。**

  对于传统的基于编解码器的模型来说，新闻文档通常太长，这往往会导致一般的和不相关的评论。在本文中，我们建议使用图到序列模型生成注释，该模型将输入新闻建模为主题交互图。

* 新闻的标题有时使用与文章内容语义不同的**夸张(hyperbolic)表达**。

* 用户在发表评论时会**关注新闻的不同方面（主题topic）**

我们建议**将长文档表示为一个主题交互图**，它将文本分解为几个==以主题为中心的文本集群==，每个文本集群代表文章的一个关键方面（主题）。

> 每个集群与主题一起在图中形成一个顶点。
>
> 顶点之间的边 是根据顶点之间的语义关系来计算的。

我们的模型能更好的**理解新闻中不同topics的联系**

我们的模型通过**将标题作为一个特殊的顶点**，对文章的标题和内容进行联合建模，这有助于得到文章的要点。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407111624531.png" alt="image-20230407111624531" style="zoom:50%;" />



**文章工作：**

> 我们用一个==**主题交互图来**表示文章==，它将文章的**句子组织成几个以主题为中心的顶点**。
>
> 我们提出了一个**基于主题交互图生成评论的图到序列模型**。

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407111657302.png" alt="image-20230407111657302" style="zoom:50%;" />



## 相关工作

GNN不仅适用于结构场景（社交网络预测系统，推荐系统，知识图谱），还适用于非结构场景（关系结构并不明确，包括图像分类，文本）

* 在文本分类任务中应用GNN，其中包括将长文档建模为图
* 在文本生成任务中使用GNN



虽然这些工作应用GNN作为编码器，但**它们是为了利用图的形式的信息（SQL查询，AMR图，依赖图），输入文本相对较短的信息**，而==我们的工作试图**将长文本文档建模为图**，这更具挑战性。==



三个人工评论：**Coherence**；**Informativeness**；**Fluency**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407112306458.png" alt="image-20230407112306458" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407112508857.png" alt="image-20230407112508857" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407112945262.png" alt="image-20230407112945262" style="zoom: 50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407113922115.png" alt="image-20230407113922115" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407114032079.png" alt="image-20230407114032079" style="zoom:50%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230407114116788.png" alt="image-20230407114116788" style="zoom:50%;" />
