# SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization

| <font style="background: Aquamarine">exposure bias</font> | <font style="background: Aquamarine"> a gap between the *objective function* and the *evaluation* *metrics*</font> | <font style="background: Aquamarine">contrastive learning</font> |
| :-------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |





SIMCLS：将文本生成制定为**由对比学习辅助的无参考评估问题（即质量估计）**。可以**弥补**目前主导的序列到序列学习框架所产生的**学习目标和评估指标之间的差距**

问题:

> 1. 目标函数和评估度量之间差距
>    因为目标函数是基于本地的、令牌级别的预测
>    而评估度量（例如ROUGE）将比较黄金参考和系统输出之间的整体相似性
> 2. exposure bias



虽然之前提议引入对比损失作为条件文本生成任务的 MLE 训练的增强

但我们反而选择通过在我们提出的框架的不同阶段引入它们来理清对比损失和 MLE 损失的功能。



we propose to use a two-stage model for abstractive summarization:（**a generate-then-evaluate two stage framework with contrastive learning**）

> 1. 首先利用MLE loss训练Seq2Seq模型**生成的候选摘要**
>
> 2. 然后训练参数化评估模型，通过**对比学习对生成的候选摘要进行排序**。



<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230424181302145.png" alt="image-20230424181302145" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230424182102729.png" alt="image-20230424182102729" style="zoom:67%;" />

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230424182114859.png" alt="image-20230424182114859" style="zoom:67%;" />