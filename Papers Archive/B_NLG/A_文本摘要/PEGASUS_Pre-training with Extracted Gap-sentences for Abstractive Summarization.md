# PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization

| <font style="background: Aquamarine">Gap Sentences Generation (GSG)</font> | <font style="background: Aquamarine">**Zero and Low-Resource**</font> | <font style="background: Aquamarine">**Masked Language Model (MLM)**</font> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |

​	

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230507174358599.png" alt="image-20230507174358599" style="zoom: 50%;" />

## 问题

1. 针对抽象文本摘要的预训练目标尚未被探索。
2. 此外，目前还缺乏跨不同领域的系统评估(systematic evaluation)



在PEGASUS中，**重要的句子被从输入文档中删除/屏蔽**，并从**剩余的句子中一起生成为一个输出序列**，类似于提取的摘要

我们的模型在**低资源总结方面**也显示出了惊人的性能

> we study pre-training objectives specififically for abstractive text summarization and **evaluate on 12 down stream datasets** spanning **news** (Hermann et al., 2015; Narayan et al., 2018; Grusky et al., 2018; Rush et al., 2015; Fabbri et al., 2019), **science** (Cohan et al., 2018), **short stories** (Kim et al., 2019), **instructions** (Koupaee & Wang, 2018), **emails** (Zhang & Tetreault, 2019), **patents** (Sharma et al., 2019), and **legislative bills** (Kornilova & Eidelman, 2019).

我们发现，从文档中屏蔽整个句子，==**并从文档的其余部分生成这些空白句子**==，可以很好地作为下游总结任务的预训练目标

> 特别是，选择相对重要的句子 胜过 领先或随机(lead or randomly)选择的句子。





## Related word

**MASS**  (Song et al., 2019)

> 提出了掩蔽的序列-序列生成(masked sequence-tosequence generation)，即在给定句子的剩余部分的情况下重建一个句子片段。一个单一的句子片段被随机选择。

**UniLM** (Dong et al., 2019)

> 提出对三种类型的语言建模任务进行联合训练：单向（从左到右和从右到左），双向（词级掩码，有下一句预测），以及序列到序列（词级掩码）预测。

**T5** (Raffel et al., 2019) 

> 将文本到文本框架推广到各种NLP任务中，并展示了扩大模型规模（达到110亿个参数）和预训练语料的优势，引入了C4，一个来自Common Crawl的大规模文本语料，我们在一些模型中也使用了它。T5是用不同掩码率和跨度大小的 随机损坏的文本跨度进行预训练。

**BART** (Lewis et al., 2019)

> 引入去噪自动编码器来预训练序列到序列模型。BART使用任意噪声函数破坏文本，并学习重建原始文本。对于生成任务，噪声函数是文本填充，它使用单个屏蔽记号来屏蔽随机采样的文本范围



与MASS、UniLM、BART和T5相比，提出的**PEGASUS屏蔽了多个完整的句子**，而不是更小的连续文本跨度。==在我们的最终目标中，我们**根据重要性确定地选择句子**，而不是随机地==。

与T5一样，PEGASUS不重建完整的输入序列，**只生成屏蔽句子作为单个输出序列**。在这项工作中，我们完全专注于下游的摘要(生成)任务，并不评估NLU分类任务。



## Gap Sentences Generation - GSG

我们假设，使用一个更接近下游任务的训练前目标，会导致更好和更快的微调性能

我们从文档中选择并掩码整个句子，并将间隙句子连接成一个伪摘要。

==**为了更接近摘要，我们选择看起来对文档重要的句子。由此产生的目标既通过经验证明了掩蔽的的好处，又预见了下游任务的形式。**==



句子被独立评分(**Ind**),并选择前m名。我们还考虑像Nallapati等人(2017)那样，通过贪婪地最大化所选句子之间的ROUGE1-F1来顺序(**Seq**)选择它们



## **Pre-training Corpus**

* **C4**, or the Colossal and Cleaned version of Common Crawl, introduced in Raffel et al. (2019); consists of text from 350M Web-pages (750GB).

* **HugeNews**, a dataset of 1.5B articles (3.8TB) collected from news and news-like websites from 2013-2019. A whitelist of domains ranging from high quality news publishers to lower-quality sites such as high-school newspapers, and blogs was curated and used to seed a web-crawler. Heuristics were used to identify news-like articles, and only the main article text was extracted as plain text.
