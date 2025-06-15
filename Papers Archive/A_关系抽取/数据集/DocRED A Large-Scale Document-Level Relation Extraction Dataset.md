# **DocRED: A Large-Scale Document-Level Relation Extraction Dataset**

DocRED 包含 5053 篇人工标注的文章以及 101873 远程监督得到的文章，其中**人工标注的 5053 篇文章中的 3053 篇文章用作训练集，1000 篇用作验证集，1000 篇用作测试集，数据集中共包含 96 种关系。**

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230528112801259.png" alt="image-20230528112801259" style="zoom:50%;" />

对于表示关系(s)的实体对，大约7%的实体对有一个以上的关系标签。



```
Example: 100%|██████████████████████████████| 3053/3053 [22:58<00:00,  2.21it/s]
# of documents 3053.
# of positive examples 35615.
# of negative examples 1163035.
Example: 100%|██████████████████████████████| 1000/1000 [00:22<00:00, 45.07it/s]
# of documents 1000.
# of positive examples 11518.
# of negative examples 385272.
Example: 100%|██████████████████████████████| 1000/1000 [00:21<00:00, 45.51it/s]
# of documents 1000.
# of positive examples 0.
# of negative examples 392158.
```



## train_annotated.json/train_distant.json

```
[
    {  # 一个文档
        "vertexSet": [
            [{'pos': [0, 4], 'type': 'ORG', 'sent_id': 0, 'name': 'Zest Airways, Inc.'}, {}, {}]  # 一个实体，包含多个提及
            [{}, {} ... ]  # 新的实体
        ],
        
        "labels": [
            {"r":"P159",  "h": 0, "t": 2, "evidence": [0]},  # 某个关系
            { ... }
        ],
        
        "title": "AirAsia Zest",
        
        "sents": [
            ["Zest", "Airways", ",", ...],  # 一个句子，已经被拆分
            [ ... ]
        ]
    }, 
    
    {  # 第二个文档
    	...
    }
]
```



##  gen_data.py

<img src="https://cdn.jsdelivr.net/gh/xin-fight/note_image@main/img/image-20230528130348413.png" alt="image-20230528130348413" style="zoom:67%;" />

```
[
    {  # 一个文档
        "vertexSet": [
            [{'pos': [0, 4], 'type': 'ORG', 'sent_id': 0, 'name': 'Zest Airways, Inc.'}, {}, {}]  # 一个实体，包含多个提及
            [{}, {} ... ]  # 新的实体
        ],
        
        "labels": [
            {"r":"P159",  "h": 0, "t": 2, "evidence": [0], 		"intrain": false, "indev_train": false},  # 某个关系
            { ... }
        ],
        
        "title": "AirAsia Zest",
        
        "na_triple": [  ## 保存没有关系的头尾	
        	[0, 1], [0, 3], ...
        ]
        
        "Ls": [0, 40, 71, 87, 98, 119, 164, 186, 197]  ## [0, 0+len(sen[0]), 0+len(sen[0])+len(sen[1])]
        
        "sents": [
            ["Zest", "Airways", ",", ...],  # 一个句子，已经被拆分
            [ ... ]
        ]
    }, 
    
    {  # 第二个文档
    	...
    }
]
```





关注于多个实体对直接的关系



DocRED is constructed with the following three features:

> 1. DocRED包含132，375个实体和56，354个关系事实，在5，053个维基百科文档上进行注释，使其成为最大的人工注释文档级RE数据集。
> 2. 由于DocRED中至少40.7%的关系事实只能从多个句子中提取，DocRED需要阅读文档中的多个句子来识别实体，并通过综合文档的所有信息来推断它们的关系。这将DocRED与那些句子级RE数据集区分开来。
> 3. 我们还提供大规模远程监督数据来支持弱监督RE研究。



我们的最终目标是从纯文本中构建一个文档级RE的数据集

> 这需要必要的信息，包括 named entity mentions, entity coreferences, and relations of all entity pairs, evidence information for relation instances.



 **Distantly Supervised Annotation Generation.**

> 具体来说，我们首先使用spaCy2来执行命名实体识别。然后，这些命名的实体提到被链接到Wikidata项，其中具有相同KBid的命名实体提到被合并。最后，通过查询维基达数据，标记文档中每个合并的命名实体对之间的关系

**Named Entity and Coreference Annotation**

> **从文档中提取关系需要首先识别命名实体提及，并识别引用文档中相同实体的提及。**
>
> 我们要求人类注释者首先审查、纠正和补充阶段1中生成的命名实体提及建议，然后合并那些引用相同实体的不同提及，从而提供额外的共同引用信息。

**Entity Linking.**

> 实体链接（Entity Linking）是将文本中的命名实体提及与知识库中相应的实体进行关联的过程，例如与Wikidata的关联。在所提供的描述中，实体链接阶段涉及以下步骤：
>
> 1. 生成**候选集**：针对每个命名实体提及，创建一个候选集。候选集包含所有在名称或别名**与命名实体提及非常相似的Wikidata实体项**。通过将提及的文本表示与Wikidata实体的标签和别名进行比较来实现。
>
> 2. 扩展候选集：通过将文档中与命名实体提及建立超链接的Wikidata实体项加入候选集来进一步扩展。这意味着如果实体提及在文档中与特定的Wikidata实体项建立了链接，该实体项将被添加到候选集中。
>
> 3. 利用实体链接工具包：实体链接过程还利用了名为TagMe的实体链接工具包。TagMe根据提及的上下文和文档等因素，为实体链接提供建议和推荐。它考虑语义匹配，包括处理数字和时间等实体。
>
> 通过结合这些技术，实体链接阶段旨在将每个命名实体提及链接到一组候选的Wikidata实体项，从而为使用远程监督的关系抽取的后续阶段提供关系推荐。

**Relation and Supporting Evidence Collection.** 

> 有两个挑战：第一个挑战来自文档中大量的潜在实体对；第二个挑战在于在我们的数据集中有大量的细粒度关系类型



**5 Experiments**

> 对于每个单词，==**输入编码器**的特征是其GloVe**单词嵌入**（Pennintonetal.，2014）、**实体类型嵌入**和**共引用嵌入**的连接。==
>
> 



我们认为以下研究方向值得遵循：

> (1)探索明确考虑推理的模型；(2)设计更具表达力的模型体系结构来收集和综合句间信息；(3)利用远距离监督数据提高文档级RE的性能。
