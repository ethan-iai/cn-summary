# cn-summary

Repostiory for generation of Chinese text summary with multiple algorithms

## features 

- TexTRank
- MMR 
- TF-IDF
- LDA
- LSI
- Bert Extractive (Bert + k-means)
- Bert Seq-to-seq (UNILM)
## Installation
```shell
pip install -r requirements.txt
pip install -e .
```

## Documentation

In developing...

## Run Summarization Methods

```python

import cn_summary as cs

document = """
习近平指出，人工智能具有多学科综合、高度复杂的特征。我们必须加强研判，统筹谋划，协同创新，稳步推进，
把增强原创能力作为重点，以关键核心技术为主攻方向，夯实新一代人工智能发展的基础。要加强基础理论研究，
支持科学家勇闯人工智能科技前沿的“无人区”，努力在人工智能发展方向和理论、方法、工具、系统等方面取得变革性、颠覆性突破，
确保我国在人工智能这个重要领域的理论研究走在前面、关键核心技术占领制高点。
要主攻关键核心技术，以问题为导向，全面增强人工智能科技创新能力，加快建立新一代人工智能关键共性技术体系，在短板上抓紧布局，
确保人工智能关键核心技术牢牢掌握在自己手里。要强化科技应用开发，紧紧围绕经济社会发展需求，
充分发挥我国海量数据和巨大市场应用规模优势，
坚持需求导向、市场倒逼的科技发展路径，积极培育人工智能创新产品和服务，推进人工智能技术产业化，
形成科技创新和产业应用互相促进的良好发展局面。要加强人才队伍建设，以更大的决心、更有力的措施，
打造多种形式的高层次人才培养平台，加强后备人才培养力度，为科技和产业发展提供更加充分的人才支撑。
"""

# extractive methods
summary = cs.mmr(document, num_sentences=2)
print(summary)

summary = cs.textrank(document)
print(summary)

summary = cs.tf_idf(document)
print(summary)

summary = cs.lda(document)
print(summary)

summary = cs.lsi(document)
print(summary)

summary = cs.bert_extract(document)
print(summary)


# abstractive method
summary = cs.bert_seq2seq(document)
print(summary)

```

### Train UNILM For Abstravice Method

cn-summary provides command line to train UNILM. For example, to train UNILM for 90 epochs on gpu 0 using CLI:

```shell
python -m cn_summary.engine.bert_seq2seq_summarizer.train ./data \
                                                    --epochs 90 \
                                                    --batch-size 32 \
                                                    --gpu 0
```