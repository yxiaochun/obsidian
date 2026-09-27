---
title: "抖音广告&电商LONGER:无GSU的End2End长序列建模"
source: "https://zhuanlan.zhihu.com/p/1927757714173460579"
author:
  - "[[州懂]]"
published:
created: 2026-09-26
description: "大家好, 我是州懂, 最近攒了好些待分享的内容, 趁这两天空闲下来陆续整理出来和大家一直分享学习。 标题: LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders 地址: https://www.arxiv.org/pd…"
tags:
  - "clippings"
---
[收录于 · AI 长风破浪](https://www.zhihu.com/column/aisailing)

130 人赞同了该文章

目录

收起

1\. 前言

2\. 方法

2.1 全局Tokens & 序列Tokens

2.1.1 全局Tokens的组成

2.1.2 全局Tokens的作用

2.1.3 序列Tokens的位置编码

2.1.4 Token映射降维

2.2 Tokens融合

2.2.1 InnerTrans模块

2.2.2 效率分析

2.3 注意力机制

2.3.1 Cross Causal Attention(第1层)

2.3.2 Self Causal Attention(其它层)

2.4 训练和部署优化

2.4.1 全同步训练框架

2.4.2 混合精度训练与重计算

2.4.3 KV Cache服务

3\. 实验部分

3.1 整体效果

3.2 消融实验

3.3 Scaling分析

3.4 线上AB测试

大家好, 我是州懂, 最近攒了好些待分享的内容, 趁这两天空闲下来陆续整理出来和大家一直分享学习。

> 标题: [LONGER](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=LONGER&zhida_source=entity): Scaling Up Long Sequence Modeling in Industrial Recommenders  
> 地址: [arxiv.org/pdf/2505.0442](https://link.zhihu.com/?target=https%3A//www.arxiv.org/pdf/2505.04421)  
> 公司: 字节跳动

## 1\. 前言

用户行为的长序列建模是推荐系统的一个核心命题, 长序列建模对于提高推荐结果的准确性,多样性,以及缓解推荐系统的信息茧房问题都至关重要。

由于线上性能的约束, 业界一般遵循以下几种策略进行长序列建模:

- **两阶段检索:** 首先使用 [GSU](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=GSU&zhida_source=entity) 从用户行为序列中检索出top- $k$ 个行为组成短序列, 再使用 [ESU](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=ESU&zhida_source=entity) 对抽取出的短序列进行精细化用户兴趣建模。这是工业界探索和落地最多的方式了, 比如阿里的 [SIM](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=SIM&zhida_source=entity) 、ETA, 美团的 [SDIM](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=SDIM&zhida_source=entity), 快手的 [TWIN](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=TWIN&zhida_source=entity) 和TWIN-V2等。

[![](https://pica.zhimg.com/v2-38c0709c557373bee2463957fd0e0c16.png?source=7e7ef6e2&needBackground=1)](https://zhuanlan.zhihu.com/p/699924066)

- **预训练用户 [Embedding](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=Embedding&zhida_source=entity):** 在源模型中预训练超长序列, 生成用户Embedding，然后迁移到下游模型使用。
- **记忆增强模型:** 通过记忆网络缓存关键信息，减少计算复杂度, 以空间换时间, 比如阿里的 [MIMN](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=MIMN&zhida_source=entity), 快手的 [MARM](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=MARM&zhida_source=entity) 等

近年来, 也有一些方法提出不依赖于中间阶段的 **直接长序列建模** 方案, 比较常见的思路是将用户行为长序列压缩or聚合成短序列进行建模:

- **长序列压缩:** 阿里今年提出LREA方法使用 [低秩矩阵分解](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=%E4%BD%8E%E7%A7%A9%E7%9F%A9%E9%98%B5%E5%88%86%E8%A7%A3&zhida_source=entity) 来简化推理时的计算
- **长序列聚合:** 快手在 [KuaiFormer](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=KuaiFormer&zhida_source=entity) (召回)中将用户历史行为序列按交互时间顺序按不同粒度分组, 再使用Mean Pooling做聚合来扩展更长的用户序列。

抖音的这篇论文也是类似的思路, 只是对长序列进行更高效的聚合以减少信息损失, 同时提高计算效率。

## 2\. 方法

作者所提LONGER长序列建模方法的整体框架如下图所示, 下面详细介绍

![](https://pica.zhimg.com/v2-97316cc04bf56a0860555cb58e4e6e9c_1440w.jpg)

### 2.1 全局Tokens & 序列Tokens

LONGER的输入分成两部分, 一部分是常规的用户行为长序列Tokens, 另一部分作者称之为"Global Tokens", 笔者认为这个"Global Tokens"的设置还是非常有技巧的。

### 2.1.1 全局Tokens的组成

全局Tokens主要包含几个信息:

- **Target Item Features:** 当前候选物品ItemID, Tag标签等
- **User Profiles:** 比如UserID, 性别等
- **Context & Cross Features:** 当前用户和候选Item的交叉特征
- **Learnable CLS Token:** 一种可学习的特殊标记，用于聚合整个序列的信息。

其中, 从框架图的可以看出, User Profiles和Context & Cross Features合并组成一个Token, 而Candidate Item Features视为单独的Token。此外, 这些全局Tokens放在 **开头位置** 的, 这个很重要。

### 2.1.2 全局Tokens的作用

这些全局Tokens的作用有两个:

- **起到全局信息锚点作用:** 全局Tokens放在开头位置, 其具有完整的注意力感受野, 使它们能够从整个序列中聚合上下文信号，同时影响所有其他序列Token, 可以进一步促进用户历史、上下文属性和候选Item间的特征交互。
- **稳定长序列注意力机制:** 在长序列建模时，注意力机制可能会出现“Attention sink”效应(StreamLLM论文的作者发现了一个神奇的现象，最开头的initial tokens尽管从整体生成内容的语义上感觉没那么重要，但是它们的attention score一直很高)。将全局Tokens放在开头位置，正好可以作为全局信息锚点，稳定注意力分布，避免模型过度关注序列的早期部分。

### 2.1.3 序列Tokens的位置编码

对于用户行为序列, 为了更好的捕获时序信息, 作者增加了两种常规的位置编码:

- **绝对时间差:** 用户历史行为交互时间戳与候选Item请求时间戳的Diff作为Side Info **拼接** 到Item Embed中
- **绝对位置编码:** 基于Item在用户行为序列中的下标得到绝对位置编码, 并直接 **加** 到Item Embed上

### 2.1.4 Token映射降维

位置编码之后, Global Tokens和序列Tokens对应的特征各自经过 [MLP](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=MLP&zhida_source=entity) 做映射降维以对齐特征维度, 得到最终的表征 $\mathbf{R} \in \mathbb{R}^{\left(m + L\right) \times d} = \left[\mathbf{G} \in \mathbb{R}^{m \times d} ; \mathbf{H} \in \mathbb{R}^{L \times d}\right]$, 抖音这里实验时, 设置 $d = 32$, $L = 2000$

### 2.2 Tokens融合

常规Transformer的时间复杂度为 $O \left(L^{2} d\right)$, Token Merge 的主要目标是减少长序列中的tokens数量, 从而降低模型的计算开销。作者将用户行为长序列按时间顺序每 $K$ 个Item合并成一组, 这样合并后的序列长度则减少为 $\frac{L}{K}$ 。

### 2.2.1 InnerTrans模块

对于组内的 $K$ 个Item, 直接concat起来会导致token间缺乏交互, 为了弥补这一不足, 对组内的 $K$ 个Item, 作者增加一个轻量级($K$ 很小所以计算开销不大)的Transformer模块。

$$
\mathbf{M}_{i} = \text{TransformerBlock} \left(\left[\mathbf{e}_{i}^{1} , . . . , \mathbf{e}_{i}^{K}\right]\right)
$$

### 2.2.2 效率分析

标准Transformer encoder的FLOPs和参数量分别为:

$\text{FLOPs}_{\text{vanilla trans}} = 24 L d^{2} + 4 L^{2} d$ $\#\text{Params}_{\text{vanilla trans}} = 12 d^{2} + 13 d$

使用Tokens融合, 计算效率的对比为:

$$
\frac{\text{FLOPs}_{\text{Merge Token}}}{\text{FLOPs}_{\text{vanilla}}} = \frac{24 L d^{2} K + \frac{4 L^{2} d}{K}}{24 L d^{2} + 4 L^{2} d} = \frac{6 d K + \frac{L}{K}}{6 d + L}
$$

例如当 $= 2048$, $= 32$ 时, 常规Transformer的FLOPs ≈ 587M, Token融合($= 4$)的FLOPs ≈ 336M, 计算开销减少42.8%。

而Token融合后token embedding size为 $K d$, 因此, Token融合后的参数量为 $\Theta_{\text{merge}} = 12 K^{2} d^{2} + 13 K d$ 。也就是说, 通过Token融合, 模型的参数规模进一步提升了, 但计算开销反而降低了。

### 2.3 注意力机制

原始的Self Attention的计算效率并不高:

$$
\text{Attention} \left(\mathbf{Q} , \mathbf{K} , \mathbf{V}\right) = \text{Softmax} \left(\frac{\mathbf{Q} \mathbf{K}^{T}}{\sqrt{d}}\right) \mathbf{V}
$$

对于 $\mathbf{Q} , \mathbf{K} , \mathbf{V} \in \mathbb{R}^{\left(m + L\right) \times d}$, 原始Self-Attention注意力计算的时间复杂度为 $O \left(\right. \left(m + L \left.\right)^{2} d\right)$, 是非常高的计算开销。

为此, 作者这里在第1层使用 [Cross Attention](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=Cross+Attention&zhida_source=entity) 用以压缩长序列, 降低计算开销, 后面接着是 $N$ 层的Self Attention以实现高阶的特征交叉。

$$
\underset{\text{compress long sequence}}{\underbrace{\text{CrossAttn} \left(\mathbf{O} , \mathbf{R}\right)}} \rightarrow \underset{\text{high}-\text{order interactions}}{\underbrace{\text{SelfAttn} \left(\cdot\right) \times N}}
$$

### 2.3.1 Cross Causal Attention(第1层)

Cross Attention是通过计算一个序列(称为查询序列，Query Sequence)与另一个序列(称为键值序列，Key-Value Sequence)之间的注意力分数，动态地从键值序列中提取相关信息，以增强查询序列的表示。

作者从原始的长序列 $\mathbf{H}$ 中采样出 $\mathbf{H}_{\mathbf{S}}$ 子序列作为部分Query序列。具体地, 作者尝试了几种采样的策略, 实验发现直接使用最近的 $k$ 个交互作为部分Query序列的策略效果最好。然后, 再在前面拼接Global Tokens的表征就得到最终的Query矩阵 $\mathbf{O} = \left[\mathbf{G} ; \mathbf{H}_{\mathbf{S}}\right] \in \mathbb{R}^{\left(m + k\right) \times d}$

论文提到, 这种混合注意力设计的动机是观察到模型性能相对于序列token的数量表现出强烈的边际效应：仅采样完整序列的40%就保留了超过95%的 [性能改进](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=%E6%80%A7%E8%83%BD%E6%94%B9%E8%BF%9B&zhida_source=entity) ，同时减少了大约50%的FLOPs.

Cross Causal Attention的具体计算如下:

其中, , , , , 为因果掩码矩阵, 而注意力计算结果会再经过 [FFN](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=FFN&zhida_source=entity) 做后续处理。

### 2.3.2 Self Causal Attention(其它层)

在Cross Causal Attention之后, 后续层使用Self Causal Attention组成。这些层旨在学习采样Tokens序列内的内部关系, 学习捕获行为token序列自身中的依赖关系和模式。关于Self Causal Attention这个应该很熟悉了, 就不介绍了。同样的, 后面也会有个FFN，这有助于进一步处理注意力机制学习到的信息。

### 2.4 训练和部署优化

论文这里也提到了一些训练和部署优化的措施。

### 2.4.1 全同步训练框架

- **统一参数存储&同步更新:** 将模型dense parameters和sparse parameters统一存储在GPU上, 并且所有GPU上的参数更新都是同步进行的, 避免了传统方法中需要外部参数服务器的开销。这种设计减少了通信延迟，提高了训练效率和训练的稳定性。
- **层次化存储:** 针对推荐系统中特征分布不均匀的特点，采用了层次化存储sparse embeddings。具体来说, 高频特征存储在 GPU 的HBM显存, 中频特征存储在CPU的 [MEM](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=MEM&zhida_source=entity) 主存中, 低频特征存储在本地固态硬盘(SSD)中。 这种分层存储方式在延迟、吞吐量和容量之间取得了良好的平衡。

### 2.4.2 混合精度训练与重计算

- **混合精度训练:** 在LONGER模型中，作者采用了BF16/FP16的混合精度训练。用户可以在模型自行配置精度，将高精度应用于关键组件，低精度应用于其他部分。
- **重计算:** 计算剃度的两种模式, 前向模式foward-mode和反向模式reverse-mode, 反向模式需要存储前向传播中的所有中间激活值（activations）。这些激活值可能会占用大量内存。为了减少内存占用，作者采用了重计算策略。在前向传播中丢弃部分激活值，在反向传播中重新计算这些值。这种策略通过增加计算量来换取内存节省，从而在有限的GPU内存下训练更大的模型。

### 2.4.3 KV Cache服务

标准的Transformer方式下, LONGER对于每个候选Item, 可能都需要做一遍完整计算, 从下面的左图可以看出, 对于用户行为序列里的, 会存在大量重复计算。

![](https://pic3.zhimg.com/v2-10309fbac64baf299ed061ba8e21b506_1440w.jpg)

因此, 在推理时, 作者这里采用两阶段策略:

- **Stage1: 缓存用户行为序列的键值对** 在推理开始时，计算并缓存用户行为序列的Key和Value。
- **Stage2: 对每个候选Item计算注意力** 对于每个候选Item，计算与候选Item有关的全局Tokens与缓存的用户行为序列之间的注意力。

## 3\. 实验部分

### 3.1 整体效果

与各baselines的离线指标对比

![](https://pic3.zhimg.com/v2-d8bb334c335eefcd3390ece0f6e8f384_1440w.jpg)

### 3.2 消融实验

做了3个方面的消融实验:

- **Token Merge和InnerTrans的影响:** 增加TokenMerge后FLOPs显著下降, 同时AUC有明显提升, 而增加InnerTrans后, FLOPs有一些上涨, 同时 [离线AUC](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=%E7%A6%BB%E7%BA%BFAUC&zhida_source=entity) 指标进一步提升
- **Query Number的影响:** 从效果上是多多益善, 但对应的开销也会更大, 作者这里做了折衷, 取Query Number=100
- **不同Query样本选择策略:** 发现直接取最近的100个Item效果最佳
![](https://pic1.zhimg.com/v2-fc2297ac030f64f28eb54608ef54bcba_1440w.jpg)

### 3.3 Scaling分析

序列长度的扩展性:

![](https://pic4.zhimg.com/v2-32116dd449a10803f6d26f3cabdba645_1440w.jpg)

参数量&计算量的扩展性:

![](https://pic1.zhimg.com/v2-67158e139981dcd12f00cf6fbcc13b36_1440w.jpg)

### 3.4 线上AB测试

- **抖音广告:** 不同内容素材的广告在 [ADSS](https://zhida.zhihu.com/search?content_id=260286422&content_type=Article&match_order=1&q=ADSS&zhida_source=entity) (Advertiser Score)和 ADVV (Advertiser Value)两个核心指标上都有不错的提升
![](https://pic2.zhimg.com/v2-87caccc4abe4e398243bb99637a89005_1440w.jpg)

- **抖音电商:** 不同电商形式下在Order/U(人均订单)和GMV/U (人均GMV)上都有显著提升
![](https://pica.zhimg.com/v2-42a47df4fa45003b739ed0e6fbd0ceda_1440w.jpg)

编辑于 2025-07-14 06:42・广东[Qwen3.8-Max首发尝鲜，个企双版超值优惠低至39元/月起](https://click.aliyun.com/m/20000000945/?cb=https%3A%2F%2Fsugar.zhihu.com%2Fplutus_adreaper_callback%3Fsi%3D0f6ba1ac-26e8-48bf-b5d8-36e3bedb0945%26os%3D3%26zid%3D1629%26zaid%3D3782460%26zcid%3D3799971%26cid%3D3799971%26event%3D__EVENTTYPE__%26value%3D__EVENTVALUE__%26score%3D__EVENTSCORE__%26ts%3D__TIMESTAMP__%26cts%3D__TS__%26mh%3Df53768d4b45e418cfd0189717b65118c%26adv%3D645640%26ocg%3D0%26cp%3D0%26ocs%3D0%26aic%3D0%26atp%3D0%26ct%3D0%26ed%3DGiBNJgVzfCMmUW9XFyEvRA8xBGxJICwkOhh0FlwxKw1aY0gnWzUoISkYdBZcPC1XVnUfO1UvKX1-AycSWDRxAV58CXoKdGlwdxVlVwJnfhYEK1wjXH5jc3sVYVMEdiBFVnwKfwhwbnVoVycODGEtCQ92WyxbbmwheBx_UwdoKB0Jcg4sFSY8dHhDN15XZXhTWWNYJl9-f3cMAGBVRTE-Vw4xZilOMQUlNlV3VQN1f3FOdwt6HXFoZXlhUcd2NPJo6ck%3D&spu=biz%3D0%26ci%3D3799971%26si%3D1a9d3bfc-6a69-468d-b77f-ef46fe9f54c2%26ts%3D1790419495%26zid%3D1629)

[

Qwen3.8-Max 首发尝鲜、上新 deepseek-v4-flash，更多模态和旗舰模型共享额度，个企双版本超值优惠低至 39 元/...

](https://click.aliyun.com/m/20000000945/?cb=https%3A%2F%2Fsugar.zhihu.com%2Fplutus_adreaper_callback%3Fsi%3D0f6ba1ac-26e8-48bf-b5d8-36e3bedb0945%26os%3D3%26zid%3D1629%26zaid%3D3782460%26zcid%3D3799971%26cid%3D3799971%26event%3D__EVENTTYPE__%26value%3D__EVENTVALUE__%26score%3D__EVENTSCORE__%26ts%3D__TIMESTAMP__%26cts%3D__TS__%26mh%3Df53768d4b45e418cfd0189717b65118c%26adv%3D645640%26ocg%3D0%26cp%3D0%26ocs%3D0%26aic%3D0%26atp%3D0%26ct%3D0%26ed%3DGiBNJgVzfCMmUW9XFyEvRA8xBGxJICwkOhh0FlwxKw1aY0gnWzUoISkYdBZcPC1XVnUfO1UvKX1-AycSWDRxAV58CXoKdGlwdxVlVwJnfhYEK1wjXH5jc3sVYVMEdiBFVnwKfwhwbnVoVycODGEtCQ92WyxbbmwheBx_UwdoKB0Jcg4sFSY8dHhDN15XZXhTWWNYJl9-f3cMAGBVRTE-Vw4xZilOMQUlNlV3VQN1f3FOdwt6HXFoZXlhUcd2NPJo6ck%3D&spu=biz%3D0%26ci%3D3799971%26si%3D1a9d3bfc-6a69-468d-b77f-ef46fe9f54c2%26ts%3D1790419495%26zid%3D1629)

赞同 130