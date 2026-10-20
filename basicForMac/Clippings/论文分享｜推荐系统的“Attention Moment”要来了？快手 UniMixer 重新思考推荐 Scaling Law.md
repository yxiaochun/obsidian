---
title: "论文分享｜推荐系统的“Attention Moment”要来了？快手 UniMixer 重新思考推荐 Scaling Law"
source: "https://mp.weixin.qq.com/s/-vLy8aZsU1S-XVFMIqSqZA?scene=334"
author:
  - "[[卓导]]"
published:
created: 2026-10-20
description: "一个推荐算法工程师的灵魂拷问：模型还能怎么变大？过去几年，大模型领域最重要的共识是什么？"
tags:
  - "clippings"
---
卓导 HiTech实验室 *2026年5月13日 14:00*

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/yLc5rtCwJnMY0fXrPwHPibLkyrO55oh2n4AgxjW7D14SVYiciacnB08fgFmJgtj1eWOMzYIjCvpZ33IPKgyvVWnkzGrPeNIrA2zB08KT5y97Ys/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

一个推荐算法工程师的灵魂拷问：模型还能怎么变大？

过去几年，大模型领域最重要的共识是什么？

不是某一个具体模型，也不是某一种训练技巧，而是一个非常朴素但强大的规律：模型越大，数据越多，算力越强，效果往往越好。

这就是 Scaling Law。

在大语言模型里，这个规律已经被反复验证。GPT、Claude、Gemini、Qwen、LLaMA 等模型的发展，本质上都离不开这条路线：更大的模型、更大的数据、更长的上下文、更强的训练系统。

但问题来了：

## 推荐系统有没有自己的 Scaling Law？

对于搜广推从业者来说，这个问题非常现实。线上推荐系统每天处理海量用户行为、商品特征、广告特征、上下文特征和实时反馈。模型规模确实在变大，特征也越来越复杂，但我们似乎还没有找到一个像 Transformer 之于大语言模型那样的统一基础架构。

语言模型可以一路堆 Transformer，那推荐系统靠什么 scale？

是继续堆 MLP？

是继续做 FM 特征交叉？

是把 Transformer 搬到推荐系统里？

还是用 RankMixer、TokenMixer 这种更轻量的 token mixing 架构？

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/yLc5rtCwJnOCaajWAUqTibxULcfBALcGLicX2gaBqnsrLRbUgZ2dneQYrhMEBbxyq5ShUZqPTN9jC6NNwnv1vaGgxt7gib4GTj9jptypcSWZuI/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

快手这篇论文UniMixer: A Unified Architecture for Scaling Laws in Recommendation Systems，正是在尝试回答这个问题。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/yLc5rtCwJnPez8SGuxtctNDhicNfwqicYBYy9ZnXgap4PkIibj8iaPMxcjBDLYaRMUvbIaMkYKFg4513AelWBUiau1w5ofXOrJHfaSoTEMkDVrlU/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

这篇论文的目标非常明确：把推荐系统里几条看似不同的 scaling 路线统一起来，并提出一个更适合工业推荐系统的高效 backbone。

它统一的对象主要有三类：

第一类是Attention-based 方法，也就是借鉴 Transformer，用 attention 建模不同特征之间的关系。

第二类是TokenMixer-based 方法，例如 RankMixer、TokenMixer-Large，用规则化的 token mixing 替代 attention。

第三类是FM-based 方法，例如 Wukong、Kunlun，延续推荐系统经典的显式特征交叉思想。

UniMixer 的野心就在于：这三类方法表面上完全不同，但本质上都在做一件事情——让不同推荐特征之间发生有效交互。

---

## 为什么推荐系统不能简单复制 Transformer？

Transformer 在语言模型里很成功，因为语言 token 天然处于相对统一的语义空间里。

“我”“喜欢”“篮球”“鞋子”这些词虽然不同，但它们都是语言 token，可以通过同一套 embedding 空间和 attention 机制进行建模。

但推荐系统不一样。

推荐系统的输入非常异构。一个样本里可能同时包含：

用户年龄、性别、地域；

商品 ID、类目、价格、店铺信息；

用户历史点击、收藏、加购、购买序列；

广告创意、素材、落地页信息；

时间、场景、设备、网络、query 等上下文特征；

以及大量人工交叉特征和统计特征。

这些特征来自完全不同的业务语义空间。

用户年龄和商品 ID 是一类东西吗？

用户过去点击的视频和当前推荐商品可以直接算 attention 相似度吗？

地域特征和广告素材 embedding 之间的内积真的有解释意义吗？

这就是推荐系统直接使用 Transformer 的困难所在。

Attention 的核心依赖是 QK 内积。它假设不同 token 经过映射后可以在同一空间里比较相似度。但在推荐系统里，很多 field 的语义天然不对齐，强行用 attention score 衡量它们之间的关系，可能并不稳定。

这也是为什么工业推荐里虽然有很多 attention 结构，但很少能像 NLP Transformer 那样成为绝对统一的 backbone。

推荐系统需要的不是照搬 Transformer，而是需要一种更适合异构特征的交互结构。

---

## Attention、TokenMixer、FM：三条路线，各有优劣

在 UniMixer 之前，推荐系统里已经有几条比较重要的 scaling 路线。

第一条是Attention-based 路线。

它的优点是表达能力强，可以动态建模特征之间的关系。不同 token 之间是否应该交互，由输入本身决定。比如用户最近点击了母婴视频，当前商品是婴儿车，attention 理论上可以学习到这两者之间的强关联。

但它的问题也很明显：计算成本高，而且在异构特征上不一定稳定。推荐系统的特征不是自然语言 token，很多 field 之间没有自然的相似度定义。

第二条是TokenMixer-based 路线。

RankMixer、TokenMixer-Large 这类方法的思路是：既然 attention 又贵又不稳定，那不如直接设计一种规则化的信息混合方式。把不同 token 的表示切分、重排、混合，让信息在特征之间流动。

这种方法的优势是高效、稳定、工程友好，很适合工业场景。

但问题是：规则是人为设计的，不够灵活。哪些特征应该交互，交互强度如何，很大程度上被结构固定住了。模型只能在既定的 mixing pattern 里学习，表达空间受限。

第三条是FM-based 路线。

FM 是推荐系统里的经典结构，它擅长建模二阶特征交叉。Wukong、Kunlun 这类模型也可以看作是在更大规模下延续 FM-style interaction。

FM 的好处是非常符合推荐系统对特征交叉的需求，但它的局限也明显：传统 FM 更偏显式低阶交互，如果推荐系统想继续向更大规模、更复杂结构发展，只靠 FM 可能不够。

所以这三条路线的问题可以概括成一句话：

Attention 太贵，TokenMixer 太死，FM 太低阶。

UniMixer 想做的，就是把三者统一起来，取长补短。

---

## UniMixer 的核心洞察：TokenMixer 不是简单重排，而是一种特征混合矩阵

UniMixer 最关键的出发点，是重新理解 TokenMixer。

![图片](https://mmbiz.qpic.cn/mmbiz_png/yLc5rtCwJnMkOziaZsK4yA0BXwcWkr2wFeiaVBA75xy5jenDk2v26EyVp37gvJHOibrHltZRX68iaAkIWh0mEVQdF1GmeoScuVOFVW3bksjf1eQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

过去我们看 RankMixer 或 TokenMixer，通常会觉得它是一种工程设计：把 token 表示切成若干块，然后按照固定规则重新排列，让不同 token 的信息混合到一起。

但 UniMixer 发现，这件事其实可以从一个更一般的角度理解：

TokenMixer 本质上是在用一个固定的混合矩阵，对所有特征表示进行重新组织。

也就是说，原来的 TokenMixer 虽然看起来是“切分 + 重排”，但本质上是一个固定的信息路由规则。每个 token 的哪一部分信息流向哪里，是事先规定好的。

这个观察很重要。

因为一旦我们把 TokenMixer 理解成一种“混合矩阵”，就可以自然提出一个问题：

既然固定的混合规则有效，那能不能把这个规则变成可学习的？

原来的 TokenMixer 像是固定线路的交换机。

UniMixer 想把它变成一个可学习的路由器。

哪些特征应该交互？

哪些 block 之间应该传递信息？

哪些交互应该加强，哪些应该抑制？

这些不再完全由人工规则决定，而是交给模型从数据中学习。

不过这里也有一个现实问题：如果直接学习一个完整的大矩阵，参数量和计算量会非常夸张，根本不适合工业推荐系统。

所以 UniMixer 没有直接暴力学习大矩阵，而是进一步提出了“局部混合 + 全局混合”的设计。

---

## UniMixing：局部混合负责细节，全局混合负责路由

UniMixing 是 UniMixer 的核心模块。

它可以理解成两层信息混合：

第一层是局部混合。

它负责一个小 block 内部的信息交互。比如某一组特征片段内部，不同维度之间如何重新组合，如何提取更有效的局部表达。

第二层是全局混合。

它负责不同 block 之间的信息交互。比如用户行为相关的 block 是否应该和商品类目相关的 block 交互，用户画像 block 是否应该和广告素材 block 交互，历史兴趣 block 是否应该和当前 item block 交互。

这就形成了一个非常清晰的结构：

局部混合解决“每个局部怎么表达”，全局混合解决“不同局部之间怎么交互”。

这个设计比原始 TokenMixer 更灵活。

原始 TokenMixer 的 mixing pattern 是固定的，模型只能在固定的信息流动路径里学习。而 UniMixing 的 mixing pattern 是可训练的，模型可以根据数据自动学习更合理的交互结构。

同时，它又比 Attention 更适合工业推荐。

因为它不依赖 QK 内积去动态计算所有 token 之间的相似度，不需要在异构特征之间强行定义 attention score。它直接学习一个结构化的混合模式，更像是在学习推荐系统中的“特征交互拓扑”。

从工程角度看，这一点非常重要。

推荐系统里真正有价值的，不一定是让所有特征两两交互，而是找到那些真正有业务意义的交互路径。

比如：

用户最近看了很多露营视频，当前商品是户外折叠椅，这两者应该强交互。

用户所在城市正在降温，当前商品是羽绒服，这两者应该强交互。

用户最近频繁浏览宠物内容，当前商品是猫粮，这两者应该强交互。

但用户设备型号和某些无关商品属性之间，可能就没有必要强交互。

UniMixing 要学习的，就是这种更稀疏、更有效的交互路径。

---

## 为什么 UniMixer 强调稀疏性？

UniMixer 不是简单地让所有特征自由混合，而是给 mixing matrix 加了一些结构约束。

其中最重要的是稀疏性。

直观理解就是：模型不应该让所有特征都和所有特征发生强交互。推荐系统里的特征非常多，如果所有信息都混在一起，模型不仅计算成本高，也容易引入大量噪声。

真正有效的特征交互往往是稀疏的。

一个用户点击篮球鞋视频，可能强相关于运动鞋、篮球装备、运动服饰；

但不一定强相关于厨房用品、办公文具、宠物零食。

一个用户最近搜索“新生儿奶瓶”，可能强相关于母婴用品、纸尿裤、婴儿推车；

但不一定和手机壳、汽车坐垫、男士香水产生强交互。

所以，推荐系统中的特征交互不应该是“越密越好”，而应该是“该交互的强交互，不该交互的尽量弱化”。

UniMixer 通过温度系数控制混合矩阵的稀疏程度。训练前期，温度较高，让模型有更多探索空间；训练后期，温度降低，让模型学到更尖锐、更明确的交互模式。

这和推荐工程里的经验非常一致：

早期要让模型充分探索，后期要让模型收敛到真正有效的特征交互。

论文的消融实验也证明，温度控制对效果影响很大。去掉温度系数后，模型效果明显下降。这说明 UniMixer 的收益不只是来自“多加了一些参数”，而是来自“可学习且稀疏的特征交互结构”。

---

## UniMixer 如何统一 Attention、TokenMixer 和 FM？

UniMixer 最有价值的地方，是它提出了一个统一视角：

推荐系统里的很多模块，本质上都在做 feature mixing。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/yLc5rtCwJnMdsY9PP5ERuHia8SINj0OciaExP4UiaUO6om09vMOg9Bf8gv3T3gA3UxMhlmZh4ZSDvo2r7VVDaf63zxUNcoS3AJr9picJ7uibDzP0/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=4)

Attention 是一种 feature mixing。

它根据输入动态计算不同 token 之间的关系，再把信息聚合起来。

TokenMixer 也是一种 feature mixing。

它不计算 attention score，而是按照固定规则把 token 信息重新排列和混合。

FM 也是一种 feature mixing。

它显式建模不同特征之间的二阶交叉。

这三者过去看起来属于不同路线，但 UniMixer 认为，它们都可以被放到同一个框架下理解：

一个模块负责决定“谁和谁交互”，另一个模块负责决定“交互后的信息怎么变换”。

Attention 的特点是：谁和谁交互，由输入动态决定。

TokenMixer 的特点是：谁和谁交互，由固定规则决定。

FM 的特点是：谁和谁交互，由显式特征交叉决定。

UniMixer 的特点是：谁和谁交互，由模型学习出来，同时受到结构约束，保证高效和稳定。

这个统一视角非常重要。

它让我们不再纠结“推荐系统到底该不该用 Transformer”，而是把问题换成：

推荐系统到底需要什么样的信息混合方式？

这个问题比“用不用 attention”更本质。

因为推荐系统的核心并不是 attention，而是特征交互。只要能高效学习异构特征之间的有效关系，它可以是 attention，可以是 mixer，也可以是 FM-style interaction。

UniMixer 的意义就在于，它把这些方法放到一个共同坐标系里，让我们可以更系统地比较不同 backbone 的表达能力、计算成本和 scaling 效率。

---

## UniMixing-Lite：工业推荐真正关心的参数效率

标准 UniMixing 已经能学习可训练的特征混合模式，但工业推荐系统还会继续追问一个问题：

效果是好了，但成本能不能接受？

推荐系统不是离线 benchmark。线上有延迟，有吞吐，有显存，有服务成本，有更新频率，有训练资源限制。模型不能只是变强，还必须变得划算。

因此论文提出了 UniMixing-Lite。

UniMixing-Lite 主要做了两件事。

第一，对局部混合矩阵进行 basis 复用。

简单理解就是，不再让每个 block 都单独学习一套完整的局部混合规则，而是学习一组基础混合模式。每个 block 根据自己的需要，对这些基础模式进行组合。

这很像推荐系统里的“专家复用”。

不同特征 block 的交互模式不一定完全不同，它们可能共享一些基础规律。比如某些 block 都需要做类似的压缩、重排、融合，只是组合方式不同。

UniMixing-Lite 用 basis composition 把这种共性提取出来，从而减少参数冗余。

第二，对全局混合矩阵做低秩压缩。

也就是说，不直接学习一个完整的大矩阵，而是用更小的矩阵组合来近似它。这样可以显著降低参数量和计算量，同时保留主要的全局交互能力。

这两个设计让 UniMixing-Lite 更适合工业落地。

它不是盲目做大模型，而是在追求更高的 scaling ROI：

用更少的参数，获得更大的效果提升。

论文实验也证明，UniMixing-Lite 的参数效率非常高。它用几十 M 的 dense 参数规模，就超过了上百 M 参数的 RankMixer 和 TokenMixer-Large。

对于工业推荐来说，这比“参数翻倍带来一点点收益”更有意义。

---

## UniMixer Block 里面还有哪些关键设计？

除了 UniMixing 和 UniMixing-Lite，UniMixer block 里还有两个值得关注的组件。

第一个是Pertoken SwiGLU。

普通 FFN 通常对所有 token 使用共享参数。但推荐系统里的 token 高度异构，用户画像 token、商品 token、行为序列 token、上下文 token 的语义完全不同。

所以 UniMixer 使用 per-token 的 SwiGLU，让不同 token 有更强的个性化表达能力。

可以理解为：

不同特征域有不同的加工方式，而不是所有特征都扔进同一个 FFN 里统一处理。

这对推荐系统非常合理。

因为用户年龄、商品类目、历史点击行为、实时上下文，本来就不应该被完全一样地处理。

第二个是SiameseNorm。

深层推荐模型经常遇到训练不稳定的问题。很多模型加深之后，不但效果不提升，甚至还会下降。

RankMixer 就有这个问题。论文实验中，RankMixer 从 2 层加到 4 层后，效果反而下降。

UniMixer 引入 SiameseNorm，是为了让模型在加深时更加稳定。它通过一种双流归一化设计，缓解深层网络中 Pre-Norm 和 Post-Norm 的矛盾，从而让模型更适合 depth scaling。

这也是 UniMixer-Lite 一个很重要的优势：

它不只是能变宽，也能更稳定地变深。

---

## 实验设置：真实快手广告留存场景，规模非常工业

这篇论文的实验背景是快手广告投放场景中的用户留存建模。

数据规模非常大，包含超过 0.7 billion 用户样本，时间跨度一年，特征包括数百种异构特征，例如 numerical features、ID features、cross features 和 sequential features。

任务是预测用户首次激活后，第二天是否回到快手 App。

评价指标包括 AUC、UAUC，同时还统计 dense 参数量、FLOPs 和 MFU 等效率指标。

对比方法覆盖了三类主流 backbone：

Attention-based 方法，包括 Heterogeneous Attention、HiFormer 和 FAT；

TokenMixer-based 方法，包括 RankMixer 和 TokenMixer-Large；

FM-based 方法，包括 Wukong。

实验在 40 张 GPU 的混合分布式训练框架上完成。

这个实验设置的价值在于，它不是一个小规模公开数据集上的 toy benchmark，而是真正面向工业广告推荐场景的大规模实验。

对于搜广推从业者来说，这类实验比单纯公开数据集上的提升更有参考价值。

---

## 离线效果：UniMixer-Lite 用更少参数超过更大模型

论文最核心的实验结论是：

UniMixer-Lite 在更少参数量下，超过了 RankMixer、TokenMixer-Large、Heterogeneous Attention 和 Wukong 等方法。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/yLc5rtCwJnMmRKkbpXjf7hFCnXcL9Umx0MtjgeowJ9s8wHzgQZGT4bxXg40O98ovYQgibFKkEqYSmrlnQ3JNw4cH2dUzdicCNbNFooLCWcF9k/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=5)

几个关键结果非常值得关注。

RankMixer 参数量约 135.5M，AUC 为 0.749329，UAUC 为 0.738938。

TokenMixer-Large 参数量约 103.3M，AUC 为 0.748410，UAUC 为 0.737940。

UniMixer-2-Blocks 参数量约 101.5M，AUC 达到 0.750238，UAUC 达到 0.739983。

更强的是 UniMixer-Lite。

UniMixer-Lite-4-Blocks 只有约 38.2M 参数，但 AUC 达到 0.752327，UAUC 达到 0.742091。

UniMixer-Lite-4-Blocks 约 84.5M 参数时，AUC 进一步达到 0.752718，UAUC 达到 0.742530。

这组结果非常关键。

它说明 UniMixer-Lite 不是靠“模型更大”取胜，而是靠“结构更高效”取胜。

在工业推荐里，这一点非常重要。因为线上模型不是越大越好，而是要看单位成本收益。

如果一个模型用更少参数、更低计算量，取得更好的 AUC 和 UAUC，那它的工程价值远高于一个单纯堆大参数的模型。

---

## Scaling Law 实验：UniMixer-Lite 的增长曲线更好

论文不仅比较了单点效果，还分析了不同模型随着参数量和计算量增长时的性能趋势。

结论是：UniMixer-Lite 的 scaling 曲线最好。

![图片](https://mmbiz.qpic.cn/mmbiz_png/yLc5rtCwJnMgia7wWbhdjGwTp5x5HumY0lwyoBQVq6Xrscqb8kA034ULrn98SgKyLJoH0eXZ2GO4wFUskjkb6ryI0aFAyBdb3cWxThjCmwDw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=6)

简单说就是，当继续增加参数量或 FLOPs 时，UniMixer-Lite 获得的性能收益比 RankMixer 和标准 UniMixer 更高。

这说明 UniMixer-Lite 更值得继续 scale。

这件事对于工业系统决策非常重要。

假设你有更多 GPU、更多训练数据、更大模型预算，你当然希望选择一个“继续变大还能持续涨效果”的结构，而不是一个很快达到收益瓶颈的结构。

RankMixer 虽然已经很强，但它的 scaling 效率不如 UniMixer-Lite。

UniMixer-Lite 的优势在于，它既有可学习的特征混合能力，又通过 basis 和低秩结构控制了参数冗余，因此在模型变大时，新增参数更容易转化为有效收益。

这也是论文标题里强调 Scaling Laws 的原因。

它不是只想证明 UniMixer 当前效果好，而是想证明 UniMixer-Lite 是一条更值得继续扩展的推荐 backbone 路线。

---

## 消融实验：真正有效的是稀疏、可学习、稳定的特征交互

论文的消融实验非常有信息量。

当去掉温度控制时，模型效果明显下降。

这说明稀疏化的 mixing pattern 很重要。模型不能让所有特征都混在一起，而是要学会选择真正重要的交互。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/yLc5rtCwJnM6JEbCfYpS89p5rIzUdPpvGR3iaARyzTJpMf7XREoRHPtdcNadOnM7XKUtM0vcRYJOhG7LqL5Syic0lIhv2kpfmSwJZzSh4ibALQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=7)

当去掉对称约束时，效果也会下降。

这说明混合矩阵不能完全自由学习，适当的结构约束可以提升稳定性。

当去掉 block-specific local mixing 时，效果下降。

这说明不同 block 需要不同的局部交互方式，不能所有 block 共用完全一样的处理逻辑。

当去掉 warm-up 训练策略时，效果也下降。

这说明低温稀疏虽然最终有效，但训练初期不能太激进。模型需要先充分学习，再逐步变得稀疏和确定。

这些结果说明 UniMixer 的成功不是因为简单“加了几个矩阵”，而是来自一个完整设计：

可学习的 mixing pattern、稀疏约束、局部个性化、稳定训练策略。

这几个因素共同让模型既有表达能力，又不会失控。

---

## 更有意思的结论：深度 scaling 比宽度 scaling 更有效

论文还分析了 UniMixer-Lite 的 basis 数量、低秩 rank 和网络深度。

一个重要结论是：增加 basis 数量比单纯增加 low-rank rank 更划算。

这说明局部交互模式的多样性非常重要。也就是说，不同特征 block 内部到底如何混合，对推荐效果影响很大。

另一个更重要的结论是：UniMixer-Lite 更适合沿深度方向扩展。

RankMixer 从 2 层增加到 4 层后，效果反而下降。

但 UniMixer-Lite 从 2 层增加到 4 层，效果明显提升；继续增加到 8 层后，效果仍有提升，只是增幅开始变小。

这说明 UniMixer-Lite 具备更好的 depth scaling 能力。

过去很多推荐模型不太敢堆太深。原因很简单：深层模型训练不稳定，收益不明显，还容易线上成本上升。

但 UniMixer-Lite 的实验说明，只要 mixing 结构和归一化设计合理，推荐模型也可能像语言模型一样，从更深的结构中持续获益。

这对未来 ranking backbone 的设计非常重要。

---

## 在线 A/B：CAD D1-D30 平均提升超过 15%

论文最后给出了线上 A/B 结果。

UniMixer 和 UniMixer-Lite 已经部署在快手多个广告投放场景中。在线指标使用 CAD，也就是 Cumulative Active Days，统计用户在 30 天窗口内的累计活跃天数，不包括安装当天。

论文报告称，在多个场景中，CAD D1-D30 平均提升超过 15%。

这个提升非常亮眼。

当然，从学术严谨性角度看，论文没有展开每个场景的实验周期、流量规模、置信区间和显著性检验，因此我们不能过度解读。

但从工业推荐角度看，一个模型能在真实广告投放场景上线，并且在用户留存指标上获得明显提升，已经说明它不是纯离线 trick，而是具备真实部署价值的 backbone。

---

## 深度思考一：推荐系统的核心不是 Attention，而是 Feature Mixing

这篇论文最值得反复思考的一点是：

推荐系统不一定需要复制 Transformer，但一定需要更强的特征混合能力。

过去几年，很多推荐模型都在尝试借鉴 NLP。

Transformer、self-attention、multi-head attention、MoE、generative recommendation、LLM4Rec，这些概念不断进入推荐系统。

但推荐系统和 NLP 的数据形态差异太大。

NLP 的基本单位是 token，推荐系统的基本单位是 field。

NLP 的 token 大体处于统一语义空间，推荐系统的 field 高度异构。

NLP 主要处理序列结构，推荐系统同时处理用户、商品、场景、行为、上下文、多任务目标。

所以推荐系统不能简单说“Transformer 成功了，我们也用 Transformer”。

更本质的问题应该是：

如何让异构特征之间高效、稳定、可扩展地发生交互？

Attention 是一种答案。

TokenMixer 是一种答案。

FM 是一种答案。

UniMixer 试图给出一个更统一的答案：用可学习、可约束、可压缩的 feature mixing 结构，替代固定规则或高成本 attention。

这可能是推荐系统下一阶段 backbone 演进的重要方向。

---

## 深度思考二：推荐模型的未来不只是变大，而是更懂得“怎么交互”

很多时候，我们讨论推荐系统 scaling，会下意识关注模型参数量。

embedding 维度加大一点；

MLP 层数加深一点；

专家数加多一点；

行为序列拉长一点；

attention head 增加一点。

但 UniMixer 提醒我们：

参数量不是关键，参数怎么组织才是关键。

一个结构如果不能有效建模特征交互，参数再多也可能只是冗余。

推荐系统真正需要的是高 ROI 的参数增长。

UniMixer-Lite 的 basis composition 和 low-rank global mixing，本质上就是在回答这个问题：如何让新增参数尽量用于有效交互，而不是浪费在冗余矩阵上。

这对工业推荐非常关键。

线上模型不是论文里的无限资源实验。每一点参数、每一点 FLOPs、每一点延迟，背后都对应真实成本。

所以未来推荐模型的竞争，不只是“谁更大”，而是：

谁能用更少成本，学到更有效的特征交互。

---

## 深度思考三：LLM 语义特征和 UniMixer 这类 backbone 很适合结合

从更大的趋势看，推荐系统正在引入越来越多大模型生成的语义信息。

比如：

用 LLM 理解用户行为背后的兴趣意图；

用多模态模型理解短视频内容；

用大模型生成商品语义标签；

用 LLM 把跨域行为转化成消费意图；

用大模型把用户历史行为压缩成语义 token。

这些语义特征进入推荐系统后，会带来一个新问题：

这些 LLM 生成的语义 token，应该如何和传统推荐特征融合？

如果只是简单 concat 到 MLP 里，可能浪费掉语义信息。

如果直接上 attention，又可能计算成本高、线上不稳定。

UniMixer 这类结构可能正好适合作为融合层。

LLM 负责离线理解和生成语义特征，UniMixer 负责在线高效建模语义特征与用户、商品、行为、上下文之间的交互。

这条路线非常适合工业推荐：

大模型不直接参与线上毫秒级推理；

LLM 生成的 intent token、content token、product semantic tag 被缓存或离线更新；

在线 ranking backbone 使用 UniMixer-Lite 这类结构进行高效融合。

这可能是推荐系统结合大模型的一条务实路径。

---

## 深度思考四：推荐系统可能正在迎来自己的“Attention Moment”

Transformer 对 NLP 的意义，不只是提出了 attention，而是提供了一个可以持续 scale 的基础架构。

在 Transformer 出现之前，NLP 也有很多模型：RNN、LSTM、CNN、Seq2Seq、Attention。它们都有效，但没有一个真正成为大规模统一 backbone。

推荐系统今天也有点类似。

我们有 FM、DeepFM、DIN、DIEN、MMoE、PLE、DCN、AutoInt、DSSM、HSTU、RankMixer、Wukong、TokenMixer 等大量结构。

但这些结构之间仍然比较割裂。

UniMixer 的意义就在于，它试图把这些路线放在一个统一视角下：大家本质上都在做 feature mixing，只是 mixing 的方式不同。

这可能是推荐系统走向统一 scaling backbone 的重要一步。

当然，UniMixer 不一定就是最终答案。

但它提出了一个非常重要的问题：

推荐系统自己的基础模块，到底应该长什么样？

它不一定是原版 Transformer。

它也不一定是传统 FM。

它可能是一种结合了可学习交互、稀疏路由、局部特征处理、低秩压缩和深度稳定性的混合架构。

UniMixer 正是在这个方向上的一次重要探索。

---

## 结语：推荐系统的下一代 backbone，可能不是 Attention，而是 Mixer

如果用一句话总结这篇论文，我会这样说：

UniMixer 的价值，不是又提出了一个推荐模型，而是把推荐系统中的 Attention、TokenMixer 和 FM 放到了同一个框架里，让我们重新思考推荐模型到底应该如何 scale。

在推荐系统里，真正重要的不是某一个具体模块，而是特征之间的信息如何流动。

谁和谁交互？

交互强度如何？

哪些交互应该保留？

哪些交互应该被抑制？

这种交互结构能不能随着参数和算力增长持续提升？

UniMixer 给出的答案是：用可学习、稀疏、可压缩的 mixing 结构，替代固定规则的 TokenMixer，也避免直接照搬高成本 attention。

对于搜广推从业者来说，这篇论文值得关注，因为它讨论的是一个非常底层的问题：

当推荐系统进入 scaling 时代，我们到底应该用什么结构承载越来越大的模型容量？

过去我们说，大模型的核心是 Transformer。

未来也许我们会说：

推荐系统的核心，不是 Attention，而是 Feature Mixing。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/yLc5rtCwJnMY0fXrPwHPibLkyrO55oh2n4AgxjW7D14SVYiciacnB08fgFmJgtj1eWOMzYIjCvpZ33IPKgyvVWnkzGrPeNIrA2zB08KT5y97Ys/640?wx_fmt=jpeg&from=appmsg&watermark=1#imgIndex=8)