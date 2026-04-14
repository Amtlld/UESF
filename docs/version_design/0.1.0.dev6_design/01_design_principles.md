# 一、设计原则

1. **三种划分原语，各司其职** — 明确区分三种划分操作：**Split**（train/val/test 三分）、**ValSplit**（train/val 二分）、**DomainPartition**（source/target 二分），每种原语有独立的配置结构和语义
2. **顶层模式分发** — `mode: regular | uda` 一级区分，避免配置散落
3. **关注点分离** — 域划分（domain partition）与域内数据划分（inner split）正交组合
4. **自动接线** — 框架自动构建 dataloader 通道映射，用户无需手动配置通道绑定
5. **最小惊讶** — 默认值合理，简单场景 YAML 极简，复杂场景渐进展开
