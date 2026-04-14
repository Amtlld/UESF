# 八、设计优势总结

1. **原语清晰**：Split、ValSplit、DomainPartition 三种划分原语各司其职，语义明确，不强行统一
2. **渐进复杂度**：简单实验 ~10 行 YAML，复杂 UDA 也不超过 ~40 行
3. **正交组合**：域划分和域内划分独立配置，可自由组合
4. **接口一致**：DomainSplitter 拆为 DatasetDomainSplitter / DimensionDomainSplitter，各自接口签名统一，分发逻辑由 UDAOrchestrator 封装
5. **可追溯**：UDASplitResult 携带 fold_info，嵌套折叠展平后仍可追溯来源
6. **最小改动**：Runner、Evaluator、BaseTrainer、BaseModel 完全不变，变更集中在 splitter 和 experiment_manager
7. **可扩展**：新的对齐策略、新的划分维度均可通过注册表模式扩展，无需修改已有代码
