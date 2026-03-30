# 09. 测试与质量保证策略 (Testing & QA Strategy)

## 1. 核心理念与技术栈设定

UESF 的所有测试与质量保证工作将以 **`pytest`** 为绝对核心来推行。不需要复杂的外部测试框架，仅仅利用 `pytest` 强大的 **Fixtures (测试夹具)** 和 **参数化 (Parametrization)** 功能，就能实现 90% 以上的框架级可靠性保障。

**测试核心原则：完全隔离与确定性**
由于 UESF 涉及大量的文件读写（SQLite 数据库、实验快照、预处理数据），**绝不能在测试中污染用户真实的全局或本地项目环境**。所有的测试行为必须发生在 `pytest` 提供的临时目录中。同时，由于包含深度学习运算，测试输入与输出必须具备确定性。

---

## 2. 基于 Pytest 的测试金字塔设计

### 2.1 单元测试 (Unit Testing) - 针对纯逻辑层
*   **目标**：隔离验证单个函数、类方法的输入输出。这是 UESF 测试的基石。
*   **如何通过 pytest 实现**：
    *   **异常捕获断言**：大量使用 `pytest.raises(UESFError)` 验证在 `08_exception_and_logging.md` 中定义的定制错误是否被正确触发（例如解析不存在的配置文件时）。
    *   **字典与 YAML/JSON 解析验证**：利用 `pytest.mark.parametrize` 传入不同的畸形或正常的配置字典，验证内置映射管理器能否正确合并或透传。
    *   **无状态组件测试**：例如针对路径拼接工具、哈希生成器等纯逻辑函数进行大量边缘输入测试。

### 2.2 集成测试 (Integration Testing) - 针对组件联动与数据库
*   **目标**：验证跨组件协作（例如 Data Manager 将数据交付给 Preprocessing Pipeline），以及必须与底层系统交互（SQLite、配置文件读写）的行为。
*   **如何通过 pytest 实现**：
    *   **纯净沙盒依赖 (`tmp_path` fixture)**：为涉及文件写入的测试传入 `pytest` 的内置测试夹具 `tmp_path`。强制 Manager 在临时沙盒目录下建库建表并生成快照，测试结束后自动销毁，以保证环境纯净。
    *   **内存数据库隔离**：对于核心 SQLite 的唯一性约束或外键逻辑验证，可以在测试初始化时连接至 `sqlite:///:memory:` (纯内存数据库) 进行超快速无盘 I/O 测试。
    *   **轻量级 Mock (`unittest.mock`)**：如果我们仅测试“实验记录被成功保存回收”，而不关心底层真实的 PyTorch 模型能否收敛，可以 `mock` 掉组件重的计算过程，从而将几小时的训练缩短为几毫秒的验证响应。

### 2.3 端到端测试 (E2E) - 针对 CLI 流水线
*   **目标**：将完整的命令行流串起来，模拟真实用户的黑盒操作并验证结果产物。
*   **如何通过 pytest 实现**：
    *   **终端命令流模拟**：使用终端库（如 Click）自带的测试套件（例：`click.testing.CliRunner`），在一个测试用例中基于纯净的临时目录模拟输入 `uesf init` -> `uesf data preprocess` -> `uesf run`。
    *   **极简直通跑通 (Dummy Fast-Forwarding)**：框架内提供超小型的 Dummy 数据集（如少量 1Hz 的生成正弦信号数组）以及单层全连接的 Dummy 模型。目的是能够在 CI 环境耗时低于 1 分钟跑完 1 个 Epoch，确保主链路健壮不断连。

---

## 3. 针对 UESF 深度学习特性的专项测试点

### 3.1 “固定随机种子” 测试 (Determinism Assertions)
*   **痛点**：预处理管线中的 ICA 或降采样操作、以及 PyTorch 配置初始化的过程若不加干涉，极易引发“第一次执行报错，第二次莫名通过”的偶发/闪烁测试 (Flaky Tests)。
*   **方案**：建立全局的随机数控制 fixture，在所有与模型或算子相关的测试前置注入：
    ```python
    @pytest.fixture(autouse=True)
    def fixed_random_seed():
        # 固定 Python、Numpy、PyTorch 的状态机
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    ```

### 3.2 预处理流的 “Fit on Train” 防泄露断言
*   **规则约束**：确保在 `04_components/06_preprocessing_pipeline.md` 中定义的规则——仅在 Train 分片上 `fit` 状态变换因子，并将此被固定因子应用至 Test / Valid。
*   **测试设计**：注入带极大差异分布的 Fake Train 与 Fake Test Set。预处理结束后提取快照配置，必须断言保存的标准差/均值参数字典严格等同于 Train 的参数，从而从机制级杜绝数据泄露。

### 3.3 轻量化防 OOM 内存阈值测试（进阶考量）
*   **痛点**：大规模受试者级数据采用“惰性分片加载 (Lazy Loading)”，必须防止堆中驻留对象的指针泄露。
*   **测试设计**：针对最容易吃内存的数据转换与批读取接口集成 `tracemalloc` 或 `psutil` 断言，如果在 Mock 出巨大假数据后读取过程引发的内存驻留量增量超越预设阈值（例如 > 50 MB），测试状态阻断。

---

## 4. 推荐在未来叠加的轻量代码质量保障工具

为了配合 `pytest` 更好的发挥威力，对于刚接触系统测试的开发者，引入极简但见效快的辅助组合也是良方：
1. **自动补漏排版 (Ruff)**：现代 Python 生态中最快的 Lint 检查器。它能像拼写检查器一样帮您在保存代码瞬间标红无用的包导入，或修正常见的拼写/缩进不良问题。
2. **可视盲点捕获 (pytest-cov)**：`pytest` 的生态衍生插件。仅需简单的 `--cov` 参数，就能自动扫描整个测试流程到底经过了核心库（如 `Experiment Manager` 或 `Preprocessing Pipeline`）里的哪些函数分支，对于那些由于遗漏始终没测到的函数分支提供极高的数据指导。
