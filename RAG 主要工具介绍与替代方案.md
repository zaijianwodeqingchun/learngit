# 🛠️ RAG 主要工具介绍与替代方案

---

## 📖 目录

- [一、LangChain 框架及替代方案](#一langchain-框架及替代方案)
- [二、向量数据库（Vector Databases）](#二向量数据库vector-databases)
- [三、Embedding 模型](#三embedding-模型)
- [四、文档加载器（Document Loaders）](#四文档加载器document-loaders)
- [五、文本分割器（Text Splitters）](#五文本分割器text-splitters)
- [六、检索与重排序（Retriever & Rerank）](#六检索与重排序retriever--rerank)

---

## 一、LangChain 框架及替代方案

### 1.1 LangChain 是什么

LangChain 是一个**开发 LLM 应用的开源框架**，提供文档加载、文本分割、向量存储、模型调用、Chain 和 Agent 等标准化组件，将各模块串联成完整工作流。

> 💡 **知识点：Chain 与 Agent 的区别**
> - **Chain（链）**：按**固定顺序**串联多个步骤（如 检索 → 拼接 → 生成），流程可预测、可控；
> - **Agent（智能体）**：让模型**自主决策**下一步调用什么工具、按什么顺序执行，灵活但需要更多 token 且结果不确定。
> RAG 问答通常是 Chain 的典型场景；Agent 适合需要多步推理、调用多种工具的复杂任务。

### 1.2 主要替代框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **LangChain** | 生态最广、集成最多 | 通用 LLM 应用开发 |
| **LlamaIndex** | 专注 RAG 和数据索引 | **RAG 专用项目** |
| **Haystack** | 专注搜索和问答 | 企业级搜索系统 |
| **CrewAI** | 多 Agent 协作 | 复杂多智能体任务 |
| **Semantic Kernel** | 微软出品、云原生 | .NET/云原生生态 |

> **选择建议**：通用 RAG 用 LangChain；纯 RAG 项目可考虑 LlamaIndex；多 Agent 场景用 CrewAI。

---

## 二、向量数据库（Vector Databases）

### 2.1 什么是向量数据库

向量数据库用于存储文本的向量表示（Embedding），并支持高效的相似度搜索，是 RAG 系统的**记忆核心**。

> 💡 **知识点：向量与相似度搜索是怎么工作的？**
> - 每个文档片段被 Embedding 模型转成一个**高维向量**（例如 384 维），语义相近的文本在向量空间中距离也近；
> - 检索时把用户问题也转成向量，然后计算它与库中所有向量的**相似度**（常用**余弦相似度**，即两个向量夹角的余弦值，越接近 1 越相似），取最相似的 top-k 个片段；
> - 向量数据库的"快"来自**近似最近邻（ANN）索引**，如 HNSW、IVF——它们用索引结构把全库扫描变成几十毫秒的查找，代价是结果只是"近似最优"（对 RAG 完全够用）。

### 2.2 主流向量数据库对比

| 特性 | ChromaDB | FAISS | Pinecone | Weaviate | Qdrant | Milvus |
|------|----------|-------|----------|----------|--------|--------|
| **开源** | ✅ | ✅ | ❌ SaaS | ✅ | ✅ | ✅ |
| **部署方式** | 本地/pip | 本地/pip | 托管云 | 自托管/云 | 自托管/云 | Docker/K8s |
| **成本** | 免费 | 免费 | 付费 | 有免费层 | 有免费层 | 开源免费 |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **性能** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **元数据过滤** | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **GPU 加速** | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **分布式** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |

> 💡 **知识点：FAISS 和 ChromaDB 到底选哪个？**
> - **ChromaDB**：开箱即用，一个 `pip install` + 一个目录就能跑起来，还自带元数据过滤，最适合**入门与原型**；
> - **FAISS**：本质是**向量索引库**而非数据库（不带持久化管理、没有元数据过滤），但检索速度极快、支持 GPU，适合**大数据量高性能**场景；
> - 进阶路径通常是：原型用 ChromaDB → 数据量大了换 FAISS/Qdrant → 生产上云用 Pinecone/Milvus。

### 2.3 选择建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| **开发/原型** | ChromaDB | 零配置、Python 集成好 |
| **高性能/大数据** | FAISS | 极速检索、支持 GPU |
| **生产级 SaaS** | Pinecone | 全托管、无需运维 |
| **企业级自托管** | Qdrant / Milvus | 功能全面、可扩展 |

---

## 三、Embedding 模型

### 3.1 什么是 Embedding 模型

Embedding 模型将文本转换为数字向量（数组），让计算机能够理解和比较文本的语义相似度。**本地 Embedding 模型完全在本地运行，不消耗 API 额度**。

> 💡 **知识点：理解 Embedding 的三个关键概念**
> - **向量维度**：每个文本被映射成多少个数字。维度越高信息量越大，但计算和存储成本也越高；
> - **输入长度上限**：Embedding 模型一次只能处理有限长度文本（如 512 token），所以文档必须先切分成小块再向量化——这就是为什么**分割器在向量化之前**；
> - **本地 vs 云端**：本地模型（sentence-transformers）免费、隐私好、速度快，适合大多数场景；云端 API（如 OpenAI embedding）质量上限更高但按量付费，且数据要出本地。

### 3.2 主流 Embedding 模型对比

| 模型 | 参数规模 | 向量维度 | 特点 | 适用场景 |
|------|----------|----------|------|----------|
| **BAAI/bge-small-zh-v1.5** | ~90MB | 384 | 轻量、快速 | **入门首选**，速度敏感 |
| **BAAI/bge-base-en-v1.5** | ~100MB | 768 | 平衡性能 | 通用场景 |
| **BAAI/bge-large-en-v1.5** | 335M | 1024 | 高质量 | 追求精度 |
| **BAAI/bge-m3** | 570M | 1024 | 多语言（100+ 语言） | 多语言场景 |
| **all-MiniLM-L6-v2** | ~80MB | 384 | 极轻量、CPU 友好 | 资源受限环境 |
| **all-mpnet-base-v2** | 110M | 768 | 高质量、中等延迟 | 英文高质量场景 |

> 💡 **注意：**
> - 换 Embedding 模型后，**向量维度会变**（如 384 → 1024），已建好的向量库无法复用，需要**重建索引**。

### 3.3 替代方案对比

| 方案 | 特点 | 适用场景 |
|------|------|----------|
| **sentence-transformers（本地）** | 开源、隐私好、零成本 | **大多数 RAG 场景** |
| **OpenAI Embeddings（云端）** | 质量高、按量付费 | 追求极致质量、无隐私顾虑 |
| **Hugging Face Inference API** | 无需本地部署 | 不想本地跑模型 |

---

## 四、文档加载器（Document Loaders）

### 4.1 什么是文档加载器

文档加载器是 RAG 系统的**数据入口**，负责从各种数据源（PDF、Word、网页、数据库等）读取数据，并将其统一转换为 LangChain 的 `Document` 对象格式。每个 `Document` 对象包含 `page_content`（文本内容）和 `metadata`（来源信息）两部分。

> 💡 **知识点：metadata 有什么用？**
> `metadata` 记录片段的来源（文件名、页码、章节等），它的价值在检索之后才体现：
> - 回答时标注"答案出自《xxx.pdf》第 3 页"实现**溯源**；
> - 检索时按 metadata 做**过滤**（如只看某本书、某年的文档），大幅提升准确率；
> - 写入向量库时，metadata 会随向量一起存储，因此**检索结果天然携带来源信息**。

### 4.2 主要加载器分类

| 类别 | 常用加载器 | 用途 |
|------|-----------|------|
| **纯文本** | `TextLoader` | 加载 `.txt` 文件 |
| **Markdown** | `MarkdownLoader` | 加载 `.md` 文件 |
| **PDF** | `PyPDFLoader`、`UnstructuredPDFLoader` | 加载 PDF 文档 |
| **Word** | `UnstructuredWordDocumentLoader` | 加载 `.docx` 文档 |
| **CSV** | `CSVLoader` | 加载表格数据 |
| **JSON** | `JSONLoader` | 加载 JSON 数据 |
| **网页** | `WebBaseLoader`、`FirecrawlLoader` | 抓取网页内容 |
| **生产力工具** | `NotionDirectoryLoader`、`GoogleDriveLoader` | 加载 Notion、Google Drive 等数据 |

### 4.3 替代方案对比

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **LangChain Loaders** | 支持 80+ 文件类型，生态丰富 | 通用 RAG 开发 |
| **LlamaIndex (LlamaHub)** | 160+ 数据连接器，PDF 解析约 6 秒 | 专注 RAG 的数据框架 |
| **Unstructured.io** | 强 OCR 能力，表格提取准确 | 复杂布局文档、扫描件 |
| **Docling (IBM)** | 开源，自托管，表格准确率 97.9% | 企业级、高隐私要求 |

> 💡 **知识点：复杂 PDF（扫描件/表格）怎么处理？**
> 普通 PDF 加载器只抽取文本层，遇到**扫描件（图片型 PDF）**会拿到空文本，必须用带 **OCR** 的方案（如 Unstructured.io、paddleocr）；
> 遇到**复杂表格**，文本抽取会打乱行列关系，建议用 Docling 这类"文档理解"工具先转成结构化 Markdown/JSON，再进入 RAG 流程。

> **选择建议**：入门开发用 LangChain 自带加载器即可；处理复杂 PDF 推荐 Unstructured.io 或 Docling；纯 RAG 项目可考虑 LlamaIndex。

---

## 五、文本分割器（Text Splitters）

### 5.1 什么是文本分割器

文本分割器负责将长文档切分成适合模型处理的小块（Chunk），解决模型上下文窗口限制问题，并提升检索精度。

> 💡 **知识点：为什么必须切分？三个原因**
> 1. **Embedding 输入有长度上限**：bge-small-zh 一次只能处理几百 token，超长文本会被截断；
> 2. **检索粒度**：整篇文档向量化后，"相似度"是整篇的，无法定位到具体段落；切成小块后能精确命中相关内容；
> 3. **上下文成本**：塞进提示词的片段越多越贵，小块能让 top-k 检索"用最少的 token 带最多的相关上下文"。

### 5.2 主要分割器对比

| 分割器 | 分割策略 | 适用场景 |
|--------|----------|----------|
| **RecursiveCharacterTextSplitter** | 递归按段落/句子切割 | 通用文本，**最常用** |
| **CharacterTextSplitter** | 按固定字符数切割 | 简单场景 |
| **TokenTextSplitter** | 按 Token 数量切割 | 需精确控制 Token 数 |
| **MarkdownTextSplitter** | 按 Markdown 结构切割 | Markdown 文档 |
| **LatexTextSplitter** | 按 LaTeX 结构切割 | 学术论文 |

> 💡 **知识点：chunk_size 和 chunk_overlap 怎么选？**
> - **chunk_size（块大小）**：200 字属于偏小的值。经验上中文文档 **300~800 字**效果较好：太小语义不完整，太大检索精度下降；
> - **chunk_overlap（重叠）**：让相邻块共享一小段文本（如 10%~20%），避免"一句话被从中间切开"导致两边都检索不到完整语义；
> - **调优方法**：先按文档结构（段落/标题）切，再在 200/400/800 几个档位试跑几轮问答，对比回答质量即可，不必追求理论最优。

### 5.3 替代方案对比

| 工具 | 特点 | 性能对比 |
|------|------|----------------------|
| **LangChain Splitters** | 生态成熟，功能全面 | 基准：1x |
| **Chonkie** | 轻量快速，专为 RAG 设计 | Token 分块比 LangChain 快 1.21x |
| **semchunk** | 语义准确，速度比 LangChain 快 90%+ | 语义分块领先 |
| **LlamaIndex NodeParser** | 与 LlamaIndex 生态深度集成 | Token 分块比 LangChain 慢 51.7x |

> **选择建议**：入门直接用 `RecursiveCharacterTextSplitter`；追求极致性能可尝试 Chonkie 或 semchunk。

---

## 六、检索与重排序（Retriever & Rerank）

> 💡 **RAG 效果瓶颈往往不在模型，而在检索。** 这一节介绍把检索质量做上去的三板斧：相似度检索 → 混合检索 → 重排序。

### 6.1 相似度检索（Vector Search）

示例代码中的 `vector_store.as_retriever(search_kwargs={"k": 3})` 就是最基础的向量相似度检索：取与问题最相似的 3 个片段。

- **优点**：实现简单、能理解语义（同义改写也能命中）；
- **缺点**：对**专有名词/精确匹配**（如代码函数名 `rag_run.py`、编号 "A-1024"）不敏感——向量可能把它们"语义化"后检索不到。

### 6.2 混合检索（Hybrid Search）

把 **BM25 关键词检索**与**向量检索**结合，两种结果合并去重：

| 检索方式 | 原理 | 擅长 |
|------|------|------|
| 向量检索 | 语义相似度 | 同义改写、口语化提问 |
| BM25 关键词 | 词频/逆文档频率（TF-IDF 升级版） | 专有名词、精确编号、代码 |

```python
# 伪代码示意：混合检索 = 向量结果 ∪ 关键词结果，再做合并排序
vector_docs = vector_retriever.invoke(question)     # 语义结果
keyword_docs = bm25_retriever.invoke(question)      # 关键词结果
merged = dedupe(vector_docs + keyword_docs)
```

> 💡 相关工具：`rank_bm25`（轻量实现）、`Elasticsearch`（生产级）、LangChain 的 `EnsembleRetriever`（官方混合检索封装）。

### 6.3 重排序（Rerank）

检索出的 top-k 片段**顺序≠相关性顺序**（尤其混合检索合并后），重排序用一个更精确的 **交叉编码器（Cross-Encoder）** 模型对每个候选片段与问题重新打分、排序，只取前 n 个进入提示词：

```
检索 top-20 → Rerank 精排 → 取 top-3 拼进提示词
```

- **推荐模型**：`BAAI/bge-reranker-base`（中文效果好、免费本地运行）；
- **好处**：用很小的成本大幅提升回答准确率，是**性价比最高的 RAG 优化手段**；
- **工具**：LangChain 的 `CrossEncoderReranker`、LlamaIndex 的 `SentenceTransformerRerank`。

> 🎯 **优化路线图（按性价比排序）**
> 1. 调 chunk 大小与 overlap（零成本）；
> 2. 加 metadata 过滤，缩小检索范围；
> 3. 混合检索（向量 + BM25）；
> 4. 引入 Rerank 重排序（本地 bge-reranker，免费）；
> 5. 以上都做了还不行，再考虑换更强的 Embedding 模型或换更大模型生成。

---

## 📊 附：RAG 工具链选型速查表

| 环节 | 入门方案（RAG1 用的） | 进阶方案 |
|------|----------------------|----------|
| 框架 | LangChain | LlamaIndex / Haystack |
| 文档加载 | `TextLoader` + `PyPDFLoader` | Unstructured.io / Docling |
| 文本分割 | `RecursiveCharacterTextSplitter` | Chonkie / semchunk |
| Embedding | `BAAI/bge-small-zh-v1.5` | `bge-m3` / `bge-large-zh-v1.5` |
| 向量库 | ChromaDB | FAISS / Qdrant / Milvus |
| 检索 | 纯向量 top-k | 混合检索 + Rerank |
| 生成模型 | DeepSeek（deepseek-v4-flash） | 同系列更大模型 / 多轮对话 + 记忆 |

