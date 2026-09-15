# 智能记事本（本地 Ollama 版）

一个基于本地大模型（Ollama）的命令行智能记事本，支持笔记增删查搜、AI 总结与基于笔记的问答。完全本地运行，不依赖云端 API。

---

## ✨ 功能清单

| 编号 | 功能 | 是否需要 Ollama |
|------|------|----------------|
| 1 | 新增笔记 | ❌ 不需要 |
| 2 | 查看全部笔记 | ❌ 不需要 |
| 3 | 关键词搜索笔记 | ❌ 不需要 |
| 4 | AI 总结全部笔记 | ✅ 需要 |
| 5 | 基于笔记向 AI 提问 | ✅ 需要 |
| 6 | 删除笔记 | ❌ 不需要 |
| 0 | 退出程序 | — |

> 基础增、查、搜、删功能可直接运行；AI 总结与问答模块需要 Ollama 本地大模型环境。

---

## 📋 环境要求

- Python 3.7+
- Ollama（本地大模型运行环境）
- 已通过 Ollama 下载的模型（默认 `qwen2.5:7b-instruct`）

---

## 🚀 安装与运行

### 1. 安装 Python 依赖

```bash
pip install requests
```

### 2. 启动 Ollama 服务

确保 Ollama 已安装并在后台运行（右下角托盘可见 Ollama 图标）。

拉取模型（如果还没下载）：

```bash
ollama pull qwen2.5:7b-instruct
```

### 3. 运行程序

```bash
python notepad_ai.py
```

---

## 📁 项目结构

```
.
├── notepad_ai.py    # 主程序，全部功能代码
├── notes.json       # 笔记数据文件（运行后自动生成）
└── README.md        # 项目说明文档
```

---

## ⚙️ 配置说明

打开 `notepad_ai.py`，顶部配置区域可根据你的环境修改：

```python
OLLAMA_URL = "http://localhost:11434/api/generate"   # Ollama 接口地址
MODEL_NAME = "qwen2.5:7b-instruct"                   # 使用的模型名称
NOTE_FILE = "notes.json"                             # 笔记存储文件名
```

如果你使用的模型不同，修改 `MODEL_NAME` 即可，例如：

```python
MODEL_NAME = "llama3.1:8b"
MODEL_NAME = "deepseek-r1:7b"
```

---

## 🧠 技术要点

### 简易 RAG 实现
将全部笔记拼接为上下文，连同用户问题一起喂给大模型，通过提示词约束模型**只能基于笔记内容回答，不得编造信息**，从而减少大模型幻觉。

### 完全本地运行
- 所有数据存储在本地 `notes.json` 文件
- 大模型推理完全在本机完成
- 不调用任何云端 API，数据隐私安全

### 原生 HTTP 调用
使用 Python 内置 `requests` 库直接调用 Ollama API，没有使用 LangChain 等封装框架，底层逻辑清晰可见。

### JSON 持久化
笔记以 JSON 格式存储，关闭程序后数据不丢失；`ensure_ascii=False` 保证中文正常显示。

---

## 📖 使用示例

```
======== 智能记事本（本地Ollama版）========
1. 新增笔记
2. 查看全部笔记
3. 关键词搜索笔记
4. AI总结全部笔记
5. 基于笔记向AI提问
6. 删除笔记
0. 退出程序

请输入选项：1

请输入笔记内容：明天下午3点小组会议，准备好项目周报
✅ 笔记保存成功！
```

---

## ❓ 常见问题

### Q: 运行时报错 "AI调用失败"？
- 检查 Ollama 是否在后台运行
- 检查模型名称是否与你本地下载的一致（`ollama list` 查看）
- 确认 Ollama 默认端口 11434 未被占用

### Q: 中文显示乱码？
Windows 用户请在 PowerShell 中先执行：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### Q: 笔记数据存在哪里？
同目录下的 `notes.json` 文件，可直接备份或编辑。
