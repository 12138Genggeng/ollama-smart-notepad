"""
用本地 Ollama 修复 OCR 手写识别结果。
只产出「AI 修复后的文本」，入库仍走 main.upsert_note_vectors。
"""
from errors import OllamaTimeoutError

# 按需求内置的修复提示词；不确定处要求模型打【?】
OCR_REPAIR_PROMPT = """下面是OCR识别手写笔记得到的文字片段，里面存在识别错误、缺字，请结合上下文推测还原笔记原本想要表达的内容，保留笔记原意，不要编造完全无关内容；同时用【?】标记识别不确定的地方。
OCR原始文本：
{ocr_text}
"""

# 过长 OCR 分片修复，避免 7B 上下文撑爆或长时间无响应
REPAIR_CHUNK_SIZE = 1800


def build_repair_prompt(ocr_text: str) -> str:
    return OCR_REPAIR_PROMPT.format(ocr_text=ocr_text)


def _split_for_repair(text: str) -> list:
    text = (text or "").strip()
    if len(text) <= REPAIR_CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + REPAIR_CHUNK_SIZE])
        start += REPAIR_CHUNK_SIZE
    return chunks


def repair_ocr_text(ocr_text: str, llm_call) -> str:
    """
    llm_call(prompt: str) -> str，由 main.ai_ask_or_raise 注入，避免循环导入。
    超时会冒泡为 OllamaTimeoutError。
    """
    ocr_text = (ocr_text or "").strip()
    if not ocr_text:
        raise ValueError("OCR 原始文本为空，无法修复")

    repaired_parts = []
    parts = _split_for_repair(ocr_text)
    for i, part in enumerate(parts, start=1):
        if len(parts) > 1:
            print(f"  🔧 正在修复 OCR 文本分段 {i}/{len(parts)}...")
        prompt = build_repair_prompt(part)
        try:
            out = llm_call(prompt)
        except OllamaTimeoutError:
            raise
        except Exception as e:
            raise OllamaTimeoutError(f"大模型修复 OCR 失败：{e}") from e
        out = (out or "").strip()
        if not out:
            raise OllamaTimeoutError("大模型返回空文本，OCR 修复失败")
        if out.startswith("AI 调用失败"):
            raise OllamaTimeoutError(out)
        repaired_parts.append(out)
    return "\n\n".join(repaired_parts)
