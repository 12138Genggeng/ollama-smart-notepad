"""业务异常：把 PDF / OCR / 大模型失败从泛化 Exception 里拆出来，方便菜单层给出可读提示。"""


class PdfDamagedError(Exception):
    """PDF 打不开、加密、0 页、文件头损坏。"""


class OcrFailedError(Exception):
    """RapidOCR 引擎异常，或整份 PDF 认不出任何文字。"""


class OllamaTimeoutError(Exception):
    """本地大模型 HTTP 超时（OCR 修复、总结、问答都可能触发）。"""
