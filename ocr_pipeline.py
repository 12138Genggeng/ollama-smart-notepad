"""
PDF → 高清图片（dpi=300）→ RapidOCR。
只负责抽出 OCR 原始文本，不写 JSON、不写向量库。
"""
import os

import numpy as np

from errors import OcrFailedError, PdfDamagedError

PDF_RENDER_DPI = 300  # 手写扫描件过低会糊，300 是识别精度和内存的折中
_ocr_engine = None


def get_ocr_engine():
    """懒加载 RapidOCR，避免一启动就下载/加载 ONNX。"""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise OcrFailedError(
                "未安装 RapidOCR，请执行：pip install rapidocr-onnxruntime"
            ) from e
        try:
            _ocr_engine = RapidOCR()
        except Exception as e:
            raise OcrFailedError(f"RapidOCR 引擎初始化失败：{e}") from e
    return _ocr_engine


def _open_pdf(pdf_path: str):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"找不到文件：{pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise PdfDamagedError("仅支持 .pdf 文件")
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise PdfDamagedError("未安装 PyMuPDF，请执行：pip install pymupdf") from e
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PdfDamagedError(f"PDF 损坏或无法打开：{e}") from e
    if doc.is_encrypted:
        doc.close()
        raise PdfDamagedError("PDF 已加密，无法解析")
    if doc.page_count == 0:
        doc.close()
        raise PdfDamagedError("PDF 没有任何页面")
    return doc


def pdf_page_to_image(page, dpi: int = PDF_RENDER_DPI) -> np.ndarray:
    """
    把单页渲染成 ndarray。
    zoom = dpi / 72，因为 PDF 用户空间默认 72dpi。
    """
    import fitz

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # RapidOCR / OpenCV 习惯 BGR
    if arr.shape[2] == 3:
        arr = arr[:, :, ::-1].copy()
    return arr


def _parse_ocr_result(raw) -> list:
    """兼容 rapidocr-onnxruntime 的 (result, elapse) 以及部分新版对象返回值。"""
    if raw is None:
        return []
    if isinstance(raw, tuple):
        raw = raw[0]
    txts = getattr(raw, "txts", None)
    if txts:
        return [str(t) for t in txts if t]
    if not raw:
        return []
    lines = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            lines.append(item)
            continue
        # 常见结构：[box, text, score]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            text = item[1]
            if text:
                lines.append(str(text))
    return lines


def ocr_image(image: np.ndarray) -> str:
    """对一页图片做 OCR，失败抛 OcrFailedError。"""
    try:
        engine = get_ocr_engine()
        raw = engine(image)
    except OcrFailedError:
        raise
    except Exception as e:
        raise OcrFailedError(f"OCR 识别失败：{e}") from e
    lines = _parse_ocr_result(raw)
    return "\n".join(lines).strip()


def extract_ocr_text_from_pdf(pdf_path: str, dpi: int = PDF_RENDER_DPI) -> str:
    """
    整份 PDF：逐页转 300dpi 图 → OCR → 按页拼接。
    返回 OCR 原始文本（尚未经大模型修复）。
    """
    doc = _open_pdf(pdf_path)
    page_texts = []
    try:
        for i, page in enumerate(doc, start=1):
            print(f"  📄 渲染并识别第 {i}/{doc.page_count} 页（dpi={dpi}）...")
            try:
                image = pdf_page_to_image(page, dpi=dpi)
                text = ocr_image(image)
            except OcrFailedError:
                raise
            except Exception as e:
                raise OcrFailedError(f"第 {i} 页 OCR 失败：{e}") from e
            if text:
                page_texts.append(f"--- 第{i}页 ---\n{text}")
    finally:
        doc.close()

    merged = "\n\n".join(page_texts).strip()
    if not merged:
        raise OcrFailedError("OCR 未识别到任何文字，请检查扫描件是否过暗或过糊")
    return merged
