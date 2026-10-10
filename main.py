"""
AI 智能记事本（本地 RAG 版）
============================
技术栈：Python + Ollama（qwen2.5 对话 + nomic-embed-text 向量）+ ChromaDB
持久化：
    - notes.json      保存笔记原文 {id, content}
    - ./chroma_db     保存文本分片后的向量；metadata 带 note_id，用于增删改同步

数据流（写入）：
    用户输入 → JSON 原文 → 文本分片 → Ollama Embedding → Chroma 向量库
数据流（问答）：
    用户提问 → Embedding → Chroma 相似度检索 → 拼上下文 → LLM 输出
"""
import json
import os
import requests
import chromadb
from chromadb.config import Settings

# ==================== 按本机 Ollama 环境修改 ====================
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE}/api/embed"   # 新版接口；旧版曾是 /api/embeddings
MODEL_NAME = "qwen2.5:7b-instruct"              # 内存不够可改成 qwen2.5:3b 或 1.5b
EMBED_MODEL = "nomic-embed-text"
NOTE_FILE = "notes.json"
CHROMA_DIR = "./chroma_db"
TOP_K = 3
# 中文按字符切分即可；过长整篇 embed 会超时/报错，过短则语义被切碎
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),  # 关闭遥测
)
collection = chroma_client.get_or_create_collection(name="notes")


def load_notes():
    """从 notes.json 读取笔记列表；文件不存在或损坏时返回空列表。"""
    if not os.path.exists(NOTE_FILE):
        return []
    try:
        with open(NOTE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ notes.json 读取失败，将按空笔记启动：{e}")
        return []


def save_notes(notes):
    """把内存中的笔记写回 JSON。ensure_ascii=False 保证中文可读。"""
    with open(NOTE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def next_note_id(notes):
    """用历史最大 ID + 1，避免「删除后 len+1 撞上旧 ID」导致 Chroma 主键冲突。"""
    if not notes:
        return 1
    return max(int(n["id"]) for n in notes) + 1


def split_text(text: str) -> list:
    """
    按字符滑动窗口分片。
    坑：早期把整篇笔记直接丢进 Embedding，长文本会超时或语义被平均稀释。
    重叠 CHUNK_OVERLAP 是为了避免句子被拦腰切断后检索不到。
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunks.append(text[start:end])
        if end >= length:
            break
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
    return chunks


def get_embedding(text: str) -> list:
    """调用 Ollama /api/embed，返回单条文本的向量。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("不能对空文本做 Embedding")
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Embedding 返回为空：{data}")
    return embeddings[0]


def ai_ask(prompt: str) -> str:
    """调用本地大模型生成回答（非流式）。"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.exceptions.ConnectionError:
        return "AI 调用失败：连不上 Ollama，请确认服务已启动（默认端口 11434）。"
    except Exception as e:
        return f"AI 调用失败：{e}"


def check_ollama():
    """启动时探测 Ollama 是否在线，避免后面一堆 HTTP 超时。"""
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        missing = [m for m in (MODEL_NAME, EMBED_MODEL) if not any(m in n for n in names)]
        if missing:
            print("⚠️ 本地缺少模型：", "、".join(missing))
            print("   请执行：ollama pull " + " 以及 ollama pull ".join(missing))
        else:
            print("✅ Ollama 在线，对话模型与 Embedding 模型已就绪")
    except Exception:
        print("⚠️ 未检测到 Ollama（http://localhost:11434）。笔记增删改仍可用，AI/向量功能会失败。")


def chroma_ids_of_note(note_id: int) -> list:
    """查出某条笔记在向量库中的全部分片 ID。"""
    result = collection.get(where={"note_id": int(note_id)})
    return result.get("ids") or []


def upsert_note_vectors(note_id: int, content: str):
    """
    以 note_id 为粒度同步向量：先删旧分片，再写入新分片。
    这是「改 JSON 却忘了改 Chroma、检索永远命中旧笔记」的修复点。
    """
    old_ids = chroma_ids_of_note(note_id)
    if old_ids:
        collection.delete(ids=old_ids)

    chunks = split_text(content)
    if not chunks:
        return

    ids, docs, embs, metas = [], [], [], []
    for idx, chunk in enumerate(chunks):
        ids.append(f"{note_id}_{idx}")
        docs.append(chunk)
        embs.append(get_embedding(chunk))
        metas.append({"note_id": int(note_id), "chunk_index": idx})
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)


def clear_collection():
    """清空当前 collection 内全部向量。"""
    existing = collection.get().get("ids") or []
    if existing:
        collection.delete(ids=existing)


def rebuild_chroma_from_json(notes):
    """
    以 JSON 为唯一真相源，全量重建向量库。
    排查 JSON/Chroma 不一致时用这个：改笔记后 RAG 仍答旧内容，多半就是两边不同步。
    """
    print("🔄 正在按 notes.json 全量重建向量库...")
    clear_collection()
    for note in notes:
        upsert_note_vectors(int(note["id"]), note["content"])
    print(f"✅ 向量库已重建，共 {len(notes)} 条笔记、{collection.count()} 个分片")


def json_note_ids(notes):
    return {int(n["id"]) for n in notes}


def chroma_note_ids():
    """从分片 metadata 还原出向量库里有哪些笔记 ID。"""
    data = collection.get(include=["metadatas"])
    ids = set()
    for meta in data.get("metadatas") or []:
        if meta and "note_id" in meta:
            ids.add(int(meta["note_id"]))
    return ids


def ensure_chroma_consistent(notes):
    """
    启动时对比 JSON 与 Chroma 的 note_id 集合。
    以前只在 collection.count()==0 时导入，一旦改过笔记但漏更新向量，重启也修不好。
    """
    j_ids = json_note_ids(notes)
    c_ids = chroma_note_ids()
    if j_ids == c_ids:
        if notes:
            print(f"📂 已加载 {len(notes)} 条笔记，向量库与 JSON 一致")
        return
    print(f"⚠️ 数据不一致：JSON={sorted(j_ids)}  Chroma={sorted(c_ids)}，开始同步")
    rebuild_chroma_from_json(notes)


def add_note(notes):
    content = input("\n请输入笔记内容：").strip()
    if not content:
        print("❌ 笔记内容不能为空")
        return
    note_id = next_note_id(notes)
    try:
        upsert_note_vectors(note_id, content)
    except Exception as e:
        print(f"❌ 向量写入失败，笔记未保存：{e}")
        return
    notes.append({"id": note_id, "content": content})
    save_notes(notes)
    print("✅ 笔记保存成功（JSON + 向量库已同步）")


def show_all_notes(notes):
    if not notes:
        print("\n📒 暂无笔记")
        return
    print("\n===== 全部笔记 =====")
    for note in notes:
        print(f"\nID:{note['id']}")
        print(f"内容：{note['content']}")


def search_note(notes):
    """关键词包含匹配（不是语义检索）。语义问答请走菜单 5。"""
    keyword = input("\n输入搜索关键词：")
    result = [n for n in notes if keyword in n["content"]]
    if not result:
        print("未找到匹配笔记")
        return
    print("\n===== 搜索结果 =====")
    for note in result:
        print(f"\nID:{note['id']}")
        print(note["content"])


def find_note(notes, note_id):
    for note in notes:
        if int(note["id"]) == note_id:
            return note
    return None


def edit_note(notes):
    if not notes:
        print("\n📒 暂无笔记，无法修改！")
        return
    try:
        edit_id = int(input("\n输入想要修改笔记的ID："))
    except ValueError:
        print("❌ ID必须输入数字！")
        return
    target = find_note(notes, edit_id)
    if target is None:
        print("❌ 没有找到这个ID的笔记")
        return
    print(f"\n原内容：{target['content']}")
    new_content = input("请输入新的笔记内容：").strip()
    if not new_content:
        print("❌ 新内容不能为空")
        return
    try:
        upsert_note_vectors(edit_id, new_content)
    except Exception as e:
        print(f"❌ 向量更新失败，原文未改动：{e}")
        return
    target["content"] = new_content
    save_notes(notes)
    print("✅ 笔记修改成功（JSON + 向量库已同步）")


def delete_note(notes):
    if not notes:
        print("\n📒 暂无笔记，无法删除！")
        return
    try:
        del_id = int(input("\n输入想要删除笔记的ID："))
    except ValueError:
        print("❌ ID必须输入数字！")
        return
    target = find_note(notes, del_id)
    if target is None:
        print("❌ 没有找到这个ID的笔记")
        return
    # 不再重排 ID：重排会让「JSON 的新 ID」和「Chroma 里还没重建的旧 ID」短暂错位
    old_ids = chroma_ids_of_note(del_id)
    if old_ids:
        collection.delete(ids=old_ids)
    notes.remove(target)
    save_notes(notes)
    print("✅ 笔记删除成功（JSON + 向量库已同步）")


def ai_summary_notes(notes):
    if not notes:
        print("暂无笔记可以总结！")
        return
    all_text = "\n".join([f"笔记{n['id']}: {n['content']}" for n in notes])
    prompt = f"下面是我的所有笔记，请精简总结核心要点：\n{all_text}"
    print("\n🤖 AI正在总结，请等待...")
    print("\n==== AI总结结果 ====")
    print(ai_ask(prompt))


def ai_chat_with_notes(notes):
    """
    RAG：问题 → Embedding → Chroma Top-K 分片 → 作为唯一上下文 → LLM。
    提示词要求「笔记没有就说未提及」，降低幻觉。
    """
    if not notes:
        print("请先添加笔记！")
        return
    if collection.count() == 0:
        print("向量库为空，正在从 JSON 重建...")
        rebuild_chroma_from_json(notes)

    question = input("\n输入你针对笔记想问AI的问题：").strip()
    if not question:
        print("❌ 问题不能为空")
        return
    print("🔍 正在检索相关笔记...")

    try:
        q_emb = get_embedding(question)
    except Exception as e:
        print(f"❌ 问题向量化失败：{e}")
        return

    n_results = min(TOP_K, max(collection.count(), 1))
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    if not docs:
        print("未检索到相关笔记")
        return

    print("\n📚 检索到以下相关分片：")
    lines = []
    for i, doc in enumerate(docs):
        nid = (metas[i] or {}).get("note_id", "?") if i < len(metas) else "?"
        preview = doc if len(doc) <= 40 else doc[:40] + "..."
        print(f"  - [笔记{nid}] {preview}")
        lines.append(f"笔记{nid}: {doc}")

    context = "\n".join(lines)
    prompt = (
        "参考下面笔记内容回答用户问题，不要编造信息；"
        "如果笔记里没有相关内容，就回答「笔记中未提及相关信息」。\n"
        f"【笔记】\n{context}\n\n用户问题：{question}"
    )
    print("\n🤖 AI思考中...")
    print("\n==== AI回答 ====")
    print(ai_ask(prompt))


def menu():
    notes = load_notes()
    check_ollama()
    ensure_chroma_consistent(notes)

    while True:
        print("\n======== 智能记事本（本地Ollama + RAG版）========")
        print("1. 新增笔记")
        print("2. 查看全部笔记")
        print("3. 关键词搜索笔记")
        print("4. AI总结全部笔记")
        print("5. 基于笔记向AI提问（语义检索RAG）")
        print("6. 修改笔记")
        print("7. 删除笔记")
        print("8. 重建向量库（JSON → Chroma 全量同步）")
        print("0. 退出程序")
        choice = input("\n请输入选项：").strip()

        if choice == "1":
            add_note(notes)
        elif choice == "2":
            show_all_notes(notes)
        elif choice == "3":
            search_note(notes)
        elif choice == "4":
            ai_summary_notes(notes)
        elif choice == "5":
            ai_chat_with_notes(notes)
        elif choice == "6":
            edit_note(notes)
        elif choice == "7":
            delete_note(notes)
        elif choice == "8":
            rebuild_chroma_from_json(notes)
        elif choice == "0":
            print("👋 程序退出，笔记已保存")
            break
        else:
            print("❌ 输入错误，请重新选择！")


if __name__ == "__main__":
    menu()
