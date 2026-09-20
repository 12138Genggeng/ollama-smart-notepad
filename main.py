import requests     # 用于发送HTTP请求，调用Ollama本地API
import json         # 处理json读写，笔记持久化存储
import os           # 文件系统操作，用来判断笔记文件是否存在
import chromadb     # 向量数据库
from chromadb.config import Settings

# ==================== 配置区域，根据你的Ollama环境修改 ====================
OLLAMA_URL = "http://localhost:11434/api/generate"          # Ollama对话接口
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"       # Ollama向量接口
MODEL_NAME = "qwen2.5:7b-instruct"                          # 对话模型
EMBED_MODEL = "nomic-embed-text"                            # embedding模型
NOTE_FILE = "notes.json"                                    # 笔记JSON文件
CHROMA_DIR = "./chroma_db"                                  # 向量库目录
TOP_K = 3                                                   # RAG检索取最相关的几条

# 初始化Chroma向量库
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(name="notes")


def load_notes():
    """从本地json加载笔记"""
    if os.path.exists(NOTE_FILE):
        with open(NOTE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_notes(notes):
    """保存笔记到json"""
    with open(NOTE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def get_embedding(text: str) -> list:
    """调用Ollama把文本转成向量"""
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def ai_ask(prompt: str) -> str:
    """调用Ollama对话接口"""
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        return f"AI调用失败：{str(e)}"


def sync_old_notes_to_chroma():
    """启动时：如果向量库空但notes.json有旧笔记，自动导入向量库"""
    notes = load_notes()
    if collection.count() == 0 and notes:
        print(f"📥 检测到 {len(notes)} 条历史笔记，正在导入向量库...")
        ids = [str(n["id"]) for n in notes]
        docs = [n["content"] for n in notes]
        embs = [get_embedding(d) for d in docs]
        collection.add(ids=ids, documents=docs, embeddings=embs)
        print("✅ 历史笔记已导入向量库\n")


def add_note(notes):
    """新增笔记：同时写JSON和向量库"""
    content = input("\n请输入笔记内容：")
    note_id = len(notes) + 1
    notes.append({"id": note_id, "content": content})
    save_notes(notes)
    collection.add(
        ids=[str(note_id)],
        documents=[content],
        embeddings=[get_embedding(content)]
    )
    print("✅ 笔记保存成功！")


def show_all_notes(notes):
    """查看全部笔记"""
    if not notes:
        print("\n📒 暂无笔记")
        return
    print("\n===== 全部笔记 =====")
    for note in notes:
        print(f"\nID:{note['id']}")
        print(f"内容：{note['content']}")


def search_note(notes):
    """关键词搜索"""
    keyword = input("\n输入搜索关键词：")
    result = [n for n in notes if keyword in n["content"]]
    if not result:
        print("未找到匹配笔记")
        return
    print("\n===== 搜索结果 =====")
    for note in result:
        print(f"\nID:{note['id']}")
        print(note["content"])


def edit_note(notes):
    """修改笔记：同步更新向量库"""
    if not notes:
        print("\n📒 暂无笔记，无法修改！")
        return
    try:
        edit_id = int(input("\n输入想要修改笔记的ID："))
    except ValueError:
        print("❌ ID必须输入数字！")
        return
    target = None
    for note in notes:
        if note["id"] == edit_id:
            target = note
            break
    if target is None:
        print("❌ 没有找到这个ID的笔记")
        return
    print(f"\n原内容：{target['content']}")
    new_content = input("请输入新的笔记内容：")
    target["content"] = new_content
    save_notes(notes)
    collection.update(
        ids=[str(edit_id)],
        documents=[new_content],
        embeddings=[get_embedding(new_content)]
    )
    print("✅ 笔记修改成功！")


def delete_note(notes):
    """删除笔记：同步删向量库，并重排ID"""
    if not notes:
        print("\n📒 暂无笔记，无法删除！")
        return
    try:
        del_id = int(input("\n输入想要删除笔记的ID："))
    except ValueError:
        print("❌ ID必须输入数字！")
        return
    target_index = None
    for idx, note in enumerate(notes):
        if note["id"] == del_id:
            target_index = idx
            break
    if target_index is None:
        print("❌ 没有找到这个ID的笔记")
        return
    notes.pop(target_index)
    for new_idx, note in enumerate(notes):
        note["id"] = new_idx + 1
    save_notes(notes)
    # 向量库全量重建（数据量小，简单可靠）
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)
    if notes:
        ids = [str(n["id"]) for n in notes]
        docs = [n["content"] for n in notes]
        embs = [get_embedding(d) for d in docs]
        collection.add(ids=ids, documents=docs, embeddings=embs)
    print("✅ 笔记删除成功！")


def ai_summary_notes(notes):
    """AI总结全部笔记"""
    if not notes:
        print("暂无笔记可以总结！")
        return
    all_text = "\n".join([f"笔记{n['id']}: {n['content']}" for n in notes])
    prompt = f"下面是我的所有笔记，请精简总结核心要点：\n{all_text}"
    print("\n🤖 AI正在总结，请等待...")
    print("\n==== AI总结结果 ====")
    print(ai_ask(prompt))


def ai_chat_with_notes(notes):
    """【真RAG】向量检索 + 大模型回答"""
    if not notes:
        print("请先添加笔记！")
        return
    question = input("\n输入你针对笔记想问AI的问题：")
    print("🔍 正在检索相关笔记...")

    q_emb = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(TOP_K, len(notes))
    )
    retrieved = results["documents"][0]

    print("\n📚 检索到以下相关笔记：")
    for i, doc in enumerate(retrieved):
        print(f"  - {doc[:30]}...")

    context = "\n".join([f"笔记{i+1}: {doc}" for i, doc in enumerate(retrieved)])
    prompt = (
        f"参考下面笔记内容回答用户问题，不要编造信息；"
        f"如果笔记里没有相关内容，就回答'笔记中未提及相关信息'。\n"
        f"【笔记】\n{context}\n\n用户问题：{question}"
    )
    print("\n🤖 AI思考中...")
    print("\n==== AI回答 ====")
    print(ai_ask(prompt))


def menu():
    notes = load_notes()
    sync_old_notes_to_chroma()
    while True:
        print("\n======== 智能记事本（本地Ollama + RAG版）========")
        print("1. 新增笔记")
        print("2. 查看全部笔记")
        print("3. 关键词搜索笔记")
        print("4. AI总结全部笔记")
        print("5. 基于笔记向AI提问（语义检索RAG）")
        print("6. 修改笔记")
        print("7. 删除笔记")
        print("0. 退出程序")
        choice = input("\n请输入选项：")

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
        elif choice == "0":
            print("👋 程序退出，笔记已保存")
            break
        else:
            print("❌ 输入错误，请重新选择！")


if __name__ == "__main__":
    menu()
