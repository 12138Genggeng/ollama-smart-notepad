"""
AI智能记事本 - 本地RAG应用
项目简介：基于Python + Ollama + Chroma向量数据库实现轻量化RAG笔记系统
功能：笔记增删改查、关键词检索、AI总结笔记、基于笔记语义检索问答
技术栈：
    Python、requests(Ollama HTTP接口)、ChromaDB向量库、Ollama(Qwen2.5大模型 + nomic-embed-text向量嵌入模型)
持久化：笔记原始文本保存到notes.json，文本向量存入Chroma持久化向量数据库
运行前提：本地Ollama已启动，提前拉取 qwen2.5:7b-instruct 和 nomic-embed-text 模型
"""
import requests     # 用于发送HTTP请求，调用Ollama本地API
import json         # 处理json读写，笔记持久化存储
import os           # 文件系统操作，用来判断笔记文件是否存在
import chromadb     # 向量数据库，存储文本向量，实现语义检索
from chromadb.config import Settings

# ==================== 配置区域，根据你的Ollama环境修改 ====================
OLLAMA_URL = "http://localhost:11434/api/generate"          # Ollama对话生成接口地址
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"       # Ollama文本向量化接口地址
MODEL_NAME = "qwen2.5:7b-instruct"                          # 用于问答、总结的大模型
EMBED_MODEL = "nomic-embed-text"                            # 文本转向量的嵌入模型
NOTE_FILE = "notes.json"                                    # 保存原始笔记的json文件名
CHROMA_DIR = "./chroma_db"                                  # Chroma向量数据库存放文件夹
TOP_K = 3                                                   # RAG检索时，取出相似度最高的3条笔记

# 创建一个持久化的 Chroma 向量库客户端，向量数据保存在`CHROMA_DIR`这个本地文件夹
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)  # 关闭遥感上报

)
# 获取/创建名为notes的向量集合，存放笔记向量
collection = chroma_client.get_or_create_collection(name="notes")


def load_notes():
    """
    从本地json文件加载所有笔记
    return: list 笔记列表，每一条笔记是字典 {id, content}
    """
    if os.path.exists(NOTE_FILE):
        with open(NOTE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # 文件不存在，返回空笔记列表
    return []


def save_notes(notes):
    """
    将笔记列表写入本地json文件持久保存
    :param notes: list 笔记列表
    """
    with open(NOTE_FILE, "w", encoding="utf-8") as f:
        # ensure_ascii=False 支持中文，indent=2格式化json方便阅读
        json.dump(notes, f, ensure_ascii=False, indent=2)


def get_embedding(text: str) -> list:
    """
    调用Ollama的embed接口，把一段文本转换成向量（embedding）
    :param text: 输入的笔记文本/用户提问文本
    :return: list 文本对应的向量数组
    """
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=60
    )
    resp.raise_for_status()  # 请求异常直接抛出错误
    return resp.json()["embeddings"][0]


def ai_ask(prompt: str) -> str:
    """
    调用Ollama大模型对话接口，传入prompt获取大模型返回文本
    :param prompt: 提示词（包含用户问题+参考笔记上下文）
    :return: 大模型回答字符串，出错返回错误信息
    """
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        return f"AI调用失败：{str(e)}"


def sync_old_notes_to_chroma():
    """
    程序启动自动执行：
    如果向量库是空，但是notes.json里面存在旧笔记，自动把历史笔记导入向量数据库
    解决：之前保存过笔记，重启程序后向量库丢失的问题
    """
    notes = load_notes()
    if collection.count() == 0 and notes:
        print(f"📥 检测到 {len(notes)} 条历史笔记，正在导入向量库...")
        ids = [str(n["id"]) for n in notes]
        docs = [n["content"] for n in notes]
        embs = [get_embedding(d) for d in docs]
        collection.add(ids=ids, documents=docs, embeddings=embs)
        print("✅ 历史笔记已导入向量库\n")


def add_note(notes):
    """
    新增笔记：写入JSON保存原始文本，同时向Chroma向量库添加向量
    :param notes: 内存中的笔记列表
    """
    content = input("\n请输入笔记内容：")
    note_id = len(notes) + 1
    notes.append({"id": note_id, "content": content})
    save_notes(notes)
    # 同步添加到向量数据库
    collection.add(
        ids=[str(note_id)],
        documents=[content],
        embeddings=[get_embedding(content)]
    )
    print("✅ 笔记保存成功！")


def show_all_notes(notes):
    """查看内存中全部笔记"""
    if not notes:
        print("\n📒 暂无笔记")
        return
    print("\n===== 全部笔记 =====")
    for note in notes:
        print(f"\nID:{note['id']}")
        print(f"内容：{note['content']}")


def search_note(notes):
    """简单关键词检索：字符串包含匹配（不是语义检索，是普通文本查找）"""
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
    """
    修改笔记：修改JSON内笔记内容，同步更新向量库里面对应的向量
    :param notes: 内存笔记列表
    """
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
    # 更新向量库中这条笔记对应的向量
    collection.update(
        ids=[str(edit_id)],
        documents=[new_content],
        embeddings=[get_embedding(new_content)]
    )
    print("✅ 笔记修改成功！")


def delete_note(notes):
    """
    删除笔记：JSON删除笔记，向量库删除对应向量；删除后重新排序所有笔记ID
    实现方案：删除后直接全量重建向量库，适合笔记数量不多的小型项目，逻辑简单不易出错
    :param notes: 内存笔记列表
    """
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
    # 内存列表删除这条笔记
    notes.pop(target_index)
    # 删除后，重新编排所有笔记ID，保证ID连续
    for new_idx, note in enumerate(notes):
        note["id"] = new_idx + 1
    save_notes(notes)

    # 全量重建向量库：清空向量库，把剩下笔记全部重新向量化存入
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
    """调用大模型，一次性总结全部笔记内容"""
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
    ✅ RAG核心函数：基于笔记做语义检索问答
    流程：用户问题转向量 → 在向量库检索相似度最高笔记 → 把检索出来的笔记作为上下文交给大模型 → 大模型基于参考资料回答
    """
    if not notes:
        print("请先添加笔记！")
        return
    question = input("\n输入你针对笔记想问AI的问题：")
    print("🔍 正在检索相关笔记...")

    # 将用户提问转为向量
    q_emb = get_embedding(question)
    # 在向量库查询TOP_K最相似笔记
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(TOP_K, len(notes))
    )
    retrieved = results["documents"][0]

    print("\n📚 检索到以下相关笔记：")
    for i, doc in enumerate(retrieved):
        print(f"  - {doc[:30]}...")

    # 拼接检索出来的笔记作为上下文
    context = "\n".join([f"笔记{i+1}: {doc}" for i, doc in enumerate(retrieved)])
    # 构造提示词，约束大模型：只能参考笔记内容，禁止编造
    prompt = (
        f"参考下面笔记内容回答用户问题，不要编造信息；"
        f"如果笔记里没有相关内容，就回答'笔记中未提及相关信息'。\n"
        f"【笔记】\n{context}\n\n用户问题：{question}"
    )
    print("\n🤖 AI思考中...")
    print("\n==== AI回答 ====")
    print(ai_ask(prompt))


def menu():
    """程序主菜单入口，循环控制台交互"""
    # 加载本地笔记
    notes = load_notes()
    # 启动自动同步历史笔记到向量库
    sync_old_notes_to_chroma()
    # 循环菜单
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


# 程序入口，运行菜单
if __name__ == "__main__":
    menu()
