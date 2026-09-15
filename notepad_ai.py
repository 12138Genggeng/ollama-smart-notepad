import requests     # 用于发送HTTP请求，调用Ollama本地API
import json         # 处理json读写，笔记持久化存储
import os           # 文件系统操作，用来判断笔记文件是否存在

# ==================== 配置区域，根据你的Ollama环境修改 ====================
OLLAMA_URL = "http://localhost:11434/api/generate"   # Ollama本地生成接口地址
MODEL_NAME = "qwen2.5:7b-instruct"                   # 使用本地已经下载好的模型名称
NOTE_FILE = "notes.json"                             # 笔记保存的本地json文件名


def load_notes():
    """
    函数功能：从本地json文件加载已经保存的笔记
    返回值：笔记列表，没有文件就返回空列表
    """
    if os.path.exists(NOTE_FILE):
        with open(NOTE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_notes(notes):
    """
    函数功能：把内存中的笔记列表写入本地notes.json持久化保存
    :param notes: 内存里的笔记列表
    """
    with open(NOTE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def ai_ask(prompt: str) -> str:
    """
    函数功能：封装对Ollama的API请求，向本地大模型发送提示词，拿到返回结果
    :param prompt: 发给大模型的完整提示词字符串
    :return: 模型输出文本；发生错误返回错误提示文本
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["response"]
    except Exception as e:
        return f"AI调用失败：{str(e)}"


def add_note(notes):
    """
    函数功能：新增一条笔记
    :param notes: 内存中笔记列表
    """
    content = input("\n请输入笔记内容：")
    note = {
        "id": len(notes) + 1,
        "content": content
    }
    notes.append(note)
    save_notes(notes)
    print("✅ 笔记保存成功！")


def show_all_notes(notes):
    """
    函数功能：打印输出全部已经存储的笔记
    :param notes: 内存笔记列表
    """
    if not notes:
        print("\n📒 暂无笔记")
        return
    print("\n===== 全部笔记 =====")
    for note in notes:
        print(f"\nID:{note['id']}")
        print(f"内容：{note['content']}")


def search_note(notes):
    """
    函数功能：根据关键词，在笔记内容中检索匹配笔记
    :param notes: 内存笔记列表
    """
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
    函数功能：根据笔记ID修改笔记内容
    :param notes: 内存里的笔记列表
    """
    if not notes:
        print("\n📒 暂无笔记，无法修改！")
        return
    try:
        edit_id = int(input("\n输入想要修改笔记的ID："))
    except ValueError:
        print("❌ ID必须输入数字！")
        return
    target_note = None
    for note in notes:
        if note["id"] == edit_id:
            target_note = note
            break
    if target_note is None:
        print("❌ 没有找到这个ID的笔记")
        return
    print(f"\n原内容：{target_note['content']}")
    new_content = input("请输入新的笔记内容：")
    target_note["content"] = new_content
    save_notes(notes)
    print("✅ 笔记修改成功！")


def delete_note(notes):
    """
    函数功能：根据笔记ID删除笔记
    :param notes: 内存里的笔记列表
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
    notes.pop(target_index)
    for new_idx, note in enumerate(notes):
        note["id"] = new_idx + 1
    save_notes(notes)
    print("✅ 笔记删除成功！")


def ai_summary_notes(notes):
    """
    函数功能：AI读取全部笔记，自动提炼总结要点（简易RAG）
    :param notes: 内存笔记列表
    """
    if not notes:
        print("暂无笔记可以总结！")
        return
    all_text = "\n".join([f"笔记{n['id']}: {n['content']}" for n in notes])
    prompt = f"下面是我的所有笔记，请精简总结核心要点：\n{all_text}"
    print("\n🤖 AI正在总结，请等待...")
    res = ai_ask(prompt)
    print("\n==== AI总结结果 ====")
    print(res)


def ai_chat_with_notes(notes):
    """
    函数功能：基于已有笔记作为参考资料，向AI提问，只允许使用笔记里面信息回答
    :param notes: 内存笔记列表
    """
    if not notes:
        print("请先添加笔记！")
        return
    all_text = "\n".join([f"笔记{n['id']}: {n['content']}" for n in notes])
    question = input("\n输入你针对笔记想问AI的问题：")
    prompt = f"参考下面笔记内容回答用户问题，不要编造信息：\n【笔记】{all_text}\n用户问题：{question}"
    print("\n🤖 AI思考中...")
    res = ai_ask(prompt)
    print("\n==== AI回答 ====")
    print(res)


def menu():
    """
    主菜单函数，控制台循环交互入口
    """
    notes = load_notes()
    while True:
        print("\n======== 智能记事本（本地Ollama版）========")
        print("1. 新增笔记")
        print("2. 查看全部笔记")
        print("3. 关键词搜索笔记")
        print("4. AI总结全部笔记")
        print("5. 基于笔记向AI提问")
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
