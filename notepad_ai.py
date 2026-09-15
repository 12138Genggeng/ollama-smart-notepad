import requests     # 用于发送HTTP请求，调用Ollama本地API
import json         # 处理json读写，笔记持久化存储
import os           # 文件系统操作，用来判断笔记文件是否存在
import time   # ==================== 配置区域，根据你的Ollama环境修改 ====================
OLLAMA_URL = "http://localhost:11434/api/generate"   # Ollama本地生成接口地址
MODEL_NAME = "qwen2.5:7b-instruct"                   # 使用本地已经下载好的模型名称
NOTE_FILE = "notes.json"                             # 笔记保存的本地json文件名


def load_notes():
    """
    函数功能：从本地json文件加载已经保存的笔记
    返回值：笔记列表，没有文件就返回空列表
    """
    # 判断磁盘上是否存在笔记文件
    if os.path.exists(NOTE_FILE):
        try:
            # utf‑8 编码打开，兼容中文
            with open(NOTE_FILE, "r", encoding="utf-8") as f:
                notes = json.load(f)
        except json.JSONDecodeError:
            print("⚠笔记文件损坏，已清空笔记")
            return []
            # 兼容旧笔记：没有create_time字段就填充为未知时间
        for note in notes:
            if "create_time" not in note:
                note["create_time"] = "未知时间"
            return notes
    # 文件不存在，返回空笔记列表
    return []


def save_notes(notes):
    """
    函数功能：把内存中的笔记列表写入本地notes.json持久化保存
    :param notes: 内存里的笔记列表
    """
    with open(NOTE_FILE, "w", encoding="utf-8") as f:
        # ensure_ascii=False：中文不会被转义；indent=2 格式化json，方便阅读
        json.dump(notes, f, ensure_ascii=False, indent=2)


def ai_ask(prompt: str) -> str:
    """
    函数功能：封装对Ollama的API请求，向本地大模型发送提示词，拿到返回结果
    :param prompt: 发给大模型的完整提示词字符串
    :return: 模型输出文本；发生错误返回错误提示文本
    """
    # 组装API请求体，Ollama /api/generate接口规定格式
    payload = {
        "model": MODEL_NAME,    # 指定使用哪个本地模型
        "prompt": prompt,      # 用户输入提示词
        "stream": False         # 关闭流式输出，一次性返回全部结果，方便代码处理
    }
    try:
        # post发送请求，设置60秒超时，笔记长的时候模型会思考更久
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()   # 如果http返回404/500，直接抛出异常，进入except捕获
        data = resp.json()        # 把接口返回字符串转为python字典
        return data["response"]   # 提取模型生成回答内容
    except Exception as e:
        # 捕获所有异常（网络、接口报错、超时）返回错误信息给界面展示
        return f"AI调用失败：{str(e)}"


def add_note(notes):
    """
    函数功能：新增一条笔记
    :param notes: 内存中笔记列表
    """
    # 控制台接收用户输入笔记文本
    content = input("\n请输入笔记内容：")
    note = {
        "id": len(notes) + 1,    # 简单自增id，用当前列表长度+1作为笔记编号
        "content": content,       # 用户写的笔记正文
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    notes.append(note)           # 将新笔记加入内存列表
    save_notes(notes)            # 立刻保存到本地json文件
    print("✅ 笔记保存成功！")


def show_all_notes(notes):
    """
    函数功能：打印输出全部已经存储的笔记
    :param notes: 内存笔记列表
    """
    # 判断笔记列表为空
    if not notes:
        print("\n📒 暂无笔记")
        return
    print("\n===== 全部笔记 =====")
    # 遍历所有笔记打印ID、创建时间和内容
    for note in notes:
        print(f"\nID:{note['id']} | 创建时间：{note['create_time']}")
        print(f"内容：{note['content']}")


def search_note(notes):
    """
    函数功能：根据关键词，在笔记内容中检索匹配笔记
    :param notes: 内存笔记列表
    """
    keyword = input("\n输入搜索关键词：")
    # 列表推导式：筛选内容包含关键词的笔记
    result = [n for n in notes if keyword in n["content"]]
    if not result:
        print("未找到匹配笔记")
        return
    print("\n===== 搜索结果 =====")
    for note in result:
        print(f"\nID:{note['id']} | 创建时间：{note['create_time']}")
        print(note["content"])


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
    # 查找对应ID
    target_index = None
    for idx, note in enumerate(notes):
        if note["id"] == del_id:
            target_index = idx
            break
    if target_index is None:
        print("❌ 没有找到这个ID的笔记")
        return
    # 删除笔记
    notes.pop(target_index)
    # 删除后重新排序ID，避免ID断层
    for new_idx, note in enumerate(notes):
        note["id"] = new_idx + 1
    save_notes(notes)
    print("✅ 笔记删除成功！")


def ai_summary_notes(notes):
    """
    函数功能：AI读取全部笔记，自动提炼总结要点（简易RAG）
    :param notes: 内存笔记列表
    """
    # 没有笔记直接返回
    if not notes:
        print("暂无笔记可以总结！")
        return
    # 将全部笔记拼接成一段文本，作为上下文喂给大模型
    all_text = "\n".join([f"笔记{n['id']} [{n['create_time']}]: {n['content']}" for n in notes])
    # 构造提示词，告诉模型任务：对笔记做精简总结
    prompt = f"下面是我的所有笔记，请精简总结核心要点：\n{all_text}"
    print("\n🤖 AI正在总结，请等待...")
    res = ai_ask(prompt)   # 调用本地大模型
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
    # 拼接全部笔记作为参考上下文（带上时间戳一起送入AI）
    all_text = "\n".join([f"笔记{n['id']} [{n['create_time']}]: {n['content']}" for n in notes])
    question = input("\n输入你针对笔记想问AI的问题：")
    # 提示词约束：只能参考笔记内容回答，不能自己编造信息，RAG核心prompt
    prompt = f"""你只能使用下面【笔记】里提供的信息回答问题。如果笔记里面没有相关信息，直接回答：[笔记中没有找到相关内容，无法回答]，严禁编造内容.
     \n【笔记】{all_text}\n用户问题：{question}"""
    print("\n🤖 AI思考中...")
    res = ai_ask(prompt)
    print("\n==== AI回答 ====")
    print(res)


def menu():
    """
    主菜单函数，控制台循环交互入口
    """
    notes = load_notes()   # 程序启动时加载磁盘保存好的笔记
    # while True 循环，直到用户选择0退出
    while True:
        print("\n======== 智能记事本（本地Ollama版）========")
        print("1. 新增笔记")
        print("2. 查看全部笔记")
        print("3. 关键词搜索笔记")
        print("4. AI总结全部笔记")
        print("5. 基于笔记向AI提问")
        print("6. 删除笔记")
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
            delete_note(notes)
        elif choice == "0":
            print("👋 程序退出，笔记已保存")
            break   # break跳出while循环，结束程序
        else:
            print("❌ 输入错误，请重新选择！")


# python程序入口，只有直接运行本文件才执行menu()，import导入不会执行
if __name__ == "__main__":
    menu()
