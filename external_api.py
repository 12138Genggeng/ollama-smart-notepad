"""
外部笔记平台对接的预留架构（Notion / Obsidian）。
当前不实现真实 HTTP 调用，只固定接口形状，方便以后补 token、数据库 ID、Vault 路径。

接入方式：笔记写入成功后调用 notify_external_sync(note)，
各 Adapter 默认 no-op，打开 enabled=True 并填配置后再实现 push/pull。
"""
from abc import ABC, abstractmethod


class NoteSyncAdapter(ABC):
    """所有外部平台共用的笔记同步接口。"""

    name = "base"

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    @abstractmethod
    def push_note(self, note: dict) -> None:
        """把本地一条笔记推到外部平台。"""

    @abstractmethod
    def pull_notes(self) -> list:
        """从外部平台拉取笔记列表。"""

    def try_push(self, note: dict) -> None:
        if not self.enabled:
            return
        self.push_note(note)


class NotionAdapter(NoteSyncAdapter):
    """
    Notion 预留：
    - api_token: Notion Internal Integration Token
    - database_id: 目标 Database
    正式实现时走 https://api.notion.com/v1/pages
    """

    name = "notion"

    def __init__(self, api_token: str = "", database_id: str = "", enabled: bool = False):
        super().__init__(enabled=enabled)
        self.api_token = api_token
        self.database_id = database_id

    def push_note(self, note: dict) -> None:
        # TODO: POST /v1/pages，把 note["content"] / ocr_repaired 写成 Notion 块
        raise NotImplementedError("Notion API 对接预留，尚未实现")

    def pull_notes(self) -> list:
        # TODO: POST /v1/databases/{id}/query
        raise NotImplementedError("Notion API 对接预留，尚未实现")


class ObsidianAdapter(NoteSyncAdapter):
    """
    Obsidian 预留：
    - vault_path: 本地 Vault 目录，或 Local REST API 地址（默认 27123）
    - api_key: Obsidian Local REST API 插件的密钥
    正式实现时可写 .md 文件，或调用 Local REST API。
    """

    name = "obsidian"

    def __init__(self, vault_path: str = "", api_key: str = "", enabled: bool = False):
        super().__init__(enabled=enabled)
        self.vault_path = vault_path
        self.api_key = api_key

    def push_note(self, note: dict) -> None:
        # TODO: 写入 vault_path / 调用 Local REST API PUT /vault/{filename}
        raise NotImplementedError("Obsidian API 对接预留，尚未实现")

    def pull_notes(self) -> list:
        # TODO: 扫描 Vault 下 .md，或 GET /vault/
        raise NotImplementedError("Obsidian API 对接预留，尚未实现")


def default_adapters():
    """主程序挂钩用。默认全部关闭，不会发网络请求。"""
    return [
        NotionAdapter(enabled=False),
        ObsidianAdapter(enabled=False),
    ]


def notify_external_sync(note: dict) -> None:
    """笔记入库后的扩展点：以后打开某个 adapter.enabled 即可同步。"""
    for adapter in default_adapters():
        if not adapter.enabled:
            continue
        try:
            adapter.try_push(note)
        except NotImplementedError:
            print(f"ℹ️ 外部同步 [{adapter.name}] 接口已预留，尚未实现，已跳过")
        except Exception as e:
            print(f"⚠️ 外部同步 [{adapter.name}] 失败（不影响本地入库）：{e}")
