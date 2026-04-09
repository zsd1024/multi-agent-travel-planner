"""
Memory Manager —— 基于 ChromaDB 的长期记忆模块。

职责:
  - 将用户偏好（如"不吃辣"、"怕热"、"喜欢看展"）持久化存储到本地向量数据库
  - 在后续规划中根据当前上下文检索相关历史偏好，注入 Agent 决策流程

设计要点:
  - 持久化目录: ./memory_db （项目根目录下，git 可忽略）
  - Collection 命名: user_preferences
  - 每条记忆包含: 文本内容 + user_id 元数据 + 自动生成的 embedding
  - ChromaDB 内置 all-MiniLM-L6-v2 做 embedding，无需额外模型

面试考点:
  - RAG 中的 R (Retrieval): 先检索相关记忆，再注入 LLM 上下文
  - 向量数据库选型: ChromaDB 轻量本地 vs Pinecone/Weaviate 云端
  - 去重策略: 相同文本不重复写入，通过 document hash 去重
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

# 持久化目录: 项目 python/ 下的 memory_db/
_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "memory_db")
_COLLECTION_NAME = "user_preferences"


class MemoryManager:
    """基于 ChromaDB 的用户偏好长期记忆管理器。"""

    def __init__(self, persist_dir: str = _PERSIST_DIR) -> None:
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure_initialized(self) -> None:
        """懒初始化: 首次调用时才 import chromadb 并创建客户端。"""
        if self._client is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.Client(ChromaSettings(
                persist_directory=self._persist_dir,
                is_persistent=True,
                anonymized_telemetry=False,
            ))
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"description": "用户旅行偏好长期记忆"},
            )
            logger.info(f"[MemoryManager] ChromaDB 初始化成功, 持久化目录: {self._persist_dir}")
            logger.info(f"[MemoryManager] 当前记忆条数: {self._collection.count()}")

        except ImportError:
            logger.warning("[MemoryManager] chromadb 未安装, 记忆功能不可用 (pip install chromadb)")
            self._client = None
            self._collection = None
        except Exception as exc:
            logger.error(f"[MemoryManager] ChromaDB 初始化失败: {exc}")
            self._client = None
            self._collection = None

    @staticmethod
    def _text_hash(text: str) -> str:
        """生成文本的短 hash，用于去重。"""
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()[:12]

    def save_preference(self, user_id: str, text: str) -> bool:
        """将一条用户偏好存入向量数据库。

        Args:
            user_id: 用户标识
            text: 偏好文本，如 "不吃辣" / "怕热不要去热带" / "喜欢看展览"

        Returns:
            是否成功写入 (去重后可能跳过)
        """
        self._ensure_initialized()
        if self._collection is None:
            return False

        text = text.strip()
        if not text:
            return False

        doc_id = f"{user_id}_{self._text_hash(text)}"

        # 去重: 检查是否已存在
        try:
            existing = self._collection.get(ids=[doc_id])
            if existing and existing["ids"]:
                logger.debug(f"[MemoryManager] 记忆已存在, 跳过: {text[:30]}...")
                return False
        except Exception:
            pass  # ID 不存在时某些版本会抛异常，忽略即可

        try:
            self._collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{"user_id": user_id, "type": "preference"}],
            )
            logger.info(f"[MemoryManager] 💾 保存记忆: [{user_id}] {text}")
            return True
        except Exception as exc:
            logger.error(f"[MemoryManager] 写入失败: {exc}")
            return False

    def retrieve_preferences(
        self, user_id: str, query: str, top_k: int = 3,
    ) -> list[str]:
        """根据当前查询检索用户的相关历史偏好。

        Args:
            user_id: 用户标识
            query: 当前上下文查询词（如 "北京出发去成都 舒适风格"）
            top_k: 返回最相关的前 N 条

        Returns:
            相关偏好文本列表
        """
        self._ensure_initialized()
        if self._collection is None:
            return []

        if self._collection.count() == 0:
            return []

        try:
            # 限制 top_k 不超过实际数量
            actual_k = min(top_k, self._collection.count())

            results = self._collection.query(
                query_texts=[query],
                n_results=actual_k,
                where={"user_id": user_id},
            )

            documents = results.get("documents", [[]])[0]
            if documents:
                logger.info(f"[MemoryManager] 🧠 检索到 {len(documents)} 条历史偏好")
                for doc in documents:
                    logger.debug(f"  ↳ {doc}")
            return documents

        except Exception as exc:
            logger.warning(f"[MemoryManager] 检索失败: {exc}")
            return []

    def get_all_preferences(self, user_id: str) -> list[str]:
        """获取用户的所有历史偏好（调试用）。"""
        self._ensure_initialized()
        if self._collection is None:
            return []

        try:
            results = self._collection.get(
                where={"user_id": user_id},
            )
            return results.get("documents", [])
        except Exception:
            return []

    def clear_user_memory(self, user_id: str) -> int:
        """清除指定用户的所有记忆（调试用）。"""
        self._ensure_initialized()
        if self._collection is None:
            return 0

        try:
            results = self._collection.get(where={"user_id": user_id})
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"[MemoryManager] 清除 {len(ids)} 条 [{user_id}] 的记忆")
            return len(ids)
        except Exception as exc:
            logger.error(f"[MemoryManager] 清除失败: {exc}")
            return 0


# ── 全局单例 ──
memory_manager = MemoryManager()
