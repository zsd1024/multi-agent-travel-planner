"""
Preference Agent —— 偏好收集 Agent（带长期记忆版）。

职责:
  1. 读取记忆：从 ChromaDB 检索该用户的历史偏好
  2. 收集偏好：解析用户当前输入（预算/风格/时间/禁忌）
  3. 注入上下文：将历史偏好注入 LLM System Prompt，影响后续规划
  4. 写入记忆：将用户本次的主观偏好存入 ChromaDB

面试考点:
  - RAG (Retrieval-Augmented Generation): 先检索历史偏好，再注入 LLM
  - 长期记忆 vs 短期记忆: ChromaDB 持久化 vs state 对象生命周期
  - 偏好提取: 识别主观偏好关键词并结构化存储
  - 记忆去重: 相同偏好不重复写入（MemoryManager 内部去重）
"""

from __future__ import annotations

from loguru import logger

from models.schemas import PlanningState, TravelPlanState, UserPreferences

from .base_agent import BaseAgent
from .memory_manager import memory_manager

# ── 偏好关键词库: 用于从 notes 中识别主观偏好 ──
# 匹配到任意关键词说明该句是主观偏好，值得持久化
PREFERENCE_KEYWORDS = [
    # 饮食
    "不吃", "吃素", "素食", "清真", "过敏", "忌口", "不喝", "无辣", "不能吃",
    "喜欢吃", "爱吃", "偏好", "口味",
    # 气候
    "怕热", "怕冷", "怕晒", "不喜欢", "讨厌", "避开",
    # 活动偏好
    "喜欢", "爱好", "热爱", "想去", "必须", "一定要", "最好有",
    "不想", "不要", "不去", "不坐", "恐高", "晕车", "晕船",
    # 住宿
    "要住", "不住", "民宿", "五星", "安静", "靠近",
    # 节奏
    "慢节奏", "快节奏", "轻松", "紧凑", "不赶", "多休息",
    # 同行人
    "带小孩", "带老人", "带宠物", "蜜月", "亲子", "无障碍",
]

# LLM System Prompt: 有历史偏好时的增强 prompt
MEMORY_SYSTEM_PROMPT = """\
你是一位贴心的旅行偏好分析师。

以下是该用户历史积累的偏好记忆（来自过去的旅行规划），这些是用户长期稳定的个人喜好和禁忌：

---用户历史偏好---
{memories}
---历史偏好结束---

请在本次规划中严格遵守这些偏好：
1. 如果用户历史中说过"不吃辣"，本次所有餐饮推荐必须避开辣菜
2. 如果用户说过"怕热"，优先推荐凉爽目的地或安排室内活动
3. 如果用户有特定饮食限制，确保在 dietary_restrictions 中标注
4. 如果用户有出行偏好（如"喜欢看展"），在 interests 中补充

请根据用户的历史偏好和本次输入，输出建议补充的 interests 和 dietary_restrictions（JSON 格式）。
如果没有需要特别调整的，返回空列表即可。

返回格式:
{{"extra_interests": ["...", "..."], "extra_dietary_restrictions": ["...", "..."], "notes_for_planning": "给后续 Agent 的一句话提醒"}}
"""


class PreferenceAgent(BaseAgent):
    name = "PreferenceAgent"

    async def execute(self, state: TravelPlanState) -> TravelPlanState:
        if state.preferences is None:
            raise ValueError("用户偏好未提供，请先设置 state.preferences")

        pref = state.preferences
        user_id = "default_user"

        # ━━━ Step 1: 读取记忆 — 从 ChromaDB 检索历史偏好 ━━━
        memory_context = self._retrieve_memory(user_id, pref)

        # ━━━ Step 2: 基于记忆增强偏好 ━━━
        if memory_context:
            logger.info(f"[{self.name}] 🧠 检索到历史偏好, 注入决策上下文")
            pref = await self._enhance_with_memory(pref, memory_context)

        # ━━━ Step 3: 默认兴趣补充 ━━━
        if not pref.interests:
            pref.interests = self._default_interests(pref.travel_style.value)
            logger.info(f"[{self.name}] 自动补充兴趣标签: {pref.interests}")

        # ━━━ Step 4: 写入记忆 — 将本次主观偏好存入 ChromaDB ━━━
        self._save_preferences_from_notes(user_id, pref)

        state.preferences = pref
        state.state = PlanningState.RECOMMENDING_DESTINATIONS
        return state

    # ── 记忆检索 ──────────────────────────────

    @staticmethod
    def _retrieve_memory(user_id: str, pref: UserPreferences) -> list[str]:
        """根据当前上下文构建查询词，检索相关历史偏好。"""
        # 拼接查询上下文: 出发城市 + 风格 + 备注
        query_parts = [
            f"出发城市: {pref.departure_city}",
            f"旅行风格: {pref.travel_style.value}",
        ]
        if pref.notes:
            query_parts.append(f"备注: {pref.notes}")
        if pref.interests:
            query_parts.append(f"兴趣: {', '.join(pref.interests)}")

        query = " ".join(query_parts)
        logger.info(f"[PreferenceAgent] 记忆检索 query: {query[:80]}...")

        memories = memory_manager.retrieve_preferences(
            user_id=user_id,
            query=query,
            top_k=5,
        )
        return memories

    # ── 记忆增强 ──────────────────────────────

    async def _enhance_with_memory(
        self, pref: UserPreferences, memories: list[str],
    ) -> UserPreferences:
        """将历史偏好注入 LLM，让 LLM 输出建议补充的 interests 和 restrictions。

        Mock 模式下直接做规则匹配，不调用 LLM。
        """
        memories_text = "\n".join(f"- {m}" for m in memories)
        logger.info(f"[{self.name}] 历史偏好:\n{memories_text}")

        # ── Mock 模式: 规则匹配，不调用 LLM ──
        if self._llm_provider == "mock":
            return self._rule_based_enhance(pref, memories)

        # ── 真实 LLM 模式: 让 LLM 分析历史偏好 ──
        try:
            import json
            sys_prompt = MEMORY_SYSTEM_PROMPT.format(memories=memories_text)
            user_prompt = f"""当前用户输入:
出发城市: {pref.departure_city}
日期: {pref.start_date} ~ {pref.end_date}
风格: {pref.travel_style.value}
已有兴趣: {pref.interests}
已有饮食限制: {pref.dietary_restrictions}
备注: {pref.notes}

请根据历史偏好和当前输入，返回 JSON。"""

            raw = await self.call_llm(prompt=user_prompt, system_prompt=sys_prompt)
            data = json.loads(raw.strip().strip("`").strip())

            extra_interests = data.get("extra_interests", [])
            extra_diet = data.get("extra_dietary_restrictions", [])
            planning_note = data.get("notes_for_planning", "")

            if extra_interests:
                pref.interests = list(set(pref.interests + extra_interests))
                logger.info(f"[{self.name}] 🧠 记忆补充兴趣: {extra_interests}")
            if extra_diet:
                pref.dietary_restrictions = list(set(pref.dietary_restrictions + extra_diet))
                logger.info(f"[{self.name}] 🧠 记忆补充饮食限制: {extra_diet}")
            if planning_note and planning_note not in pref.notes:
                pref.notes = f"{pref.notes} [历史偏好提醒] {planning_note}".strip()

        except Exception as exc:
            logger.warning(f"[{self.name}] LLM 记忆增强失败: {exc}, 回退规则匹配")
            pref = self._rule_based_enhance(pref, memories)

        return pref

    @staticmethod
    def _rule_based_enhance(pref: UserPreferences, memories: list[str]) -> UserPreferences:
        """Mock/降级模式: 基于关键词规则匹配历史偏好，直接修改 pref。"""
        combined = " ".join(memories).lower()

        # 饮食相关
        diet_rules = {
            "不吃辣": "不吃辣", "无辣": "不吃辣",
            "吃素": "素食", "素食": "素食",
            "清真": "清真", "过敏": "食物过敏",
            "不喝酒": "不饮酒",
        }
        for keyword, restriction in diet_rules.items():
            if keyword in combined and restriction not in pref.dietary_restrictions:
                pref.dietary_restrictions.append(restriction)
                logger.info(f"[PreferenceAgent] 🧠 从记忆补充饮食限制: {restriction}")

        # 兴趣相关
        interest_rules = {
            "看展": "展览", "博物馆": "博物馆", "美术馆": "艺术",
            "摄影": "摄影", "拍照": "摄影",
            "徒步": "徒步", "爬山": "登山",
            "温泉": "温泉", "泡汤": "温泉",
            "购物": "购物", "逛街": "购物",
        }
        for keyword, interest in interest_rules.items():
            if keyword in combined and interest not in pref.interests:
                pref.interests.append(interest)
                logger.info(f"[PreferenceAgent] 🧠 从记忆补充兴趣: {interest}")

        # 气候/环境相关 → 注入 notes
        climate_notes = []
        if "怕热" in combined:
            climate_notes.append("用户怕热，避免高温目的地")
        if "怕冷" in combined:
            climate_notes.append("用户怕冷，避免寒冷目的地")
        if "恐高" in combined:
            climate_notes.append("用户恐高，避免高空活动")

        if climate_notes:
            note_text = " | ".join(climate_notes)
            if note_text not in pref.notes:
                pref.notes = f"{pref.notes} [历史偏好] {note_text}".strip()

        return pref

    # ── 记忆写入 ──────────────────────────────

    @staticmethod
    def _save_preferences_from_notes(user_id: str, pref: UserPreferences) -> None:
        """从用户的 notes 和 dietary_restrictions 中提取主观偏好，存入 ChromaDB。"""
        saved_count = 0

        # 1. 从 notes 中按句拆分，匹配关键词
        if pref.notes:
            # 按常见分隔符拆句
            separators = ["，", ",", "。", "；", ";", "、", "\n", "|"]
            sentences = [pref.notes]
            for sep in separators:
                new_sentences = []
                for s in sentences:
                    new_sentences.extend(s.split(sep))
                sentences = new_sentences

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 2:
                    continue
                # 排除系统注入的标记
                if sentence.startswith("[历史偏好"):
                    continue

                # 检查是否包含偏好关键词
                is_preference = any(kw in sentence for kw in PREFERENCE_KEYWORDS)
                if is_preference:
                    if memory_manager.save_preference(user_id, sentence):
                        saved_count += 1

        # 2. 饮食限制直接存储
        for restriction in pref.dietary_restrictions:
            text = f"饮食限制: {restriction}"
            if memory_manager.save_preference(user_id, text):
                saved_count += 1

        # 3. 无障碍需求直接存储
        for need in pref.accessibility_needs:
            text = f"无障碍需求: {need}"
            if memory_manager.save_preference(user_id, text):
                saved_count += 1

        if saved_count > 0:
            logger.info(f"[PreferenceAgent] 💾 本次新增 {saved_count} 条记忆到 ChromaDB")

    # ── 默认兴趣 ──────────────────────────────

    @staticmethod
    def _default_interests(style: str) -> list[str]:
        mapping = {
            "budget": ["免费景点", "街头美食", "步行游览"],
            "comfort": ["经典景点", "当地美食", "文化体验"],
            "luxury": ["米其林餐厅", "私人导游", "SPA"],
            "adventure": ["徒步", "潜水", "极限运动"],
            "cultural": ["博物馆", "历史遗迹", "传统手工艺"],
            "relaxation": ["海滩", "温泉", "瑜伽"],
        }
        return mapping.get(style, ["经典景点", "美食"])
