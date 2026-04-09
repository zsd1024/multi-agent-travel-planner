"""
Activity Agent —— 景点推荐 Agent（高德地图 POI 版）。

职责: 推荐景点和体验活动，生成每日行程安排。
在第二并行阶段执行（Flight + Hotel 完成后），与 FoodAgent 同时运行。
依赖 state.hotel_result 中已确定的酒店地理位置，只推荐酒店周边的景点。

架构:
  1. 调用高德地图 POI 搜索获取目的地真实景点结构化数据
  2. 将 POI 数据作为 <amap_context> 喂给 LLM
  3. LLM 严格基于高德数据提取/编排景点（真实名称+真实门票+真实地址）
  4. 容错: 高德 API 失败 → LLM 降级推演（带城市约束警告）→ Mock 兜底

面试考点:
  - LBS 架构: 高德 POI 返回结构化数据（坐标/评分/价格），比 Tavily 网页摘要精准
  - Grounding: POI 作为事实锚点，消除 LLM 地名幻觉
  - extensions=all: 获取 biz_ext 扩展字段（评分、人均消费）
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from loguru import logger

from config.settings import settings
from models.schemas import Activity, ActivitySearchResult, DayPlan, TravelPlanState
from utils.amap_client import format_pois_as_context, search_amap_poi

from .base_agent import BaseAgent


# ━━━━━━━━━━━━━━━━━━ LLM Prompt ━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT_WITH_AMAP = """\
你是一位专业的旅行景点规划师。用户会提供由高德地图 API 返回的目的地真实 POI（兴趣点）数据。

⚠️ 核心规则（违反即为严重错误）：
1. 你的推荐必须 100% 来源于提供的 <amap_context> 数据库。请直接使用里面提供的真实名称、评分和门票价格。
2. 绝不允许捏造地名或生搬硬套其他城市的景点。数据里没有的景点，你不能推荐。
3. 门票价格：如果 <amap_context> 中有「人均」数据，直接使用；如果显示「暂无」，标注 0（免费或价格未知）。
4. 如果数据中缺少详细描述，你可以基于你的常识进行适度补充说明（如游玩时长、出片机位、小贴士），但地名和价格绝不允许捏造！
5. 每个时段（morning/afternoon/evening）安排 1 个活动。

返回格式为严格 JSON 数组：
[
  {
    "name": "景点名称（必须与高德数据一致）",
    "category": "sightseeing 或 experience",
    "location": "高德数据中的地址",
    "duration_hours": 游玩时长(浮点数,你的经验估计),
    "price": 门票价格(数字,来自高德人均或填0),
    "rating": 评分(来自高德数据,无则填8.0),
    "time_slot": "morning / afternoon / evening",
    "photo_spots": ["出片机位1", "出片机位2"],
    "tour_guide_tips": "一段向导风格的贴心贴士"
  }
]

要求：
- 每天生成 3 个活动（上午/下午/晚上各1个）
- 只返回 JSON 数组，不要包含任何其他文字
"""

SYSTEM_PROMPT_FALLBACK = """\
你是一位专业的旅行景点规划师。当前无法调用地图 API，需要你基于已有知识推荐景点。

⚠️ 极其严格的约束（违反即为致命错误）：
1. 你推荐的所有景点必须确实存在于目的地城市「{city}」！
2. 绝对禁止推荐其他城市的景点！例如目的地是南京，就不能出现成都的景点。
3. 景点名称必须是该城市真实存在的知名景点。
4. 门票价格请基于你的知识给出合理估价（国内景点通常 0-200 元）。
5. 如果你对某个城市不熟悉，宁可推荐少一些但保证真实，也不要编造。

返回 JSON 数组格式同上。每天 3 个活动（morning/afternoon/evening）。
只返回 JSON 数组，不要包含任何其他文字。
"""


def _build_activity_prompt(
    city: str, days: list[str], interests: list[str],
    hotel_loc: str, amap_context: str,
    daily_budget: float,
) -> str:
    prompt = f"""请为以下旅行生成每日景点行程：

目的地城市: {city}
酒店位置: {hotel_loc}
旅行天数: {len(days)} 天 ({', '.join(days)})
兴趣标签: {', '.join(interests) if interests else '综合'}
每日活动预算(参考): ¥{daily_budget:.0f}/人
"""
    if amap_context:
        prompt += f"""
以下是高德地图 API 返回的「{city}」真实景点 POI 数据，请严格基于此数据推荐：
<amap_context>
{amap_context}
</amap_context>

请从高德数据中选择合适的景点，安排到每天的行程中。名称和价格必须与数据一致。
"""
    else:
        prompt += f"""
地图 API 未返回数据。请基于你对「{city}」的知识推荐，但必须确保每个景点都是该城市真实存在的！
"""

    prompt += f"\n请为每一天生成 3 个活动 (morning/afternoon/evening)，总共 {len(days)} 天，返回一个 JSON 数组。"
    return prompt


def _parse_activities_json(raw: str, days: list[str]) -> list[DayPlan]:
    """解析 LLM 返回的 JSON 为 DayPlan 列表。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)
    if not isinstance(data, list):
        data = data.get("activities", data.get("day_plans", []))

    day_plans: list[DayPlan] = []
    for i, date_str in enumerate(days):
        day_activities = data[i * 3: (i + 1) * 3] if len(data) >= (i + 1) * 3 else data[i * 3:]
        activities = []
        for a in day_activities:
            activities.append(Activity(
                name=a.get("name", "未知景点"),
                category=a.get("category", "sightseeing"),
                location=a.get("location", ""),
                duration_hours=float(a.get("duration_hours", 2.0)),
                price=float(a.get("price", 0)),
                rating=float(a.get("rating", 8.0)),
                description=f"{date_str} {a.get('time_slot', '')} - {a.get('name', '')}",
                time_slot=a.get("time_slot", "morning"),
                photo_spots=a.get("photo_spots", []),
                tour_guide_tips=a.get("tour_guide_tips", ""),
            ))
        day_cost = sum(act.price for act in activities)
        day_plans.append(DayPlan(date=date_str, activities=activities, day_cost=day_cost))

    return day_plans


# ━━━━━━━━━━━━━━━━━━ Mock 兜底数据 ━━━━━━━━━━━━━━━━━━

FALLBACK_ACTIVITIES = [
    {"name": "城市地标游览", "cat": "sightseeing", "hours": 2.0, "price": 0, "slot": "morning",
     "spots": ["广场正前方（低角度拍摄）", "入口左侧30米（全景位）"],
     "tips": "上午10点前人少，光线好。"},
    {"name": "博物馆/展览馆", "cat": "sightseeing", "hours": 2.5, "price": 50, "slot": "morning",
     "spots": ["主展厅中央楼梯", "天窗走廊（自然光人像）"],
     "tips": "建议跟讲解员，体验完全不同。周一可能闭馆。"},
    {"name": "历史文化街区", "cat": "sightseeing", "hours": 2.0, "price": 0, "slot": "afternoon",
     "spots": ["巷口回望（纵深感强）", "特色门窗拍摄"],
     "tips": "下午3-4点光线斜射进巷子最好看，主街商业化严重，走小巷更有味道。"},
    {"name": "特色体验活动", "cat": "experience", "hours": 2.0, "price": 150, "slot": "afternoon",
     "spots": ["制作过程抓拍", "成品合影"],
     "tips": "选下午场，老师傅更有耐心。"},
    {"name": "日落观景点", "cat": "sightseeing", "hours": 1.5, "price": 30, "slot": "evening",
     "spots": ["观景台左侧（天际线+晚霞）", "栏杆处（夕阳剪影）"],
     "tips": "日落前1小时到占好位置。带外套，日落后降温快。"},
    {"name": "夜间演出/夜景", "cat": "experience", "hours": 2.0, "price": 200, "slot": "evening",
     "spots": ["入场前海报墙", "谢幕时舞台全景"],
     "tips": "选中间偏前的座位体验最佳。"},
]


# ━━━━━━━━━━━━━━━━━━ Agent 本体 ━━━━━━━━━━━━━━━━━━

class ActivityAgent(BaseAgent):
    name = "ActivityAgent"

    async def execute(self, state: TravelPlanState) -> TravelPlanState:
        pref = state.preferences
        dest = state.selected_destination
        if pref is None or dest is None:
            raise ValueError("缺少偏好或目的地信息")

        hotel = state.hotel_result.recommended if state.hotel_result else None
        hotel_loc = hotel.address if hotel else f"{dest.city}市中心"
        city = dest.city

        days = self._get_travel_days(pref.start_date, pref.end_date)
        daily_budget = (pref.budget * 0.20) / max(len(days), 1) / pref.num_travelers

        logger.info(f"[{self.name}] 基于酒店位置 [{hotel_loc}] 搜索「{city}」周边景点")

        # ── Step 1: 高德地图 POI 搜索 ──
        interest_kw = " ".join(pref.interests[:3]) if pref.interests else "必去景点"
        search_keyword = f"{city} {interest_kw}"

        pois = await search_amap_poi(keyword=search_keyword, city=city)
        amap_context = format_pois_as_context(pois, label=f"{city}景点")

        has_amap = bool(amap_context)
        logger.info(f"[{self.name}] 搜索状态: {'有高德 POI 数据 ({len(pois)} 条)' if has_amap else '无数据(降级推演)'}")

        # ── Step 2: LLM 结构化编排 ──
        day_plans = await self._extract_with_llm(
            city, days, pref.interests, hotel_loc, amap_context, daily_budget,
        )

        # 补充人数计算
        total_cost = 0.0
        for plan in day_plans:
            plan.day_cost = sum(a.price for a in plan.activities) * pref.num_travelers
            total_cost += plan.day_cost

        state.activity_result = ActivitySearchResult(
            day_plans=day_plans,
            total_activity_cost=total_cost,
        )
        logger.info(f"[{self.name}] 生成 {len(day_plans)} 天行程, 活动总费用: ¥{total_cost:.0f}")
        return state

    async def _extract_with_llm(
        self, city: str, days: list[str], interests: list[str],
        hotel_loc: str, amap_context: str, daily_budget: float,
    ) -> list[DayPlan]:
        """LLM 编排景点数据，含三级容错。"""
        sys_prompt = SYSTEM_PROMPT_WITH_AMAP if amap_context else SYSTEM_PROMPT_FALLBACK.format(city=city)

        user_prompt = _build_activity_prompt(
            city, days, interests, hotel_loc, amap_context, daily_budget,
        )

        try:
            raw = await self.call_llm(prompt=user_prompt, system_prompt=sys_prompt)

            if settings.LLM_PROVIDER == "mock":
                logger.info(f"[{self.name}] Mock 模式 → 使用兜底数据")
                return self._fallback_plans(city, days, daily_budget, interests)

            day_plans = _parse_activities_json(raw, days)
            if day_plans:
                return day_plans

            logger.warning(f"[{self.name}] LLM 返回空数据, 使用兜底")
            return self._fallback_plans(city, days, daily_budget, interests)

        except json.JSONDecodeError as exc:
            logger.warning(f"[{self.name}] JSON 解析失败: {exc}, 使用兜底")
            return self._fallback_plans(city, days, daily_budget, interests)
        except Exception as exc:
            logger.error(f"[{self.name}] LLM 调用异常: {exc}, 使用兜底")
            return self._fallback_plans(city, days, daily_budget, interests)

    @staticmethod
    def _fallback_plans(
        city: str, days: list[str], daily_budget: float, interests: list[str],
    ) -> list[DayPlan]:
        """硬兜底: 生成通用景点数据，名称带城市前缀防幻觉。"""
        day_plans: list[DayPlan] = []
        for date_str in days:
            activities: list[Activity] = []
            slots_used = []
            for fa in FALLBACK_ACTIVITIES:
                if fa["slot"] in slots_used:
                    continue
                slots_used.append(fa["slot"])
                activities.append(Activity(
                    name=f"{city}{fa['name']}",
                    category=fa["cat"],
                    location=f"{city}市内",
                    duration_hours=fa["hours"],
                    price=float(fa["price"]),
                    rating=round(random.uniform(7.5, 9.0), 1),
                    description=f"{date_str} {fa['slot']} - {city}{fa['name']}",
                    time_slot=fa["slot"],
                    photo_spots=fa["spots"],
                    tour_guide_tips=fa["tips"],
                ))
                if len(slots_used) >= 3:
                    break
            day_plans.append(DayPlan(
                date=date_str,
                activities=activities,
                day_cost=sum(a.price for a in activities),
            ))
        return day_plans

    @staticmethod
    def _get_travel_days(start: str, end: str) -> list[str]:
        try:
            d1 = datetime.strptime(start, "%Y-%m-%d")
            d2 = datetime.strptime(end, "%Y-%m-%d")
            days_count = max((d2 - d1).days, 1)
            return [(d1 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_count)]
        except (ValueError, TypeError):
            return ["2026-01-01", "2026-01-02", "2026-01-03"]
