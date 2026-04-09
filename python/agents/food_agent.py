"""
Food Agent —— 美食推荐 Agent（高德地图 POI 版）。

职责: 根据用户偏好和酒店地理位置，推荐目的地城市的真实特色美食与餐厅。
在第二并行阶段执行（Flight + Hotel 完成后），与 ActivityAgent 同时运行。

架构:
  1. 调用高德地图 POI 搜索获取目的地真实餐厅结构化数据
  2. 将 POI 数据作为 <amap_context> 喂给 LLM
  3. LLM 严格基于高德数据提取/编排美食推荐（真实店名+真实人均+真实地址）
  4. 容错: 高德 API 失败 → LLM 降级推演（带城市约束警告）→ Mock 兜底

面试考点:
  - LBS 架构: 高德 POI 包含 biz_ext.cost（人均消费）和 biz_ext.rating（评分）
  - 消除城市错配幻觉: POI 搜索设置 citylimit=true 强制限城
  - 预算分配: 美食费用 ≈ 总预算的 15%
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from loguru import logger

from config.settings import settings
from models.schemas import FoodItem, FoodSearchResult, TravelPlanState
from utils.amap_client import format_pois_as_context, search_amap_poi

from .base_agent import BaseAgent


# ━━━━━━━━━━━━━━━━━━ LLM Prompt ━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT_WITH_AMAP = """\
你是一位资深美食向导。用户会提供由高德地图 API 返回的目的地真实餐厅 POI 数据。

⚠️ 核心规则（违反即为严重错误）：
1. 你的推荐必须 100% 来源于提供的 <amap_context> 数据库。请直接使用里面提供的真实餐厅名称、评分和人均消费。
2. 绝不允许出现目的地与餐厅所在城市不符的情况！数据里没有的餐厅，你不能推荐。
3. 人均价格：如果 <amap_context> 中有「人均」数据，直接使用；如果显示「暂无」，根据餐厅类型合理估价（早餐 15-30，午餐 50-120，晚餐 80-200）。
4. must_try_dishes：基于你对该餐厅或菜系的常识补充招牌菜推荐。
5. insider_tips：可以基于常识适度补充，写得生动有向导感，但地名和价格绝不允许捏造！

每天安排: 1 个 breakfast + 1 个 lunch + 1 个 dinner + 可选 1 个 snack。

返回格式为严格 JSON 数组：
[
  {
    "name": "菜品/餐厅类型名称",
    "cuisine": "菜系(local/chinese/western/japanese等)",
    "restaurant": "完整的餐厅名称（必须与高德数据一致）",
    "address": "高德数据中的地址",
    "price_per_person": 人均价格(数字,来自高德人均或合理估价),
    "rating": 评分(来自高德数据,无则填8.0),
    "meal_type": "breakfast / lunch / dinner / snack",
    "must_try_dishes": ["推荐菜1", "推荐菜2"],
    "insider_tips": "一段向导风格的吃货贴士"
  }
]

要求：
- 只返回 JSON 数组，不要包含任何其他文字
"""

SYSTEM_PROMPT_FALLBACK = """\
你是一位资深美食向导。当前无法调用地图 API，需要你基于已有知识推荐美食。

⚠️ 极其严格的约束（违反即为致命错误）：
1. 你推荐的所有餐厅和美食必须确实存在于目的地城市「{city}」！
2. 绝对禁止出现其他城市的餐厅！例如目的地是南京，不能出现成都的串串香店。
3. 餐厅名称必须是该城市真实存在的知名餐厅。
4. 人均价格必须基于该城市的真实消费水平合理估价。
5. 如果你对某个城市的餐饮不熟悉，推荐通用的连锁餐饮也可以，但不要编造本地店名。
6. 推荐的特色菜必须是该城市真正的特色美食，而不是其他城市的。

每天: breakfast + lunch + dinner + 可选 snack。
返回 JSON 数组格式同上。只返回 JSON 数组，不要包含任何其他文字。
"""


def _build_food_prompt(
    city: str, days: list[str], hotel_loc: str,
    amap_context: str, daily_food_budget: float,
    style: str, dietary_restrictions: list[str],
) -> str:
    prompt = f"""请为以下旅行生成每日美食推荐：

目的地城市: {city}
酒店位置: {hotel_loc}
旅行天数: {len(days)} 天 ({', '.join(days)})
旅行风格: {style}
每日美食预算(参考): ¥{daily_food_budget:.0f}/人
"""
    if dietary_restrictions:
        prompt += f"饮食限制: {', '.join(dietary_restrictions)}\n"

    if amap_context:
        prompt += f"""
以下是高德地图 API 返回的「{city}」真实餐厅 POI 数据，请严格基于此数据推荐：
<amap_context>
{amap_context}
</amap_context>

请从高德数据中选择合适的餐厅，安排到每天的三餐中。名称和价格必须与数据一致。
"""
    else:
        prompt += f"""
地图 API 未返回数据。请基于你对「{city}」的美食知识推荐，但必须确保每家餐厅都是该城市真实存在的！
"""

    prompt += f"\n请为每一天各生成 3-4 道美食推荐 (breakfast+lunch+dinner+可选snack)，总共 {len(days)} 天，返回一个 JSON 数组。"
    return prompt


def _parse_food_json(raw: str, days: list[str]) -> list[FoodItem]:
    """解析 LLM 返回的 JSON 为 FoodItem 列表。"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)
    if not isinstance(data, list):
        data = data.get("food_items", data.get("foods", []))

    items_per_day = max(len(data) // max(len(days), 1), 3)
    food_items: list[FoodItem] = []

    for i, date_str in enumerate(days):
        day_foods = data[i * items_per_day: (i + 1) * items_per_day]
        if not day_foods and data:
            day_foods = data[:items_per_day]

        for f in day_foods:
            food_items.append(FoodItem(
                name=f.get("name", "特色美食"),
                cuisine=f.get("cuisine", "local"),
                restaurant=f.get("restaurant", ""),
                address=f.get("address", ""),
                price_per_person=float(f.get("price_per_person", 50)),
                rating=float(f.get("rating", 8.0)),
                description=f"{date_str} {f.get('meal_type', 'lunch')} - {f.get('name', '')}",
                meal_type=f.get("meal_type", "lunch"),
                distance_to_hotel_km=round(random.uniform(0.2, 3.0), 1),
                must_try_dishes=f.get("must_try_dishes", []),
                insider_tips=f.get("insider_tips", ""),
            ))

    return food_items


# ━━━━━━━━━━━━━━━━━━ Mock 兜底数据 ━━━━━━━━━━━━━━━━━━

FALLBACK_MEALS = [
    {"name": "当地特色早点", "cuisine": "local", "rest": "老字号早餐店",
     "price": 25, "meal": "breakfast",
     "dishes": ["特色包子", "豆浆油条"], "tips": "早上7点前到不排队。"},
    {"name": "当地特色午餐", "cuisine": "local", "rest": "百年老店",
     "price": 75, "meal": "lunch",
     "dishes": ["招牌菜", "特色小炒"], "tips": "11:30前到免排队。"},
    {"name": "特色晚餐", "cuisine": "local", "rest": "人气餐厅",
     "price": 120, "meal": "dinner",
     "dishes": ["招牌主菜", "特色甜品"], "tips": "提前1天预约靠窗位。"},
    {"name": "下午茶甜品", "cuisine": "western", "rest": "咖啡馆",
     "price": 45, "meal": "snack",
     "dishes": ["手冲咖啡", "招牌蛋糕"], "tips": "下午2-4点是黄金时段。"},
]


# ━━━━━━━━━━━━━━━━━━ Agent 本体 ━━━━━━━━━━━━━━━━━━

class FoodAgent(BaseAgent):
    name = "FoodAgent"

    async def execute(self, state: TravelPlanState) -> TravelPlanState:
        pref = state.preferences
        dest = state.selected_destination
        if pref is None or dest is None:
            raise ValueError("缺少偏好或目的地信息")

        hotel = state.hotel_result.recommended if state.hotel_result else None
        hotel_loc = hotel.address if hotel else f"{dest.city}市中心"
        city = dest.city

        days = self._get_travel_days(pref.start_date, pref.end_date)
        daily_food_budget = (pref.budget * 0.15) / max(len(days), 1) / pref.num_travelers

        logger.info(f"[{self.name}] 基于酒店位置 [{hotel_loc}] 搜索「{city}」周边美食")

        # ── Step 1: 高德地图 POI 搜索 ──
        diet_kw = " ".join(pref.dietary_restrictions[:2]) if pref.dietary_restrictions else ""
        search_keyword = f"特色美食 餐厅 {diet_kw}".strip()

        pois = await search_amap_poi(keyword=search_keyword, city=city)
        amap_context = format_pois_as_context(pois, label=f"{city}美食餐厅")

        has_amap = bool(amap_context)
        logger.info(f"[{self.name}] 搜索状态: {'有高德 POI 数据 ({len(pois)} 条)' if has_amap else '无数据(降级推演)'}")

        # ── Step 2: LLM 结构化编排 ──
        food_items = await self._extract_with_llm(
            city, days, hotel_loc, amap_context, daily_food_budget,
            pref.travel_style.value, pref.dietary_restrictions,
        )

        logger.info(f"[{self.name}] 提取到 {len(food_items)} 条美食推荐")

        total_cost = sum(f.price_per_person * pref.num_travelers for f in food_items)
        restaurant_set = {f.restaurant for f in food_items if f.restaurant}

        state.food_result = FoodSearchResult(
            food_items=food_items,
            recommended_restaurants=sorted(restaurant_set),
            total_food_cost=total_cost,
        )
        logger.info(
            f"[{self.name}] 涉及 {len(restaurant_set)} 家餐厅，"
            f"美食总费用: ¥{total_cost:.0f}"
        )
        return state

    async def _extract_with_llm(
        self, city: str, days: list[str], hotel_loc: str,
        amap_context: str, daily_food_budget: float,
        style: str, dietary_restrictions: list[str],
    ) -> list[FoodItem]:
        """LLM 编排美食数据，含三级容错。"""
        sys_prompt = SYSTEM_PROMPT_WITH_AMAP if amap_context else SYSTEM_PROMPT_FALLBACK.format(city=city)

        user_prompt = _build_food_prompt(
            city, days, hotel_loc, amap_context, daily_food_budget,
            style, dietary_restrictions,
        )

        try:
            raw = await self.call_llm(prompt=user_prompt, system_prompt=sys_prompt)

            if settings.LLM_PROVIDER == "mock":
                logger.info(f"[{self.name}] Mock 模式 → 使用兜底数据")
                return self._fallback_foods(city, days, daily_food_budget, style)

            food_items = _parse_food_json(raw, days)
            if food_items:
                return food_items

            logger.warning(f"[{self.name}] LLM 返回空数据, 使用兜底")
            return self._fallback_foods(city, days, daily_food_budget, style)

        except json.JSONDecodeError as exc:
            logger.warning(f"[{self.name}] JSON 解析失败: {exc}, 使用兜底")
            return self._fallback_foods(city, days, daily_food_budget, style)
        except Exception as exc:
            logger.error(f"[{self.name}] LLM 调用异常: {exc}, 使用兜底")
            return self._fallback_foods(city, days, daily_food_budget, style)

    @staticmethod
    def _fallback_foods(
        city: str, days: list[str], daily_budget: float, style: str,
    ) -> list[FoodItem]:
        """硬兜底: 生成通用美食数据，名称带城市前缀防幻觉。"""
        price_mult = {
            "budget": 0.6, "comfort": 1.0, "luxury": 1.8,
            "adventure": 0.8, "cultural": 1.0, "relaxation": 1.2,
        }
        mult = price_mult.get(style, 1.0)
        food_items: list[FoodItem] = []

        for date_str in days:
            for fm in FALLBACK_MEALS:
                adj_price = round(fm["price"] * mult, 0)
                food_items.append(FoodItem(
                    name=f"{city}{fm['name']}",
                    cuisine=fm["cuisine"],
                    restaurant=f"{city}{fm['rest']}",
                    address=f"{city}市内",
                    price_per_person=adj_price,
                    rating=round(random.uniform(7.5, 9.0), 1),
                    description=f"{date_str} {fm['meal']} - {city}{fm['name']}",
                    meal_type=fm["meal"],
                    distance_to_hotel_km=round(random.uniform(0.2, 2.5), 1),
                    must_try_dishes=fm["dishes"],
                    insider_tips=fm["tips"],
                ))

        return food_items

    @staticmethod
    def _get_travel_days(start: str, end: str) -> list[str]:
        try:
            d1 = datetime.strptime(start, "%Y-%m-%d")
            d2 = datetime.strptime(end, "%Y-%m-%d")
            days_count = max((d2 - d1).days, 1)
            return [(d1 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_count)]
        except (ValueError, TypeError):
            return ["2026-01-01", "2026-01-02", "2026-01-03"]
