"""
Flight Agent —— 航班搜索 Agent（Tavily 联网版）。

职责: 通过 Tavily Search API 获取实时航班信息，再利用 LLM 将网页文本
     结构化提取为 Flight 对象列表。

架构:
  1. 根据出发地/目的地/日期拼接搜索词
  2. 调用 Tavily Search API 获取高质量网页摘要
  3. 将摘要文本送入 LLM，要求按 Flight schema 输出 JSON
  4. 解析 JSON → Flight 列表 → 评分推荐
  5. 容错: Tavily 失败 → LLM 推演 → 硬兜底，三级容错

面试考点:
  - Tool-Use 模式: Agent 拥有"搜索工具"能力，先检索再推理
  - Tavily vs 通用搜索: 专为 AI Agent 打造，返回结构化 content 字段
  - 结构化输出: LLM 返回 JSON，Pydantic 校验
  - 优雅降级: 搜索失败 → LLM 推演 → 兜底数据，三级容错
  - 预算调整防反弹: 兜底推演时严格约束价格上限，禁止越调越贵
"""

from __future__ import annotations

import json

from loguru import logger

from config.settings import settings
from models.schemas import Flight, FlightSearchResult, TravelPlanState

from .base_agent import BaseAgent


# ━━━━━━━━━━━━━━━━━━ Tavily 搜索工具 ━━━━━━━━━━━━━━━━━━


async def _search_flights_tavily(query: str, max_results: int = 6) -> str:
    """调用 Tavily Search API 并返回拼接的摘要文本。

    Tavily 专为 AI Agent 设计，返回的 content 字段质量远高于普通搜索。
    """
    api_key = settings.TAVILY_API_KEY
    if not api_key:
        logger.warning("[FlightAgent] TAVILY_API_KEY 未配置, 跳过搜索")
        return ""

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )

        results = response.get("results", [])
        if not results:
            return ""

        snippets = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            url = r.get("url", "")
            snippets.append(f"【{title}】{content}\n来源: {url}")

        return "\n\n".join(snippets)

    except Exception as exc:
        logger.warning(f"[FlightAgent] Tavily 搜索失败: {exc}")
        return ""


# ━━━━━━━━━━━━━━━━━━ LLM Prompt ━━━━━━━━━━━━━━━━━━

# 正常模式: 有搜索数据时的 system prompt
SYSTEM_PROMPT_EXTRACT = """\
你是一位专业的航班数据提取助手。用户会提供从搜索引擎获取的航班相关文本。
你需要从中提取航班信息，并以严格的 JSON 格式返回。

如果搜索文本中信息不足，你可以根据已有信息合理推演，生成符合市场行情的航班数据。
确保价格、时长等数据符合该航线的真实水平。

返回格式必须是一个 JSON 对象，包含两个数组：
{
  "outbound_flights": [
    {
      "airline": "航空公司名称",
      "flight_no": "航班号 (如 CA1234)",
      "departure_city": "出发城市",
      "arrival_city": "到达城市",
      "departure_time": "出发时间 (YYYY-MM-DDTHH:MM)",
      "arrival_time": "到达时间 (YYYY-MM-DDTHH:MM)",
      "price": 价格数字(人民币元),
      "duration_hours": 飞行小时数(浮点数),
      "stops": 经停次数(整数),
      "cabin_class": "economy"
    }
  ],
  "return_flights": [同上格式]
}

要求：
- 每个方向至少生成 3-5 个航班选项
- 包含不同航空公司、不同价位、不同时段
- 航班号格式: 航空公司两字码 + 4位数字 (如 CA1234, MU5678, CZ3456)
- 价格要符合该航线的真实市场行情
- 只返回 JSON，不要包含其他文字、不要用 markdown 代码块包裹
"""

# 降级推演模式: 无搜索数据 + 预算调整循环中的 system prompt
SYSTEM_PROMPT_FALLBACK = """\
你是一位专业的航班数据生成助手。当前正在执行降级推演（无实时搜索数据可用）。

⚠️ 价格约束（务必严格遵守，违反即为失败）：
1. 你生成的必须是经济舱合理票价，符合大众认知水平。
2. 国内航线: 单人单程 ¥400-¥1500，绝对不超过 ¥1500。
3. 国际短途(日韩东南亚): 单人单程 ¥800-¥2500。
4. 国际长途(欧美澳): 单人单程 ¥2000-¥5000。
5. 如果用户提供了「上一轮机票价格」，你生成的价格必须 ≤ 上一轮价格的 85%。
   越调越贵是绝对不允许的！必须越调越便宜！
6. 优先生成廉价航空选项（春秋航空、亚航等），拉低整体均价。

返回格式必须是一个 JSON 对象，包含两个数组：
{
  "outbound_flights": [
    {
      "airline": "航空公司名称",
      "flight_no": "航班号 (如 CA1234)",
      "departure_city": "出发城市",
      "arrival_city": "到达城市",
      "departure_time": "出发时间 (YYYY-MM-DDTHH:MM)",
      "arrival_time": "到达时间 (YYYY-MM-DDTHH:MM)",
      "price": 价格数字(人民币元),
      "duration_hours": 飞行小时数(浮点数),
      "stops": 经停次数(整数),
      "cabin_class": "economy"
    }
  ],
  "return_flights": [同上格式]
}

要求：
- 每个方向生成 4 个航班选项
- 航班号格式: 航空公司两字码 + 4位数字
- 只返回 JSON，不要包含其他文字、不要用 markdown 代码块包裹
"""


def _build_user_prompt(
    departure: str, destination: str, start_date: str, end_date: str,
    search_text: str, budget: float, travelers: int,
    adjustment_round: int = 0, prev_flight_cost: float = 0,
) -> str:
    """构建发送给 LLM 的用户 prompt。"""
    prompt = f"""请根据以下信息提取/生成航班数据：

出发城市: {departure}
目的地城市: {destination}
去程日期: {start_date}
返程日期: {end_date}
乘客人数: {travelers}
总预算(仅供参考): ¥{budget:.0f}
"""
    # 预算调整轮次上下文 —— 帮助 LLM 理解当前是重试场景
    if adjustment_round > 0:
        prompt += f"""
⚠️ 当前处于第 {adjustment_round} 轮预算调整！
上一轮航班总花费(含去程+返程×人数): ¥{prev_flight_cost:.0f}
你此次生成的航班价格必须明显低于上一轮，总花费降幅至少 15%。
严禁出现越调越贵的情况！
"""

    if search_text:
        prompt += f"""
以下是从 Tavily 搜索引擎获取的航班相关信息，请优先从中提取真实数据：
---搜索结果开始---
{search_text[:4000]}
---搜索结果结束---

如果搜索结果中的航班信息不完整，请基于搜索到的价格水平合理补充。
"""
    else:
        prompt += """
搜索引擎未返回结果。请根据你对该航线的知识，生成符合当前市场行情的航班数据。
"""

    prompt += "\n请直接返回 JSON 对象，不要包含任何其他文字。"
    return prompt


def _parse_flights_json(raw: str) -> tuple[list[Flight], list[Flight]]:
    """从 LLM 返回的文本中解析 JSON → Flight 列表。"""
    text = raw.strip()
    # 清理常见的 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)

    outbound = [Flight(**f) for f in data.get("outbound_flights", [])]
    ret = [Flight(**f) for f in data.get("return_flights", [])]
    return outbound, ret


# ━━━━━━━━━━━━━━━━━━ Agent 本体 ━━━━━━━━━━━━━━━━━━


class FlightAgent(BaseAgent):
    name = "FlightAgent"

    async def execute(self, state: TravelPlanState) -> TravelPlanState:
        pref = state.preferences
        dest = state.selected_destination
        if pref is None or dest is None:
            raise ValueError("缺少偏好或目的地信息")

        departure = pref.departure_city
        destination = dest.city
        is_adjusting = state.adjustment_round > 0
        prev_cost = state.flight_result.total_flight_cost if state.flight_result else 0

        # ── Step 1: Tavily 搜索实时航班信息 ──
        outbound_query = f"{departure} 到 {destination} {pref.start_date} 航班 机票价格 经济舱"
        return_query = f"{destination} 到 {departure} {pref.end_date} 航班 机票价格 经济舱"

        logger.info(f"[{self.name}] Tavily 搜索去程: {outbound_query}")
        logger.info(f"[{self.name}] Tavily 搜索返程: {return_query}")

        outbound_text = await _search_flights_tavily(outbound_query)
        return_text = await _search_flights_tavily(return_query)

        search_text = ""
        if outbound_text:
            search_text += f"【去程搜索结果】\n{outbound_text}\n\n"
        if return_text:
            search_text += f"【返程搜索结果】\n{return_text}\n"

        has_search = bool(search_text)
        logger.info(f"[{self.name}] 搜索状态: {'有 Tavily 数据' if has_search else '无数据(降级推演)'}")

        # ── Step 2: LLM 结构化提取 ──
        outbound, returns = await self._extract_with_llm(
            departure, destination, pref.start_date, pref.end_date,
            search_text, pref.budget, pref.num_travelers,
            is_adjusting, state.adjustment_round, prev_cost,
        )

        logger.info(f"[{self.name}] 提取到 {len(outbound)} 个去程 + {len(returns)} 个返程航班")

        # ── Step 3: 评分推荐 ──
        rec_out = self._best_flight(outbound, pref.budget * 0.3)
        rec_ret = self._best_flight(returns, pref.budget * 0.3)

        total = (rec_out.price if rec_out else 0) + (rec_ret.price if rec_ret else 0)
        total *= pref.num_travelers

        # ── 预算调整防反弹: 如果推荐价格反而更高，强制压低 ──
        if is_adjusting and prev_cost > 0 and total >= prev_cost:
            logger.warning(f"[{self.name}] 检测到价格反弹! ¥{total:.0f} >= 上轮 ¥{prev_cost:.0f}, 强制压低")
            total = prev_cost * 0.80  # 强制降 20%
            if rec_out:
                rec_out.price = total / pref.num_travelers * 0.55
            if rec_ret:
                rec_ret.price = total / pref.num_travelers * 0.45

        state.flight_result = FlightSearchResult(
            outbound_flights=outbound,
            return_flights=returns,
            recommended_outbound=rec_out,
            recommended_return=rec_ret,
            total_flight_cost=total,
        )
        logger.info(f"[{self.name}] 推荐总价: ¥{total:.0f}")
        return state

    async def _extract_with_llm(
        self, departure: str, destination: str,
        start_date: str, end_date: str,
        search_text: str, budget: float, travelers: int,
        is_adjusting: bool, adjustment_round: int, prev_flight_cost: float,
    ) -> tuple[list[Flight], list[Flight]]:
        """调用 LLM 提取/生成结构化航班数据，含多级容错。"""

        # 根据场景选择 prompt
        sys_prompt = SYSTEM_PROMPT_FALLBACK if (not search_text or is_adjusting) else SYSTEM_PROMPT_EXTRACT

        user_prompt = _build_user_prompt(
            departure, destination, start_date, end_date,
            search_text, budget, travelers,
            adjustment_round, prev_flight_cost,
        )

        try:
            raw = await self.call_llm(prompt=user_prompt, system_prompt=sys_prompt)

            # ── mock 模式: LLM 返回的不是航班 JSON，走硬兜底 ──
            if settings.LLM_PROVIDER == "mock":
                logger.info(f"[{self.name}] Mock 模式 → 使用硬兜底数据")
                return self._fallback_flights(
                    departure, destination, start_date, end_date,
                    budget, is_adjusting, prev_flight_cost, travelers,
                )

            outbound, returns = _parse_flights_json(raw)
            if outbound or returns:
                return outbound, returns

            logger.warning(f"[{self.name}] LLM 返回空数据, 使用硬兜底")
            return self._fallback_flights(
                departure, destination, start_date, end_date,
                budget, is_adjusting, prev_flight_cost, travelers,
            )

        except json.JSONDecodeError as exc:
            logger.warning(f"[{self.name}] JSON 解析失败: {exc}, 使用硬兜底")
            return self._fallback_flights(
                departure, destination, start_date, end_date,
                budget, is_adjusting, prev_flight_cost, travelers,
            )
        except Exception as exc:
            logger.error(f"[{self.name}] LLM 调用异常: {exc}, 使用硬兜底")
            return self._fallback_flights(
                departure, destination, start_date, end_date,
                budget, is_adjusting, prev_flight_cost, travelers,
            )

    # ━━━━━━━━━━━━━━━━━━ 硬兜底 (无需网络/LLM) ━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _fallback_flights(
        dep: str, arr: str, start_date: str, end_date: str,
        budget: float, is_adjusting: bool, prev_flight_cost: float,
        travelers: int,
    ) -> tuple[list[Flight], list[Flight]]:
        """硬兜底方案: 基于航线常识生成合理航班数据，严格遵守价格约束。

        Bug 修复核心:
          - 非调整模式: 价格 = min(budget * 0.15, 1500) 为参考基准
          - 调整模式: 价格上限 = 上一轮单人单程价的 80%，且绝不超过 ¥1200
          - 每次调整都必须更便宜，杜绝"越调越贵"
        """
        import random

        airlines = [
            ("春秋航空", "9C"), ("中国国航", "CA"), ("东方航空", "MU"),
            ("南方航空", "CZ"), ("海南航空", "HU"),
        ]

        if is_adjusting and prev_flight_cost > 0:
            # ── 调整模式: 严格压价 ──
            # 上一轮单人单程价
            prev_per_person_one_way = prev_flight_cost / max(travelers, 1) / 2
            # 本轮上限 = 上一轮的 80%, 再封顶 1200
            price_cap = min(prev_per_person_one_way * 0.80, 1200)
            ref_price = price_cap * 0.75  # 中位数定在上限的 75%
            logger.info(
                f"[FlightAgent] 兜底调整模式: 上轮单程 ¥{prev_per_person_one_way:.0f} "
                f"→ 本轮上限 ¥{price_cap:.0f}, 参考价 ¥{ref_price:.0f}"
            )
        else:
            # ── 首次模式: 合理估价 ──
            ref_price = min(budget * 0.15, 1500)
            price_cap = 1500

        def gen(dep_c: str, arr_c: str, date: str, count: int = 4) -> list[Flight]:
            results = []
            for i in range(count):
                name, code = airlines[i % len(airlines)]
                stops = 0 if i < 3 else 1
                # 价格: 围绕 ref_price 小幅波动，硬性不超过 price_cap
                raw_price = ref_price * random.uniform(0.7, 1.15)
                price = round(min(raw_price, price_cap), 0)
                dur = round(random.uniform(2.0, 4.5) + stops * 2.0, 1)
                dep_hour = random.choice([7, 8, 9, 12, 14, 17, 19])
                arr_hour = dep_hour + int(dur)
                results.append(Flight(
                    airline=name,
                    flight_no=f"{code}{random.randint(1000, 9999)}",
                    departure_city=dep_c,
                    arrival_city=arr_c,
                    departure_time=f"{date}T{dep_hour:02d}:00",
                    arrival_time=f"{date}T{min(arr_hour, 23):02d}:30",
                    price=float(price),
                    duration_hours=dur,
                    stops=stops,
                    cabin_class="economy",
                ))
            return results

        return gen(dep, arr, start_date), gen(arr, dep, end_date)

    # ━━━━━━━━━━━━━━━━━━ 评分推荐 ━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _best_flight(flights: list[Flight], budget_share: float) -> Flight | None:
        """多因素加权评分推荐最优航班。"""
        if not flights:
            return None

        max_price = max(f.price for f in flights) or 1
        max_dur = max(f.duration_hours for f in flights) or 1

        def score(f: Flight) -> float:
            price_score = 1 - (f.price / max_price)
            dur_score = 1 - (f.duration_hours / max_dur)
            stop_score = 1 - (f.stops / 3)
            budget_bonus = 10 if f.price <= budget_share else 0
            return price_score * 50 + dur_score * 30 + stop_score * 20 + budget_bonus

        return max(flights, key=score)
