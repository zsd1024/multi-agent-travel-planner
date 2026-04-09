"""
Pipeline 编排器 —— 串联整个行程规划流程。

架构（两阶段并行）:
  用户输入 → PreferenceAgent → DestinationAgent
  → 第一并行阶段: [FlightAgent + HotelAgent]
  → 第二并行阶段: [ActivityAgent + FoodAgent]  (依赖酒店位置)
  → BudgetAgent (预算校验)
  ↓ (超预算则循环调整)
  输出最终行程

面试考点:
  - 两阶段并行 vs 全并行: Flight/Hotel 无依赖可并行；Activity/Food 依赖 Hotel 的
    地理位置结果，必须等第一阶段完成后再启动。
  - 拓扑排序 / DAG: 本质上是 DAG 中的层级调度，同层无依赖 → 并行，跨层有依赖 → 串行
  - ParallelExecutor 复用: 同一个 ParallelExecutor 类支撑两阶段，仅传入不同 Agent 列表
  - 错误传播: 前序 Agent 失败则后序不执行，错误信息记录在 state 中
"""

from __future__ import annotations

from loguru import logger

from agents import (
    ActivityAgent,
    BudgetAgent,
    DestinationAgent,
    FlightAgent,
    FoodAgent,
    HotelAgent,
    PreferenceAgent,
)
from models.schemas import PlanningState, TravelPlanState, UserPreferences

from .budget_loop import BudgetLoopController
from .parallel import ParallelExecutor


class TravelPlanningPipeline:
    """主编排器: 串联所有 Agent 完成行程规划。"""

    def __init__(self) -> None:
        self.preference_agent = PreferenceAgent()
        self.destination_agent = DestinationAgent()
        self.flight_agent = FlightAgent()
        self.hotel_agent = HotelAgent()
        self.activity_agent = ActivityAgent()
        self.food_agent = FoodAgent()
        self.budget_agent = BudgetAgent()

        # ── 第一阶段并行: Flight + Hotel（互不依赖）──
        self.parallel_phase1 = ParallelExecutor(
            agents=[self.flight_agent, self.hotel_agent],
        )
        # ── 第二阶段并行: Activity + Food（依赖 Hotel 位置）──
        self.parallel_phase2 = ParallelExecutor(
            agents=[self.activity_agent, self.food_agent],
        )
        self.budget_loop = BudgetLoopController(
            phase1_executor=self.parallel_phase1,
            phase2_executor=self.parallel_phase2,
            budget_agent=self.budget_agent,
        )

    async def run(self, preferences: UserPreferences) -> TravelPlanState:
        state = TravelPlanState(preferences=preferences)
        logger.info("=" * 60)
        logger.info("🚀 行程规划 Pipeline 启动")
        logger.info("=" * 60)

        # ── 阶段 1: 偏好收集 ──
        state = await self.preference_agent.run(state)
        if state.state == PlanningState.FAILED:
            return state

        # ── 阶段 2: 目的地推荐 ──
        state = await self.destination_agent.run(state)
        if state.state == PlanningState.FAILED:
            return state

        # ── 阶段 3: 两阶段并行搜索 + 预算循环 ──
        state = await self.budget_loop.run(state)

        logger.info("=" * 60)
        logger.info(f"Pipeline 完成, 状态: {state.state.value}")
        if state.budget_breakdown:
            bb = state.budget_breakdown
            logger.info(f"总费用: ¥{bb.total_cost:.0f} / 预算: ¥{bb.budget:.0f}")
        logger.info("=" * 60)

        return state


async def quick_plan(
    budget: float = 10000,
    departure: str = "北京",
    start: str = "2026-05-01",
    end: str = "2026-05-05",
    style: str = "comfort",
    travelers: int = 1,
) -> TravelPlanState:
    """快速规划入口，便于 CLI / 测试调用。"""
    from models.schemas import TravelStyle

    prefs = UserPreferences(
        budget=budget,
        travel_style=TravelStyle(style),
        departure_city=departure,
        start_date=start,
        end_date=end,
        num_travelers=travelers,
    )
    pipeline = TravelPlanningPipeline()
    return await pipeline.run(prefs)
