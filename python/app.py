"""
AI 专属旅行管家 — 智能行程定制平台
Streamlit Web 界面 · 现代 OTA 商业风格

运行方式:
  cd python
  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from models.schemas import TravelPlanState, TravelStyle, UserPreferences
from orchestrator.pipeline import TravelPlanningPipeline

# ━━━━━━━━━━━━━━━━━━ 页面配置 ━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="AI 旅行管家 · 智能行程定制",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━ 现代 OTA 风格 CSS ━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;600;700&display=swap');

    /* ── 全局基底 ── */
    .stApp {
        background: #f5f7fa !important;
        font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'PingFang SC',
                     'Helvetica Neue', Arial, sans-serif !important;
    }
    .stApp > header { background: transparent !important; }

    /* ── 侧边栏 ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0066cc 0%, #0052a3 100%) !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.08) !important;
    }

    /* 侧边栏: 所有文字纯白 (广覆盖) */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stCaption p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span,
    [data-testid="stSidebar"] [data-baseweb] span {
        color: #ffffff !important;
    }
    /* 侧边栏 hr */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.20) !important;
    }

    /* 侧边栏: 输入框 — 白底深字 */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stTextArea textarea,
    [data-testid="stSidebar"] .stNumberInput input {
        background-color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.40) !important;
        border-radius: 10px !important;
        color: #333333 !important;
    }
    [data-testid="stSidebar"] .stTextInput input:focus,
    [data-testid="stSidebar"] .stTextArea textarea:focus,
    [data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.25) !important;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder,
    [data-testid="stSidebar"] .stTextArea textarea::placeholder {
        color: #999999 !important;
    }

    /* 侧边栏: 下拉菜单 / 多选框 — 白底深字 */
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="tag"] span {
        color: #333333 !important;
    }
    /* 日期选择器输入框 */
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background-color: #ffffff !important;
        color: #333333 !important;
        border-radius: 10px !important;
    }
    /* Slider 数值显示 */
    [data-testid="stSidebar"] [data-testid="stThumbValue"],
    [data-testid="stSidebar"] [data-testid="stTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stTickBarMax"] {
        color: #ffffff !important;
    }

    /* ── 主内容区 ── */
    .stApp h1 {
        color: #1a1a2e !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
        letter-spacing: -0.02em !important;
    }
    .stApp h2 {
        color: #1a1a2e !important;
        font-weight: 600 !important;
        font-size: 1.45rem !important;
    }
    .stApp h3 {
        color: #333 !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }
    .stApp p, .stApp li { color: #444 !important; line-height: 1.7 !important; }
    .stApp strong, .stApp b { color: #1a1a2e !important; }

    /* code tag 标签 */
    .stApp code {
        background: #eef3f9 !important;
        color: #0066cc !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 2px 8px !important;
        font-family: -apple-system, sans-serif !important;
        font-size: 0.88em !important;
        font-weight: 500 !important;
    }

    /* ── Metric 卡片 ── */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
        transition: box-shadow 0.2s ease, transform 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        box-shadow: 0 6px 24px rgba(0, 102, 204, 0.10) !important;
        transform: translateY(-1px) !important;
    }
    [data-testid="stMetric"] label {
        color: #888 !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0066cc !important;
        font-weight: 700 !important;
    }

    /* ── Expander: 行程卡片 ── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: none !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 12px !important;
        overflow: hidden !important;
    }
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 20px rgba(0, 102, 204, 0.09) !important;
    }
    [data-testid="stExpander"] summary {
        padding: 4px 0 !important;
    }
    [data-testid="stExpander"] summary span {
        color: #1a1a2e !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
    }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        border-top: 1px solid #f0f2f5 !important;
        padding-top: 12px !important;
    }

    /* ── Alert 气泡: 柔和色调 ── */
    /* info */
    [data-testid="stAlert"] > div:first-child {
        border-radius: 12px !important;
    }

    /* ── 按钮 ── */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #0066cc, #0080ff) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(0, 102, 204, 0.30) !important;
        transition: all 0.25s ease !important;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0052a3, #0066cc) !important;
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.40) !important;
        transform: translateY(-1px) !important;
    }

    /* ── 进度条 ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0066cc, #00aaff) !important;
        border-radius: 20px !important;
    }
    .stProgress > div {
        background: #e8edf2 !important;
        border-radius: 20px !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #e8ecf1 !important;
        margin: 1.2rem 0 !important;
    }

    /* ── 隐藏默认 footer ── */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━ 侧边栏 ━━━━━━━━━━━━━━━━━━

STYLE_MAP = {
    "comfort": "🛋️ 舒适休闲",
    "budget": "💰 经济实惠",
    "luxury": "👑 奢华尊享",
    "adventure": "🏔️ 探险刺激",
    "cultural": "🏛️ 人文深度",
    "relaxation": "🧘 悠闲度假",
}

SPINNER_MESSAGES = [
    "✈️ AI 管家正在为您精心规划行程…",
    "🔍 正在搜索最优航班与精选酒店…",
    "🍜 美食侦探出动，搜罗地道好店…",
    "📸 正在标注绝佳出片机位…",
    "🌍 多位 AI 助手正在协力为您服务…",
]

with st.sidebar:
    st.markdown("## ✈️ 旅行配置")
    st.caption("填好偏好，AI 管家帮你搞定一切")
    st.markdown("---")

    budget = st.slider(
        "💰 总预算",
        min_value=2000, max_value=100000, value=10000, step=1000,
        format="¥%d",
    )

    departure = st.text_input("🏙️ 出发城市", value="北京")
    target_dest = st.text_input(
        "📍 期望目的地",
        value="",
        placeholder="例如：巴黎（留空则由 AI 推荐）",
    )

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input("🛫 出发日期", value=date(2026, 5, 1))
    with col_d2:
        end_date = st.date_input("🛬 返回日期", value=date(2026, 5, 5))

    style = st.selectbox(
        "🎨 旅行风格",
        list(STYLE_MAP.keys()),
        format_func=lambda x: STYLE_MAP[x],
    )

    travelers = st.number_input("👥 出行人数", min_value=1, max_value=20, value=1)

    interests = st.multiselect(
        "❤️ 兴趣标签",
        ["美食", "历史", "艺术", "自然", "购物", "摄影", "运动", "建筑", "夜生活"],
    )

    notes = st.text_area("📝 备注", placeholder="例: 不吃辣、带小孩、需要无障碍设施…")

    st.markdown("---")
    plan_btn = st.button("✨ 开启专属定制之旅", type="primary", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━ 辅助函数 ━━━━━━━━━━━━━━━━━━

MEAL_EMOJI = {"breakfast": "🌅", "lunch": "☀️", "dinner": "🌙", "snack": "🍵"}
MEAL_LABEL = {"breakfast": "早餐", "lunch": "午餐", "dinner": "晚餐", "snack": "下午茶"}
SLOT_LABEL = {"morning": "🌤️ 上午", "afternoon": "☀️ 下午", "evening": "🌆 傍晚"}
MEAL_TO_SLOT = {"breakfast": "morning", "lunch": "afternoon", "snack": "afternoon", "dinner": "evening"}
MEAL_ORDER = {"breakfast": 0, "lunch": 1, "snack": 2, "dinner": 3}
SLOT_ORDER = {"morning": 0, "afternoon": 1, "evening": 2}


def build_food_by_date(state: TravelPlanState) -> dict[str, list]:
    food_by_date: dict[str, list] = {}
    if state.food_result:
        for item in state.food_result.food_items:
            date_str = item.description.split(" ")[0] if item.description else ""
            food_by_date.setdefault(date_str, []).append(item)
    return food_by_date


def render_activity(act) -> None:
    price_str = f"¥{act.price:.0f}" if act.price > 0 else "免费"
    slot_label = SLOT_LABEL.get(act.time_slot, act.time_slot)

    st.markdown(f"**🎯 {act.name}**")
    st.markdown(f"{slot_label} · {act.duration_hours}h · {price_str} · ⭐ {act.rating}")

    if act.photo_spots:
        st.info(f"📸 **出片机位**　{'　·　'.join(act.photo_spots)}")
    if act.tour_guide_tips:
        st.success(f"💡 **小贴士**　{act.tour_guide_tips}")


def render_food(food) -> None:
    emoji = MEAL_EMOJI.get(food.meal_type, "🍽️")
    label = MEAL_LABEL.get(food.meal_type, food.meal_type)
    price_str = f"¥{food.price_per_person:.0f}/人" if food.price_per_person > 0 else "含早"
    dist_str = f" · 📍 距酒店 {food.distance_to_hotel_km}km" if food.distance_to_hotel_km > 0 else ""

    st.markdown(f"**{emoji} {label} · {food.name}**")
    st.markdown(f"{food.restaurant} · {price_str}{dist_str} · ⭐ {food.rating}")

    if food.must_try_dishes:
        st.warning(f"🔥 **必点推荐**　{' / '.join(food.must_try_dishes)}")
    if food.insider_tips:
        st.success(f"💡 **吃货贴士**　{food.insider_tips}")


# ━━━━━━━━━━━━━━━━━━ 主界面 ━━━━━━━━━━━━━━━━━━

st.markdown("# ✈️ AI 专属旅行管家")
st.markdown("智能行程定制 · 7 位 AI 助手协力为您服务，从机票酒店到美食探店，一站搞定")
st.markdown("---")


if not plan_btn:
    # ── 首屏引导 ──
    st.markdown("### 👋 欢迎！在左侧填写您的旅行偏好，点击 **开启专属定制之旅**")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div style="background:#fff;border-radius:16px;padding:28px 24px;'
            'box-shadow:0 2px 12px rgba(0,0,0,0.05);text-align:center;">'
            '<div style="font-size:2.4rem;margin-bottom:10px;">🤖</div>'
            '<div style="font-weight:600;font-size:1.05rem;color:#1a1a2e;margin-bottom:6px;">智能协作</div>'
            '<div style="color:#666;font-size:0.88rem;line-height:1.6;">'
            '7 位 AI 助手各司其职<br>航班·酒店·美食·景点<br>一站式智能规划</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="background:#fff;border-radius:16px;padding:28px 24px;'
            'box-shadow:0 2px 12px rgba(0,0,0,0.05);text-align:center;">'
            '<div style="font-size:2.4rem;margin-bottom:10px;">⚡</div>'
            '<div style="font-weight:600;font-size:1.05rem;color:#1a1a2e;margin-bottom:6px;">高效并行</div>'
            '<div style="color:#666;font-size:0.88rem;line-height:1.6;">'
            '两阶段并行搜索<br>秒级出结果<br>酒店周边精准推荐</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div style="background:#fff;border-radius:16px;padding:28px 24px;'
            'box-shadow:0 2px 12px rgba(0,0,0,0.05);text-align:center;">'
            '<div style="font-size:2.4rem;margin-bottom:10px;">💰</div>'
            '<div style="font-weight:600;font-size:1.05rem;color:#1a1a2e;margin-bottom:6px;">预算无忧</div>'
            '<div style="color:#666;font-size:0.88rem;line-height:1.6;">'
            '超预算自动优化<br>渐进式智能调整<br>花最少钱玩最多</div></div>',
            unsafe_allow_html=True,
        )

else:
    # ── 执行 Pipeline ──
    import random
    spinner_msg = random.choice(SPINNER_MESSAGES)

    with st.spinner(spinner_msg):
        prefs = UserPreferences(
            budget=float(budget),
            travel_style=TravelStyle(style),
            departure_city=departure,
            target_destination=target_dest if target_dest.strip() else None,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            num_travelers=travelers,
            interests=interests,
            notes=notes,
        )
        pipeline = TravelPlanningPipeline()
        state: TravelPlanState = asyncio.run(pipeline.run(prefs))

    # ── 规划完成 ──
    st.balloons()
    st.toast("🎉 行程定制完成！向下查看您的专属方案", icon="✅")

    # ━━━━ 目的地 ━━━━
    if state.selected_destination:
        d = state.selected_destination
        st.markdown(f"## 🌍 {d.city}，{d.country}")
        st.markdown(f"> {d.description}")
        if d.highlights:
            tags = "　".join([f"`{h}`" for h in d.highlights])
            st.markdown(f"✨ **亮点**　{tags}")
    st.markdown("---")

    # ━━━━ 航班 + 酒店 ━━━━
    st.markdown("### ✈️ 机票与住宿")
    col_flight, col_hotel = st.columns(2)

    with col_flight:
        if state.flight_result:
            fr = state.flight_result
            st.metric("✈️ 机票总价", f"¥{fr.total_flight_cost:.0f}")
            if fr.recommended_outbound:
                o = fr.recommended_outbound
                st.markdown(f"**去程**　{o.airline} {o.flight_no}　{o.departure_city} → {o.arrival_city}　`¥{o.price:.0f}`")
            if fr.recommended_return:
                r = fr.recommended_return
                st.markdown(f"**返程**　{r.airline} {r.flight_no}　{r.arrival_city} → {r.departure_city}　`¥{r.price:.0f}`")
        else:
            st.info("暂无机票数据")

    with col_hotel:
        if state.hotel_result and state.hotel_result.recommended:
            h = state.hotel_result.recommended
            hr = state.hotel_result
            st.metric("🏨 住宿总价", f"¥{hr.total_hotel_cost:.0f}")
            st.markdown(f"**{h.name}**　{'⭐' * int(h.star_rating)} {h.star_rating}星")
            st.markdown(f"¥{h.price_per_night:.0f}/晚 × {hr.total_nights}晚　·　📍 {h.address}")
            if h.amenities:
                st.markdown(f"{'　·　'.join(h.amenities)}")
        else:
            st.info("暂无住宿数据")

    st.markdown("---")

    # ━━━━ 每日行程 ━━━━
    st.markdown("### 📅 每日行程")
    st.caption("展开查看每天的景点与美食安排，包含出片机位和贴心小贴士")

    food_by_date = build_food_by_date(state)
    day_plans = state.activity_result.day_plans if state.activity_result else []
    all_dates = sorted({day.date for day in day_plans} | set(food_by_date.keys()))

    for idx, dt in enumerate(all_dates):
        day_plan = next((d for d in day_plans if d.date == dt), None)
        day_foods = food_by_date.get(dt, [])

        act_cost = day_plan.day_cost if day_plan else 0
        food_cost = sum(f.price_per_person for f in day_foods)
        total_day = act_cost + food_cost

        entries: list[tuple[int, str, object]] = []
        if day_plan:
            for act in day_plan.activities:
                key = SLOT_ORDER.get(act.time_slot, 1) * 10 + 5
                entries.append((key, "activity", act))
        for food in day_foods:
            mapped_slot = MEAL_TO_SLOT.get(food.meal_type, "afternoon")
            key = SLOT_ORDER.get(mapped_slot, 1) * 10 + MEAL_ORDER.get(food.meal_type, 1)
            entries.append((key, "food", food))
        entries.sort(key=lambda x: x[0])

        with st.expander(f"📆 第 {idx + 1} 天　{dt}　·　日花费 ¥{total_day:.0f}", expanded=(idx == 0)):
            prev_slot = None
            for order_key, entry_type, data in entries:
                if entry_type == "activity":
                    cur_slot = data.time_slot
                else:
                    cur_slot = MEAL_TO_SLOT.get(data.meal_type, "afternoon")
                if prev_slot is not None and cur_slot != prev_slot:
                    st.markdown("---")
                prev_slot = cur_slot

                if entry_type == "activity":
                    render_activity(data)
                else:
                    render_food(data)
                st.markdown("")

    st.markdown("---")

    # ━━━━ 预算明细 ━━━━
    st.markdown("### 💰 费用总览")

    if state.budget_breakdown:
        bb = state.budget_breakdown

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("✈️ 机票", f"¥{bb.flight_cost:.0f}")
        mc2.metric("🏨 住宿", f"¥{bb.hotel_cost:.0f}")
        mc3.metric("🎯 活动", f"¥{bb.activity_cost:.0f}")
        mc4.metric("🍽️ 美食", f"¥{bb.food_cost:.0f}")

        st.markdown("")

        tc1, tc2 = st.columns([2, 1])
        with tc1:
            remaining = bb.remaining
            delta_label = f"节省 ¥{remaining:.0f}" if remaining >= 0 else f"超出 ¥{abs(remaining):.0f}"
            st.metric(
                "总花费 / 预算",
                f"¥{bb.total_cost:.0f} / ¥{bb.budget:.0f}",
                delta=delta_label,
                delta_color="normal" if remaining >= 0 else "inverse",
            )
        with tc2:
            if state.adjustment_round > 0:
                st.info(f"🔄 经过 **{state.adjustment_round}** 轮智能调整")

        progress = min(bb.total_cost / max(bb.budget, 1), 1.0)
        st.markdown(f"**预算使用 {progress * 100:.1f}%**")
        st.progress(progress)

        if bb.is_within_budget:
            st.success(f"✅ 预算充裕！还剩 ¥{bb.remaining:.0f} 可自由支配")
        else:
            st.error(f"⚠️ 超预算 ¥{bb.over_budget_amount:.0f}，已通过 {state.adjustment_round} 轮智能优化")

        if bb.suggestions:
            with st.expander("💡 省钱小贴士"):
                for s in bb.suggestions:
                    st.markdown(f"- {s}")

    # ── 警告 ──
    if state.error_messages:
        st.markdown("---")
        st.markdown("### ⚠️ 温馨提示")
        for msg in state.error_messages:
            st.warning(msg)

    # ── 底部 ──
    st.markdown("---")
    st.caption("✈️ AI 专属旅行管家 · 7 位 AI 助手智能协作 · 让每一次出行都完美无忧")
