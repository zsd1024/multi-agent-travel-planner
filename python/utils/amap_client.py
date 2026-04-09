"""
高德地图 Web 服务 API 客户端 —— POI 搜索能力封装。

职责:
  - 调用高德地图「搜索 POI」接口（/v3/place/text）获取结构化 POI 数据
  - 解析 pois 列表，提取名称、坐标、地址、评分、人均消费
  - 组装成精简的上下文文本，供 LLM 做结构化推荐

接口文档: https://lbs.amap.com/api/webservice/guide/api/search

面试考点:
  - LBS (Location-Based Service): 基于位置的服务，商业级旅游 App 核心能力
  - 高德 vs Tavily: 高德返回结构化 POI（含坐标/评分/价格），Tavily 返回网页摘要
  - extensions=all: 必须带此参数才能获取 biz_ext 扩展字段（评分、人均消费）
"""

from __future__ import annotations

import httpx
from loguru import logger

from config.settings import settings

# 高德 POI 搜索接口
AMAP_PLACE_URL = "https://restapi.amap.com/v3/place/text"


async def search_amap_poi(
    keyword: str,
    city: str,
    offset: int = 10,
    page: int = 1,
) -> list[dict]:
    """调用高德地图 POI 搜索，返回精简的 POI 列表。

    Args:
        keyword: 搜索关键词，如 "必去景点" / "特色美食 餐厅"
        city: 城市名，如 "南京" / "成都"
        offset: 每页返回数量（最大 25）
        page: 页码

    Returns:
        精简 POI 列表，每项包含:
          name, address, location(经纬度), rating, cost(人均), type_desc
    """
    api_key = settings.AMAP_API_KEY
    if not api_key:
        logger.warning("[AmapClient] AMAP_API_KEY 未配置, 跳过搜索")
        return []

    params = {
        "key": api_key,
        "keywords": keyword,
        "city": city,
        "citylimit": "true",       # 强制限制在该城市内搜索
        "offset": str(offset),
        "page": str(page),
        "extensions": "all",       # 必须带 all 才能获取 biz_ext
        "output": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(AMAP_PLACE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        # 高德返回状态检查
        if data.get("status") != "1":
            info = data.get("info", "未知错误")
            infocode = data.get("infocode", "")
            logger.warning(f"[AmapClient] 高德 API 返回错误: {info} (code={infocode})")
            return []

        pois_raw = data.get("pois", [])
        if not pois_raw:
            logger.info(f"[AmapClient] 关键词「{keyword}」在「{city}」无搜索结果")
            return []

        # 解析精简 POI
        pois: list[dict] = []
        for p in pois_raw:
            biz = p.get("biz_ext", {}) or {}
            poi = {
                "name": p.get("name", ""),
                "address": p.get("address", ""),
                "location": p.get("location", ""),          # "116.397499,39.908722"
                "type_desc": p.get("type", ""),              # "风景名胜;公园"
                "rating": _safe_float(biz.get("rating")),    # "4.5" → 4.5
                "cost": _safe_float(biz.get("cost")),        # "85" → 85.0
                "tel": p.get("tel", ""),
            }
            pois.append(poi)

        logger.info(f"[AmapClient] 「{keyword}」@「{city}」返回 {len(pois)} 条 POI")
        return pois

    except httpx.TimeoutException:
        logger.warning(f"[AmapClient] 请求超时: keyword={keyword}, city={city}")
        return []
    except httpx.HTTPStatusError as exc:
        logger.warning(f"[AmapClient] HTTP 错误 {exc.response.status_code}: {exc}")
        return []
    except Exception as exc:
        logger.error(f"[AmapClient] 搜索异常: {exc}")
        return []


def format_pois_as_context(pois: list[dict], label: str = "POI") -> str:
    """将 POI 列表格式化为 LLM 可消费的上下文文本。

    示例输出:
      【南京博物院】地址: 中山东路321号 | 评分: 4.8 | 人均: ¥0(免费) | 类型: 科教文化服务;博物馆
    """
    if not pois:
        return ""

    lines = [f"以下是高德地图返回的「{label}」真实 POI 数据（共 {len(pois)} 条）："]
    for i, p in enumerate(pois, 1):
        name = p.get("name", "未知")
        addr = p.get("address", "未知")
        rating = p.get("rating", 0)
        cost = p.get("cost", 0)
        type_desc = p.get("type_desc", "")
        location = p.get("location", "")

        rating_str = f"评分: {rating}" if rating > 0 else "评分: 暂无"
        cost_str = f"人均: ¥{cost:.0f}" if cost > 0 else "人均: 暂无"

        lines.append(
            f"[{i}] 【{name}】"
            f"地址: {addr} | {rating_str} | {cost_str} | "
            f"类型: {type_desc} | 坐标: {location}"
        )

    return "\n".join(lines)


def _safe_float(val) -> float:
    """安全转换为 float，高德返回的可能是空字符串/None/[]。"""
    if not val or val == [] or val == "[]":
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0
