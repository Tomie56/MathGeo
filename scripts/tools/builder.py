import json
import random
import sympy as sp
import logging
from sympy import symbols, cos, sin, pi, simplify, sqrt, tan, Eq, solve, Min, Max, Ge, Le, Rational
from typing import List, Dict, Tuple, Optional, Union, Set, Generator
import os
import threading
import time

logger = logging.getLogger('builder')


class RandomGeometryBuilder:
    def __init__(self, base_json: Dict, base_id: str = "", max_points: int = 500, max_lines: int = 300):
        """初始化builder，加载基础图形数据"""
        self.data = base_json.copy()
        self.x, self.y = symbols('x y')
        
        # 新增：点和线段数量上限（可配置）
        self.max_points = max_points  # 最大点数量
        self.max_lines = max_lines    # 最大线段数量
        self.max_expr_length = 50
        
        # 基础数据结构与索引
        self.points = self.data["points"]
        self.lines = self.data["lines"]
        self.arcs = self.data.get("arcs", []) 
        self.entities = self.data["entities"]
        
        self.point_id_map = {p["id"]: p for p in self.points}
        self.line_id_map = {l["id"]: l for l in self.lines}
        self.arc_id_map = {a["id"]: a for a in self.arcs}
        self.entity_id_map = {e["id"]: e for e in self.entities}
        
        # 操作计数器（生成唯一ID）
        self.operation_counter = 0
        self.entity_vertices_cache = self._cache_entity_vertices()
        self.entity_lines_cache = self._cache_entity_lines()
        self.entity_arcs_cache = self._cache_entity_arcs()
        self.on_segment_points_cache: Set[str] = set(self._cache_on_segment_points()) 
        self.all_points_cache: Set[str] = set(self.point_id_map.keys())
        self.line_pairs: Set[Tuple[str, str]] = self._build_line_pairs()
        
        self.enhancement_history = []
        self.MAX_RETRIES = 4  # 重试次数
        self.base_id = base_id
        self.only_circles = self._is_only_circles()

        # 检查初始数量是否已超标
        if len(self.points) >= self.max_points:
            raise ValueError(f"初始点数量({len(self.points)})超过上限({self.max_points})")
        if len(self.lines) >= self.max_lines:
            raise ValueError(f"初始线段数量({len(self.lines)})超过上限({self.max_lines})")

    # ------------------------------ 基础工具方法（增量缓存） ------------------------------
    
    def _is_only_circles(self) -> bool:
        """判断基础图形是否仅包含圆形实体（无其他线段/多边形等）"""
        if len(self.entities) == 0:
            return False
        all_circles = all(e["type"] == "circle" for e in self.entities)
        has_non_circle_lines = len(self.lines) > 0 and not all(
            line.get("type") in ["diameter", "circle_related"] for line in self.lines
        )
        return all_circles

    def _cache_entity_vertices(self) -> Dict[str, List[str]]:
        cache = {}
        for e in self.entities:
            eid = e["id"]
            cache[eid] = [c for c in e["components"] if c in self.point_id_map]
        return cache

    def _cache_entity_lines(self) -> Dict[str, List[str]]:
        cache = {}
        for e in self.entities:
            eid = e["id"]
            cache[eid] = [c for c in e["components"] if c in self.line_id_map]
        return cache

    def _cache_entity_arcs(self) -> Dict[str, List[str]]:
        cache = {}
        for e in self.entities:
            eid = e["id"]
            cache[eid] = [c for c in e["components"] if c in self.arc_id_map]
        return cache

    def _cache_on_segment_points(self) -> List[str]:
        """初始化线上点缓存（仅首次调用）"""
        on_segment = set()
        # 1. 添加所有线段的端点
        for line in self.lines:
            on_segment.add(line["start_point_id"])
            on_segment.add(line["end_point_id"])
        # 2. 添加所有线段上的非端点
        for pid in self.point_id_map:
            if pid in on_segment:
                continue
            px, py = self.get_point_coords(pid)
            for line in self.lines:
                s_id, e_id = line["start_point_id"], line["end_point_id"]
                s_x, s_y = self.get_point_coords(s_id)
                e_x, e_y = self.get_point_coords(e_id)
                if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                    on_segment.add(pid)
                    break  # 找到即停止
        return list(on_segment)

    def _build_line_pairs(self) -> Set[Tuple[str, str]]:
        """构建线段点对哈希表（存储有序对），O(1)判断两点是否已连接"""
        pairs = set()
        for line in self.lines:
            p1, p2 = line["start_point_id"], line["end_point_id"]
            if p1 > p2:
                p1, p2 = p2, p1
            pairs.add((p1, p2))
        return pairs
    
    def _is_expr_too_long(self, pid: str) -> bool:
        """检查点的x/y坐标表达式是否超过长度上限"""
        p = self.point_id_map[pid]
        x_expr = p["x"]["expr"]
        y_expr = p["y"]["expr"]
        # 检查x或y表达式长度是否超标
        return len(x_expr) > self.max_expr_length or len(y_expr) > self.max_expr_length

    def _is_point_on_any_segment(self, pid: str) -> bool:
        return pid in self.on_segment_points_cache
    
    def _is_point_on_other_edge(self, point_id: str, exclude_line_id: str) -> bool:
        """检查点是否在除exclude_line_id之外的任何线段上"""
        if point_id not in self.point_id_map:
            return False
            
        px, py = self.get_point_coords(point_id)
        
        for line_id, line in self.line_id_map.items():
            if line_id == exclude_line_id:
                continue
                
            s_id, e_id = line["start_point_id"], line["end_point_id"]
            s_x, s_y = self.get_point_coords(s_id)
            e_x, e_y = self.get_point_coords(e_id)
            
            if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                return True
        return False

    def _get_unique_id(self, prefix: str) -> str:
        uid = f"{prefix}{self.operation_counter}"
        self.operation_counter += 1
        return uid

    def _parse_expr(self, expr_str: str) -> sp.Expr:
        return sp.sympify(expr_str)  # 延迟化简，仅解析

    def _serialize_expr(self, expr: sp.Expr) -> str:
        """仅对复杂表达式化简，简单表达式直接转换"""
        if expr.has(sp.sin, sp.cos, sp.sqrt, sp.tan):
            expr = simplify(expr)
        return str(expr)

    # ------------------------------ 点和线段操作（增量更新+数量控制） ------------------------------
    def get_point_coords(self, pid: str) -> Tuple[sp.Expr, sp.Expr]:
        p = self.point_id_map[pid]
        return self._parse_expr(p["x"]["expr"]), self._parse_expr(p["y"]["expr"])

    def add_new_point(self, 
                     x_expr: sp.Expr, 
                     y_expr: sp.Expr, 
                     prefix: str = "P",
                     point_type: Optional[str] = None,
                     related_vertex: Optional[str] = None,
                     related_edge: Optional[str] = None) -> str:
        # 新增：检查点数量是否超过上限
        if len(self.points) >= self.max_points:
            raise OverflowError(f"点数量超过上限({self.max_points})，放弃本轮生成")
        
        x_simplified = x_expr  # 延迟化简
        y_simplified = y_expr
        x_str = self._serialize_expr(x_simplified)
        y_str = self._serialize_expr(y_simplified)
        
        # 快速查重（用符号等式判断，避免全量化简）
        for p in self.points:
            existing_x = self._parse_expr(p["x"]["expr"])
            existing_y = self._parse_expr(p["y"]["expr"])
            
            x_equal = sp.simplify(existing_x - x_simplified) == 0
            y_equal = sp.simplify(existing_y - y_simplified) == 0
            
            if x_equal and y_equal:
                if point_type and "type" not in p:
                    p["type"] = point_type
                if related_vertex and "related_vertex" not in p:
                    p["related_vertex"] = related_vertex
                if related_edge and "related_edge" not in p:
                    p["related_edge"] = related_edge
                return p["id"]
        
        pid = self._get_unique_id(prefix)
        new_point = {
            "id": pid,
            "x": {"expr": x_str, "latex": sp.latex(x_simplified)},
            "y": {"expr": y_str, "latex": sp.latex(y_simplified)}
        }
        if point_type:
            new_point["type"] = point_type
        if related_vertex:
            new_point["related_vertex"] = related_vertex
        if related_edge:
            new_point["related_edge"] = related_edge
        
        self.points.append(new_point)
        self.point_id_map[pid] = new_point
        self.all_points_cache.add(pid)
        
        # 增量更新线上点缓存：仅检查新点是否在任何线段上
        px, py = x_simplified, y_simplified
        for line in self.lines:
            s_id, e_id = line["start_point_id"], line["end_point_id"]
            s_x, s_y = self.get_point_coords(s_id)
            e_x, e_y = self.get_point_coords(e_id)
            if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                self.on_segment_points_cache.add(pid)
                break
        
        # 增量更新实体顶点缓存
        for eid in self.entity_vertices_cache:
            self.entity_vertices_cache[eid].append(pid)
        
        return pid

    def add_new_line(self, 
                    start_pid: str, 
                    end_pid: str, 
                    prefix: str = "L", 
                    line_type: Optional[str] = None,
                    description: Optional[str] = None) -> str:
        # 新增：检查线段数量是否超过上限
        if len(self.lines) >= self.max_lines:
            raise OverflowError(f"线段数量超过上限({self.max_lines})，放弃本轮生成")
        
        # 用哈希表快速判断线段是否已存在
        p1, p2 = start_pid, end_pid
        if p1 > p2:
            p1, p2 = p2, p1
        if (p1, p2) in self.line_pairs:
            for line in self.lines:
                s, e = line["start_point_id"], line["end_point_id"]
                if (s == start_pid and e == end_pid) or (s == end_pid and e == start_pid):
                    if line_type and "type" not in line:
                        line["type"] = line_type
                    if description and "description" not in line:
                        line["description"] = description
                    return line["id"]
        
        lid = self._get_unique_id(prefix)
        new_line = {
            "id": lid,
            "type": line_type if line_type else "line",
            "start_point_id": start_pid,
            "end_point_id": end_pid
        }
        if description:
            new_line["description"] = description
        
        self.lines.append(new_line)
        self.line_id_map[lid] = new_line
        self.line_pairs.add((p1, p2))  # 更新线段哈希表
        
        # 增量更新线上点缓存：添加线段端点
        self.on_segment_points_cache.add(start_pid)
        self.on_segment_points_cache.add(end_pid)
        
        # 增量更新实体线段缓存
        for eid in self.entity_lines_cache:
            self.entity_lines_cache[eid].append(lid)
        
        return lid

    # ------------------------------ 随机选择方法 ------------------------------
    def _randomly_select_entity(self, entity_types: List[str] = None) -> str:
        candidates = []
        for e in self.entities:
            if not entity_types or e["type"] in entity_types:
                candidates.append(e["id"])
        if not candidates:
            raise ValueError("无符合条件的实体")
        return random.choice(candidates)

    def _randomly_select_points(self, 
                            entity_id: str = None, 
                            count: int = 2, 
                            distinct: bool = True, 
                            filter_isolated: bool = True) -> List[str]:
        """选择点，确保返回不重复的点，仅支持过滤孤立点（不允许孤立点补充）"""
        # 1. 获取候选点
        if entity_id:
            candidates = self.entity_vertices_cache[entity_id]
        else:
            candidates = list(self.all_points_cache)
        
        # 2. 过滤孤立点（严格过滤，不允许fallback）
        if filter_isolated:
            candidates = [pid for pid in candidates if self._is_point_on_any_segment(pid)]

        candidates = [pid for pid in candidates if not self._is_expr_too_long(pid)]
        if not candidates:
            raise ValueError(f"无符合条件的点（已过滤表达式过长的点，上限{self.max_expr_length}字符）")
        
        # 3. 检查数量是否足够（不足则直接失败）
        if count > len(candidates):
            raise ValueError(f"线上点数量不足（需要{count}，现有{len(candidates)}）")
        
        # 4. 随机选择（使用sample确保不重复）
        return random.sample(candidates, count)

    def _randomly_select_lines(self, entity_id: str = None, count: int = 2, distinct: bool = True) -> List[str]:
        if entity_id:
            candidates = self.entity_lines_cache[entity_id]
        else:
            candidates = list(self.line_id_map.keys())
        
        if count > len(candidates):
            if entity_id:
                logger.debug(f"实体{entity_id}内线段不足，尝试跨实体选择")
                return self._randomly_select_lines(entity_id=None, count=count, distinct=distinct)
            raise ValueError(f"线段数量不足（需要{count}，现有{len(candidates)}）")
        
        if distinct:
            return random.sample(candidates, count)
        return [random.choice(candidates) for _ in range(count)]

    def _randomly_select_circle_entities(self) -> Tuple[str, str]:
        circle_entities = [e for e in self.entities if e["type"] == "circle"]
        if not circle_entities:
            raise ValueError("无circle类型实体")
        
        valid_circles = []
        for entity in circle_entities:
            for comp_id in entity["components"]:
                if comp_id in self.arc_id_map:
                    circle = self.arc_id_map[comp_id]
                    if ("center_id" in circle and "radius_expr" in circle and 
                        circle["center_id"] in self.point_id_map):
                        valid_circles.append((entity["id"], comp_id))
                        break
        
        if not valid_circles:
            raise ValueError("无包含有效圆心和半径的圆形实体")
        return random.choice(valid_circles)

    # ------------------------------ 操作可行性与选择（动态抽样） ------------------------------
    def _is_operation_feasible(self, op_type: str, constraints: Dict) -> bool:
        try:
            if op_type == "connect_points":
                on_segment_points = list(self.on_segment_points_cache)
                return len(on_segment_points) >= 2
            
            elif op_type == "connect_midpoints":
                lines = list(self.line_id_map.keys())
                on_segment_points = list(self.on_segment_points_cache)
                return (len(lines) >= 2) or (len(lines) >= 1 and len(on_segment_points) >= 1)
            
            elif op_type == "draw_perpendicular":
                on_segment_points = list(self.on_segment_points_cache)
                return len(self.line_id_map) >= 1 and len(on_segment_points) >= 1
            
            elif op_type == "draw_diameter":
                # 适配约束：仅从配置指定的实体类型中选择（默认["circle"]）
                target_entity_types = constraints.get("entity_types", ["circle"])
                circle_entities = [e for e in self.entities if e["type"] in target_entity_types]
                if not circle_entities:
                    return False
                
                for entity in circle_entities:
                    for comp_id in entity["components"]:
                        if comp_id in self.arc_id_map:
                            circle = self.arc_id_map[comp_id]
                            # 确保圆心存在且有效
                            if ("center_id" in circle and "radius" in circle and 
                                circle["center_id"] in self.point_id_map):
                                # 兼容radius字段可能的存储格式（expr或直接值）
                                radius_expr = circle["radius"]["expr"] if isinstance(circle["radius"], dict) else circle["radius"]
                                if radius_expr:
                                    return True
                return False
            
            else:
                return False
        except Exception as e:
            logger.debug(f"操作{op_type}可行性检查失败: {str(e)}")
            return False

    def _select_feasible_operation_type(self, operation_probs: Dict[str, float], constraints: Dict) -> str:
        adjusted_probs = operation_probs.copy()
        
        # 优化1：扩大draw_diameter优先级提升场景（不仅限于仅圆形图形）
        # 只要存在可行的圆形实体，就提高draw_diameter概率（确保能被选中）
        draw_diameter_feasible = self._is_operation_feasible("draw_diameter", constraints.get("draw_diameter", {}))
        if draw_diameter_feasible and "draw_diameter" in adjusted_probs:
            # 调整概率：draw_diameter占比0.4，其余操作按原比例分配剩余0.6
            diameter_prob = 0.4
            other_total = sum(p for k, p in adjusted_probs.items() if k != "draw_diameter")
            if other_total > 0:
                scale = (1 - diameter_prob) / other_total
                for k in adjusted_probs:
                    if k != "draw_diameter":
                        adjusted_probs[k] *= scale
                adjusted_probs["draw_diameter"] = diameter_prob
            else:
                # 仅draw_diameter可行时，概率设为1.0
                adjusted_probs["draw_diameter"] = 1.0
            logger.debug(f"检测到可行的圆形实体，提高draw_diameter操作优先级（概率：{diameter_prob}）")

        feasible_types = []
        feasible_probs = []
        
        # 优化2：去掉失败惩罚系数（删除recent_failures相关逻辑）
        for op_type, prob in adjusted_probs.items():
            op_constraints = constraints.get(op_type, {})
            if self._is_operation_feasible(op_type, op_constraints):
                feasible_types.append(op_type)
                feasible_probs.append(prob)
        
        if not feasible_types:
            raise ValueError("无可行的操作类型")
        
        # 归一化概率
        total = sum(feasible_probs)
        feasible_probs = [p / total for p in feasible_probs]
        return random.choices(feasible_types, weights=feasible_probs, k=1)[0]

    # ------------------------------ 操作生成（严格过滤孤立点） ------------------------------
    def _generate_random_operation(self, op_type: str, op_constraints: Dict) -> Dict:
        if op_type == "connect_points":
            constraints = op_constraints.get("connect_points", {})
            target_entity_types = constraints.get("entity_types", ["polygon", "composite"])
            allow_isolated = constraints.get("allow_isolated", False)
            for retry in range(self.MAX_RETRIES):
                try:
                    entity_id = self._randomly_select_entity(target_entity_types) if (retry < 5) else None
                    p1_id, p2_id = self._randomly_select_points(
                        entity_id=entity_id,
                        count=2,
                        filter_isolated=not allow_isolated
                    )
                    
                    # 严格检查两点间是否已有线段（双向检查，确保无重复）
                    a, b = (p1_id, p2_id) if p1_id < p2_id else (p2_id, p1_id)
                    if (a, b) not in self.line_pairs and (b, a) not in self.line_pairs:
                        logger.debug(f"connect_points重试{retry}次成功：连接点{p1_id}和{p2_id}")
                        return {
                            "type": op_type,
                            "entity_id": entity_id,
                            "point_ids": [p1_id, p2_id]
                        }
                    logger.debug(f"connect_points重试{retry}次失败：点对({p1_id},{p2_id})已存在线段")
                except ValueError as e:
                    logger.debug(f"connect_points重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到可连接的线上点对")
        
        elif op_type == "connect_midpoints":
            constraints = op_constraints.get("connect_midpoints", {})
            target_entity_types = constraints.get("entity_types", ["polygon", "composite"])
            entity_id = self._randomly_select_entity(target_entity_types) if (target_entity_types and random.random() < 0.7) else None
            
            lines = self.entity_lines_cache.get(entity_id, list(self.line_id_map.keys())) if entity_id else list(self.line_id_map.keys())
            lines = list(set(lines))
            
            on_segment_vertices = [pid for pid in self.all_points_cache if self._is_point_on_any_segment(pid)]
            
            available_modes = []
            if len(lines) >= 2:
                available_modes.append("two_midpoints")
            if len(lines) >= 1 and len(on_segment_vertices) >= 1:
                available_modes.append("vertex_and_midpoint")
            if not available_modes:
                raise ValueError("无可用的connect_midpoints模式（线上点不足）")
            
            mode_probs = constraints.get("mode_probs", {"two_midpoints": 0.5, "vertex_and_midpoint": 0.5})
            mode = random.choices(
                available_modes,
                weights=[mode_probs[mode] for mode in available_modes],
                k=1
            )[0]

            if mode == "two_midpoints":
                for retry in range(self.MAX_RETRIES):
                    try:
                        line1_id, line2_id = random.sample(lines, 2)
                        if line1_id == line2_id:
                            continue
                            
                        s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
                        x1, y1 = self.get_point_coords(s1)
                        x2, y2 = self.get_point_coords(e1)
                        mid1_x = (x1 + x2) / 2
                        mid1_y = (y1 + y2) / 2
                        
                        s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
                        x3, y3 = self.get_point_coords(s2)
                        x4, y4 = self.get_point_coords(e2)
                        mid2_x = (x3 + x4) / 2
                        mid2_y = (y3 + y4) / 2
                        
                        temp_mid1_id = f"temp_mid1_{retry}"
                        temp_mid2_id = f"temp_mid2_{retry}"
                        self.point_id_map[temp_mid1_id] = {"x": {"expr": str(mid1_x)}, "y": {"expr": str(mid1_y)}}
                        self.point_id_map[temp_mid2_id] = {"x": {"expr": str(mid2_x)}, "y": {"expr": str(mid2_y)}}
                        
                        mid1_on_other = self._is_point_on_other_edge(temp_mid1_id, line1_id)
                        mid2_on_other = self._is_point_on_other_edge(temp_mid2_id, line2_id)
                        
                        del self.point_id_map[temp_mid1_id]
                        del self.point_id_map[temp_mid2_id]
                        
                        if mid1_on_other or mid2_on_other:
                            logger.debug(f"connect_midpoints重试{retry}次失败：中点在另一条边上")
                            continue
                            
                        logger.debug(f"connect_midpoints（two_midpoints）重试{retry}次成功")
                        return {
                            "type": op_type,
                            "mode": mode,
                            "entity_id": entity_id,
                            "line_ids": [line1_id, line2_id]
                        }
                    except ValueError as e:
                        logger.debug(f"connect_midpoints重试{retry}次失败：{str(e)}")
                    
                    raise ValueError(f"超过最大重试次数，未找到可连接中点的线段对")
            
            else:  # mode == "vertex_and_midpoint"
                allow_endpoint = constraints.get("allow_endpoint", False)
                for retry in range(self.MAX_RETRIES):
                    try:
                        line_id = random.choice(lines)
                        vertex_id = random.choice(on_segment_vertices)
                        line = self.line_id_map[line_id]
                        
                        is_endpoint = (vertex_id == line["start_point_id"]) or (vertex_id == line["end_point_id"])
                        
                        if allow_endpoint or not is_endpoint:
                            s, e = line["start_point_id"], line["end_point_id"]
                            x1, y1 = self.get_point_coords(s)
                            x2, y2 = self.get_point_coords(e)
                            mid_x = (x1 + x2) / 2
                            mid_y = (y1 + y2) / 2
                            
                            temp_mid_id = f"temp_mid_{retry}"
                            self.point_id_map[temp_mid_id] = {"x": {"expr": str(mid_x)}, "y": {"expr": str(mid_y)}}
                            
                            vertex_on_other = self._is_point_on_other_edge(vertex_id, line_id)
                            mid_on_other = self._is_point_on_other_edge(temp_mid_id, line_id)
                            
                            del self.point_id_map[temp_mid_id]
                            
                            if vertex_on_other or mid_on_other:
                                logger.debug(f"connect_midpoints重试{retry}次失败：顶点或中点在另一条边上")
                                continue
                                
                            logger.debug(f"connect_midpoints（vertex_and_midpoint）重试{retry}次成功")
                            return {
                                "type": op_type,
                                "mode": mode,
                                "entity_id": entity_id,
                                "line_id": line_id,
                                "vertex_id": vertex_id
                            }
                        else:
                            logger.debug(f"connect_midpoints重试{retry}次失败：顶点是端点且不允许")
                    except Exception as e:
                        logger.debug(f"connect_midpoints重试{retry}次失败：{str(e)}")
                    
                    raise ValueError(f"超过最大重试次数，未找到可连接的顶点和线段")
            
        elif op_type == "draw_perpendicular":
            constraints = op_constraints.get("draw_perpendicular", {})
            target_entity_types = constraints.get("entity_types", ["polygon", "composite"])
            
            # 辅助函数：判断点是否在直线（含延长线）上
            def is_point_on_line(px, py, s_x, s_y, e_x, e_y):
                # 向量叉积为0则共线（点在直线上）
                cross = (e_x - s_x) * (py - s_y) - (e_y - s_y) * (px - s_x)
                return sp.simplify(cross) == 0
            
            # 辅助函数：计算垂足坐标
            def calculate_foot_of_perpendicular(px, py, s_x, s_y, e_x, e_y):
                # 线段向量
                se_x = e_x - s_x
                se_y = e_y - s_y
                # 点到起点向量
                sp_x = px - s_x
                sp_y = py - s_y
                # 投影参数 t = (sp · se) / |se|²
                dot_product = sp_x * se_x + sp_y * se_y
                se_sq = se_x**2 + se_y**2
                t = dot_product / se_sq
                # 垂足坐标
                foot_x = s_x + t * se_x
                foot_y = s_y + t * se_y
                return sp.simplify(foot_x), sp.simplify(foot_y)
            
            # 辅助函数：检查是否存在与目标坐标重合的点
            def find_matching_point(target_x, target_y):
                for pid, p in self.point_id_map.items():
                    px = self._parse_expr(p["x"]["expr"])
                    py = self._parse_expr(p["y"]["expr"])
                    if sp.simplify(px == target_x) and sp.simplify(py == target_y):
                        return pid
                return None
            
            for retry in range(self.MAX_RETRIES):
                try:
                    entity_id = self._randomly_select_entity(target_entity_types) if (retry < 5) else None
                    lines = self.entity_lines_cache.get(entity_id, list(self.line_id_map.keys())) if entity_id else list(self.line_id_map.keys())
                    line_id = random.choice(lines)
                    line = self.line_id_map[line_id]
                    
                    point_id = self._randomly_select_points(
                        entity_id=entity_id,
                        count=1,
                        filter_isolated=True
                    )[0]
                    
                    # 获取点和线段坐标
                    px, py = self.get_point_coords(point_id)
                    s, e = line["start_point_id"], line["end_point_id"]
                    s_x, s_y = self.get_point_coords(s)
                    e_x, e_y = self.get_point_coords(e)
                    
                    # 校验1：点是否在直线（含延长线）上（在则作垂线无意义）
                    if is_point_on_line(px, py, s_x, s_y, e_x, e_y):
                        logger.debug(f"draw_perpendicular重试{retry}次失败：点{point_id}在直线{line_id}（含延长线）上")
                        continue
                    
                    # 计算垂足
                    foot_x, foot_y = calculate_foot_of_perpendicular(px, py, s_x, s_y, e_x, e_y)
                    
                    # 校验2：垂足是否已存在
                    foot_pid = find_matching_point(foot_x, foot_y)
                    if foot_pid:
                        # 校验3：垂足与点之间是否已有线段
                        a, b = (point_id, foot_pid) if point_id < foot_pid else (foot_pid, point_id)
                        if (a, b) in self.line_pairs or (b, a) in self.line_pairs:
                            logger.debug(f"draw_perpendicular重试{retry}次失败：垂足{foot_pid}与点{point_id}已存在线段")
                            continue
                    
                    logger.debug(f"draw_perpendicular重试{retry}次成功")
                    return {
                        "type": op_type,
                        "line_id": line_id,
                        "point_id": point_id,
                        "foot_coords": (foot_x, foot_y)  # 附加垂足坐标供后续使用
                    }
                except Exception as e:
                    logger.debug(f"draw_perpendicular重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到合适的垂线参数")
        
        elif op_type == "draw_diameter":
            # 保持原有逻辑
            constraints = op_constraints.get("draw_diameter", {})
            target_entity_types = constraints.get("entity_types", ["circle"])
            directions = constraints.get("directions", ["horizontal", "vertical"])
            for retry in range(self.MAX_RETRIES):
                try:
                    entity_id, circle_id = self._randomly_select_circle_entities(target_entity_types)
                    circle = self.arc_id_map[circle_id]
                    if "center_id" in circle and circle["center_id"] in self.point_id_map:
                        direction = random.choice(directions)
                        logger.debug(f"draw_diameter重试{retry}次成功")
                        return {
                            "type": op_type,
                            "entity_id": entity_id,
                            "circle_id": circle_id,
                            "center_id": circle["center_id"],
                            "radius_expr": circle["radius_expr"],
                            "direction": direction
                        }
                except Exception as e:
                    logger.debug(f"draw_diameter重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到有效圆形")
        
        else:
            raise ValueError(f"不支持的操作类型: {op_type}")

    # ------------------------------ 交点检测与几何计算（全符号） ------------------------------
    def _point_on_segment(self, px: sp.Expr, py: sp.Expr, 
                        x1: sp.Expr, y1: sp.Expr, 
                        x2: sp.Expr, y2: sp.Expr) -> bool:
        """检查点(px, py)是否在线段(x1,y1)-(x2,y2)上"""
        collinear = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        collinear_simplified = sp.simplify(collinear)
        if not sp.Eq(collinear_simplified, 0):
            return False
        
        x_min = sp.Min(x1, x2)
        x_max = sp.Max(x1, x2)
        y_min = sp.Min(y1, y2)
        y_max = sp.Max(y1, y2)
        
        x_in_range = sp.And(sp.Ge(px, x_min), sp.Le(px, x_max))
        y_in_range = sp.And(sp.Ge(py, y_min), sp.Le(py, y_max))
        
        on_segment = sp.And(x_in_range, y_in_range)
        simplified = sp.simplify(on_segment)
        
        return bool(simplified) if isinstance(simplified, bool) else False

    def _line_arc_intersection(self, line_id: str, arc_id: str) -> List[Tuple[sp.Expr, sp.Expr]]:
        """计算线段与圆弧的交点"""
        arc = self.arc_id_map[arc_id]
        if "center_id" not in arc or "radius_expr" not in arc:
            return []
        
        center_id = arc["center_id"]
        cx, cy = self.get_point_coords(center_id)
        radius = self._parse_expr(arc["radius_expr"])
        
        s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s)
        x2, y2 = self.get_point_coords(e)
        
        t = sp.Symbol('t')
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        circle_eq = (x - cx)**2 + (y - cy)** 2 - radius**2
        circle_eq = sp.simplify(circle_eq)
        
        solutions = sp.solve(circle_eq, t)
        intersections = []
        
        for sol in solutions:
            if isinstance(sol, dict):
                continue
            t_val = sp.simplify(sol)
            if sp.im(t_val) != 0:
                continue
            t_val = sp.re(t_val)
            
            if sp.simplify(t_val >= 0) and sp.simplify(t_val <= 1):
                ix = sp.simplify(x.subs(t, t_val))
                iy = sp.simplify(y.subs(t, t_val))
                
                if "start_angle" in arc and "end_angle" in arc:
                    start_angle = self._parse_expr(arc["start_angle"])
                    end_angle = self._parse_expr(arc["end_angle"])
                    dx = ix - cx
                    dy = iy - cy
                    angle = sp.atan2(dy, dx)
                    if sp.simplify(start_angle <= end_angle):
                        in_angle_range = (angle >= start_angle) & (angle <= end_angle)
                    else:
                        in_angle_range = (angle >= start_angle) | (angle <= end_angle)
                    if not sp.simplify(in_angle_range):
                        continue
                
                intersections.append((ix, iy))
        
        return intersections

    def detect_new_intersections(self, new_line_id: str) -> List[str]:
        new_ips = []
        new_line = self.line_id_map[new_line_id]
        s_new, e_new = new_line["start_point_id"], new_line["end_point_id"]
        x1_new, y1_new = self.get_point_coords(s_new)
        x2_new, y2_new = self.get_point_coords(e_new)
        
        new_min_x = Min(x1_new, x2_new)
        new_max_x = Max(x1_new, x2_new)
        new_min_y = Min(y1_new, y2_new)
        new_max_y = Max(y1_new, y2_new)
        
        # 检测与现有线段的交点
        for line_id in self.line_id_map:
            if line_id == new_line_id:
                continue
            line = self.line_id_map[line_id]
            s, e = line["start_point_id"], line["end_point_id"]
            x1, y1 = self.get_point_coords(s)
            x2, y2 = self.get_point_coords(e)
            
            if (Max(x1, x2) < new_min_x) or (Min(x1, x2) > new_max_x) or \
               (Max(y1, y2) < new_min_y) or (Min(y1, y2) > new_max_y):
                continue
            
            intersections = self._line_intersection(new_line_id, line_id)
            for x, y in intersections:
                x_simplified = sp.simplify(x)
                y_simplified = sp.simplify(y)
                pid = self.add_new_point(x_simplified, y_simplified, prefix="I", point_type="intersection")
                new_ips.append(pid)
        
        # 检测与现有圆弧的交点
        for arc_id in self.arc_id_map:
            arc = self.arc_id_map[arc_id]
            if "center_id" in arc and "radius_expr" in arc:
                cx, cy = self.get_point_coords(arc["center_id"])
                r = self._parse_expr(arc["radius_expr"])
                arc_min_x = cx - r
                arc_max_x = cx + r
                arc_min_y = cy - r
                arc_max_y = cy + r
                
                if (arc_max_x < new_min_x) or (arc_min_x > new_max_x) or \
                   (arc_max_y < new_min_y) or (arc_min_y > new_max_y):
                    continue
            
            intersections = self._line_arc_intersection(new_line_id, arc_id)
            for x, y in intersections:
                x_simplified = sp.simplify(x)
                y_simplified = sp.simplify(y)
                pid = self.add_new_point(x_simplified, y_simplified, prefix="I", point_type="arc_intersection")
                new_ips.append(pid)
        
        return new_ips

    def _line_intersection(self, line1_id: str, line2_id: str) -> List[Tuple[sp.Expr, sp.Expr]]:
        s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s1)
        x2, y2 = self.get_point_coords(e1)
        
        s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
        x3, y3 = self.get_point_coords(s2)
        x4, y4 = self.get_point_coords(e2)
        
        def points_equal(p1: Tuple[sp.Expr, sp.Expr], p2: Tuple[sp.Expr, sp.Expr]) -> bool:
            return sp.Eq(p1[0], p2[0]) and sp.Eq(p1[1], p2[1])
        
        if (points_equal((x1, y1), (x3, y3)) or points_equal((x1, y1), (x4, y4)) or
            points_equal((x2, y2), (x3, y3)) or points_equal((x2, y2), (x4, y4))):
            return []
        
        t, s = symbols('t s')
        eq1 = Eq(x1 + t*(x2 - x1), x3 + s*(x4 - x3))
        eq2 = Eq(y1 + t*(y2 - y1), y3 + s*(y4 - y3))
        
        solutions = solve((eq1, eq2), (t, s), dict=True)
        intersections = []
        for sol in solutions:
            if t not in sol or s not in sol:
                continue
            
            t_val = sol[t]
            s_val = sol[s]
            
            t_valid = Ge(t_val, 0) and Le(t_val, 1)
            s_valid = Ge(s_val, 0) and Le(s_val, 1)
            
            if t_valid and s_valid:
                x = simplify(x1 + t_val*(x2 - x1))
                y = simplify(y1 + t_val*(y2 - y1))
                intersections.append((x, y))
        
        return intersections

    def _calculate_foot_of_perpendicular(self, point_id: str, line_id: str) -> Tuple[sp.Expr, sp.Expr, bool]:
        px, py = self.get_point_coords(point_id)
        s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s)
        x2, y2 = self.get_point_coords(e)
        
        dx = x2 - x1
        dy = y2 - y1
        px_vec = px - x1
        py_vec = py - y1
        
        if sp.Eq(dx**2 + dy**2, 0):
            return (x1, y1, True)
        
        t = (px_vec * dx + py_vec * dy) / (dx**2 + dy**2)
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy
        on_segment = Ge(t, 0) and Le(t, 1)
        return (foot_x, foot_y, bool(on_segment))

    # ------------------------------ 操作执行（全符号计算） ------------------------------
    def execute_operation(self, op: Dict) -> Dict:
        op_type = op["type"]
        result = {"operation": op_type, "details": op, "new_elements": []}
        
        if op_type == "connect_points":
            p1_id, p2_id = op["point_ids"]
            desc = f"Connect points {p1_id} and {p2_id} (both on segments)"
            line_id = self.add_new_line(p1_id, p2_id, prefix="ConnP", line_type="connection", description=desc)
            new_ips = self.detect_new_intersections(line_id)
            result["new_elements"].append(("line", line_id))
            result["new_elements"].extend([("point", pid) for pid in new_ips])
            result["description"] = desc
        
        elif op_type == "connect_midpoints":
            mode = op["mode"]
            
            if mode == "two_midpoints":
                line1_id, line2_id = op["line_ids"]
                s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
                x1, y1 = self.get_point_coords(s1)
                x2, y2 = self.get_point_coords(e1)
                mid1_x = (x1 + x2) / 2
                mid1_y = (y1 + y2) / 2
                mid1_id = self.add_new_point(mid1_x, mid1_y, prefix="M", point_type="midpoint", related_edge=line1_id)
                
                s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
                x3, y3 = self.get_point_coords(s2)
                x4, y4 = self.get_point_coords(e2)
                mid2_x = (x3 + x4) / 2
                mid2_y = (y3 + y4) / 2
                mid2_id = self.add_new_point(mid2_x, mid2_y, prefix="M", point_type="midpoint", related_edge=line2_id)
                
                desc = f"Connect midpoints {mid1_id} (of line {line1_id}) and {mid2_id} (of line {line2_id})"
                line_id = self.add_new_line(mid1_id, mid2_id, prefix="MidL", line_type="midline", description=desc)
                new_ips = self.detect_new_intersections(line_id)
                result["new_elements"].extend([("point", mid1_id), ("point", mid2_id)])
                result["new_elements"].append(("line", line_id))
                result["new_elements"].extend([("point", pid) for pid in new_ips])
                result["description"] = desc
            
            else:  # mode == "vertex_and_midpoint"
                line_id = op["line_id"]
                vertex_id = op["vertex_id"]
                
                s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
                x1, y1 = self.get_point_coords(s)
                x2, y2 = self.get_point_coords(e)
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                mid_id = self.add_new_point(mid_x, mid_y, prefix="M", point_type="midpoint", related_edge=line_id)
                
                desc = f"Connect vertex {vertex_id} with midpoint {mid_id} (of line {line_id})"
                line_id = self.add_new_line(vertex_id, mid_id, prefix="VMidL", line_type="vertex_midline", description=desc)
                new_ips = self.detect_new_intersections(line_id)
                result["new_elements"].append(("point", mid_id))
                result["new_elements"].append(("line", line_id))
                result["new_elements"].extend([("point", pid) for pid in new_ips])
                result["description"] = desc
        
        elif op_type == "draw_perpendicular":
            target_line_id = op["line_id"]
            start_point_id = op["point_id"]
            allow_on_segment = op.get("allow_on_segment", False)
            
            foot_x, foot_y, on_segment = self._calculate_foot_of_perpendicular(start_point_id, target_line_id)
            
            foot_id = self.add_new_point(
                foot_x, foot_y, 
                prefix="F",
                point_type="perpendicular_foot",
                related_vertex=start_point_id,
                related_edge=target_line_id
            )
            
            new_elements = [("point", foot_id)]
            if not on_segment:
                s, e = self.line_id_map[target_line_id]["start_point_id"], self.line_id_map[target_line_id]["end_point_id"]
                s_dist = (foot_x - self.get_point_coords(s)[0])**2 + (foot_y - self.get_point_coords(s)[1])** 2
                e_dist = (foot_x - self.get_point_coords(e)[0])**2 + (foot_y - self.get_point_coords(e)[1])** 2
                extend_from_id = s if s_dist < e_dist else e
                
                extend_desc = f"Extend line {target_line_id} to perpendicular foot {foot_id}"
                extend_line_id = self.add_new_line(
                    extend_from_id, foot_id,
                    prefix="ExtL",
                    line_type="extension",
                    description=extend_desc
                )
                new_elements.append(("line", extend_line_id))
                desc = f"Draw perpendicular from point {start_point_id} to line {target_line_id} (foot {foot_id} on extension)"
            else:
                desc = f"Draw perpendicular from point {start_point_id} to line {target_line_id} (foot {foot_id} on segment)"
            
            perp_line_id = self.add_new_line(
                start_point_id, foot_id, 
                prefix="PerpL", 
                line_type="perpendicular",
                description=desc
            )
            new_elements.append(("line", perp_line_id))
            
            new_ips = self.detect_new_intersections(perp_line_id)
            if not on_segment:
                new_ips.extend(self.detect_new_intersections(extend_line_id))
            new_elements.extend([("point", pid) for pid in new_ips])
            
            result["new_elements"] = new_elements
            result["description"] = desc
        
        elif op_type == "draw_diameter":
            circle_id = op["circle_id"]
            center_id = op["center_id"]
            radius_expr = op["radius_expr"]
            direction = op["direction"]
            center_x, center_y = self.get_point_coords(center_id)
            radius = self._parse_expr(radius_expr)
            
            if direction == "horizontal":
                end1_x = center_x + radius
                end1_y = center_y
                end2_x = center_x - radius
                end2_y = center_y
            else:  # vertical
                end1_x = center_x
                end1_y = center_y + radius
                end2_x = center_x
                end2_y = center_y - radius
            
            end1_id = self.add_new_point(end1_x, end1_y, prefix="Diam", point_type="diameter_end")
            end2_id = self.add_new_point(end2_x, end2_y, prefix="Diam", point_type="diameter_end")
            
            desc = f"Draw {direction} diameter of circle {circle_id} with endpoints {end1_id} and {end2_id}"
            diam_line_id = self.add_new_line(
                end1_id, end2_id, 
                prefix="DiamL", 
                line_type="diameter",
                description=desc
            )
            
            new_ips = self.detect_new_intersections(diam_line_id)
            result["new_elements"].extend([("point", end1_id), ("point", end2_id)])
            result["new_elements"].append(("line", diam_line_id))
            result["new_elements"].extend([("point", pid) for pid in new_ips])
            result["description"] = desc
        
        return result
    
    # ------------------------------ 增强图形生成 ------------------------------
    def generate_enhancements(self, config: Dict) -> List[Dict]:
        """
        生成增强图形（严格遵循 rounds_distribution，如{"0":1,"1":3} → 4个独立图形）
        直接调用 run_rounds 处理完整分布，避免重复解析
        """
        # 校验核心配置
        if "rounds_distribution" not in config:
            raise ValueError("config 必须包含 rounds_distribution 配置")
        
        # 计算总增强数量（sum(分布值)）
        rounds_dist = {int(k): int(v) for k, v in config["rounds_distribution"].items()}
        num_enhancements = sum(rounds_dist.values())
        
        # 构造 run_rounds 所需完整配置（一次性传入完整分布）
        run_config = {
            **config,
            "num_enhancements": num_enhancements,  # 总结果数=分布总和
            "single_enhance_timeout": config.get("single_enhance_timeout", 30)
        }
        
        # 调用 run_rounds 生成所有增强结果（4个：1个0轮+3个1轮）
        all_results = self.run_rounds(run_config)
        
        # 整理结果：提取 final_geometry 并补充标识
        enhancements = []
        for result in all_results:
            if not isinstance(result, dict) or "final_geometry" not in result:
                logger.warning(f"跳过无效增强结果: {result}")
                continue
            
            final_geo = result["final_geometry"]
            # 补充核心标识（与全流程一致）
            final_geo["is_base"] = (result["execution_summary"] == [])  # 0轮为基础图形
            final_geo["timeout_occurred"] = result.get("timeout_occurred", False)
            final_geo["completed_rounds"] = len(result.get("execution_summary", []))
            final_geo["base_id"] = self.base_id  # 关联原始基础图形ID
            final_geo["enhance_id"] = f"{self.base_id}_enhance_{result['enhance_idx']:03d}"  # 唯一标识
            
            enhancements.append(final_geo)
        
        logger.info(f"生成 {len(enhancements)} 个独立增强图形（预期 {num_enhancements} 个）")
        return enhancements

    def run_rounds(self, config: Dict) -> List[Dict]:
        """生成所有增强结果（确保操作产生的新元素被写入self.data）"""
        rounds_distribution = config["rounds_distribution"]
        min_ops_per_round = config["min_operations_per_round"]
        max_ops_per_round = config["max_operations_per_round"]
        num_enhancements = config["num_enhancements"]
        single_enhance_timeout = config.get("single_enhance_timeout", 30)

        # 解析轮数分布为列表
        try:
            rounds_list = []
            for round_key, count in rounds_distribution.items():
                round_num = int(round_key)
                rounds_list.extend([round_num] * count)
        except ValueError as e:
            raise ValueError(f"轮数分布键必须为整数（如'0','1'）：{str(e)}") from e

        if len(rounds_list) != num_enhancements:
            raise ValueError(f"轮数分布总和（{len(rounds_list)}）与 num_enhancements（{num_enhancements}）不匹配")

        all_enhance_results = []
        # 深拷贝原始数据（确保每次增强从原始状态开始）
        original_data = json.loads(json.dumps(self.data))
        original_desc = original_data.get("description", "无原始描述")
        base_seed = config.get("seed", 42)

        for enh_idx, num_rounds in enumerate(rounds_list):
            # 每个增强结果独立种子（避免重复）
            enh_seed = base_seed + enh_idx * 100
            random.seed(enh_seed)
            
            # 关键修复1：重置为原始数据后，重新绑定 self.points/self.lines/self.arcs 引用
            self.data = json.loads(json.dumps(original_data))
            # 重新绑定到新 self.data 的列表（必须做！否则新增元素写不到 self.data 里）
            self.points = self.data["points"]
            self.lines = self.data["lines"]
            self.arcs = self.data.get("arcs", [])
            # 重新初始化缓存和索引（确保新增元素能被正确识别）
            self.point_id_map = {p["id"]: p for p in self.points}
            self.line_id_map = {l["id"]: l for l in self.lines}
            self.arc_id_map = {a["id"]: a for a in self.arcs}
            self.line_pairs = self._build_line_pairs()
            self.on_segment_points_cache = set(self._cache_on_segment_points())
            self.all_points_cache = set(self.point_id_map.keys())
            
            self.enhancement_history = []
            self.operation_counter = 0

            enhancement_desc = [
                f"原始图形描述: {original_desc}",
                f"增强结果序号: {enh_idx + 1}/{num_enhancements}",
                f"本结果操作轮数: {num_rounds}"
            ]

            timeout_occurred = False
            round_results = []
            start_time = time.time()

            if num_rounds == 0:
                enhancement_desc.append("未执行增强操作（轮数为0）")
                self.data["description"] = "\n".join(enhancement_desc)
                all_enhance_results.append({
                    "final_geometry": self.data,
                    "execution_summary": round_results,
                    "timeout_occurred": timeout_occurred,
                    "enhance_idx": enh_idx
                })
                continue

            # 确保有可行操作
            try:
                self._select_feasible_operation_type(config["operation_probs"], config["operation_constraints"])
            except ValueError as e:
                logger.warning(f"增强结果{enh_idx + 1}无可行操作，跳过")
                continue

            # 执行轮数操作
            for round_idx in range(num_rounds):
                if time.time() - start_time > single_enhance_timeout:
                    timeout_occurred = True
                    logger.warning(f"增强结果{enh_idx + 1}超时，已完成{round_idx}/{num_rounds}轮")
                    break

                logger.info(f"增强结果{enh_idx + 1}：执行第 {round_idx + 1}/{num_rounds} 轮操作")
                num_ops = random.randint(min_ops_per_round, max_ops_per_round)
                round_result = {"round": round_idx, "operations": []}
                round_desc = [f"第 {round_idx + 1} 轮操作（共 {num_ops} 步）:"]

                for op_idx in range(num_ops):
                    if time.time() - start_time > single_enhance_timeout:
                        timeout_occurred = True
                        break

                    # 操作重试机制
                    for op_retry in range(self.MAX_RETRIES):
                        try:
                            # 1. 选择操作类型
                            op_type = self._select_feasible_operation_type(
                                config["operation_probs"], config["operation_constraints"]
                            )
                            # 2. 生成操作参数
                            op = self._generate_random_operation(op_type, config["operation_constraints"].get(op_type, {}))
                            # 3. 执行操作（add_new_line/add_new_point 会直接修改 self.lines/self.points）
                            op_result = self.execute_operation(op)

                            # 关键修复2：读取正确的 new_elements 键（而非不存在的 new_lines 等）
                            # 无需手动追加，因为 self.lines 是 self.data["lines"] 的引用，add_new_line 已修改
                            logger.debug(f"操作{op_type}成功，新增元素：{op_result['new_elements']}")

                            # 记录操作历史
                            round_result["operations"].append(op_result)
                            round_desc.append(f"  第 {op_idx + 1} 步: {op_result['description']}")
                            self.enhancement_history.append(op_result)
                            break  # 操作成功，退出重试
                        except OverflowError as e:
                            logger.warning(f"操作触发数量上限：{str(e)}，终止本轮")
                            timeout_occurred = True
                            break
                        except ValueError as e:
                            logger.debug(f"操作重试{op_retry}次失败：{str(e)}")
                    else:
                        logger.warning(f"操作超过最大重试次数，终止生成")
                        timeout_occurred = True
                        break

                    if timeout_occurred:
                        break

                if timeout_occurred:
                    break

                round_results.append(round_result)
                enhancement_desc.extend(round_desc)

            # 更新复合实体（整合新元素）
            self._update_composite_entity()
            if timeout_occurred:
                enhancement_desc.append(f"\n超时（{single_enhance_timeout}秒），保留部分操作")
            self.data["description"] = "\n".join(enhancement_desc)

            all_enhance_results.append({
                "final_geometry": self.data,
                "execution_summary": round_results,
                "timeout_occurred": timeout_occurred,
                "enhance_idx": enh_idx
            })

        return all_enhance_results

    def _update_composite_entity(self):
        composite_id = "enhanced_composite"
        if composite_id not in self.entity_id_map:
            self.entities.append({
                "id": composite_id,
                "type": "composite",
                "components": []
            })
            self.entity_id_map[composite_id] = self.entities[-1]
        
        composite = self.entity_id_map[composite_id]
        
        all_components = []
        for p in self.points:
            if p["id"] not in composite["components"]:
                composite["components"].append(p["id"])
        for l in self.lines:
            if l["id"] not in composite["components"]:
                composite["components"].append(l["id"])
        for a in self.arcs:
            if a["id"] not in composite["components"]:
                composite["components"].append(a["id"])
        
        composite["components"] = all_components

        self.entity_vertices_cache[composite_id] = [
            c for c in all_components if c in self.point_id_map
        ]

    @property
    def base_json(self) -> Dict:
        return {
            "points": self.points.copy(),
            "lines": self.lines.copy(),
            "arcs": self.arcs.copy(),
            "entities": self.entities.copy(),
            "description": self.data.get("description", "")
        }