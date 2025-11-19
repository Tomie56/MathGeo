import json
import random
import sympy as sp
import logging
from sympy import symbols, cos, sin, pi, simplify, sqrt, tan, Eq, solve, Min, Max, Ge, Le, Rational, ask, Q
from typing import List, Dict, Tuple, Optional, Union, Set, Generator
import os
import threading
import time

logger = logging.getLogger('builder')


class RandomGeometryBuilder:
    def __init__(self, base_json: Dict, base_id: str = "", max_points: int = 500, max_lines: int = 300):
        """初始化builder，加载基础图形数据并添加level属性"""
        self.data = base_json.copy()
        self.x, self.y = symbols('x y')
        self.max_points = max_points 
        self.max_lines = max_lines 
        
        self.max_expr_length = 50
        
        # 初始化并补全level属性（默认基础元素level=1）
        self.points = self.data["points"]
        for p in self.points:
            if "level" not in p:
                p["level"] = 1
        
        self.lines = self.data["lines"]
        for l in self.lines:
            if "level" not in l:
                l["level"] = 1
        
        self.arcs = self.data.get("arcs", [])
        for a in self.arcs:
            if "level" not in a:
                a["level"] = 1
        
        self.entities = self.data["entities"]
        
        self.point_id_map = {p["id"]: p for p in self.points}
        self.line_id_map = {l["id"]: l for l in self.lines}
        self.arc_id_map = {a["id"]: a for a in self.arcs}
        self.entity_id_map = {e["id"]: e for e in self.entities}
        
        self.operation_counter = 0
        self.entity_vertices_cache = self._cache_entity_vertices()
        self.entity_lines_cache = self._cache_entity_lines()
        self.entity_arcs_cache = self._cache_entity_arcs()
        self.on_segment_points_cache: Set[str] = set(self._cache_on_segment_points()) 
        self.all_points_cache: Set[str] = set(self.point_id_map.keys())
        self.line_pairs: Set[Tuple[str, str]] = self._build_line_pairs()
        
        self.enhancement_history = []
        
        self.MAX_RETRIES = 4
        self.base_id = base_id
        self.only_circles = self._is_only_circles()

        if len(self.points) >= self.max_points:
            raise ValueError(f"初始点数量({len(self.points)})超过上限({self.max_points})")
        if len(self.lines) >= self.max_lines:
            raise ValueError(f"初始线段数量({len(self.lines)})超过上限({self.max_lines})")

    # ------------------------------ 基础工具方法 ------------------------------
    
    def _is_only_circles(self) -> bool:
        if len(self.entities) == 0:
            return False
        all_circles = all(e["type"] == "circle" for e in self.entities)
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
        on_segment = set()
        for line in self.lines:
            on_segment.add(line["start_point_id"])
            on_segment.add(line["end_point_id"])
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
                    break
        return list(on_segment)

    def _build_line_pairs(self) -> Set[Tuple[str, str]]:
        pairs = set()
        for line in self.lines:
            p1, p2 = line["start_point_id"], line["end_point_id"]
            if p1 > p2:
                p1, p2 = p2, p1
            pairs.add((p1, p2))
        return pairs
    
    def _is_expr_too_long(self, pid: str) -> bool:
        p = self.point_id_map[pid]
        x_expr = p["x"]["expr"]
        y_expr = p["y"]["expr"]
        return len(x_expr) > self.max_expr_length or len(y_expr) > self.max_expr_length

    def _is_point_on_any_segment(self, pid: str) -> bool:
        return pid in self.on_segment_points_cache
    
    def _is_point_on_other_edge(self, point_id: str, exclude_line_id: str) -> bool:
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
        return sp.sympify(expr_str)

    def _serialize_expr(self, expr: sp.Expr) -> str:
        if expr.has(sp.sin, sp.cos, sp.sqrt, sp.tan):
            expr = simplify(expr)
        return str(expr)

    # ------------------------------ 点和线段操作（含level维护） ------------------------------
    def get_point_coords(self, pid: str) -> Tuple[sp.Expr, sp.Expr]:
        p = self.point_id_map[pid]
        return self._parse_expr(p["x"]["expr"]), self._parse_expr(p["y"]["expr"])

    def add_new_point(self, 
                     x_expr: sp.Expr, 
                     y_expr: sp.Expr, 
                     prefix: str = "P",
                     point_type: Optional[str] = None,
                     related_vertex: Optional[str] = None,
                     related_edge: Optional[str] = None,
                     level: int = 1) -> str:  # 新增level参数
        
        if len(self.points) >= self.max_points:
            raise OverflowError(f"点数量超过上限({self.max_points})，放弃本轮生成")
        
        x_simplified = x_expr
        y_simplified = y_expr
        x_str = self._serialize_expr(x_simplified)
        y_str = self._serialize_expr(y_simplified)
        
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
                # 重合点取最高level
                if p["level"] < level:
                    p["level"] = level
                return p["id"]
        
        pid = self._get_unique_id(prefix)
        new_point = {
            "id": pid,
            "x": {"expr": x_str, "latex": sp.latex(x_simplified)},
            "y": {"expr": y_str, "latex": sp.latex(y_simplified)},
            "level": level  # 存储level
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
        
        px, py = x_simplified, y_simplified
        for line in self.lines:
            s_id, e_id = line["start_point_id"], line["end_point_id"]
            s_x, s_y = self.get_point_coords(s_id)
            e_x, e_y = self.get_point_coords(e_id)
            if self._point_on_segment(px, py, s_x, s_y, e_x, e_y):
                self.on_segment_points_cache.add(pid)
                break
        
        for eid in self.entity_vertices_cache:
            self.entity_vertices_cache[eid].append(pid)
        
        return pid

    def add_new_line(self, 
                    start_pid: str, 
                    end_pid: str, 
                    prefix: str = "L", 
                    line_type: Optional[str] = None,
                    description: Optional[str] = None,
                    level: int = 1) -> str:  # 新增level参数
        
        if len(self.lines) >= self.max_lines:
            raise OverflowError(f"线段数量超过上限({self.max_lines})，放弃本轮生成")
        
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
                    # 重合线取最高level
                    if line["level"] < level:
                        line["level"] = level
                    return line["id"]
        
        lid = self._get_unique_id(prefix)
        new_line = {
            "id": lid,
            "type": line_type if line_type else "line",
            "start_point_id": start_pid,
            "end_point_id": end_pid,
            "level": level  # 存储level
        }
        if description:
            new_line["description"] = description
        
        self.line_id_map[lid] = new_line
        
        # 立即验证映射是否成功
        if self.line_id_map.get(lid) != new_line:
            raise RuntimeError(f"线段 {lid} 添加到line_id_map失败，映射同步异常")
        self.lines.append(new_line)
        self.line_pairs.add((p1, p2))
        
        self.on_segment_points_cache.add(start_pid)
        self.on_segment_points_cache.add(end_pid)
        
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
        if entity_id:
            candidates = self.entity_vertices_cache[entity_id]
        else:
            candidates = list(self.all_points_cache)
        
        if filter_isolated:
            candidates = [pid for pid in candidates if self._is_point_on_any_segment(pid)]

        candidates = [pid for pid in candidates if not self._is_expr_too_long(pid)]
        if not candidates:
            raise ValueError(f"无符合条件的点（已过滤表达式过长的点，上限{self.max_expr_length}字符）")
        
        if count > len(candidates):
            raise ValueError(f"线上点数量不足（需要{count}，现有{len(candidates)}）")
        
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

    # ------------------------------ 操作可行性与选择 ------------------------------
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
                target_entity_types = constraints.get("entity_types", ["circle"])
                circle_entities = [e for e in self.entities if e["type"] in target_entity_types]
                if not circle_entities:
                    return False
                
                for entity in circle_entities:
                    for comp_id in entity["components"]:
                        if comp_id in self.arc_id_map:
                            circle = self.arc_id_map[comp_id]
                            if ("center_id" in circle and "radius" in circle and 
                                circle["center_id"] in self.point_id_map):
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
        
        draw_diameter_feasible = self._is_operation_feasible("draw_diameter", constraints.get("draw_diameter", {}))
        if draw_diameter_feasible and "draw_diameter" in adjusted_probs:
            diameter_prob = 0.4
            other_total = sum(p for k, p in adjusted_probs.items() if k != "draw_diameter")
            if other_total > 0:
                scale = (1 - diameter_prob) / other_total
                for k in adjusted_probs:
                    if k != "draw_diameter":
                        adjusted_probs[k] *= scale
                adjusted_probs["draw_diameter"] = diameter_prob
            else:
                adjusted_probs["draw_diameter"] = 1.0
            logger.debug(f"检测到可行的圆形实体，提高draw_diameter操作优先级（概率：{diameter_prob}）")

        feasible_types = []
        feasible_probs = []
        
        for op_type, prob in adjusted_probs.items():
            op_constraints = constraints.get(op_type, {})
            if self._is_operation_feasible(op_type, op_constraints):
                feasible_types.append(op_type)
                feasible_probs.append(prob)
        
        if not feasible_types:
            raise ValueError("无可行的操作类型")
        
        total = sum(feasible_probs)
        feasible_probs = [p / total for p in feasible_probs]
        return random.choices(feasible_types, weights=feasible_probs, k=1)[0]

    # ------------------------------ 操作生成 ------------------------------
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
            
            def is_point_on_line(px, py, s_x, s_y, e_x, e_y):
                cross = (e_x - s_x) * (py - s_y) - (e_y - s_y) * (px - s_x)
                return sp.simplify(cross) == 0
            
            def calculate_foot_of_perpendicular(px, py, s_x, s_y, e_x, e_y):
                se_x = e_x - s_x
                se_y = e_y - s_y
                sp_x = px - s_x
                sp_y = py - s_y
                dot_product = sp_x * se_x + sp_y * se_y
                se_sq = se_x**2 + se_y**2
                t = dot_product / se_sq
                foot_x = s_x + t * se_x
                foot_y = s_y + t * se_y
                return sp.simplify(foot_x), sp.simplify(foot_y)
            
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
                    
                    px, py = self.get_point_coords(point_id)
                    s, e = line["start_point_id"], line["end_point_id"]
                    s_x, s_y = self.get_point_coords(s)
                    e_x, e_y = self.get_point_coords(e)
                    
                    if is_point_on_line(px, py, s_x, s_y, e_x, e_y):
                        logger.debug(f"draw_perpendicular重试{retry}次失败：点{point_id}在直线{line_id}（含延长线）上")
                        continue
                    
                    foot_x, foot_y = calculate_foot_of_perpendicular(px, py, s_x, s_y, e_x, e_y)
                    
                    foot_pid = find_matching_point(foot_x, foot_y)
                    if foot_pid:
                        a, b = (point_id, foot_pid) if point_id < foot_pid else (foot_pid, point_id)
                        if (a, b) in self.line_pairs or (b, a) in self.line_pairs:
                            logger.debug(f"draw_perpendicular重试{retry}次失败：垂足{foot_pid}与点{point_id}已存在线段")
                            continue
                    
                    logger.debug(f"draw_perpendicular重试{retry}次成功")
                    return {
                        "type": op_type,
                        "line_id": line_id,
                        "point_id": point_id,
                        "foot_coords": (foot_x, foot_y)
                    }
                except Exception as e:
                    logger.debug(f"draw_perpendicular重试{retry}次失败：{str(e)}")
            
            raise ValueError(f"超过最大重试次数（{self.MAX_RETRIES}次），未找到合适的垂线参数")
        
        elif op_type == "draw_diameter":
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

    # ------------------------------ 交点检测与几何计算（含level处理） ------------------------------
    def _point_on_segment(self, px: sp.Expr, py: sp.Expr, 
                        x1: sp.Expr, y1: sp.Expr, 
                        x2: sp.Expr, y2: sp.Expr) -> bool:
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

    def _line_arc_intersection(self, line_id: str, arc_id: str) -> List[Tuple[sp.Expr, sp.Expr, int]]:
        """计算线段与圆弧的交点，返回(坐标x, 坐标y, 交点level)"""
        arc = self.arc_id_map[arc_id]
        # 检查圆弧必要参数
        if "center_id" not in arc or "radius_expr" not in arc:
            return []
        
        center_id = arc["center_id"]
        # 验证圆心存在
        if center_id not in self.point_id_map:
            logger.warning(f"圆弧{arc_id}的圆心{center_id}不存在")
            return []
            
        cx, cy = self.get_point_coords(center_id)
        radius = self._parse_expr(arc["radius_expr"])
        arc_level = arc["level"]
        
        # 验证线段存在
        if line_id not in self.line_id_map:
            logger.warning(f"线段{line_id}不存在")
            return []
            
        s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s)
        x2, y2 = self.get_point_coords(e)
        line_level = self.line_id_map[line_id]["level"]
        
        # 交点level为两者最大值
        intersection_level = max(line_level, arc_level)
        
        # 参数化线段方程
        t = sp.Symbol('t', real=True)  
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        # 圆方程
        circle_eq = (x - cx)**2 + (y - cy)** 2 - radius**2
        circle_eq = sp.simplify(circle_eq)
        
        solutions = sp.solve(circle_eq, t)
        intersections = []
        
        for sol in solutions:
            # 处理不同形式的解
            if isinstance(sol, dict):
                sol = sol.get(t, None)
            if sol is None:
                continue
                
            try:
                t_val = sp.simplify(sol)
                # 过滤复数解
                if sp.im(t_val) != 0:
                    continue
                t_val = sp.re(t_val)
                
                # 检查t是否在线段范围内[0,1]
                t_valid = sp.And(sp.Ge(t_val, 0), sp.Le(t_val, 1))
                if not sp.simplify(t_valid):
                    continue
                
                # 计算交点坐标
                ix = sp.simplify(x.subs(t, t_val))
                iy = sp.simplify(y.subs(t, t_val))
                
                # 检查是否在圆弧角度范围内
                angle_check_passed = True
                if "start_angle" in arc and "end_angle" in arc:
                    try:
                        start_angle = self._parse_expr(arc["start_angle"])
                        end_angle = self._parse_expr(arc["end_angle"])
                        dx = ix - cx
                        dy = iy - cy
                        
                        # 计算交点相对于圆心的角度
                        angle = sp.atan2(dy, dx)
                        # 标准化角度到[0, 2π)范围
                        angle = sp.simplify(angle % (2 * sp.pi))
                        start_angle = sp.simplify(start_angle % (2 * sp.pi))
                        end_angle = sp.simplify(end_angle % (2 * sp.pi))
                        
                        # 检查角度是否在圆弧范围内
                        if sp.simplify(start_angle <= end_angle):
                            in_angle_range = sp.And(angle >= start_angle, angle <= end_angle)
                        else:
                            in_angle_range = sp.Or(angle >= start_angle, angle <= end_angle)
                            
                        angle_check_passed = sp.simplify(in_angle_range)
                    except Exception as e:
                        logger.warning(f"角度范围计算失败: {str(e)}")
                        angle_check_passed = False
                
                if angle_check_passed:
                    is_new_point = True
                    for endpoint in [s, e]:
                        ex, ey = self.get_point_coords(endpoint)
                        if sp.simplify(ix - ex) == 0 and sp.simplify(iy - ey) == 0:
                            is_new_point = False
                            break
                            
                    if is_new_point:
                        intersections.append((ix, iy, intersection_level))
            
            except Exception as e:
                logger.warning(f"处理交点解时出错: {str(e)}")
                continue
        
        return intersections


    def detect_new_intersections(self, new_line_id: str) -> List[str]:
        new_ips = []
        new_line = self.line_id_map[new_line_id]
        new_line_level = new_line["level"]
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
            line_level = line["level"]
            s, e = line["start_point_id"], line["end_point_id"]
            x1, y1 = self.get_point_coords(s)
            x2, y2 = self.get_point_coords(e)
            
            all_symbols = set()
            for expr in [x1, y1, x2, y2, new_min_x, new_max_x, new_min_y, new_max_y]:
                all_symbols.update(expr.free_symbols)
            all_symbols = list(all_symbols)
            
            # 2. 构建假设：所有符号都是实数
            assumptions = Q.real(*all_symbols)
            
            # 3. 使用 ask() 来判断边界框是否重叠
            # 如果 ask 返回 True，说明线段在该方向上没有交集，可以直接跳过
            cond1 = ask(Max(x1, x2) < new_min_x, assumptions)
            cond2 = ask(Min(x1, x2) > new_max_x, assumptions)
            cond3 = ask(Max(y1, y2) < new_min_y, assumptions)
            cond4 = ask(Min(y1, y2) > new_max_y, assumptions)
            
            if cond1 or cond2 or cond3 or cond4:
                continue
            
            # if (Max(x1, x2) < new_min_x) or (Min(x1, x2) > new_max_x) or \
            #    (Max(y1, y2) < new_min_y) or (Min(y1, y2) > new_max_y):
            #     continue
            
            intersections = self._line_intersection(new_line_id, line_id)
            for x, y, level in intersections:
                x_simplified = sp.simplify(x)
                y_simplified = sp.simplify(y)
                pid = self.add_new_point(
                    x_simplified, y_simplified, 
                    prefix="I", 
                    point_type="intersection",
                    level=level  # 使用计算的交点level
                )
                new_ips.append(pid)
        
        # 检测与现有圆弧的交点
        for arc_id in self.arc_id_map:
            arc = self.arc_id_map[arc_id]
            arc_level = arc["level"]
            if "center_id" in arc and "radius_expr" in arc:
                cx, cy = self.get_point_coords(arc["center_id"])
                r = self._parse_expr(arc["radius_expr"])
                arc_min_x = cx - r
                arc_max_x = cx + r
                arc_min_y = cy - r
                arc_max_y = cy + r
                
                all_symbols = set()
                for expr in [arc_min_x, arc_max_x, arc_min_y, arc_max_y, new_min_x, new_max_x, new_min_y, new_max_y]:
                    all_symbols.update(expr.free_symbols)
                all_symbols = list(all_symbols)
                
                # 2. 构建假设
                assumptions = Q.real(*all_symbols)
                
                # 3. 使用 ask() 来判断边界框是否重叠
                cond1 = ask(arc_max_x < new_min_x, assumptions)
                cond2 = ask(arc_min_x > new_max_x, assumptions)
                cond3 = ask(arc_max_y < new_min_y, assumptions)
                cond4 = ask(arc_min_y > new_max_y, assumptions)
                
                if cond1 or cond2 or cond3 or cond4:
                    continue
                
                # if (arc_max_x < new_min_x) or (arc_min_x > new_max_x) or \
                #    (arc_max_y < new_min_y) or (arc_min_y > new_max_y):
                #     continue
            
            intersections = self._line_arc_intersection(new_line_id, arc_id)
            for x, y, level in intersections:
                x_simplified = sp.simplify(x)
                y_simplified = sp.simplify(y)
                pid = self.add_new_point(
                    x_simplified, y_simplified, 
                    prefix="I", 
                    point_type="arc_intersection",
                    level=level  # 使用计算的交点level
                )
                new_ips.append(pid)
        
        return new_ips

    def _line_intersection(self, line1_id: str, line2_id: str) -> List[Tuple[sp.Expr, sp.Expr, int]]:
        """计算两线段交点，返回(坐标x, 坐标y, 交点level)"""
        s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s1)
        x2, y2 = self.get_point_coords(e1)
        line1_level = self.line_id_map[line1_id]["level"]
        
        s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
        x3, y3 = self.get_point_coords(s2)
        x4, y4 = self.get_point_coords(e2)
        line2_level = self.line_id_map[line2_id]["level"]
        
        # 交点level为两者最大值
        intersection_level = max(line1_level, line2_level)
        
        def points_equal(p1: Tuple[sp.Expr, sp.Expr], p2: Tuple[sp.Expr, sp.Expr]) -> bool:
            return sp.Eq(p1[0], p2[0]) and sp.Eq(p1[1], p2[1])
        
        if (points_equal((x1, y1), (x3, y3)) or points_equal((x1, y1), (x4, y4)) or
            points_equal((x2, y2), (x3, y3)) or points_equal((x2, y2), (x4, y4))):
            return []
        
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        cross_product = sp.simplify(dx1 * dy2 - dx2 * dy1)
        if sp.Eq(cross_product, 0):
            logger.debug(f"线段{line1_id}与{line2_id}平行或重合，无交点")
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
                intersections.append((x, y, intersection_level))
        
        return intersections

    def _calculate_foot_of_perpendicular(self, point_id: str, line_id: str) -> Tuple[sp.Expr, sp.Expr, bool, int]:
        """计算垂足，返回(坐标x, 坐标y, 是否在线段上, 垂足level)"""
        px, py = self.get_point_coords(point_id)
        point_level = self.point_id_map[point_id]["level"]
        
        s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
        x1, y1 = self.get_point_coords(s)
        x2, y2 = self.get_point_coords(e)
        line_level = self.line_id_map[line_id]["level"]
        
        # 垂足level = 点level + 线level
        foot_level = point_level + line_level
        
        dx = x2 - x1
        dy = y2 - y1
        px_vec = px - x1
        py_vec = py - y1
        
        if sp.Eq(dx**2 + dy**2, 0):
            return (x1, y1, True, foot_level)
        
        t = (px_vec * dx + py_vec * dy) / (dx**2 + dy**2)
        foot_x = x1 + t * dx
        foot_y = y1 + t * dy
        on_segment = Ge(t, 0) and Le(t, 1)
        return (foot_x, foot_y, bool(on_segment), foot_level)

    # ------------------------------ 操作执行（含level计算） ------------------------------
    def execute_operation(self, op: Dict) -> Dict:
        op_type = op["type"]
        result = {"operation": op_type, "details": op, "new_elements": [], "new_line": "", "description": ""}
        
        if op_type == "connect_points":
            p1_id, p2_id = op["point_ids"]
            p1_level = self.point_id_map[p1_id]["level"]
            p2_level = self.point_id_map[p2_id]["level"]
            # 新线level = 两点level和
            line_level = p1_level + p2_level
            
            desc = f"Connect points {p1_id} and {p2_id} (both on segments)"
            line_id = self.add_new_line(
                p1_id, p2_id, 
                prefix="ConnP", 
                line_type="connection", 
                description=desc,
                level=line_level  # 设置新线level
            )
            new_ips = self.detect_new_intersections(line_id)
            result["new_elements"].append(("line", line_id))
            result["new_elements"].extend([("point", pid) for pid in new_ips])
            result["description"] = desc
        
        elif op_type == "connect_midpoints":
            mode = op["mode"]
            
            if mode == "two_midpoints":
                line1_id, line2_id = op["line_ids"]
                line1_level = self.line_id_map[line1_id]["level"]
                line2_level = self.line_id_map[line2_id]["level"]
                
                # 中点level = 对应边的level
                s1, e1 = self.line_id_map[line1_id]["start_point_id"], self.line_id_map[line1_id]["end_point_id"]
                x1, y1 = self.get_point_coords(s1)
                x2, y2 = self.get_point_coords(e1)
                mid1_x = (x1 + x2) / 2
                mid1_y = (y1 + y2) / 2
                mid1_id = self.add_new_point(
                    mid1_x, mid1_y, 
                    prefix="M", 
                    point_type="midpoint", 
                    related_edge=line1_id,
                    level=line1_level  # 中点level=边level
                )
                
                s2, e2 = self.line_id_map[line2_id]["start_point_id"], self.line_id_map[line2_id]["end_point_id"]
                x3, y3 = self.get_point_coords(s2)
                x4, y4 = self.get_point_coords(e2)
                mid2_x = (x3 + x4) / 2
                mid2_y = (y3 + y4) / 2
                mid2_id = self.add_new_point(
                    mid2_x, mid2_y, 
                    prefix="M", 
                    point_type="midpoint", 
                    related_edge=line2_id,
                    level=line2_level  # 中点level=边level
                )
                
                # 新线level = 两中点level和
                line_level = self.point_id_map[mid1_id]["level"] + self.point_id_map[mid2_id]["level"]
                desc = f"Connect midpoints {mid1_id} (of line {line1_id}) and {mid2_id} (of line {line2_id})"
                line_id = self.add_new_line(
                    mid1_id, mid2_id, 
                    prefix="MidL", 
                    line_type="midline", 
                    description=desc,
                    level=line_level  # 设置新线level
                )
                new_ips = self.detect_new_intersections(line_id)
                result["new_elements"].extend([("point", mid1_id), ("point", mid2_id)])
                result["new_elements"].append(("line", line_id))
                result["new_elements"].extend([("point", pid) for pid in new_ips])
                result["description"] = desc
            
            else:  # mode == "vertex_and_midpoint"
                line_id = op["line_id"]
                vertex_id = op["vertex_id"]
                line_level = self.line_id_map[line_id]["level"]
                vertex_level = self.point_id_map[vertex_id]["level"]
                
                # 中点level = 边的level
                s, e = self.line_id_map[line_id]["start_point_id"], self.line_id_map[line_id]["end_point_id"]
                x1, y1 = self.get_point_coords(s)
                x2, y2 = self.get_point_coords(e)
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                mid_id = self.add_new_point(
                    mid_x, mid_y, 
                    prefix="M", 
                    point_type="midpoint", 
                    related_edge=line_id,
                    level=line_level  # 中点level=边level
                )
                
                # 新线level = 顶点level + 中点level
                line_level = vertex_level + self.point_id_map[mid_id]["level"]
                desc = f"Connect vertex {vertex_id} with midpoint {mid_id} (of line {line_id})"
                line_id = self.add_new_line(
                    vertex_id, mid_id, 
                    prefix="VMidL", 
                    line_type="vertex_midline", 
                    description=desc,
                    level=line_level  # 设置新线level
                )
                new_ips = self.detect_new_intersections(line_id)
                result["new_elements"].append(("point", mid_id))
                result["new_elements"].append(("line", line_id))
                result["new_elements"].extend([("point", pid) for pid in new_ips])
                result["description"] = desc
        
        elif op_type == "draw_perpendicular":
            target_line_id = op["line_id"]
            start_point_id = op["point_id"]
            allow_on_segment = op.get("allow_on_segment", False)
            
            # 计算垂足及level（点level + 底边level）
            foot_x, foot_y, on_segment, foot_level = self._calculate_foot_of_perpendicular(start_point_id, target_line_id)
            target_line_level = self.line_id_map[target_line_id]["level"]
            start_point_level = self.point_id_map[start_point_id]["level"]
            
            # 垂足level已在计算方法中确定
            foot_id = self.add_new_point(
                foot_x, foot_y, 
                prefix="F",
                point_type="perpendicular_foot",
                related_vertex=start_point_id,
                related_edge=target_line_id,
                level=foot_level  # 设置垂足level
            )
            
            new_elements = [("point", foot_id)]
            # 垂线level = 点level + 底边level
            perp_line_level = start_point_level + target_line_level
            
            if not on_segment:
                s, e = self.line_id_map[target_line_id]["start_point_id"], self.line_id_map[target_line_id]["end_point_id"]
                s_dist = (foot_x - self.get_point_coords(s)[0])**2 + (foot_y - self.get_point_coords(s)[1])** 2
                e_dist = (foot_x - self.get_point_coords(e)[0])**2 + (foot_y - self.get_point_coords(e)[1])** 2
                extend_from_id = s if s_dist < e_dist else e
                
                # 延长线level = 底边level + 垂足level
                extend_line_level = target_line_level + foot_level
                extend_desc = f"Extend line {target_line_id} to perpendicular foot {foot_id}"
                extend_line_id = self.add_new_line(
                    extend_from_id, foot_id,
                    prefix="ExtL",
                    line_type="extension",
                    description=extend_desc,
                    level=extend_line_level  # 设置延长线level
                )
                new_elements.append(("line", extend_line_id))
                desc = f"Draw perpendicular from point {start_point_id} to line {target_line_id} (foot {foot_id} on extension)"
            else:
                desc = f"Draw perpendicular from point {start_point_id} to line {target_line_id} (foot {foot_id} on segment)"
            
            perp_line_id = self.add_new_line(
                start_point_id, foot_id, 
                prefix="PerpL", 
                line_type="perpendicular",
                description=desc,
                level=perp_line_level  # 设置垂线level
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
            circle_level = self.arc_id_map[circle_id]["level"]
            center_level = self.point_id_map[center_id]["level"]
            
            # 直径端点level = 圆心level + 圆level
            end_level = center_level + circle_level
            
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
            
            end1_id = self.add_new_point(
                end1_x, end1_y, 
                prefix="Diam", 
                point_type="diameter_end",
                level=end_level  # 端点level
            )
            end2_id = self.add_new_point(
                end2_x, end2_y, 
                prefix="Diam", 
                point_type="diameter_end",
                level=end_level  # 端点level
            )
            
            # 直径线level = 两端点level和
            diam_line_level = self.point_id_map[end1_id]["level"] + self.point_id_map[end2_id]["level"]
            desc = f"Draw {direction} diameter of circle {circle_id} with endpoints {end1_id} and {end2_id}"
            diam_line_id = self.add_new_line(
                end1_id, end2_id, 
                prefix="DiamL", 
                line_type="diameter",
                description=desc,
                level=diam_line_level  # 直径线level
            )
            
            new_ips = self.detect_new_intersections(diam_line_id)
            result["new_elements"].extend([("point", end1_id), ("point", end2_id)])
            result["new_elements"].append(("line", diam_line_id))
            result["new_elements"].extend([("point", pid) for pid in new_ips])
            result["description"] = desc
        
        return result
    
    # ------------------------------ 增强图形生成 ------------------------------
    def generate_enhancements(self, config: Dict) -> List[Dict]:
        if "rounds_distribution" not in config:
            raise ValueError("config 必须包含 rounds_distribution 配置")
        
        rounds_dist = {int(k): int(v) for k, v in config["rounds_distribution"].items()}
        num_enhancements = sum(rounds_dist.values())
        
        run_config = {
            **config,
            "num_enhancements": num_enhancements,
            "single_enhance_timeout": config.get("single_enhance_timeout", 256)
        }
        
        all_results = self.run_rounds(run_config)
        
        enhancements = []
        for result in all_results:
            if not isinstance(result, dict) or "final_geometry" not in result:
                logger.warning(f"跳过无效增强结果: {result}")
                continue
            
            final_geo = result["final_geometry"]
            final_geo["is_base"] = (result["execution_summary"] == [])
            final_geo["timeout_occurred"] = result.get("timeout_occurred", False)
            final_geo["completed_rounds"] = len(result.get("execution_summary", []))
            final_geo["base_id"] = self.base_id
            final_geo["enhance_id"] = f"{self.base_id}_enhance_{result['enhance_idx']:03d}"
            
            enhancements.append(final_geo)
        
        logger.info(f"生成 {len(enhancements)} 个独立增强图形（预期 {num_enhancements} 个）")
        return enhancements

    def run_rounds(self, config: Dict) -> List[Dict]:
        rounds_distribution = config["rounds_distribution"]
        min_ops_per_round = config["min_operations_per_round"]
        max_ops_per_round = config["max_operations_per_round"]
        num_enhancements = config["num_enhancements"]
        single_enhance_timeout = config.get("single_enhance_timeout", 256)

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
        original_data = json.loads(json.dumps(self.data))
        original_desc = original_data.get("description", "无原始描述")
        base_seed = config.get("seed", 42)

        for enh_idx, num_rounds in enumerate(rounds_list):
            enh_seed = base_seed + enh_idx * 100
            random.seed(enh_seed)
            self.data = json.loads(json.dumps(original_data))
            self.points = self.data["points"]
            self.lines = self.data["lines"]
            self.arcs = self.data.get("arcs", [])
            self.point_id_map = {p["id"]: p for p in self.points}
            self.line_id_map = {l["id"]: l for l in self.lines}
            self.arc_id_map = {a["id"]: a for a in self.arcs}
            self.line_pairs = self._build_line_pairs()
            self.on_segment_points_cache = set(self._cache_on_segment_points())
            self.all_points_cache = set(self.point_id_map.keys())
            self.entity_vertices_cache = self._cache_entity_vertices()
            self.entity_lines_cache = self._cache_entity_lines()
            self.entity_arcs_cache = self._cache_entity_arcs()    
            
            self.enhancement_history = []
            self.operation_counter = 0

            enhancement_desc = [
                f"Original graphic description: {original_desc}",
                f"Enhancement index: {enh_idx + 1}/{num_enhancements}",
                f"Number of rounds: {num_rounds}"
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

            try:
                self._select_feasible_operation_type(config["operation_probs"], config["operation_constraints"])
            except ValueError as e:
                logger.warning(f"增强结果{enh_idx + 1}无可行操作，跳过")
                continue

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

                    for op_retry in range(self.MAX_RETRIES):
                        try:
                            op_type = self._select_feasible_operation_type(
                                config["operation_probs"], config["operation_constraints"]
                            )
                            op = self._generate_random_operation(op_type, config["operation_constraints"].get(op_type, {}))
                            op_result = self.execute_operation(op)
                            logger.debug(f"操作{op_type}成功，新增元素：{op_result['new_elements']}")

                            round_result["operations"].append(op_result)
                            round_desc.append(f"  第 {op_idx + 1} 步: {op_result['description']}")
                            self.enhancement_history.append(op_result)
                            break
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
                
                self.entity_vertices_cache = self._cache_entity_vertices()
                self.entity_lines_cache = self._cache_entity_lines()
                self.entity_arcs_cache = self._cache_entity_arcs()
                self.point_id_map = {p["id"]: p for p in self.points}  # 包含本轮新增的点
                self.line_id_map = {l["id"]: l for l in self.lines}    # 包含本轮新增的线
                self.arc_id_map = {a["id"]: a for a in self.arcs}      # 包含本轮新增的圆弧

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