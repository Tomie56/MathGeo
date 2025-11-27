import json
import random
import os
import sympy as sp
from sympy import S, symbols, cos, sin, pi, simplify, sqrt, Eq, atan2, tan, nsimplify, Rational, Min, Max, solve, Ge, Le
from typing import List, Dict, Tuple, Optional, Set, Iterable
import re


class TemplateGenerator:
    def __init__(self, config: Dict, seed: Optional[int] = None):
        """初始化生成器，支持随机种子"""
        self.config = self._validate_config(config)
        self.seed = seed if seed is not None else random.randint(0, 10000)
        random.seed(self.seed)
        self.data = {"points": [], "lines": [], "arcs": [], "entities": [], "description": ""}
        self.description_parts = []
        self.point_ids: Set[str] = set()
        self.line_ids: Set[str] = set()
        self.arc_ids: Set[str] = set()
        self.entity_ids: Set[str] = set()
        self.letter_counter = 0 
        self.circle_point_counter = 0
        self.center_point_counter = 0
        self.line_counter = 0
        self.arc_counter = 0
        self.x, self.y = symbols('x y')
        self.base_entity_id: Optional[str] = None
        self.current_entities: List[str] = []
        self.complete_circle_arcs: Set[str] = set()
        self.points_counter = 0
        self.lines_counter = 0


    # ------------------------------ 配置验证 ------------------------------
    def _validate_config(self, config: Dict) -> Dict:
        required_keys = {
            "n": int,
            "points_limit": int,
            "lines_limit": int,
            "seed": (int, type(None)),
            "output_dir": str,
            "output_filename": str,
            "base": dict,
            "derivation": dict
        }
        for key, type_ in required_keys.items():
            if key not in config or not isinstance(config[key], type_):
                raise ValueError(f"配置缺失或类型错误：{key}（需为{type_.__name__}）")
        os.makedirs(config["output_dir"], exist_ok=True)
        return config

    # ------------------------------ 表达式格式化 ------------------------------
    def _format_expr(self, expr: sp.Expr) -> str:
        simplified = simplify(expr)
        if simplified.is_Integer:
            return str(int(simplified))
        return str(simplified)
    
    def _serialize_expr(self, expr: sp.Expr) -> str:
        if expr is None:
            return ""
        expr = nsimplify(expr, rational=True)
        if expr.has(sp.sin, sp.cos, sp.sqrt, sp.tan):
            expr = simplify(expr)
        expr_str = str(expr).replace(".0", "")
        return expr_str


    # ------------------------------ ID生成工具 ------------------------------
    def _get_letter_id(self) -> str:
        """生成普通点ID（A,B,C...）"""
        counter = self.letter_counter
        letters = []
        while True:
            remainder = counter % 26
            letters.append(chr(65 + remainder))
            counter = counter // 26 - 1
            if counter < 0:
                break
        self.letter_counter += 1
        return ''.join(reversed(letters))

    def _get_circle_point_id(self) -> str:
        """生成完整圆初始点ID（circle_1,2...）"""
        self.circle_point_counter += 1
        return f"circle_{self.circle_point_counter}"

    def _get_center_point_id(self) -> str:
        """生成圆心/中心ID（O1,O2...）"""
        self.center_point_counter += 1
        return f"O{self.center_point_counter}"

    def _get_unique_line_id(self) -> str:
        lid = f"L{self.line_counter}"
        self.line_counter += 1
        self.line_ids.add(lid)
        return lid

    def _get_unique_arc_id(self) -> str:
        aid = f"Arc{self.arc_counter}"
        self.arc_counter += 1
        self.arc_ids.add(aid)
        return aid

    def _get_unique_entity_id(self, base_id: str) -> str:
        if base_id not in self.entity_ids:
            self.entity_ids.add(base_id)
            return base_id
        counter = 1
        while f"{base_id}_{counter}" in self.entity_ids:
            counter += 1
        new_id = f"{base_id}_{counter}"
        self.entity_ids.add(new_id)
        return new_id

    # ------------------------------ 点工具 ------------------------------
    def _is_point(self, component_id: str) -> bool:
        """判断一个组件ID是否为点"""
        # 点的ID格式: 普通点(A, B, C...), 中心点(O1, O2...), 圆初始点(circle_1, circle_2...)
        return (component_id.isalpha() or 
                component_id.startswith('O') and component_id[1:].isdigit() or
                component_id.startswith('circle_'))

    def _is_line(self, component_id: str) -> bool:
        """判断一个组件ID是否为线段"""
        # 线的ID格式: L1, L2, L3...
        return component_id.startswith('L') and component_id[1:].isdigit()

    def _is_arc(self, component_id: str) -> bool:
        """判断一个组件ID是否为弧"""
        # 弧的ID格式: Arc1, Arc2, Arc3...
        return component_id.startswith('Arc') and component_id[3:].isdigit()
    
    def _update_references_to_old_point(self, old_pid: str, new_pid: str):
        """更新所有引用老点（circle_x）的边、弧、实体"""
        # 1. 更新边的端点
        for line in self.data["lines"]:
            if line["start_point_id"] == old_pid:
                line["start_point_id"] = new_pid
            if line["end_point_id"] == old_pid:
                line["end_point_id"] = new_pid

        # 2. 更新弧的端点
        for arc in self.data["arcs"]:
            if arc["start_point_id"] == old_pid:
                arc["start_point_id"] = new_pid
            if arc["end_point_id"] == old_pid:
                arc["end_point_id"] = new_pid
            if arc["center_point_id"] == old_pid:
                arc["center_point_id"] = new_pid

        # 3. 更新实体组件
        for entity in self.data["entities"]:
            if "vertices" in entity:
                for component in entity["vertices"]:
                    if component == old_pid:
                        component = new_pid
            
            if "center_id" in entity and entity["center_id"] == old_pid:
                entity["center_id"] = new_pid

    def _add_point(self, x_expr: sp.Expr, y_expr: sp.Expr, is_center: bool = False, is_circle_init: bool = False, level: int = 1) -> str:
 
        """
        添加点，支持三种类型：
        - is_center=True：圆心/中心（O1,O2...）
        - is_circle_init=True：完整圆初始点（circle_1,2...）
        - 其他：普通点（A,B,C...）+ 后续圆上交点/切点
        核心逻辑：查重，circle_x与普通点重合时用普通点替换并更新关联边
        """
        x_expr = simplify(x_expr)
        y_expr = simplify(y_expr)

        # 第一步：查重（检查是否与已有点重合）
        for idx, old_point in enumerate(self.data["points"]):
            old_x = simplify(sp.sympify(old_point["x"]["expr"]))
            old_y = simplify(sp.sympify(old_point["y"]["expr"]))
            
            x_diff = sp.simplify(old_x - x_expr)
            y_diff = sp.simplify(old_y - y_expr)
            x_equal = x_diff.is_zero
            y_equal = y_diff.is_zero
            update_tag = old_point.get("is_circle_init") or old_point.get("is_center")
            
            if x_equal and y_equal:
                old_pid = old_point["id"]
                new_level = min(level, old_point["level"])
                if old_point["level"] != new_level:
                    self.data["points"][idx]["level"] = new_level
                    
                if update_tag and not is_circle_init and not is_center:
                    old_related_edges = old_point["related_edges"].copy()
                    del self.data["points"][idx]
                    self.point_ids.remove(old_pid)
                    if is_center:
                        new_pid = self._get_center_point_id()
                    else:
                        new_pid = self._get_letter_id()
                        
                    new_point = {
                        "id": new_pid,
                        "x": {"expr": self._format_expr(x_expr), "latex": sp.latex(x_expr)},
                        "y": {"expr": self._format_expr(y_expr), "latex": sp.latex(y_expr)},
                        "related_edges": old_related_edges,
                        "is_center": is_center,
                        "is_circle_init": False,
                        "level": new_level
                    }
                    self.data["points"].insert(idx, new_point)
                    self.point_ids.add(new_pid)
                    self._update_references_to_old_point(old_pid, new_pid)
                    return new_pid
                else:
                    return old_pid
                
        self.points_counter += 1
        if(self.points_counter >= self.config["points_limit"]):
            raise Exception("点数量超过限制，图形可能过于复杂，放弃生成")

        # 第二步：无重合，生成新点
        if is_center:
            pid = self._get_center_point_id()
        elif is_circle_init:
            pid = self._get_circle_point_id()
        else:
            pid = self._get_letter_id()

        self.point_ids.add(pid)
        self.data["points"].append({
            "id": pid,
            "x": {"expr": self._format_expr(x_expr), "latex": sp.latex(x_expr)},
            "y": {"expr": self._format_expr(y_expr), "latex": sp.latex(y_expr)},
            "related_edges": [],
            "is_center": is_center,
            "is_circle_init": is_circle_init,
            "level": level
        })
        return pid

    def get_point_coords(self, pid: str) -> Tuple[sp.Expr, sp.Expr]:
        p = next(p for p in self.data["points"] if p["id"] == pid)
        return simplify(sp.sympify(p["x"]["expr"])), simplify(sp.sympify(p["y"]["expr"]))

    def get_entity(self, entity_id: str) -> Dict:
        return next(e for e in self.data["entities"] if e["id"] == entity_id)

    def _get_center_id(self, entity: Dict) -> str:
        center_id = entity.get("center_id")
        if center_id is None:
            raise ValueError(f"实体 {entity['id']} 缺少 center_id")
        return center_id


    # ------------------------------ 边/弧查重工具 ------------------------------
    def _check_line_duplicate(self, start_pid: str, end_pid: str) -> Optional[str]:
        """检查线段是否重复（AB与BA视为同一条），返回已存在的线段ID或None"""
        for line in self.data["lines"]:
            s = line["start_point_id"]
            e = line["end_point_id"]
            if (s == start_pid and e == end_pid) or (s == end_pid and e == start_pid):
                return line["id"]
        return None

    def _check_arc_duplicate(self, center_pid: str, start_pid: str, end_pid: str, radius_expr: str) -> Optional[str]:
        """检查弧是否重复（圆心、半径、起终点相同），返回已存在的弧ID或None"""
        for arc in self.data["arcs"]:
            if (arc["center_point_id"] == center_pid and
                arc["start_point_id"] == start_pid and
                arc["end_point_id"] == end_pid and
                arc["radius"]["expr"] == radius_expr):
                return arc["id"]
        return None

    def _add_line(self, start_pid: str, end_pid: str) -> str:
        """添加线段"""
        
        start_point = next(p for p in self.data["points"] if p["id"] == start_pid)
        end_point = next(p for p in self.data["points"] if p["id"] == end_pid)
        line_level = max(start_point["level"], end_point["level"])
        
        dup_lid = self._check_line_duplicate(start_pid, end_pid)
        if dup_lid:
            dup_line = next(l for l in self.data["lines"] if l["id"] == dup_lid)
            if dup_line["level"] < line_level:
                dup_line["level"] = line_level
            return dup_lid
        
        self.lines_counter += 1
        if(self.lines_counter >= self.config["lines_limit"]):
            raise Exception("线段数量超过限制，图形可能过于复杂，放弃生成")
        
        lid = self._get_unique_line_id()
        self.data["lines"].append({
            "id": lid,
            "type": "line",
            "start_point_id": start_pid,
            "end_point_id": end_pid,
            "level": line_level
        })
        # 更新端点的关联边
        for pid in [start_pid, end_pid]:
            p = next(p for p in self.data["points"] if p["id"] == pid)
            if lid not in p["related_edges"]:
                p["related_edges"].append(lid)
        return lid

    def _add_arc(self, start_pid: str, end_pid: str, center_pid: str, radius_expr: sp.Expr, angle_expr: sp.Expr, is_complete: bool) -> str:
        """添加弧"""
        
        start_point = next(p for p in self.data["points"] if p["id"] == start_pid)
        end_point = next(p for p in self.data["points"] if p["id"] == end_pid)
        arc_level = max(start_point["level"], end_point["level"])
        
        radius_str = self._format_expr(radius_expr)
        dup_aid = self._check_arc_duplicate(center_pid, start_pid, end_pid, radius_str)
        if dup_aid:
            dup_arc = next(a for a in self.data["arcs"] if a["id"] == dup_aid)
            if dup_arc["level"] < arc_level:
                dup_arc["level"] = arc_level
            return dup_aid
        
        aid = self._get_unique_arc_id()
        self.data["arcs"].append({
            "id": aid,
            "type": "arc",
            "start_point_id": start_pid,
            "end_point_id": end_pid,
            "center_point_id": center_pid,
            "radius": {"expr": radius_str, "latex": sp.latex(radius_expr)},
            "angle": {"expr": self._format_expr(angle_expr), "latex": sp.latex(angle_expr)},
            "is_complete": is_complete,
            "level": arc_level 
        })
        for pid in [start_pid, end_pid]:
            p = next(p for p in self.data["points"] if p["id"] == pid)
            if aid not in p["related_edges"]:
                p["related_edges"].append(aid)
        return aid

    def _get_line_vertices(self, line_id: str) -> Tuple[str, str]:
        """
        根据边的ID查找并返回其起点和终点的ID。
        
        :param line_id: 边的ID (e.g., "L1")
        :return: 一个包含两个点ID的元组 (start_point_id, end_point_id)
        :raises ValueError: 如果找不到对应的边
        """
        for line in self.data["lines"]:
            if line["id"] == line_id:
                return line["start_point_id"], line["end_point_id"]
        raise ValueError(f"在边列表中未找到ID为 '{line_id}' 的边。")

    # ------------------------------ 几何检查与交点计算 ------------------------------
    def _is_point_on_segment(self, px: sp.Expr, py: sp.Expr, 
                           x1: sp.Expr, y1: sp.Expr, 
                           x2: sp.Expr, y2: sp.Expr) -> bool:
        """判断点是否在线段上"""
        collinear = sp.simplify((px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)) == 0
        if not collinear:
            return False
        x_min = Min(x1, x2)
        x_max = Max(x1, x2)
        y_min = Min(y1, y2)
        y_max = Max(y1, y2)
        in_x_range = sp.And(px >= x_min, px <= x_max)
        in_y_range = sp.And(py >= y_min, py <= y_max)
        return bool(sp.simplify(in_x_range & in_y_range))

    def _segment_circle_intersection(self, 
                                        x1: sp.Expr, y1: sp.Expr,  # 线段起点
                                        x2: sp.Expr, y2: sp.Expr,  # 线段终点
                                        cx: sp.Expr, cy: sp.Expr,  # 圆心
                                        r: sp.Expr) -> List[Tuple[sp.Expr, sp.Expr]]:
            """计算线段与圆的交点（修复虚数解比较错误）"""
            t = sp.symbols('t')
            x = x1 + t*(x2 - x1)
            y = y1 + t*(y2 - y1)
            
            eq = sp.Eq((x - cx)**2 + (y - cy)** 2, r**2)
            eq_simplified = sp.simplify(eq)
            solutions = sp.solve(eq_simplified, t, dict=True)
            intersections = []
            
            for sol in solutions:
                if t not in sol:
                    continue
                    
                t_val = sol[t]
                # 1. 先检查是否为实数解（关键修复：过滤虚数解）
                if not sp.simplify(t_val.is_real):
                    continue
                    
                # 2. 仅对实数解进行范围判断
                t_valid = sp.And(sp.Ge(t_val, 0), sp.Le(t_val, 1))
                if sp.simplify(t_valid):
                    x_val = sp.simplify(x.subs(t, t_val))
                    y_val = sp.simplify(y.subs(t, t_val))
                    
                    # 再次确认坐标为实数
                    if sp.simplify(x_val.is_real) and sp.simplify(y_val.is_real):
                        intersections.append((x_val, y_val))
            
            # 去重处理
            # unique_intersections = []
            # seen = set()
            # for x, y in intersections:
            #     key = (str(sp.simplify(x)), str(sp.simplify(y)))
            #     if key not in seen:
            #         seen.add(key)
            #         unique_intersections.append((x, y))
            
            return intersections

    def _circle_circle_intersection(self, 
                                cx1: sp.Expr, cy1: sp.Expr, r1: sp.Expr,  # 圆1
                                cx2: sp.Expr, cy2: sp.Expr, r2: sp.Expr) -> List[Tuple[sp.Expr, sp.Expr]]:
        """计算两圆的交点（仅保留实数解，完全基于符号表达式计算）"""
        dx = cx2 - cx1
        dy = cy2 - cy1
        d_sq = dx**2 + dy**2
        d = sqrt(d_sq)
        
        if not sp.simplify(d.is_real):
            return []
        
        sum_r = r1 + r2
        diff_r = sp.Abs(r1 - r2)
        if sp.simplify(d > sum_r) or sp.simplify(d < diff_r):
            return []
        
        if sp.simplify(d == 0) and not sp.simplify(r1 == r2):
            return []
        
        try:
            a = (r1**2 - r2**2 + d_sq) / (2 * d)
            h_sq = r1**2 - a**2
            
            if not sp.simplify(h_sq.is_real) or sp.simplify(h_sq < 0):
                return []
            h = sqrt(h_sq)
            
            x2 = cx1 + (a * dx) / d
            y2 = cy1 + (a * dy) / d
            
            x3 = x2 - (h * dy) / d
            y3 = y2 + (h * dx) / d
            x4 = x2 + (h * dy) / d
            y4 = y2 - (h * dx) / d
            
            def is_real_coord(coord):
                return sp.simplify(coord.is_real) and sp.simplify(sp.im(coord) == 0)
            
            valid_points = []
            if is_real_coord(x3) and is_real_coord(y3):
                valid_points.append((sp.simplify(x3), sp.simplify(y3)))
                
            if not sp.simplify(h_sq == 0) and is_real_coord(x4) and is_real_coord(y4):
                valid_points.append((sp.simplify(x4), sp.simplify(y4)))
            
            return valid_points
        except:
            return []
        
    def _segment_segment_intersection(
        self,
        x1: sp.Expr, y1: sp.Expr,
        x2: sp.Expr, y2: sp.Expr,
        x3: sp.Expr, y3: sp.Expr,
        x4: sp.Expr, y4: sp.Expr
    ) -> List[Tuple[sp.Expr, sp.Expr]]:
        """计算两条线段的交点（仅保留线段内部的交点，符号化计算）"""
        def points_equal(p1: Tuple[sp.Expr, sp.Expr], p2: Tuple[sp.Expr, sp.Expr]) -> bool:
            return sp.simplify(sp.Eq(p1[0], p2[0])) and sp.simplify(sp.Eq(p1[1], p2[1]))
        
        if (points_equal((x1, y1), (x3, y3)) or points_equal((x1, y1), (x4, y4)) or
            points_equal((x2, y2), (x3, y3)) or points_equal((x2, y2), (x4, y4))):
            return []
        
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        cross_product = sp.simplify(dx1 * dy2 - dx2 * dy1)
        para = cross_product.is_zero
        if para:
            return []
        
        t, s = sp.symbols('t s')
        eq1 = sp.Eq(x1 + t*(x2 - x1), x3 + s*(x4 - x3))
        eq2 = sp.Eq(y1 + t*(y2 - y1), y3 + s*(y4 - y3))
        
        solutions = sp.solve((eq1, eq2), (t, s), dict=True)
        intersections = []
        
        for sol in solutions:
            if t not in sol or s not in sol:
                continue
            
            t_val = sol[t]
            s_val = sol[s]
            
            if not (sp.simplify(t_val.is_real) and sp.simplify(s_val.is_real)):
                continue
            
            t_valid = sp.Ge(t_val, 0) and sp.Le(t_val, 1)
            s_valid = sp.Ge(s_val, 0) and sp.Le(s_val, 1)
            
            if sp.simplify(t_valid) and sp.simplify(s_valid):
                x = sp.simplify(x1 + t_val*(x2 - x1))
                y = sp.simplify(y1 + t_val*(y2 - y1))
                intersections.append((x, y))
        
        unique_intersections = []
        seen = set()
        for x, y in intersections:
            key = (str(sp.simplify(x)), str(sp.simplify(y)))
            if key not in seen:
                seen.add(key)
                unique_intersections.append((x, y))
        
        return unique_intersections

    def _find_all_intersections(self):
        """查找所有线段与弧、弧与弧的交点"""
        # 1. 收集所有圆（弧）和线段
        circles = [] 
        for arc in self.data["arcs"]:
            center_id = arc["center_point_id"]
            radius_expr = sp.sympify(arc["radius"]["expr"])
            angle_expr = sp.sympify(arc["angle"]["expr"])
            is_complete = sp.simplify(angle_expr) == 2 * pi
            circles.append((center_id, radius_expr, arc["id"], is_complete))
        
        lines = []
        for line in self.data["lines"]:
            lines.append((line["start_point_id"], line["end_point_id"], line["id"]))
            
        for i in range(len(lines)):
            s1_id, e1_id, line1_id = lines[i]
            x1, y1 = self.get_point_coords(s1_id)
            x2, y2 = self.get_point_coords(e1_id)
            for j in range(i + 1, len(lines)):
                s2_id, e2_id, line2_id = lines[j]
                x3, y3 = self.get_point_coords(s2_id)
                x4, y4 = self.get_point_coords(e2_id)
                
                intersections = self._segment_segment_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                line1 = next(l for l in self.data["lines"] if l["id"] == line1_id)
                line2 = next(l for l in self.data["lines"] if l["id"] == line2_id)
                intersection_level = max(line1["level"], line2["level"])
                
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level)
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    p["related_edges"].extend([line1_id, line2_id])

        # 2. 线段与圆的交点
        for (s_id, e_id, line_id) in lines:
            sx, sy = self.get_point_coords(s_id)
            ex, ey = self.get_point_coords(e_id)
            for (c_id, r_expr, arc_id, _) in circles:
                cx, cy = self.get_point_coords(c_id)
                intersections = self._segment_circle_intersection(sx, sy, ex, ey, cx, cy, r_expr)
                
                line = next(l for l in self.data["lines"] if l["id"] == line_id)
                arc = next(a for a in self.data["arcs"] if a["id"] == arc_id)
                intersection_level = max(line["level"], arc["level"])
                
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level) 
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    p["related_edges"].extend([line_id, arc_id])

        # 3. 圆与圆的交点
        for i in range(len(circles)):
            c1_id, r1_expr, arc1_id, _ = circles[i]
            c1x, c1y = self.get_point_coords(c1_id)
            for j in range(i + 1, len(circles)):
                c2_id, r2_expr, arc2_id, _ = circles[j]
                c2x, c2y = self.get_point_coords(c2_id)
                intersections = self._circle_circle_intersection(c1x, c1y, r1_expr, c2x, c2y, r2_expr)
                arc1 = next(a for a in self.data["arcs"] if a["id"] == arc1_id)
                arc2 = next(a for a in self.data["arcs"] if a["id"] == arc2_id)
                intersection_level = max(arc1["level"], arc2["level"])
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level)
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    p["related_edges"].extend([arc1_id, arc2_id])

        self.description_parts.append(f"Found {len(self.data['points'])} total points (including intersections).")

    # ------------------------------ 基础形状生成 ------------------------------
    def generate_base_shape(self) -> str:
        
        # 选择基础形状类型(目前支持圆和多边形)
        # 后续想要支持自定义的base_shape:
        # 1. 预模版，添加新的基础形状类型到base_types列表中
        # 2. 自动生成：在generate_base_shape方法中添加对应的逻辑处理
        
        base_types = self.config["base"]["types"]
        type_weights = self.config["base"].get("type_weights", {})

        weights = []
        for typ in base_types:
            if typ in type_weights:
                weights.append(type_weights[typ])
            elif typ == "polygon":
                weights.append(0.6)
            else:
                weights.append(0.4 / (len(base_types) - (1 if "polygon" in base_types else 0)))

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        base_type = random.choices(base_types, weights=weights, k=1)[0]

        # 基础原点：圆心/中心
        # 代表数学坐标的原点，通常是图形的中心或圆心，特殊图形有特殊设计
        # origin_id = self._add_point(0, 0, is_center=True)
        temp_origin_id = self._add_point(0, 0, is_center=True)

        # if base_type == "polygon":
        #     n = random.choice(self.config["base"]["polygon"]["n_choices"])
        #     side_min, side_max = self.config["base"]["polygon"]["side_range"]
        #     side_length = sp.Integer(random.randint(side_min, side_max))
        #     rot_type = self.config["base"]["polygon"]["rotation_choices_type"]
        #     rotations = [0, sp.pi/(n)] if rot_type == "pi_over_2n" else [0]
        #     # rotation = random.choice(rotations)
        #     # sp.pi/(n) 表示底边平行于x轴
        #     rotation = random.choices(rotations, weights=[0.7, 0.3] if len(rotations) == 2 else [1.0], k=1)[0]
        #     entity_id = self._get_unique_entity_id(f"base_polygon_n{n}")
        #     self.generate_regular_polygon(
        #         n=n,
        #         side_length=side_length,
        #         center_id=origin_id,
        #         rotation=rotation,
        #         entity_id=entity_id,
        #         is_base=True
        #     )
        #     self.description_parts.append(
        #         f"Base shape is a regular {n}-sided polygon (center {origin_id}) with side length {side_length}"
        #     )

        # if base_type == "circle": 
        #     r_min, r_max = self.config["base"]["circle"]["radius_range"]
        #     radius = sp.Integer(random.randint(r_min, r_max))
        #     start_angle = self.config["base"]["circle"]["start_angle"]
        #     end_angle = 2 * pi
        #     entity_id = self._get_unique_entity_id("base_circle")
        #     arc_id = self.generate_circle(
        #         radius=radius,
        #         center_id=origin_id,
        #         start_angle=start_angle,
        #         end_angle=end_angle,
        #         entity_id=entity_id,
        #         is_base=True
        #     )
        #     self.complete_circle_arcs.add(arc_id)
        #     self.description_parts.append(
        #         f"Base shape is a complete circle (center {origin_id}, init point circle_1) with radius {radius}."
        #     )
  
        if base_type == "polygon":
            n = random.choice(self.config["base"]["polygon"]["n_choices"])
            side_min, side_max = self.config["base"]["polygon"]["side_range"]
            side_length = sp.Integer(random.randint(side_min, side_max))
            rot_type = self.config["base"]["polygon"]["rotation_choices_type"]
            rotations = [0, sp.pi/(n)] if rot_type == "pi_over_2n" else [0]
            rotation = random.choices(rotations, weights=[0.7, 0.3] if len(rotations) == 2 else [1.0], k=1)[0]
            # rotation = random.choice(rotations)
            # sp.pi/(n) 表示底边平行于x轴
            
            entity_id = self._get_unique_entity_id(f"base_polygon_n{n}")
            self.generate_regular_polygon(
                n=n,
                side_length=side_length,
                center_id=temp_origin_id,
                rotation=rotation,
                entity_id=entity_id,
                is_base=True
            )
            self.description_parts.append(
                f"Base shape is a regular {n}-sided polygon (center {temp_origin_id}) with side length {side_length}"
            )

        elif base_type == "circle":
            r_min, r_max = self.config["base"]["circle"]["radius_range"]
            radius = sp.Integer(random.randint(r_min, r_max))
            start_angle = self.config["base"]["circle"]["start_angle"]
            end_angle = 2 * sp.pi
            entity_id = self._get_unique_entity_id("base_circle")
            arc_id = self.generate_circle(
                radius=radius,
                center_id=temp_origin_id,
                start_angle=start_angle,
                end_angle=end_angle,
                entity_id=entity_id,
                is_base=True
            )
            self.complete_circle_arcs.add(arc_id)
            self.description_parts.append(
                f"Base shape is a complete circle (center {temp_origin_id}) with radius {radius}."
            )

        elif base_type == "special_triangle":
            tri_subtype = random.choice(["isosceles", "right"])
            config = self.config["base"]["special_triangle"][tri_subtype]

            if tri_subtype == "isosceles":
                top_angle = random.choice(config["angle_choices"])
                length_type = random.choice(["waist", "base"]) if config["length_type"] == "either" else config["length_type"]
                # length_type = random.choice(["waist", "base"])
                len_min, len_max = config["length_range"]
                is_root = random.choice([False, True])
                if is_root:
                    root_val = random.randint(len_min, len_max)
                    length = sp.sympify(f"sqrt({root_val})")
                else:
                    length = sp.Integer(random.randint(len_min, len_max)) 

                # Calculate parameters (infer other values from waist/base length)
                top_angle_rad = sp.rad(top_angle)
                # base_angle_rad = sp.rad((180 - top_angle) / 2)
                base_angle_deg_sym = (S(180) - top_angle) / 2
                base_angle_rad = sp.rad(base_angle_deg_sym)
                
                if length_type == "waist":
                    waist_length = length
                    base_length = sp.sympify(2 * waist_length * sp.sin(top_angle_rad / 2))
                else:
                    base_length = length
                    waist_length = sp.sympify((base_length / 2) / sp.cos(base_angle_rad))

                # Rotation config (if enabled)
                rotate_mode = "original"
                if config["rotate"]["enable"]:
                    rotate_mode = random.choice(config["rotate"]["modes"])

                # Generate isosceles triangle (origin_id = base midpoint)
                entity_id = self._get_unique_entity_id(f"base_isosceles_triangle_{top_angle}deg")
                self.generate_special_triangle(
                    triangle_type="isosceles",
                    params={
                        "top_angle": top_angle,
                        "waist_length": waist_length,
                        "base_length": base_length,
                        "rotate_mode": rotate_mode
                    },
                    origin_id=temp_origin_id,  # Midpoint of the base
                    entity_id=entity_id,
                    is_base=True
                )
                self.description_parts.append(
                    f"Base shape is an isosceles triangle (base midpoint {temp_origin_id}) with top angle {top_angle}°, "
                    f"{length_type} length {length}, rotate mode {rotate_mode}"
                )

            else:  # Right triangle
                # 1.2 Right triangle: origin_id is the midpoint of the hypotenuse
                ratio_mode = random.choice(["manual", "random"]) if config["ratio_mode"] == "either" else config["ratio_mode"]
                if ratio_mode == "manual":
                    # Select ratio manually
                    ratio = random.choice(config["manual_ratios"])
                    # Parse square root expressions in ratio (e.g., "sqrt(3)" → sp.sqrt(3))
                    ratio = [sp.sympify(x) if isinstance(x, str) else x for x in ratio]
                    leg1, leg2 = ratio
                else:
                    # Random ratio combination (select 2 from integers and square roots)
                    pool = []
                    # Integer part: 1-10
                    pool.extend([sp.Integer(i) for i in range(config["random_pool_range"][0], config["random_pool_range"][1]+1)])
                    # Square root part: √1-√100
                    pool.extend([sp.sympify(f"sqrt({i})") for i in range(1, 101)])
                    leg1, leg2 = random.sample(pool, 2)

                # Calculate hypotenuse and its midpoint (origin_id)
                hypotenuse = sp.sqrt(leg1**2 + leg2**2)
                rotate_mode = "original"
                if config["rotate"]["enable"]:
                    rotate_mode = random.choice(config["rotate"]["modes"])

                # Generate right triangle (origin_id = hypotenuse midpoint)
                entity_id = self._get_unique_entity_id(f"base_right_triangle_{leg1}:{leg2}")
                self.generate_special_triangle(
                    triangle_type="right",
                    params={
                        "leg1": leg1,
                        "leg2": leg2,
                        "hypotenuse": hypotenuse,
                        "rotate_mode": rotate_mode
                    },
                    origin_id=temp_origin_id, 
                    entity_id=entity_id,
                    is_base=True
                )
                self.description_parts.append(
                    f"Base shape is a right triangle (hypotenuse midpoint {temp_origin_id}) with legs {leg1}, {leg2}, "
                    f"rotate mode {rotate_mode}"
                )

        elif base_type == "special_rectangle":
            # 2. Special rectangle: origin_id is the center
            config = self.config["base"]["special_rectangle"]
            ratio_mode = random.choice(["manual", "random"]) if config["ratio_mode"] == "either" else config["ratio_mode"]

            if ratio_mode == "manual":
                width, length = random.choice(config["manual_ratios"])
            else:
                width, length = random.sample(range(config["random_range"][0], config["random_range"][1]+1), 2)
                width, length = sorted([width, length])

            # Rotation config (swap width/length if enabled)
            rotate_mode = "original"
            if config["rotate"]["enable"]:
                rotate_mode = random.choice(config["rotate"]["modes"])
                if rotate_mode == "swap_ratio":
                    width, length = length, width 
            
            # width = sp.simplify(width)
            # length = sp.simplify(length)

            # Generate rectangle (origin_id = center)
            entity_id = self._get_unique_entity_id(f"base_rectangle_{width}:{length}")
            self.generate_special_rectangle(
                width=width,
                length=length,
                center_id=temp_origin_id,  # Center
                rotate_mode=rotate_mode,
                entity_id=entity_id,
                is_base=True
            )
            self.description_parts.append(
                f"Base shape is a rectangle (center {temp_origin_id}) with width {width}, length {length}, rotate mode {rotate_mode}"
            )

        elif base_type == "parallelogram":
            # 3. Parallelogram: origin_id is the center
            config = self.config["base"]["parallelogram"]
            base = random.randint(*config["base_range"])
            height = random.randint(*config["height_range"])
            angle_deg = random.choice(config["angle_choices"])
            angle_rad = sp.rad(angle_deg)

            # Rotation config (horizontal flip if enabled)
            rotate_mode = "original"
            if config["rotate"]["enable"]:
                rotate_mode = random.choice(config["rotate"]["modes"])
                if rotate_mode == "flip_horizontal":
                    angle_deg = 180 - angle_deg 

            # Generate parallelogram (origin_id = center)
            entity_id = self._get_unique_entity_id(f"base_parallelogram_{base}x{height}")
            self.generate_parallelogram(
                base=base,
                height=height,
                angle=angle_rad,
                center_id=temp_origin_id, 
                rotate_mode=rotate_mode,
                entity_id=entity_id,
                is_base=True
            )
            self.description_parts.append(
                f"Base shape is a parallelogram (center {temp_origin_id}) with base {base}, height {height}, angle {angle_deg}°, "
                f"rotate mode {rotate_mode}"
            )

        elif base_type == "trapezoid":
            # 4. Isosceles trapezoid: origin_id is the center
            config = self.config["base"]["trapezoid"]["isosceles"]
            # Ensure upper base < lower base
            base1 = random.randint(*config["base1_range"])  # Lower base (longer)
            base2_max = min(base1 - 1, config["base2_range"][1])
            base2 = random.randint(config["base2_range"][0], base2_max)  # Upper base (shorter)
            height = random.randint(*config["height_range"])
            angle_deg = random.choice(config["angle_choices"])
            angle_rad = sp.rad(angle_deg)

            # Rotation config (vertical flip if enabled)
            rotate_mode = "original"
            if config["rotate"]["enable"]:
                rotate_mode = random.choice(config["rotate"]["modes"])

            # Generate isosceles trapezoid (origin_id = center)
            entity_id = self._get_unique_entity_id(f"base_trapezoid_{base1}:{base2}")
            self.generate_trapezoid(
                base1=base1,  # Lower base (longer)
                base2=base2,  # Upper base (shorter)
                height=height,
                angle=angle_rad,
                center_id=temp_origin_id,  # Center
                rotate_mode=rotate_mode,
                entity_id=entity_id,
                is_base=True
            )
            self.description_parts.append(
                f"Base shape is an isosceles trapezoid (center {temp_origin_id}) with bases {base1}, {base2}, height {height}, "
                f"angle {angle_deg}°, rotate mode {rotate_mode}"
            )

            
        self.base_entity_id = entity_id
        self.current_entities.append(entity_id)
        return entity_id

    def generate_regular_polygon(
        self,
        n: int,
        side_length: sp.Expr,
        center_id: str,
        rotation: sp.Expr,
        entity_id: Optional[str] = None,
        is_base: bool = False
    ) -> str:
        cx, cy = self.get_point_coords(center_id)
        r_expr = simplify(side_length / (2 * sin(pi / n)))
        r_in_expr = simplify(r_expr * cos(pi / n))

        vertices = []
        for i in range(n):
            angle = simplify(rotation + 2 * pi * i / n)
            x = simplify(cx + r_expr * cos(angle))
            y = simplify(cy + r_expr * sin(angle))
            vertices.append(self._add_point(x, y, is_circle_init=False, level=1 if is_base else 2))

        lines = []
        for i in range(n):
            start_id = vertices[i]
            end_id = vertices[(i + 1) % n]
            line_id = self._add_line(start_id, end_id)
            lines.append(line_id)

        entity_id = entity_id or self._get_unique_entity_id(f"polygon_n{n}")
        self.data["entities"].append({
            "id": entity_id,
            "type": "polygon",
            "center_id": center_id,
            "vertices": vertices,
            "lines": lines,
            "n": n,
            "side_length": {"expr": self._format_expr(side_length), "latex": sp.latex(side_length)},
            "radius": {"expr": self._format_expr(r_expr), "latex": sp.latex(r_expr)},
            "inner_radius": {"expr": self._format_expr(r_in_expr), "latex": sp.latex(r_in_expr)},
            "rotation": {"expr": self._format_expr(rotation), "latex": sp.latex(rotation)},
            "is_base": is_base
        })
        return entity_id

    def generate_circle(
        self,
        radius: sp.Expr,
        center_id: str,
        start_angle: sp.Expr = 0,
        end_angle: sp.Expr = 2*pi,
        entity_id: Optional[str] = None,
        is_base: bool = False
    ) -> str:
        """生成圆（圆心O系列，完整圆初始点circle_x，弧先查重）"""
        cx, cy = self.get_point_coords(center_id)
        r_expr = simplify(radius)
        if r_expr < 0:
            raise ValueError("半径不能为负")

        is_complete = sp.simplify(end_angle - start_angle) == 2 * pi
        start_x = simplify(cx + r_expr * cos(start_angle))
        start_y = simplify(cy + r_expr * sin(start_angle))
        start_id = self._add_point(start_x, start_y, is_circle_init=is_complete, level=1 if is_base else 2)

        end_x = simplify(cx + r_expr * cos(end_angle))
        end_y = simplify(cy + r_expr * sin(end_angle))
        end_id = start_id if is_complete else self._add_point(end_x, end_y, is_circle_init=False)

        angle_expr = simplify(end_angle - start_angle)
        arc_id = self._add_arc(
            start_pid=start_id,
            end_pid=end_id,
            center_pid=center_id,
            radius_expr=r_expr,
            angle_expr=angle_expr,
            is_complete=is_complete
        )

        entity_id = entity_id or self._get_unique_entity_id("circle")
        self.data["entities"].append({
            "id": entity_id,
            "type": "circle",
            "center_id": center_id,
            "start_id": start_id,
            "end_id": end_id,
            "arcs": arc_id,
            "radius": {"expr": self._format_expr(r_expr), "latex": sp.latex(r_expr)},
            "start_angle": {"expr": self._format_expr(start_angle), "latex": sp.latex(start_angle)},
            "end_angle": {"expr": self._format_expr(end_angle), "latex": sp.latex(end_angle)},
            "is_complete": is_complete,
            "is_base": is_base
        })
        return arc_id

    def generate_special_triangle(
        self,
        triangle_type: str,  # "isosceles" or "right"
        params: dict,
        origin_id: str, 
        entity_id: Optional[str] = None,
        is_base: bool = False
    ) -> str:
        """生成特殊三角形（等腰/直角），统一参数格式为expr+latex，兼容符号计算"""
        level = 1 if is_base else 2
        entity_id = entity_id or self._get_unique_entity_id(f"special_triangle_{triangle_type}")
        
        # --- 核心修正：获取基准点 origin_id 的世界坐标 ---
        ox, oy = self.get_point_coords(origin_id)
        
        rotate_mode = params["rotate_mode"]
        entity_data = {
            "id": entity_id,
            "type": "special_triangle",
            "subtype": triangle_type,
            "center_id": origin_id,
            "rotate_mode": rotate_mode,
            "is_base": is_base
        }

        if triangle_type == "isosceles":
            # 等腰三角形参数处理
            top_angle = params["top_angle"]
            waist_length = simplify(params["waist_length"]) 
            base_length = simplify(params["base_length"])
            
            # 计算相对于 origin_id 的顶点坐标
            base_half = simplify(S(base_length) / 2)
            # 底边左点相对坐标
            rx_a, ry_a = -base_half, 0
            # 底边右点相对坐标
            rx_b, ry_b = base_half, 0
            
            height = simplify(sp.sqrt(waist_length**2 - base_half**2))  # 高
            y_coord_rel = height if rotate_mode == "original" else -height
            # 顶角顶点相对坐标
            rx_c, ry_c = 0, y_coord_rel
            
            # --- 核心修正：计算最终世界坐标 ---
            point_a = self._add_point(ox + rx_a, oy + ry_a, level=level)
            point_b = self._add_point(ox + rx_b, oy + ry_b, level=level)
            point_c = self._add_point(ox + rx_c, oy + ry_c, level=level)
            
            # 添加边
            line_ab = self._add_line(point_a, point_b)
            line_bc = self._add_line(point_b, point_c)
            line_ca = self._add_line(point_c, point_a)
            
            # 更新组件和实体数据
            entity_data.update({
                "vertices": [point_a, point_b, point_c],
                "lines": [line_ab, line_bc, line_ca],
                "top_angle": top_angle,
                "top_point": point_c,
                "waist_length": {
                    "expr": self._format_expr(waist_length),
                    "latex": sp.latex(waist_length)
                },
                "base_length": {
                    "expr": self._format_expr(base_length),
                    "latex": sp.latex(base_length)
                },
                "height": {
                    "expr": self._format_expr(height),
                    "latex": sp.latex(height)
                }
            })

        elif triangle_type == "right":
            # 直角三角形参数处理
            leg1 = simplify(params["leg1"])
            leg2 = simplify(params["leg2"])
            hypotenuse = simplify(params["hypotenuse"])
            
            # --- 核心修正：所有坐标计算都基于相对坐标 ---
            rx_a, ry_a, rx_b, ry_b, rx_c, ry_c = 0, 0, 0, 0, 0, 0
            
            leg1_half = simplify(S(leg1) / 2)
            leg2_half = simplify(S(leg2) / 2)

            if rotate_mode == "original":
                rx_a, ry_a = -leg1_half, leg2_half
                rx_b, ry_b = leg1_half, -leg2_half  # 直角顶点
                rx_c, ry_c = leg1_half, leg2_half
            elif rotate_mode == "rotate_90":
                rx_a, ry_a = -leg2_half, -leg1_half
                rx_b, ry_b = leg2_half, leg1_half    # 直角顶点
                rx_c, ry_c = -leg2_half, leg1_half
            elif rotate_mode == "rotate_180":
                rx_a, ry_a = leg1_half, -leg2_half
                rx_b, ry_b = -leg1_half, leg2_half   # 直角顶点
                rx_c, ry_c = -leg1_half, -leg2_half
            elif rotate_mode == "rotate_270":
                rx_a, ry_a = leg2_half, leg1_half
                rx_b, ry_b = -leg2_half, -leg1_half  # 直角顶点
                rx_c, ry_c = leg2_half, -leg1_half
            
            
            # --- 核心修正：计算最终世界坐标 ---
            point_a = self._add_point(ox + rx_a, oy + ry_a, level=level)
            point_b = self._add_point(ox + rx_b, oy + ry_b, level=level)
            point_c = self._add_point(ox + rx_c, oy + ry_c, level=level)
            
            # 添加边
            line_ab = self._add_line(point_a, point_b)  # 斜边
            line_bc = self._add_line(point_b, point_c)  # 直角边1
            line_ca = self._add_line(point_c, point_a)  # 直角边2
            
            # 更新组件和实体数据
            entity_data.update({
                "vertices": [point_a, point_b, point_c],
                "lines": [line_ab, line_bc, line_ca],
                "leg1": {
                    "expr": self._format_expr(leg1),
                    "latex": sp.latex(leg1)
                },
                "leg2": {
                    "expr": self._format_expr(leg2),
                    "latex": sp.latex(leg2)
                },
                "hypotenuse": {
                    "expr": self._format_expr(hypotenuse),
                    "latex": sp.latex(hypotenuse)
                }
            })

        self.data["entities"].append(entity_data)
        return entity_id

    def generate_special_rectangle(
        self,
        width: sp.Expr,
        length: sp.Expr,
        center_id: str,
        rotate_mode: str,
        entity_id: Optional[str] = None,
        is_base: bool = False
    ) -> str:
        """生成特殊矩形，统一参数格式为expr+latex，基于中心对称"""
        level = 1 if is_base else 2
        entity_id = entity_id or self._get_unique_entity_id(f"special_rectangle_{width}:{length}")
        
        # --- 核心修正：获取基准点 center_id 的世界坐标 ---
        cx, cy = self.get_point_coords(center_id)
        
        # 符号化简参数
        width = simplify(width)
        length = simplify(length)
        half_w = simplify(width / 2)
        half_l = simplify(length / 2)
        
        # --- 核心修正：计算相对于中心点的相对坐标 ---
        # 左下
        rx_a, ry_a = -half_w, -half_l
        # 右下
        rx_b, ry_b = half_w, -half_l
        # 右上
        rx_c, ry_c = half_w, half_l
        # 左上
        rx_d, ry_d = -half_w, half_l
        
        # --- 核心修正：计算最终世界坐标 ---
        point_a = self._add_point(cx + rx_a, cy + ry_a, level=level)
        point_b = self._add_point(cx + rx_b, cy + ry_b, level=level)
        point_c = self._add_point(cx + rx_c, cy + ry_c, level=level)
        point_d = self._add_point(cx + rx_d, cy + ry_d, level=level)
        
        # 添加边
        line_ab = self._add_line(point_a, point_b)
        line_bc = self._add_line(point_b, point_c)
        line_cd = self._add_line(point_c, point_d)
        line_da = self._add_line(point_d, point_a)
        
        # 实体数据（统一expr+latex格式）
        self.data["entities"].append({
            "id": entity_id,
            "type": "special_rectangle",
            "center_id": center_id,
            "vertices": [point_a, point_b, point_c, point_d],
            "lines": [line_ab, line_bc, line_cd, line_da],
            "width": {
                "expr": self._format_expr(width),
                "latex": sp.latex(width)
            },
            "length": {
                "expr": self._format_expr(length),
                "latex": sp.latex(length)
            },
            "rotate_mode": rotate_mode,
            "is_base": is_base
        })
        return entity_id

    def generate_parallelogram(
        self,
        base: sp.Expr,
        height: sp.Expr,
        angle: sp.Expr,
        center_id: str,
        rotate_mode: str,
        entity_id: Optional[str] = None,
        is_base: bool = False
    ) -> str:
        """生成平行四边形，统一参数格式为expr+latex，支持符号计算"""
        level = 1 if is_base else 2
        entity_id = entity_id or self._get_unique_entity_id(f"parallelogram_{base}x{height}")
        
        # 符号化简参数
        base = simplify(base)
        height = simplify(height)
        angle = simplify(angle)
        
        # 计算侧边长度和水平偏移
        side_len = simplify(height / sp.sin(angle))
        offset_x = simplify(side_len * sp.cos(angle))
        
        # 翻转处理（水平翻转时偏移取反）
        actual_offset = offset_x if rotate_mode == "original" else -offset_x
        
        # 获取中心点坐标
        cx, cy = self.get_point_coords(center_id)
        
        # 重新计算四个顶点坐标（基于对角线交点）
        # 计算对角线的一半长度
        half_diag1_x = simplify(base / 2)
        half_diag1_y = 0
        
        half_diag2_x = simplify(actual_offset / 2)
        half_diag2_y = simplify(height / 2)
        
        # 四个顶点坐标（通过对角线中点计算）
        point_a = self._add_point(
            cx - half_diag1_x - half_diag2_x,
            cy - half_diag1_y - half_diag2_y,
            level=level
        )
        point_b = self._add_point(
            cx + half_diag1_x - half_diag2_x,
            cy + half_diag1_y - half_diag2_y,
            level=level
        )
        point_c = self._add_point(
            cx + half_diag1_x + half_diag2_x,
            cy + half_diag1_y + half_diag2_y,
            level=level
        )
        point_d = self._add_point(
            cx - half_diag1_x + half_diag2_x,
            cy - half_diag1_y + half_diag2_y,
            level=level
        )
        
        # 添加边
        line_ab = self._add_line(point_a, point_b)
        line_bc = self._add_line(point_b, point_c)
        line_cd = self._add_line(point_c, point_d)
        line_da = self._add_line(point_d, point_a)
        
        # 实体数据
        self.data["entities"].append({
            "id": entity_id,
            "type": "parallelogram",
            "center_id": center_id,
            "vertices": [point_a, point_b, point_c, point_d],
            "lines": [line_ab, line_bc, line_cd, line_da],
            "base": {
                "expr": self._format_expr(base),
                "latex": sp.latex(base)
            },
            "height": {
                "expr": self._format_expr(height),
                "latex": sp.latex(height)
            },
            "angle_rad": {  # 弧度制
                "expr": self._format_expr(angle),
                "latex": sp.latex(angle)
            },
            "angle_deg": {  # 角度制（便于阅读）
                "expr": self._format_expr(sp.deg(angle)),
                "latex": sp.latex(sp.deg(angle))
            },
            "rotate_mode": rotate_mode,
            "is_base": is_base
        })
        return entity_id

    def generate_trapezoid(
        self,
        base1: sp.Expr,  # 下底（长边）
        base2: sp.Expr,  # 上底（短边）
        height: sp.Expr,
        angle: sp.Expr,
        center_id: str,
        rotate_mode: str,
        entity_id: Optional[str] = None,
        is_base: bool = False
    ) -> str:
        """生成等腰梯形，统一参数格式为expr+latex，校验上下底关系"""
        base1 = simplify(base1)
        base2 = simplify(base2)
        if base2 >= base1:
            raise ValueError(f"Upper base (base2={base2}) must be shorter than lower base (base1={base1})")
        
        level = 1 if is_base else 2
        entity_id = entity_id or self._get_unique_entity_id(f"trapezoid_{base1}:{base2}")
        
        cx, cy = self.get_point_coords(center_id)
        
        # 翻转处理（垂直翻转时交换上下底）
        if rotate_mode == "flip_vertical":
            base1, base2 = base2, base1
        
        # 符号化简参数
        height = simplify(height)
        angle = simplify(angle)
        half_base1 = simplify(base1 / 2)
        half_base2 = simplify(base2 / 2)
        overhang = simplify((base1 - base2) / 2)  # 下底超出上底的长度
        leg_len = simplify(height / sp.sin(angle))  # 腰长
        
        rx_a, ry_a = -half_base1, -height/2
        rx_b, ry_b = half_base1, -height/2
        rx_c, ry_c = half_base2, height/2
        rx_d, ry_d = -half_base2, height/2
        
        # --- 核心修正：计算最终世界坐标（相对坐标 + 中心点坐标） ---
        point_a = self._add_point(cx + rx_a, cy + ry_a, level=level)  # 左下
        point_b = self._add_point(cx + rx_b, cy + ry_b, level=level)  # 右下
        point_c = self._add_point(cx + rx_c, cy + ry_c, level=level)  # 右上
        point_d = self._add_point(cx + rx_d, cy + ry_d, level=level)  # 左上
        
        # 添加边
        line_ab = self._add_line(point_a, point_b)  # 下底
        line_bc = self._add_line(point_b, point_c)  # 右腰
        line_cd = self._add_line(point_c, point_d)  # 上底
        line_da = self._add_line(point_d, point_a)  # 左腰
        
        # 实体数据
        self.data["entities"].append({
            "id": entity_id,
            "type": "trapezoid",
            "subtype": "isosceles",
            "center_id": center_id,
            "vertices": [point_a, point_b, point_c, point_d],
            "lines": [line_ab, line_bc, line_cd, line_da],
            "base1": {  # 下底（长边）
                "expr": self._format_expr(base1),
                "latex": sp.latex(base1)
            },
            "base2": {  # 上底（短边）
                "expr": self._format_expr(base2),
                "latex": sp.latex(base2)
            },
            "height": {
                "expr": self._format_expr(height),
                "latex": sp.latex(height)
            },
            "angle_rad": {
                "expr": self._format_expr(angle),
                "latex": sp.latex(angle)
            },
            "angle_deg": {
                "expr": self._format_expr(sp.deg(angle)),
                "latex": sp.latex(sp.deg(angle))
            },
            "leg_length": {  # 腰长
                "expr": self._format_expr(leg_len),
                "latex": sp.latex(leg_len)
            },
            "rotate_mode": rotate_mode,
            "is_base": is_base
        })
        return entity_id

    def _calculate_new_entity_intersections(self, new_id: str):
        """计算新生成entity的边/弧与所有已有元素的交点"""
        # 提取新entity的所有边和弧ID
        line_id_pattern = re.compile(r'^L\d+$')
        new_entity = self.get_entity(new_id)
        new_arcs = []
        new_lines = []
        
        if "lines" in new_entity:
            new_lines = [cid for cid in new_entity["lines"] if line_id_pattern.match(cid)]
        if "arcs" in new_entity:
            new_arcs = [cid for cid in new_entity["arcs"] if cid.startswith("Arc")]
        
        # 收集所有已有元素
        all_existing_lines = [
            (line["start_point_id"], line["end_point_id"], line["id"])
            for line in self.data["lines"]
            if line["id"] not in new_lines
        ]
        all_existing_circles = [
            (arc["center_point_id"], sp.sympify(arc["radius"]["expr"]), arc["id"], arc["is_complete"])
            for arc in self.data["arcs"]
            if arc["id"] not in new_arcs
        ]
        
        # 新边 ↔ 已有边 的交点
        for new_line_id in new_lines:
            new_line = next(l for l in self.data["lines"] if l["id"] == new_line_id)
            sx, sy = self.get_point_coords(new_line["start_point_id"])
            ex, ey = self.get_point_coords(new_line["end_point_id"])
            
            for (s_id, e_id, old_line_id) in all_existing_lines:
                old_line = next(l for l in self.data["lines"] if l["id"] == old_line_id)
                intersection_level = max(new_line["level"], old_line["level"])
                
                ox1, oy1 = self.get_point_coords(s_id)
                ox2, oy2 = self.get_point_coords(e_id)
                intersections = self._segment_segment_intersection(sx, sy, ex, ey, ox1, oy1, ox2, oy2)
                
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level)
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    if new_line_id not in p["related_edges"]:
                        p["related_edges"].append(new_line_id)
                    if old_line_id not in p["related_edges"]:
                        p["related_edges"].append(old_line_id)
        
        # 新边 ↔ 已有圆 的交点
        for new_line_id in new_lines:
            new_line = next(l for l in self.data["lines"] if l["id"] == new_line_id)
            sx, sy = self.get_point_coords(new_line["start_point_id"])
            ex, ey = self.get_point_coords(new_line["end_point_id"])
            
            for (c_id, r_expr, arc_id, _) in all_existing_circles:
                old_arc = next(a for a in self.data["arcs"] if a["id"] == arc_id)
                intersection_level = max(new_line["level"], old_arc["level"])
                
                cx, cy = self.get_point_coords(c_id)
                intersections = self._segment_circle_intersection(sx, sy, ex, ey, cx, cy, r_expr)
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level)
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    if new_line_id not in p["related_edges"]:
                        p["related_edges"].append(new_line_id)
                    if arc_id not in p["related_edges"]:
                        p["related_edges"].append(arc_id)
        
        # 新弧 ↔ 已有边/圆 的交点
        new_circles = [
            (arc["center_point_id"], sp.sympify(arc["radius"]["expr"]), arc["id"], arc["is_complete"])
            for arc in self.data["arcs"]
            if arc["id"] in new_arcs
        ]
        
        for (c_id, r_expr, new_arc_id, _) in new_circles:
            new_arc = next(a for a in self.data["arcs"] if a["id"] == new_arc_id)
            cx, cy = self.get_point_coords(c_id)
            for (s_id, e_id, old_line_id) in all_existing_lines:
                old_line = next(l for l in self.data["lines"] if l["id"] == old_line_id)
                intersection_level = max(new_arc["level"], old_line["level"])
                
                ox1, oy1 = self.get_point_coords(s_id)
                ox2, oy2 = self.get_point_coords(e_id)
                intersections = self._segment_circle_intersection(ox1, oy1, ox2, oy2, cx, cy, r_expr)
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level)
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    if new_arc_id not in p["related_edges"]:
                        p["related_edges"].append(new_arc_id)
                    if old_line_id not in p["related_edges"]:
                        p["related_edges"].append(old_line_id)
            
            # 新弧 ↔ 已有圆
            for (oc_id, or_expr, old_arc_id, _) in all_existing_circles:
                old_arc = next(a for a in self.data["arcs"] if a["id"] == old_arc_id)
                intersection_level = max(new_arc["level"], old_arc["level"])
                
                ocx, ocy = self.get_point_coords(oc_id)
                intersections = self._circle_circle_intersection(cx, cy, r_expr, ocx, ocy, or_expr)
                for (x, y) in intersections:
                    pid = self._add_point(x, y, is_circle_init=False, level=intersection_level)
                    p = next(p for p in self.data["points"] if p["id"] == pid)
                    if new_arc_id not in p["related_edges"]:
                        p["related_edges"].append(new_arc_id)
                    if old_arc_id not in p["related_edges"]:
                        p["related_edges"].append(old_arc_id)

    # ------------------------------ 衍生规则 ------------------------------
    def _get_rule_params(self, base_entity: Dict, rule_name: str) -> Dict:
        
        entity_type = base_entity["type"]
        supported_rules = self.config["derivation"]["rules"][entity_type]["supported_rules"]

        for rule in supported_rules:
            if rule["name"] == rule_name:
                return rule.copy()
        
    def _rule_concentric(self, base_entity: Dict) -> str:
        base_type = base_entity["type"]
        center_id = self._get_center_id(base_entity)
        
        rule_params = self._get_rule_params(base_entity, "concentric")
        scale_choices = rule_params.get("scale_choices", [1])
        scale = Rational(random.choice(scale_choices))

        if base_type == "polygon":
            n = base_entity["n"]
            side_expr = nsimplify(sp.sympify(base_entity["side_length"]["expr"]))
            new_side = nsimplify(side_expr * scale)
            rotation = simplify(sp.sympify(base_entity["rotation"]["expr"]))
            entity_id = self._get_unique_entity_id(f"concentric_polygon_n{n}")
            new_id = self.generate_regular_polygon(
                n=n, side_length=new_side, center_id=center_id, rotation=rotation, entity_id=entity_id, is_base=False
            )
            self.description_parts.append(
                f"Concentric derivation: {n}-sided polygon (center {center_id}) scaled by {scale}."
            )

        else:
            r_expr = simplify(sp.sympify(base_entity["radius"]["expr"]))
            new_r = simplify(r_expr * scale)
            start_angle = simplify(sp.sympify(base_entity["start_angle"]["expr"]))
            end_angle = 2*pi if base_entity.get("is_complete") else simplify(sp.sympify(base_entity["end_angle"]["expr"]))
            entity_id = self._get_unique_entity_id("concentric_circle")
            arc_id = self.generate_circle(
                radius=new_r, center_id=center_id, start_angle=start_angle, end_angle=end_angle, entity_id=entity_id, is_base=False
            )
            new_id = entity_id
            if end_angle == 2*pi:
                self.complete_circle_arcs.add(arc_id)
            self.description_parts.append(
                f"Concentric derivation: circle (center {center_id}) scaled by {scale} (new radius {new_r})."
            )

        return new_id
    
    def _rule_translation(self, base_entity: Dict) -> str:
        base_type = base_entity["type"]
        base_center_id = self._get_center_id(base_entity)
        base_cx, base_cy = self.get_point_coords(base_center_id)
        translate_mode = random.choices(["touch", "half"], weights=[0.7, 0.3], k=1)[0]
        
        if base_type == "polygon":
            
            # 只有沿着某边方向和垂直某边方向，首先如果是touch这个模式的话，移动的距离和多边形的n和方向有关；
            # 如果方向是垂直边，n为奇数，则移动外接圆半径加内接圆半径，n为偶数，则移动2倍内接圆半径；
            # 如果方向是沿着边，这是移动边长的长度；
            # 除了touch是half模式，就移动touch的一半距离
            
            n = base_entity["n"]
            r = simplify(sp.sympify(base_entity["radius"]["expr"]))
            r_in = simplify(sp.sympify(base_entity["inner_radius"]["expr"])) 
            side_len = simplify(sp.sympify(base_entity["side_length"]["expr"]))
            rotation = simplify(sp.sympify(base_entity["rotation"]["expr"]))
            
            directions = [f"along_side_{i}" for i in range(n)] + [f"perp_side_{i}" for i in range(n)] + ["up", "down", "left", "right"]
            direction = random.choice(directions)
            translate_mode = random.choices(["touch", "half"], weights=[0.6, 0.4], k=1)[0]
            
            if direction.startswith("perp_side_"):
                side_idx = int(direction.split("_")[-1])
                vertices = [c for c in base_entity["vertices"] if any(p["id"] == c and not p["is_center"] for p in self.data["points"])]
                p1_id, p2_id = vertices[side_idx], vertices[(side_idx + 1) % n]
                x1, y1 = self.get_point_coords(p1_id)
                x2, y2 = self.get_point_coords(p2_id)
                
                edge_dx = simplify(x2 - x1)
                edge_dy = simplify(y2 - y1)
                edge_len = simplify(sqrt(edge_dx**2 + edge_dy**2))
                perp_dx = simplify(-edge_dy / edge_len)
                perp_dy = simplify(edge_dx / edge_len)
                
                if translate_mode == "touch":
                    dist = simplify(r_in + r) if n % 2 == 1 else simplify(2 * r_in)
                else:
                    dist = simplify((r_in + r) / 2) if n % 2 == 1 else simplify(r_in)
                
                tx = simplify(dist * perp_dx)
                ty = simplify(dist * perp_dy)
                dir_desc = f"perpendicular to side {side_idx + 1}"
                if translate_mode == "touch":
                    dist_desc = "with distance equal to sum of inscribed and circumscribed circle radii" if n % 2 == 1 else "with distance equal to twice the inscribed circle radius"
                else:
                    dist_desc = "with distance equal to half the sum of inscribed and circumscribed circle radii" if n % 2 == 1 else "with distance equal to the inscribed circle radius"
            
            elif direction.startswith("along_side_"):
                side_idx = int(direction.split("_")[-1])
                vertices = [c for c in base_entity["vertices"] if any(p["id"] == c and not p["is_center"] for p in self.data["points"])]
                p1_id, p2_id = vertices[side_idx], vertices[(side_idx + 1) % n]
                x1, y1 = self.get_point_coords(p1_id)
                x2, y2 = self.get_point_coords(p2_id)
                
                edge_dx = simplify(x2 - x1)
                edge_dy = simplify(y2 - y1)
                edge_len = simplify(sqrt(edge_dx**2 + edge_dy**2))
                along_dx = simplify(edge_dx / edge_len)
                along_dy = simplify(edge_dy / edge_len)
                
                dist = simplify(side_len) if translate_mode == "touch" else simplify(side_len / 2)
                tx = simplify(dist * along_dx)
                ty = simplify(dist * along_dy)
                dir_desc = f"along side {side_idx + 1}"
                dist_desc = f"with distance equal to the {'' if translate_mode == 'touch' else 'half '}side length"
            
            else:
                dir_angle_map = {
                    "up": pi/2,
                    "down": pi/2,
                    "left": pi,
                    "right": 0
                }
                angle = dir_angle_map[direction]
                dir_desc = direction
                
                distance_types = [
                    ("side length", side_len),
                    ("inscribed circle radius", r_in),
                    ("circumscribed circle radius", r)
                ]
                dist_name, base_dist = random.choice(distance_types)
                
                if translate_mode == "touch":
                    dist = simplify(base_dist)
                    dist_desc = f"with distance equal to the {dist_name}"
                else:
                    dist = simplify(base_dist * 2)
                    dist_desc = f"with distance equal to twice the {dist_name}"
                
                tx = simplify(dist * cos(angle))
                ty = simplify(dist * sin(angle))
            
            new_cx = simplify(base_cx + tx)
            new_cy = simplify(base_cy + ty)
            new_center_id = self._add_point(new_cx, new_cy, is_center=True)
            entity_id = self._get_unique_entity_id(f"translated_polygon_n{n}")
            new_id = self.generate_regular_polygon(
                n=n, side_length=side_len, center_id=new_center_id, rotation=rotation, entity_id=entity_id, is_base=False
            )
            self.description_parts.append(
                f"Translation derivation: {n}-sided polygon (new center {new_center_id}) translated {dir_desc}, {dist_desc}."
            )
        
        else:
            r = simplify(sp.sympify(base_entity["radius"]["expr"]))
            direction = random.choice(["up", "down", "left", "right"])
            dist = simplify(2 * r if translate_mode == "touch" else r)
            dir_vector_map = {
                "up": (0, dist),
                "down": (0, -dist),
                "left": (-dist, 0),
                "right": (dist, 0)
            }
            tx, ty = dir_vector_map[direction]
            dir_desc = direction
            dist_desc = f"with distance equal to the {'' if translate_mode == 'touch' else ''}diameter (2×radius)" if translate_mode == "touch" else f"with distance equal to the radius"
            
            new_cx = simplify(base_cx + tx)
            new_cy = simplify(base_cy + ty)
            new_center_id = self._add_point(new_cx, new_cy, is_center=True)
            start_angle = simplify(sp.sympify(base_entity["start_angle"]["expr"]))
            end_angle = 2*pi if base_entity.get("is_complete") else simplify(sp.sympify(base_entity["end_angle"]["expr"]))
            entity_id = self._get_unique_entity_id("translated_circle")
            arc_id = self.generate_circle(
                radius=r, center_id=new_center_id, start_angle=start_angle, end_angle=end_angle,
                entity_id=entity_id, is_base=False
            )
            new_id = entity_id
            if end_angle == 2*pi:
                self.complete_circle_arcs.add(arc_id)
            self.description_parts.append(
                f"Translation derivation: circle (new center {new_center_id}) translated {dir_desc}, {dist_desc}."
            )
        
        return new_id
    
    def _rule_circum_inscribe(self, base_entity: Dict) -> str:
        
        # 允许内外接圆和内外接多边形
        # 当n<=6时允许选择内外接多边形；n>6时生成圆（n太大的时候生成的内外接多边形太密集）
        
        base_type = base_entity["type"]
        base_center_id = self._get_center_id(base_entity)

        if base_type == "polygon":
            n = base_entity["n"]
            r_expr = simplify(sp.sympify(base_entity["radius"]["expr"]))
            is_circum = random.choice([True, False])
            rel_type = "circumscribed" if is_circum else "inscribed" 
            
            if n <= 6:
                generate_polygon = random.choice([True, False])
            else:
                generate_polygon = False

            if not generate_polygon:
                new_r = r_expr if is_circum else simplify(r_expr * cos(pi / n))
                entity_id = self._get_unique_entity_id(f"{rel_type}circle_polygon")
                arc_id = self.generate_circle(
                    radius=new_r, center_id=base_center_id, start_angle=0, end_angle=2*pi, entity_id=entity_id, is_base=False
                )
                new_id = entity_id
                self.complete_circle_arcs.add(arc_id)
                self.description_parts.append(
                    f"{rel_type.capitalize()} circle (center {base_center_id}) around {n}-sided polygon."
                )
            else:
                new_n = n
                new_side = (simplify(2 * r_expr * sin(pi / new_n)) 
                            if is_circum 
                            else simplify(2 * r_expr * sin(pi / (2 * new_n))))
                rotation = simplify(sp.sympify(base_entity["rotation"]["expr"]))
                entity_id = self._get_unique_entity_id(f"{rel_type}polygon_n{new_n}")
                new_id = self.generate_regular_polygon(
                    n=new_n, side_length=new_side, center_id=base_center_id, rotation=rotation, entity_id=entity_id, is_base=False
                )
                self.description_parts.append(
                    f"{rel_type.capitalize()} {new_n}-sided polygon (center {base_center_id}) around {n}-sided polygon."
                )

            new_entity = self.get_entity(new_id)
            new_entity["rel_type"] = rel_type

        else:
            r_expr = simplify(sp.sympify(base_entity["radius"]["expr"]))
            rule_params = self._get_rule_params(base_entity, "circum_inscribe")
            n_choices = rule_params.get("n_choices", [1])
            new_n = random.choice(n_choices)
            
            is_circum = random.choice([True, False])
            rel_type = "circumscribed" if is_circum else "inscribed"
            new_side = (simplify(2 * r_expr * sin(pi / new_n)) 
                        if is_circum 
                        else simplify(2 * r_expr * tan(pi / new_n)))
            entity_id = self._get_unique_entity_id(f"polygon_{rel_type}circle_n{new_n}")
            new_id = self.generate_regular_polygon(
                n=new_n, side_length=new_side, center_id=base_center_id, rotation=0, entity_id=entity_id, is_base=False
            )
            
            new_entity = self.get_entity(new_id)
            new_entity["rel_type"] = rel_type
            self.description_parts.append(
                f"{rel_type.capitalize()} {new_n}-sided polygon (center {base_center_id}) around circle."
            )

        return new_id

    def _rule_vertex_on_center(self, base_entity: Dict) -> str:
        """顶点在中心衍生"""
        base_center_id = self._get_center_id(base_entity)
        base_cx, base_cy = self.get_point_coords(base_center_id)
        
        
        rule_params = self._get_rule_params(base_entity, "vertex_on_center")
        n_choices = rule_params.get("n_choices", [1])
        param_choices = rule_params.get("param_choices", ["side_length", "radius"])

        # n_choices = self.config["derivation"]["rules"]["vertex_on_center"].get("n_choices", [3, 4, 5, 6])
        # param_choices = self.config["derivation"]["rules"]["vertex_on_center"].get("param_choices", ["side_length", "radius"])
        
        new_n = random.choice(n_choices)
        param_type = random.choice(param_choices)

        base_type = base_entity["type"]
        base_r = None 
        base_side = None

        if base_type == "polygon":
            # 对于多边形，我们只能直接获取边长
            base_side = simplify(sp.sympify(base_entity["side_length"]["expr"]))
            base_n = base_entity["n"] # 获取基础多边形的边数

            # 如果 param_type 是 "radius"，我们需要根据边长和边数计算出外接圆半径
            # 正多边形外接圆半径公式: R = side_length / (2 * sin(π / n))
            if param_type == "radius":
                base_r = simplify(base_side / (2 * sp.sin(pi / base_n)))
                param_desc = f"radius equal to base {base_n}-gon's circumradius"
            else: # param_type == "side_length"
                param_desc = f"side length equal to base {base_n}-gon side"

        elif base_type == "circle":
            # 对于圆，我们可以直接获取半径
            base_r = simplify(sp.sympify(base_entity["radius"]["expr"]))
            if param_type == "radius":
                param_desc = f"radius equal to base circle radius"
            else: # param_type == "side_length"
                param_desc = f"side length equal to base circle radius"
        else:
            # 如果基础实体既不是多边形也不是圆，则抛出错误
            raise NotImplementedError(f"'vertex_on_center' rule not implemented for base entity type '{base_type}'.")
        # --- 核心修改部分结束 ---

        # 3. 计算新多边形的尺寸
        if param_type == "radius" and base_r is not None:
            new_r = base_r
            new_side = simplify(2 * new_r * sp.sin(pi / new_n))
        else: # param_type == "side_length" and base_side is not None
            # 当 param_type 是 "side_length" 时，对于圆，我们用其半径作为新多边形的边长
            new_side = base_side if base_side is not None else base_r
            new_r = simplify(new_side / (2 * sp.sin(pi / new_n)))

        # 4. 计算新多边形的中心位置（确保一个顶点落在基础实体中心）
        rotation = random.choice([0, pi/new_n])
        new_cx = simplify(base_cx - new_r * sp.cos(rotation))
        new_cy = simplify(base_cy - new_r * sp.sin(rotation))
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        # 5. 生成新的正多边形
        entity_id = self._get_unique_entity_id(f"vertex_on_center_polygon_n{new_n}")
        new_id = self.generate_regular_polygon(
            n=new_n, side_length=new_side, center_id=new_center_id, rotation=rotation, entity_id=entity_id, is_base=False
        )
        
        self.description_parts.append(f"Vertex-on-center derivation: {new_n}-gon (center {new_center_id}) with {param_desc}.")

        return new_id
    
    def _rule_special_triangle_circum_inscribe(self, base_entity: Dict) -> str:
        """特殊三角形的内外接圆衍生（等腰/直角三角形专用）- 几何构造法"""
        base_subtype = base_entity["subtype"]
        is_circum = random.choice([True, False])
        rel_type = "circumscribed" if is_circum else "inscribed"

        # --- 提取三角形的三个顶点 ---
        vertices = [c for c in base_entity["vertices"] if self._is_point(c) and not c.startswith('O')]
        if len(vertices) != 3:
            raise ValueError(f"Special triangle {base_entity['id']} has an incorrect number of vertices: {len(vertices)}.")
        
        p1_id, p2_id, p3_id = vertices
        
        if(base_entity["subtype"] == "isosceles"):
            if p1_id == base_entity["top_point"]:
                p1_id, p3_id = p3_id, p1_id
            elif p2_id == base_entity["top_point"]:
                p2_id, p3_id = p3_id, p2_id
        
        # 获取顶点坐标
        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)
        p3x, p3y = self.get_point_coords(p3_id)

        center_id = None
        radius_expr = None

        if is_circum: # --- 计算外接圆 (Circumcircle) ---
            
            if base_subtype == "right":
                # 直角三角形：外心在斜边的中点。
                # 1. 确定哪条边是斜边（最长的边）
                def distance_sq(p1, p2):
                    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                
                d12_sq = distance_sq((p1x,p1y), (p2x,p2y))
                d13_sq = distance_sq((p1x,p1y), (p3x,p3y))
                d23_sq = distance_sq((p2x,p2y), (p3x,p3y))

                max_d_sq = max(d12_sq, d13_sq, d23_sq)
                
                if max_d_sq == d12_sq:
                    hypotenuse_p1, hypotenuse_p2 = (p1x,p1y,p1_id), (p2x,p2y,p2_id)
                elif max_d_sq == d13_sq:
                    hypotenuse_p1, hypotenuse_p2 = (p1x,p1y,p1_id), (p3x,p3y,p3_id)
                else: # max_d_sq == d23_sq
                    hypotenuse_p1, hypotenuse_p2 = (p2x,p2y,p2_id), (p3x,p3y,p3_id)

                # 2. 计算斜边中点（外心）
                center_x = simplify((hypotenuse_p1[0] + hypotenuse_p2[0]) / 2)
                center_y = simplify((hypotenuse_p1[1] + hypotenuse_p2[1]) / 2)
                center_id = self._add_point(center_x, center_y, is_center=True)

                # 3. 半径是斜边的一半（用表达式表示，避免开方）
                # 我们直接用斜边的一个端点到中心的距离作为半径的表达式
                radius_expr = simplify(sp.sqrt((hypotenuse_p1[0] - center_x)**2 + (hypotenuse_p1[1] - center_y)**2))

            elif base_subtype == "isosceles":
                # 等腰三角形：外心在底边的垂直平分线上。
                # 利用对称性，外心到三个顶点的距离相等 (OA = OB = OC = R)。
                
                # 1. 计算底边 p1p2 的中点 M
                mid_base_x = simplify((p1x + p2x) / 2)
                mid_base_y = simplify((p1y + p2y) / 2)
                
                # 2. 计算顶点到底边中点的距离 (h')，这是等腰三角形的高
                # 注意：这不是三角形的高 h，除非外心在三角形内部。
                # h' 是从 M 到 P3 的向量长度
                h_prime_sq = simplify((p3x - mid_base_x)**2 + (p3y - mid_base_y)**2)
                
                # 3. 计算底边一半的长度
                half_base_len_sq = simplify(((p2x - p1x)/2)**2 + ((p2y - p1y)/2)**2)
                
                # 4. 利用勾股定理计算外心 O 到底边中点 M 的距离 d
                # 在外心 O，有 OA^2 = OM^2 + AM^2
                # 同时，OA = OP3 (因为 O 在垂直平分线上，且是外心)
                # OP3^2 = (h' - d)^2 (如果 O 在 M 和 P3 之间) 或 (h' + d)^2 (如果 O 在 M 的另一侧)
                # 这里我们取绝对值，因为距离是标量
                # (h' - d)^2 = d^2 + half_base_len_sq
                # 展开并化简: h'^2 - 2 h' d + d^2 = d^2 + half_base_len_sq
                # h'^2 - 2 h' d = half_base_len_sq
                # 解得: d = (h'^2 - half_base_len_sq) / (2 * h')
                
                # 为了避免除以零，我们先检查 h_prime 是否为零（此时 P3 在底边线上，构不成三角形）
                if h_prime_sq == 0:
                    raise ValueError("The vertex P3 lies on the base line P1P2, cannot form a triangle.")
                    
                h_prime = simplify(sp.sqrt(h_prime_sq))
                numerator = simplify(h_prime_sq - half_base_len_sq)
                d = simplify(numerator / (2 * h_prime))
                
                # 5. 计算外心 O 的坐标
                # O 在从 M 指向 P3 的方向上，距离 M 点为 d
                # 单位向量 u = (P3 - M) / |P3 - M|
                # O = M + d * u
                if h_prime == 0:
                    # 避免除以零，但理论上前面已经检查过 h_prime_sq
                    center_x, center_y = mid_base_x, mid_base_y
                else:
                    ux = simplify((p3x - mid_base_x) / h_prime)
                    uy = simplify((p3y - mid_base_y) / h_prime)
                    
                    center_x = simplify(mid_base_x + d * ux)
                    center_y = simplify(mid_base_y + d * uy)
                
                center_id = self._add_point(center_x, center_y, is_center=True)

                # 6. 计算外接圆半径 R (外心到任意顶点的距离)
                radius_expr = simplify(sp.sqrt((p1x - center_x)**2 + (p1y - center_y)**2))

        else: # --- 计算内切圆 (Incircle) ---
            
            # 对于两种三角形，内心都在内部，且到三边距离相等。
            # 我们可以通过计算面积和半周长来得到半径 r = A/s
            # 然后利用几何性质确定内心坐标。

            # 1. 计算边长 (用表达式表示)
            side_a = simplify(sp.sqrt((p2x - p3x)**2 + (p2y - p3y)**2)) # a = |BC|
            side_b = simplify(sp.sqrt((p1x - p3x)**2 + (p1y - p3y)**2)) # b = |AC|
            side_c = simplify(sp.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)) # c = |AB|
            
            # 2. 计算半周长
            semi_perim = simplify((side_a + side_b + side_c) / 2)

            # 3. 计算面积 (鞋带公式，避免开方)
            area = simplify(abs((p1x*(p2y-p3y) + p2x*(p3y-p1y) + p3x*(p1y-p2y)) / 2))
            
            # 4. 计算内切圆半径
            radius_expr = simplify(area / semi_perim)

            # 5. 确定内心坐标
            if base_subtype == "right":
                # 直角三角形的内心坐标有一个简单的公式。
                # 假设 p3 是直角顶点，p3p1, p3p2 是直角边。
                # 内心到两直角边的距离都是 r。
                # 这里我们通过判断哪两条边垂直来确定直角顶点。
                # 向量点积为0则垂直。
                dot_p1p2_p1p3 = (p2x-p1x)*(p3x-p1x) + (p2y-p1y)*(p3y-p1y)
                dot_p1p2_p2p3 = (p1x-p2x)*(p3x-p2x) + (p1y-p2y)*(p3y-p2y)
                dot_p1p3_p2p3 = (p1x-p3x)*(p2x-p3x) + (p1y-p3y)*(p2y-p3y)

                if dot_p1p3_p2p3 == 0:
                    right_angle_vertex = (p3x, p3y)
                    leg1_vec = (p1x-p3x, p1y-p3y)
                    leg2_vec = (p2x-p3x, p2y-p3y)
                elif dot_p1p2_p1p3 == 0:
                    right_angle_vertex = (p1x, p1y)
                    leg1_vec = (p2x-p1x, p2y-p1y)
                    leg2_vec = (p3x-p1x, p3y-p1y)
                elif dot_p1p2_p2p3 == 0:
                    right_angle_vertex = (p2x, p2y)
                    leg1_vec = (p1x-p2x, p1y-p2y)
                    leg2_vec = (p3x-p2x, p3y-p2y)

                # 从直角顶点向两直角边方向移动 r 的距离，即为内心
                # 首先单位化直角边向量
                leg1_len = simplify(sp.sqrt(leg1_vec[0]**2 + leg1_vec[1]**2))
                leg2_len = simplify(sp.sqrt(leg2_vec[0]**2 + leg2_vec[1]**2))
                
                if leg1_len == 0 or leg2_len == 0:
                    raise ValueError("Zero length leg in right triangle.")

                unit_leg1_x, unit_leg1_y = leg1_vec[0]/leg1_len, leg1_vec[1]/leg1_len
                unit_leg2_x, unit_leg2_y = leg2_vec[0]/leg2_len, leg2_vec[1]/leg2_len

                center_x = simplify(right_angle_vertex[0] + radius_expr * unit_leg1_x + radius_expr * unit_leg2_x)
                center_y = simplify(right_angle_vertex[1] + radius_expr * unit_leg1_y + radius_expr * unit_leg2_y)

            elif base_subtype == "isosceles":
                # 等腰三角形的内心在底边的垂直平分线上。
                # 我们可以用加权平均法，这本身就是一个几何构造。
                center_x = simplify((side_a * p1x + side_b * p2x + side_c * p3x) / (side_a + side_b + side_c))
                center_y = simplify((side_a * p1y + side_b * p2y + side_c * p3y) / (side_a + side_b + side_c))

            center_id = self._add_point(center_x, center_y, is_center=True)

        # --- 生成圆 ---
        if center_id is None or radius_expr is None:
            raise RuntimeError(f"Failed to compute {rel_type} circle for triangle {base_entity['id']}.")

        entity_id = self._get_unique_entity_id(f"{rel_type}circle_triangle_{base_subtype}")
        arc_id = self.generate_circle(
            radius=radius_expr, # 传入符号表达式
            center_id=center_id,
            start_angle=0,
            end_angle=2*sp.pi,
            entity_id=entity_id,
            is_base=False
        )
        self.complete_circle_arcs.add(arc_id)
        
        self.description_parts.append(
            f"{rel_type.capitalize()} circle for {base_subtype} triangle. Center: {center_id}, Radius: {radius_expr}."
        )
        return entity_id

    def _rule_special_triangle_flip(self, base_entity: Dict) -> str:
        """特殊三角形沿三条边翻转（对称衍生）"""
        base_subtype = base_entity["subtype"]
        origin_id = base_entity["center_id"]  # 底边中点/斜边中点
        rotate_mode = base_entity["rotate_mode"]

        # --- 核心修改部分开始 ---
        # 1. 直接从实体组件中提取所有边
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 3:
            raise ValueError(f"Triangle {base_entity['id']} has invalid edges. Expected 3, got {len(edges)}.")
        
        # 2. 随机选择一条边
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        
        # 3. 获取所选边的两个顶点
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        # --- 核心修改部分结束 ---

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        # 计算原点（origin_id）关于该边的对称点（新形状的origin）
        dx = simplify(p2x - p1x)
        dy = simplify(p2y - p1y)
        ox, oy = self.get_point_coords(origin_id)
        numerator = simplify(2 * (dx*(p1y - oy) - dy*(p1x - ox)))
        denominator = simplify(dx**2 + dy**2)
        new_ox = simplify(ox - (dx * numerator) / denominator)
        new_oy = simplify(oy - (dy * numerator) / denominator)
        new_origin_id = self._add_point(new_ox, new_oy, is_center=True)

        # 生成翻转后的三角形（后续逻辑不变）
        if base_subtype == "isosceles":
            entity_id = self._get_unique_entity_id(f"flipped_isosceles_triangle_edge{edge_idx}")
            new_id = self.generate_special_triangle(
                triangle_type="isosceles",
                params={
                    "top_angle": base_entity["top_angle"],
                    "waist_length": sp.sympify(base_entity["waist_length"]["expr"]),
                    "base_length": sp.sympify(base_entity["base_length"]["expr"]),
                    "rotate_mode": rotate_mode
                },
                origin_id=new_origin_id,
                entity_id=entity_id,
                is_base=False
            )
        else:  # 直角三角形
            entity_id = self._get_unique_entity_id(f"flipped_right_triangle_edge{edge_idx}")
            new_id = self.generate_special_triangle(
                triangle_type="right",
                params={
                    "leg1": sp.sympify(base_entity["leg1"]["expr"]),
                    "leg2": sp.sympify(base_entity["leg2"]["expr"]),
                    "hypotenuse": sp.sympify(base_entity["hypotenuse"]["expr"]),
                    "rotate_mode": rotate_mode
                },
                origin_id=new_origin_id,
                entity_id=entity_id,
                is_base=False
            )

        self.description_parts.append(
            f"Flipped {base_subtype} triangle over edge {selected_edge_id} (new origin {new_origin_id})."
        )
        return new_id

    def _rule_special_triangle_translation(self, base_entity: Dict) -> str:
        """特殊三角形沿边平移（全边长/半边长）"""
        base_subtype = base_entity["subtype"]
        origin_id = base_entity["center_id"]
        ox, oy = self.get_point_coords(origin_id)

        # 1. 直接从实体组件中提取所有边
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 3:
            raise ValueError(f"Triangle {base_entity['id']} has invalid edges. Expected 3, got {len(edges)}.")
        
        # 2. 随机选择一条边
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        
        # 3. 获取所选边的两个顶点
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        # 计算边的方向向量和长度（后续逻辑不变）
        edge_dx = simplify(p2x - p1x)
        edge_dy = simplify(p2y - p1y)
        edge_len = simplify(sp.sqrt(edge_dx**2 + edge_dy**2))
        dir_dx = simplify(edge_dx / edge_len)  # 单位方向向量
        dir_dy = simplify(edge_dy / edge_len)

        # 平移距离：全边长或半边长
        translate_mode = random.choice(["full", "half"])
        dist = edge_len if translate_mode == "full" else simplify(edge_len / 2)
        tx = simplify(dir_dx * dist)
        ty = simplify(dir_dy * dist)

        # 新原点坐标
        new_ox = simplify(ox + tx)
        new_oy = simplify(oy + ty)
        new_origin_id = self._add_point(new_ox, new_oy, is_center=True)

        # 生成平移后的三角形（后续逻辑不变）
        if base_subtype == "isosceles":
            entity_id = self._get_unique_entity_id(f"translated_isosceles_triangle_edge{edge_idx}")
            new_id = self.generate_special_triangle(
                triangle_type="isosceles",
                params={
                    "top_angle": base_entity["top_angle"],
                    "waist_length": sp.sympify(base_entity["waist_length"]["expr"]),
                    "base_length": sp.sympify(base_entity["base_length"]["expr"]),
                    "rotate_mode": base_entity["rotate_mode"]
                },
                origin_id=new_origin_id,
                entity_id=entity_id,
                is_base=False
            )
        else:
            entity_id = self._get_unique_entity_id(f"translated_right_triangle_edge{edge_idx}")
            new_id = self.generate_special_triangle(
                triangle_type="right",
                params={
                    "leg1": sp.sympify(base_entity["leg1"]["expr"]),
                    "leg2": sp.sympify(base_entity["leg2"]["expr"]),
                    "hypotenuse": sp.sympify(base_entity["hypotenuse"]["expr"]),
                    "rotate_mode": base_entity["rotate_mode"]
                },
                origin_id=new_origin_id,
                entity_id=entity_id,
                is_base=False
            )

        self.description_parts.append(
            f"Translated {base_subtype} triangle along edge {selected_edge_id} by {translate_mode} length (new origin {new_origin_id})."
        )
        return new_id

    def _rule_special_rectangle_concentric(self, base_entity: Dict) -> str:
        """特殊矩形的同心衍生（同中心缩放）"""
        center_id = base_entity["center_id"]
        width = simplify(sp.sympify(base_entity["width"]["expr"]))
        length = simplify(sp.sympify(base_entity["length"]["expr"]))
        scale = Rational(random.choice([0.5, 2]))  # 缩放比例（同圆的配置）

        # 缩放后长宽保持比例
        new_width = simplify(width * scale)
        new_length = simplify(length * scale)
        entity_id = self._get_unique_entity_id(f"concentric_rectangle")
        new_id = self.generate_special_rectangle(
            width=new_width,
            length=new_length,
            center_id=center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Concentric rectangle (center {center_id}) scaled by {scale} (new size {new_width}×{new_length})."
        )
        return new_id

    def _rule_special_rectangle_circum_inscribe(self, base_entity: Dict) -> str:
        """特殊矩形的内外接圆（内接圆半径=短边/2，外接圆半径=对角线/2）"""
        center_id = base_entity["center_id"]
        width = simplify(sp.sympify(base_entity["width"]["expr"]))
        length = simplify(sp.sympify(base_entity["length"]["expr"]))
        is_circum = random.choice([True, False])
        rel_type = "circumscribed" if is_circum else "inscribed"

        # 计算半径：内接圆（切于四边）取短边/2；外接圆（过四顶点）取对角线/2
        if is_circum:
            diagonal = simplify(sp.sqrt(width**2 + length**2))
            radius = simplify(diagonal / 2)
        else:
            short_side = min(width, length)
            radius = simplify(short_side / 2)

        # 生成圆
        entity_id = self._get_unique_entity_id(f"{rel_type}circle_rectangle")
        arc_id = self.generate_circle(
            radius=radius,
            center_id=center_id,
            start_angle=0,
            end_angle=2*pi,
            entity_id=entity_id,
            is_base=False
        )
        self.complete_circle_arcs.add(arc_id)
        self.description_parts.append(
            f"{rel_type.capitalize()} circle (center {center_id}) for rectangle, radius {radius}."
        )
        return entity_id

    def _rule_special_rectangle_flip(self, base_entity: Dict) -> str:
        """特殊矩形沿四条边翻转（对称衍生）"""
        center_id = base_entity["center_id"]
        cx, cy = self.get_point_coords(center_id)
        width = simplify(sp.sympify(base_entity["width"]["expr"]))
        length = simplify(sp.sympify(base_entity["length"]["expr"]))

        # --- 核心修改部分开始 ---
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 4:
            raise ValueError(f"Rectangle {base_entity['id']} has invalid edges. Expected 4, got {len(edges)}.")
        
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        # --- 核心修改部分结束 ---

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        dx = simplify(p2x - p1x)
        dy = simplify(p2y - p1y)
        numerator = simplify(2 * (dx*(p1y - cy) - dy*(p1x - cx)))
        denominator = simplify(dx**2 + dy**2)
        new_cx = simplify(cx - (dx * numerator) / denominator)
        new_cy = simplify(cy - (dy * numerator) / denominator)
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        entity_id = self._get_unique_entity_id(f"flipped_rectangle_edge{edge_idx}")
        new_id = self.generate_special_rectangle(
            width=width,
            length=length,
            center_id=new_center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Flipped rectangle over edge {selected_edge_id} (new center {new_center_id})."
        )
        return new_id
    
    def _rule_special_rectangle_translation(self, base_entity: Dict) -> str:
        """特殊矩形沿边平移（全边长/半边长）"""
        center_id = base_entity["center_id"]
        cx, cy = self.get_point_coords(center_id)
        width = simplify(sp.sympify(base_entity["width"]["expr"]))
        length = simplify(sp.sympify(base_entity["length"]["expr"]))
        
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 4:
            raise ValueError(f"Rectangle {base_entity['id']} has invalid edges. Expected 4, got {len(edges)}.")
        
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        edge_dx = simplify(p2x - p1x)
        edge_dy = simplify(p2y - p1y)
        edge_len = simplify(sp.sqrt(edge_dx**2 + edge_dy**2))
        dir_dx = simplify(edge_dx / edge_len)
        dir_dy = simplify(edge_dy / edge_len)

        translate_mode = random.choice(["full", "half"])
        dist = edge_len if translate_mode == "full" else simplify(edge_len / 2)
        tx = simplify(dir_dx * dist)
        ty = simplify(dir_dy * dist)

        new_cx = simplify(cx + tx)
        new_cy = simplify(cy + ty)
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        entity_id = self._get_unique_entity_id(f"translated_rectangle_edge{edge_idx}")
        new_id = self.generate_special_rectangle(
            width=width,
            length=length,
            center_id=new_center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Translated rectangle along edge {selected_edge_id} by {translate_mode} length (new center {new_center_id})."
        )
        return new_id

    def _rule_parallelogram_concentric(self, base_entity: Dict) -> str:
        """平行四边形的同心衍生（同中心缩放）"""
        center_id = base_entity["center_id"]
        base = simplify(sp.sympify(base_entity["base"]["expr"]))
        height = simplify(sp.sympify(base_entity["height"]["expr"]))
        angle = simplify(sp.sympify(base_entity["angle_rad"]["expr"]))
        scale = Rational(random.choice([0.5, 2]))

        # 缩放后参数
        new_base = simplify(base * scale)
        new_height = simplify(height * scale)
        entity_id = self._get_unique_entity_id(f"concentric_parallelogram")
        new_id = self.generate_parallelogram(
            base=new_base,
            height=new_height,
            angle=angle,
            center_id=center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Concentric parallelogram (center {center_id}) scaled by {scale} (new base {new_base}, height {new_height})."
        )
        return new_id

    def _rule_parallelogram_flip(self, base_entity: Dict) -> str:
        """平行四边形沿四条边翻转（对称衍生）"""
        center_id = base_entity["center_id"]
        cx, cy = self.get_point_coords(center_id)
        base = simplify(sp.sympify(base_entity["base"]["expr"]))
        height = simplify(sp.sympify(base_entity["height"]["expr"]))
        angle = simplify(sp.sympify(base_entity["angle_rad"]["expr"]))

        # --- 核心修改部分开始 ---
        # 1. 直接从实体组件中提取所有边
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 4:
            raise ValueError(f"Parallelogram {base_entity['id']} has invalid edges. Expected 4, got {len(edges)}.")
        
        # 2. 随机选择一条边
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        
        # 3. 获取所选边的两个顶点
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        # --- 核心修改部分结束 ---

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        # 计算中心对称点（后续逻辑不变）
        dx = simplify(p2x - p1x)
        dy = simplify(p2y - p1y)
        numerator = simplify(2 * (dx*(p1y - cy) - dy*(p1x - cx)))
        denominator = simplify(dx**2 + dy**2)
        new_cx = simplify(cx - (dx * numerator) / denominator)
        new_cy = simplify(cy - (dy * numerator) / denominator)
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        # 生成翻转后的平行四边形（后续逻辑不变）
        entity_id = self._get_unique_entity_id(f"flipped_parallelogram_edge{edge_idx}")
        new_id = self.generate_parallelogram(
            base=base,
            height=height,
            angle=angle,
            center_id=new_center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Flipped parallelogram over edge {selected_edge_id} (new center {new_center_id})."
        )
        return new_id

    def _rule_parallelogram_translation(self, base_entity: Dict) -> str:
        """平行四边形沿边平移（全边长/半边长）"""
        center_id = base_entity["center_id"]
        cx, cy = self.get_point_coords(center_id)
        base = simplify(sp.sympify(base_entity["base"]["expr"]))
        height = simplify(sp.sympify(base_entity["height"]["expr"]))
        angle = simplify(sp.sympify(base_entity["angle_rad"]["expr"]))

        # --- 核心修改部分开始 ---
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 4:
            raise ValueError(f"Parallelogram {base_entity['id']} has invalid edges. Expected 4, got {len(edges)}.")
        
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        # --- 核心修改部分结束 ---

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        edge_dx = simplify(p2x - p1x)
        edge_dy = simplify(p2y - p1y)
        edge_len = simplify(sp.sqrt(edge_dx**2 + edge_dy**2))
        dir_dx = simplify(edge_dx / edge_len)
        dir_dy = simplify(edge_dy / edge_len)
        
        translate_mode = random.choice(["full", "half"])
        dist = edge_len if translate_mode == "full" else simplify(edge_len / 2)
        tx = simplify(dir_dx * dist)
        ty = simplify(dir_dy * dist)

        new_cx = simplify(cx + tx)
        new_cy = simplify(cy + ty)
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        entity_id = self._get_unique_entity_id(f"translated_parallelogram_edge{edge_idx}")
        new_id = self.generate_parallelogram(
            base=base,
            height=height,
            angle=angle,
            center_id=new_center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Translated parallelogram along edge {selected_edge_id} by {translate_mode} length (new center {new_center_id})."
        )
        return new_id

    def _rule_trapezoid_concentric(self, base_entity: Dict) -> str:
        """等腰梯形的同心衍生（同中心缩放）"""
        center_id = base_entity["center_id"]
        base1 = simplify(sp.sympify(base_entity["base1"]["expr"]))  # 下底
        base2 = simplify(sp.sympify(base_entity["base2"]["expr"]))  # 上底
        height = simplify(sp.sympify(base_entity["height"]["expr"]))
        angle = simplify(sp.sympify(base_entity["angle_rad"]["expr"]))
        scale = Rational(random.choice([0.5, 2]))

        # 缩放后保持比例（上底<下底）
        new_base1 = simplify(base1 * scale)
        new_base2 = simplify(base2 * scale)
        new_height = simplify(height * scale)
        if new_base2 >= new_base1:
            new_base2 = simplify(new_base1 * 0.8)  # 确保上底仍较短

        entity_id = self._get_unique_entity_id(f"concentric_trapezoid")
        new_id = self.generate_trapezoid(
            base1=new_base1,
            base2=new_base2,
            height=new_height,
            angle=angle,
            center_id=center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Concentric trapezoid (center {center_id}) scaled by {scale} (new bases {new_base1}, {new_base2}; height {new_height})."
        )
        return new_id

    def _rule_trapezoid_flip(self, base_entity: Dict) -> str:
        """等腰梯形沿四条边翻转（对称衍生）"""
        center_id = base_entity["center_id"]
        cx, cy = self.get_point_coords(center_id)
        base1 = simplify(sp.sympify(base_entity["base1"]["expr"]))
        base2 = simplify(sp.sympify(base_entity["base2"]["expr"]))
        height = simplify(sp.sympify(base_entity["height"]["expr"]))
        angle = simplify(sp.sympify(base_entity["angle_rad"]["expr"]))

        # --- 核心修改部分开始 ---
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 4:
            raise ValueError(f"Trapezoid {base_entity['id']} has invalid edges. Expected 4, got {len(edges)}.")
        
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        # --- 核心修改部分结束 ---

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        dx = simplify(p2x - p1x)
        dy = simplify(p2y - p1y)
        numerator = simplify(2 * (dx*(p1y - cy) - dy*(p1x - cx)))
        denominator = simplify(dx**2 + dy**2)
        new_cx = simplify(cx - (dx * numerator) / denominator)
        new_cy = simplify(cy - (dy * numerator) / denominator)
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        entity_id = self._get_unique_entity_id(f"flipped_trapezoid_edge{edge_idx}")
        new_id = self.generate_trapezoid(
            base1=base1,
            base2=base2,
            height=height,
            angle=angle,
            center_id=new_center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Flipped trapezoid over edge {selected_edge_id} (new center {new_center_id})."
        )
        return new_id

    def _rule_trapezoid_translation(self, base_entity: Dict) -> str:
        """等腰梯形沿边平移（全边长/半边长）"""
        center_id = base_entity["center_id"]
        cx, cy = self.get_point_coords(center_id)
        base1 = simplify(sp.sympify(base_entity["base1"]["expr"]))
        base2 = simplify(sp.sympify(base_entity["base2"]["expr"]))
        height = simplify(sp.sympify(base_entity["height"]["expr"]))
        angle = simplify(sp.sympify(base_entity["angle_rad"]["expr"]))

        # --- 核心修改部分开始 ---
        edges = [c for c in base_entity["lines"] if self._is_line(c)]
        if len(edges) != 4:
            raise ValueError(f"Trapezoid {base_entity['id']} has invalid edges. Expected 4, got {len(edges)}.")
        
        edge_idx = random.randint(0, len(edges)-1)
        selected_edge_id = edges[edge_idx]
        p1_id, p2_id = self._get_line_vertices(selected_edge_id)
        # --- 核心修改部分结束 ---

        p1x, p1y = self.get_point_coords(p1_id)
        p2x, p2y = self.get_point_coords(p2_id)

        edge_dx = simplify(p2x - p1x)
        edge_dy = simplify(p2y - p1y)
        edge_len = simplify(sp.sqrt(edge_dx**2 + edge_dy**2))
        dir_dx = simplify(edge_dx / edge_len)
        dir_dy = simplify(edge_dy / edge_len)
        
        translate_mode = random.choice(["full", "half"])
        dist = edge_len if translate_mode == "full" else simplify(edge_len / 2)
        tx = simplify(dir_dx * dist)
        ty = simplify(dir_dy * dist)

        new_cx = simplify(cx + tx)
        new_cy = simplify(cy + ty)
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)

        entity_id = self._get_unique_entity_id(f"translated_trapezoid_edge{edge_idx}")
        new_id = self.generate_trapezoid(
            base1=base1,
            base2=base2,
            height=height,
            angle=angle,
            center_id=new_center_id,
            rotate_mode=base_entity["rotate_mode"],
            entity_id=entity_id,
            is_base=False
        )

        self.description_parts.append(
            f"Translated trapezoid along edge {selected_edge_id} by {translate_mode} length (new center {new_center_id})."
        )
        return new_id


    # ------------------------------ 生成后整理步骤 ------------------------------
    def _finalize_geometry(self):
        """最终整理：分割生成的边/弧先查重，避免重复"""
        # 1. 查找所有交点并设置正确level
        self._find_all_intersections()

        # 2. 处理点与边的关联（线段+弧）
        for point in self.data["points"]:
            pid = point["id"]
            px, py = self.get_point_coords(pid)
            point["related_edges"] = []
            
            for line in self.data["lines"]:
                lid = line["id"]
                s_id = line["start_point_id"]
                e_id = line["end_point_id"]
                sx, sy = self.get_point_coords(s_id)
                ex, ey = self.get_point_coords(e_id)
                if self._is_point_on_segment(px, py, sx, sy, ex, ey):
                    point["related_edges"].append(lid)
            
            for arc in self.data["arcs"]:
                aid = arc["id"]
                c_id = arc["center_point_id"]
                r_expr = sp.sympify(arc["radius"]["expr"])
                cx, cy = self.get_point_coords(c_id)
                dist_sq = (px - cx)**2 + (py - cy)** 2
                if sp.simplify(dist_sq) == sp.simplify(r_expr**2):
                    point["related_edges"].append(aid)

        # 3. 处理线段分割：提取最小单元并标注
        # 为了高效查找，建立一个从线段ID到线段对象的映射
        line_id_map = {line["id"]: line for line in self.data["lines"]}
        
        # 用于存储所有已生成的最小单元线段的ID，防止重复添加
        minimal_segment_ids = set()

        # 遍历所有原始线段
        for original_line_id in line_id_map.keys():
            original_line = line_id_map[original_line_id]
            s_id = original_line["start_point_id"]
            e_id = original_line["end_point_id"]
            sx, sy = self.get_point_coords(s_id)
            ex, ey = self.get_point_coords(e_id)

            # 收集这条原始线段上的所有点
            points_on_line = []
            for point in self.data["points"]:
                pid = point["id"]
                # 检查点是否在当前原始线段上
                if original_line_id in point["related_edges"]:
                    px, py = self.get_point_coords(pid)
                    # 计算点在线段参数方程上的t值，用于排序
                    if not sp.Eq(sx, ex):
                        t = sp.simplify((px - sx) / (ex - sx))
                    elif not sp.Eq(sy, ey):
                        t = sp.simplify((py - sy) / (ey - sy))
                    else:
                        t = 0  # 处理单点线段的特殊情况
                    
                    # 过滤掉t值明显在[0, 1]范围外的点（可能是计算误差）
                    if sp.simplify(t >= 0) and sp.simplify(t <= 1):
                        points_on_line.append((t, pid))

            # 按t值排序，确保点在线段上的顺序是从起点到终点
            points_on_line.sort(key=lambda x: x[0])
            
            # 提取排序后的点ID列表，并去重（避免因计算精度导致的重复点）
            unique_point_ids = []
            seen_t_values = set()
            for t, pid in points_on_line:
                t_str = str(sp.simplify(t))
                if t_str not in seen_t_values:
                    seen_t_values.add(t_str)
                    unique_point_ids.append(pid)
            
            # 从排序后的点列表中生成最小单元线段 (AB, BC, CD...)
            if len(unique_point_ids) >= 2:
                for i in range(len(unique_point_ids) - 1):
                    start_pid = unique_point_ids[i]
                    end_pid = unique_point_ids[i + 1]
                    
                    # 关键：检查由 (start_pid, end_pid) 组成的线段是否已存在
                    # 我们利用 _check_line_duplicate 来判断
                    existing_line_id = self._check_line_duplicate(start_pid, end_pid)

                    if existing_line_id:
                        # 如果已存在，直接给它打上最小单元的标记
                        existing_line = line_id_map[existing_line_id]
                        existing_line["is_minimal"] = True
                        minimal_segment_ids.add(existing_line_id)
                    else:
                        # 如果不存在，创建它并打上标记
                        new_lid = self._add_line(start_pid, end_pid)
                        # _add_line 会将新线添加到 self.data["lines"]，我们需要找到它并更新
                        new_line = next(l for l in self.data["lines"] if l["id"] == new_lid)
                        new_line["is_minimal"] = True
                        # 同时更新我们的本地映射
                        line_id_map[new_lid] = new_line
                        minimal_segment_ids.add(new_lid)

        # =====================================================================

        # 4. 处理弧分割（这部分你的逻辑可以保持不变）
        # ... (此处省略你原有的弧分割代码) ...
        arcs_to_process = self.data["arcs"].copy()
        new_arcs = []
        for arc in arcs_to_process:
            aid = arc["id"]
            c_id = arc["center_point_id"]
            start_id = arc["start_point_id"]
            end_id = arc["end_point_id"]
            r_expr = sp.sympify(arc["radius"]["expr"])
            angle_expr = sp.sympify(arc["angle"]["expr"])
            is_complete = arc["is_complete"]
            cx, cy = self.get_point_coords(c_id)

            points_on_arc = []
            for point in self.data["points"]:
                pid = point["id"]
                if aid in point["related_edges"]:
                    px, py = self.get_point_coords(pid)
                    dx = px - cx
                    dy = py - cy
                    angle = simplify(atan2(dy, dx))
                    points_on_arc.append((angle, pid))

            points_on_arc.sort(key=lambda x: x[0])
            unique_points = []
            seen_angle = set()
            for angle, pid in points_on_arc:
                if is_complete:
                    angle = simplify(angle % (2 * pi))
                angle_str = str(sp.simplify(angle))
                if angle_str not in seen_angle:
                    seen_angle.add(angle_str)
                    unique_points.append((angle, pid))

            if len(unique_points) > 2:
                for i in range(len(unique_points)):
                    curr_angle, curr_pid = unique_points[i]
                    next_angle, next_pid = unique_points[(i + 1) % len(unique_points)]
                    if is_complete:
                        sub_angle = simplify(next_angle - curr_angle if next_angle >= curr_angle else (next_angle + 2*pi) - curr_angle)
                    else:
                        sub_angle = simplify(next_angle - curr_angle)
                    if sp.simplify(sub_angle) <= 0:
                        continue
                    new_aid = self._add_arc(
                        start_pid=curr_pid,
                        end_pid=next_pid,
                        center_pid=c_id,
                        radius_expr=r_expr,
                        angle_expr=sub_angle,
                        is_complete=False
                    )
                    if new_aid not in [a["id"] for a in arcs_to_process + new_arcs]:
                        new_arcs.append(next(a for a in self.data["arcs"] if a["id"] == new_aid))


        self.data["description"] = " ".join(self.description_parts)


    # ------------------------------ 衍生流程 ------------------------------
    def generate_derivations(self) -> None:
        num_rounds = random.randint(*self.config["derivation"]["round_range"])
        derivation_mode = self.config["derivation"]["mode"]
        
        for round_idx in range(num_rounds):
            if not self.current_entities:
                break
            
            if derivation_mode == "sequential":
                base_entity_id = self.current_entities[-1]
            else:
                base_entity_id = random.choice(self.current_entities)
            
            base_entity = self.get_entity(base_entity_id)
            base_type = base_entity["type"]

            # 检查配置中是否存在当前实体类型的规则
            if base_type not in self.config["derivation"]["rules"]:
                continue

            supported_rules = self.config["derivation"]["rules"][base_type]["supported_rules"]
            rule_names = [rule["name"] for rule in supported_rules]
            rule_probs = [rule["prob"] for rule in supported_rules]

            try:
                selected_rule_name = random.choices(rule_names, weights=rule_probs, k=1)[0]
            except ValueError:
                continue

            new_id = None
            
            self.description_parts.append(f"Round {round_idx+1}: applying '{selected_rule_name}' rule.")
            try:
            # if True:
                if base_type == "circle" or base_type == "polygon":
                    if selected_rule_name == "concentric":
                        new_id = self._rule_concentric(base_entity)
                    elif selected_rule_name == "translation":
                        new_id = self._rule_translation(base_entity)
                    elif selected_rule_name == "circum_inscribe":
                        new_id = self._rule_circum_inscribe(base_entity)
                    elif selected_rule_name == "vertex_on_center":
                        new_id = self._rule_vertex_on_center(base_entity)
                
                elif base_type == "special_triangle":
                    if selected_rule_name == "circum_inscribe":
                        new_id = self._rule_special_triangle_circum_inscribe(base_entity)
                    elif selected_rule_name == "flip":
                        new_id = self._rule_special_triangle_flip(base_entity)
                    elif selected_rule_name == "translation":
                        new_id = self._rule_special_triangle_translation(base_entity)
                    
                elif base_type == "special_rectangle":
                    if selected_rule_name == "concentric":
                        new_id = self._rule_special_rectangle_concentric(base_entity)
                    elif selected_rule_name == "circum_inscribe":
                        new_id = self._rule_special_rectangle_circum_inscribe(base_entity)
                    elif selected_rule_name == "flip":
                        new_id = self._rule_special_rectangle_flip(base_entity)
                    elif selected_rule_name == "translation":
                        new_id = self._rule_special_rectangle_translation(base_entity)
                    
                elif base_type == "parallelogram":
                    if selected_rule_name == "concentric":
                        new_id = self._rule_parallelogram_concentric(base_entity)
                    elif selected_rule_name == "flip":
                        new_id = self._rule_parallelogram_flip(base_entity)
                    elif selected_rule_name == "translation":
                        new_id = self._rule_parallelogram_translation(base_entity)
                    
                elif base_type == "trapezoid":
                    if selected_rule_name == "concentric":
                        new_id = self._rule_trapezoid_concentric(base_entity)
                    elif selected_rule_name == "flip":
                        new_id = self._rule_trapezoid_flip(base_entity)
                    elif selected_rule_name == "translation":
                        new_id = self._rule_trapezoid_translation(base_entity)
                        
            except Exception as e:
                print(f"Error applying rule '{selected_rule_name}': {e}")

            if new_id and new_id not in self.current_entities:
                self.current_entities.append(new_id)
                self._calculate_new_entity_intersections(new_id)

        # self._finalize_geometry()
    
    def export_json(self) -> Dict:
        return self.data.copy()