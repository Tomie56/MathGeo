import json
import random
import os
import sympy as sp
from sympy import symbols, cos, sin, pi, simplify, sqrt, Eq, atan2, tan, nsimplify, Rational, Min, Max, solve, Ge, Le
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


    # ------------------------------ ID生成工具（增强：新增中心点O系列） ------------------------------
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

        # 3. 更新实体组件
        for entity in self.data["entities"]:
            for component in entity["components"]:
                if component == old_pid:
                    component = new_pid

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
            x_equal = x_diff.is_zero  # 返回True/False，避免Relational对象
            y_equal = y_diff.is_zero
            
            if x_equal and y_equal:
                old_pid = old_point["id"]
                new_level = min(level, old_point["level"])
                if old_point["level"] != new_level:
                    self.data["points"][idx]["level"] = new_level
                    
                if old_point.get("is_circle_init") and not is_circle_init:
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
        if entity["type"] == "polygon":
            return next(c for c in entity["components"] if any(p["id"] == c and p["is_center"] for p in self.data["points"]))
        else:
            return next(c for c in entity["components"] if any(p["id"] == c and p["is_center"] for p in self.data["points"]))


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
        """添加线段（先查重，重复则返回已有ID）"""
        
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
        """添加弧（先查重，重复则返回已有ID）"""
        
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
        """查找所有线段与圆、圆与圆的交点"""
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
        # 代表数学坐标的原点
        origin_id = self._add_point(0, 0, is_center=True)

        if base_type == "polygon":
            n = random.choice(self.config["base"]["polygon"]["n_choices"])
            side_min, side_max = self.config["base"]["polygon"]["side_range"]
            side_length = sp.Integer(random.randint(side_min, side_max))
            rot_type = self.config["base"]["polygon"]["rotation_choices_type"]
            rotations = [0, sp.pi/(n)] if rot_type == "pi_over_2n" else [0]
            # rotation = random.choice(rotations)
            # sp.pi/(n) 表示底边平行于x轴
            rotation = random.choices(rotations, weights=[0.7, 0.3] if len(rotations) == 2 else [1.0], k=1)[0]
            entity_id = self._get_unique_entity_id(f"base_polygon_n{n}")
            self.generate_regular_polygon(
                n=n,
                side_length=side_length,
                center_id=origin_id,
                rotation=rotation,
                entity_id=entity_id,
                is_base=True
            )
            self.description_parts.append(
                f"Base shape is a regular {n}-sided polygon (center {origin_id}) with side length {side_length}"
            )

        else:  
            r_min, r_max = self.config["base"]["circle"]["radius_range"]
            radius = sp.Integer(random.randint(r_min, r_max))
            start_angle = self.config["base"]["circle"]["start_angle"]
            end_angle = 2 * pi
            entity_id = self._get_unique_entity_id("base_circle")
            arc_id = self.generate_circle(
                radius=radius,
                center_id=origin_id,
                start_angle=start_angle,
                end_angle=end_angle,
                entity_id=entity_id,
                is_base=True
            )
            self.complete_circle_arcs.add(arc_id)
            self.description_parts.append(
                f"Base shape is a complete circle (center {origin_id}, init point circle_1) with radius {radius}."
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
            "components": [center_id] + vertices + lines,
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
            "components": [center_id, start_id, end_id, arc_id],
            "radius": {"expr": self._format_expr(r_expr), "latex": sp.latex(r_expr)},
            "start_angle": {"expr": self._format_expr(start_angle), "latex": sp.latex(start_angle)},
            "end_angle": {"expr": self._format_expr(end_angle), "latex": sp.latex(end_angle)},
            "is_complete": is_complete,
            "is_base": is_base
        })
        return arc_id

    def _calculate_new_entity_intersections(self, new_id: str):
        """计算新生成entity的边/弧与所有已有元素的交点"""
        # 提取新entity的所有边和弧ID
        line_id_pattern = re.compile(r'^L\d+$')
        new_entity = self.get_entity(new_id)
        new_lines = [cid for cid in new_entity["components"] if line_id_pattern.match(cid)]
        new_arcs = [cid for cid in new_entity["components"] if cid.startswith("Arc")]
        
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
    def _rule_concentric(self, base_entity: Dict) -> str:
        base_type = base_entity["type"]
        center_id = self._get_center_id(base_entity)
        scale = Rational(random.choice(self.config["derivation"]["rules"]["concentric"]["scale_choices"]))

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
                vertices = [c for c in base_entity["components"] if any(p["id"] == c and not p["is_center"] for p in self.data["points"])]
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
                vertices = [c for c in base_entity["components"] if any(p["id"] == c and not p["is_center"] for p in self.data["points"])]
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
            new_n = random.choice(self.config["derivation"]["rules"]["circum_inscribe"]["n_choices"])
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
        base_center_id = self._get_center_id(base_entity)
        base_cx, base_cy = self.get_point_coords(base_center_id)
        new_n = random.choice(self.config["derivation"]["rules"]["vertex_on_center"]["n_choices"])
        param_type = random.choice(self.config["derivation"]["rules"]["vertex_on_center"]["param_choices"])
        base_type = base_entity["type"]

        if base_type == "polygon":
            base_side = simplify(sp.sympify(base_entity["side_length"]["expr"]))
            base_r = simplify(sp.sympify(base_entity["radius"]["expr"]))
            base_n = base_entity["n"]
            if param_type == "radius":
                new_r = base_r
                new_side = simplify(2 * new_r * sin(pi / new_n))
                param_desc = f"radius equal to base {base_n}-gon radius"
            else:
                new_side = base_side
                new_r = simplify(new_side / (2 * sin(pi / new_n)))
                param_desc = f"side length equal to base {base_n}-gon side"
        else:
            base_r = simplify(sp.sympify(base_entity["radius"]["expr"]))
            if param_type == "radius":
                new_r = base_r
                new_side = simplify(2 * new_r * sin(pi / new_n))
                param_desc = f"radius equal to base circle radius"
            else:
                new_side = base_r
                new_r = simplify(new_side / (2 * sin(pi / new_n)))
                param_desc = f"side length equal to base circle radius"

        rotation = random.choice([0, pi/new_n])
        new_cx = simplify(base_cx - new_r * cos(rotation))
        new_cy = simplify(base_cy - new_r * sin(rotation))
        new_center_id = self._add_point(new_cx, new_cy, is_center=True)
        entity_id = self._get_unique_entity_id(f"vertex_on_center_polygon_n{new_n}")
        new_id = self.generate_regular_polygon(
            n=new_n, side_length=new_side, center_id=new_center_id, rotation=rotation, entity_id=entity_id, is_base=False
        )
        self.description_parts.append(f"Vertex-on-center derivation: {new_n}-gon (center {new_center_id}) with {param_desc}.")

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

        # 3. 处理线段分割（依赖_add_line确保新线level=端点max）
        lines_to_process = self.data["lines"].copy()
        new_lines = []
        for line in lines_to_process:
            lid = line["id"]
            s_id = line["start_point_id"]
            e_id = line["end_point_id"]
            sx, sy = self.get_point_coords(s_id)
            ex, ey = self.get_point_coords(e_id)

            points_on_line = []
            for point in self.data["points"]:
                pid = point["id"]
                if lid in point["related_edges"]:
                    px, py = self.get_point_coords(pid)
                    if not sp.Eq(sx, ex):
                        t = sp.simplify((px - sx) / (ex - sx))
                    elif not sp.Eq(sy, ey):
                        t = sp.simplify((py - sy) / (ey - sy))
                    else:
                        t = 0
                    points_on_line.append((t, pid))

            points_on_line.sort(key=lambda x: x[0])
            unique_points = []
            seen_t = set()
            for t, pid in points_on_line:
                t_str = str(sp.simplify(t))
                if t_str not in seen_t:
                    seen_t.add(t_str)
                    unique_points.append(pid)

            if len(unique_points) > 2:
                for i in range(len(unique_points) - 1):
                    start_pid = unique_points[i]
                    end_pid = unique_points[i + 1]
                    new_lid = self._add_line(start_pid, end_pid)
                    if new_lid not in [l["id"] for l in lines_to_process + new_lines]:
                        new_lines.append(next(l for l in self.data["lines"] if l["id"] == new_lid))

        # 4. 处理弧分割（依赖_add_arc确保新弧level=端点max）
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

        # 5. 更新实体组件
        # for entity in self.data["entities"]:
            # all_lines = [l["id"] for l in self.data["lines"]]
            # all_arcs = [a["id"] for a in self.data["arcs"]]
            # entity["components"] = list(set(entity["components"] + all_lines + all_arcs))

        # 6. 补充描述
        self.description_parts.append(
            f"Geometry finalized: {len(new_lines)} new lines and {len(new_arcs)} new arcs generated by splitting. "
            f"Total lines: {len(self.data['lines'])}, total arcs: {len(self.data['arcs'])}, total points: {len(self.data['points'])}."
        )
        self.data["description"] = " ".join(self.description_parts)


    # ------------------------------ 衍生流程 ------------------------------
    def generate_derivations(self) -> None:
        num_rounds = random.randint(*self.config["derivation"]["round_range"])
        self.description_parts.append(f"Total {num_rounds} derivation rounds.")
        
        rules = list(self.config["derivation"]["rules"].items())
        rule_probs = [cfg["prob"] for _, cfg in rules]
        rule_list = [rule for rule, _ in rules]

        for round_idx in range(num_rounds):
            base_entity_id = random.choice(self.current_entities)
            base_entity = self.get_entity(base_entity_id)
            rule = random.choices(rule_list, weights=rule_probs)[0]
            self.description_parts.append(f"Round {round_idx+1}: applying '{rule}' rule.")

            if rule == "concentric":
                new_id = self._rule_concentric(base_entity)
            elif rule == "translation":
                new_id = self._rule_translation(base_entity)
            elif rule == "circum_inscribe":
                new_id = self._rule_circum_inscribe(base_entity)
            elif rule == "vertex_on_center":
                new_id = self._rule_vertex_on_center(base_entity)
            else:
                raise ValueError(f"未知规则：{rule}")

            if new_id not in self.current_entities:
                self.current_entities.append(new_id)
                
            self._calculate_new_entity_intersections(new_id)
            
        self._finalize_geometry()

    def export_json(self) -> Dict:
        return self.data.copy()