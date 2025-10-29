import os
import json
import logging
import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict 
from datetime import datetime
import sympy as sp
import re
from .drawer import GeometryDrawer
from PIL import Image, ImageDraw, ImageFont
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)

from .region import RegionExtractor, RegionExtractConfig
from .shaders import SHADERS

# 共享类型与坐标转换
PointMath = Tuple[float, float]
PointPixel = Tuple[int, int]


COLOR_NAME_MAP = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "white": (255, 255, 255), "black": (0, 0, 0), "yellow": (255, 255, 0)
}


class GeometryAnnotator:
    """几何标注工具：修正min_dist计算范围，严格确保无冲突优先"""
    def __init__(self, annotator_config: Dict):
        self.config = self._validate_annotator_config(annotator_config)
        try:
            self.font = ImageFont.truetype("arial.ttf", 12)
        except:
            self.font = ImageFont.load_default()
        self.point_id_pattern = re.compile(r'^(?!circle_\d+$).+$')

    def _validate_annotator_config(self, cfg: Dict) -> Dict:
        default = {
            "point": {
                "enabled": True,
                "radius": 4,
                "color": (255, 0, 0),
                "text": {
                    "enabled": True,
                    "size": 12,
                    "color": (0, 0, 0),
                    "offset": 8,
                    "edge_margin": 5,  # 文本与图像边缘的最小像素距离
                    "min_pixel_dist": 3  # 文本与所有元素的最小安全距离
                }
            },
            "line": {
                "enabled": False,
                "color": (0, 255, 0),
                "width": 2,
                "text": {
                    "enabled": True,
                    "size": 12,
                    "color": (0, 0, 0),
                    "offset": 12,
                    "edge_margin": 5,
                    "min_pixel_dist": 3
                }
            },
            "perpendicular": {
                "enabled": False,
                "color": (0, 0, 255),
                "width": 2,
                "square_side": 10  # 垂直符号L形边长
            }
        }

        def fix_color(color) -> Tuple[int, int, int]:
            if isinstance(color, str) and color.lower() in COLOR_NAME_MAP:
                return COLOR_NAME_MAP[color.lower()]
            if isinstance(color, str):
                try:
                    color_str = color.replace(" ", "").strip('()')
                    channels = list(map(int, color_str.split(',')))[:3]
                except:
                    return (255, 0, 0)
            elif isinstance(color, (list, tuple)):
                channels = list(color)[:3]
            else:
                return (255, 0, 0)
            return tuple(max(0, min(255, int(c))) for c in channels) if len(channels) == 3 else (255, 0, 0)

        config = {
            "enabled": cfg.get("enabled", True),
            "point": {**default["point"],** cfg.get("point", {})},
            "line": {**default["line"],** cfg.get("line", {})},
            "perpendicular": {**default["perpendicular"],** cfg.get("perpendicular", {})}
        }

        if "bg_color" in config["point"]["text"]:
            del config["point"]["text"]["bg_color"]
        if "bg_color" in config["line"]["text"]:
            del config["line"]["text"]["bg_color"]

        config["point"]["color"] = fix_color(config["point"]["color"])
        config["point"]["text"]["color"] = fix_color(config["point"]["text"]["color"])
        config["line"]["color"] = fix_color(config["line"]["color"])
        config["line"]["text"]["color"] = fix_color(config["line"]["text"]["color"])
        config["perpendicular"]["color"] = fix_color(config["perpendicular"]["color"])

        return config

    def _parse_math_expr(self, expr_str: str) -> float:
        try:
            return float(sp.sympify(expr_str.replace(" ", "")).evalf())
        except Exception as e:
            logger.warning(f"表达式解析失败: {expr_str}，错误: {e}")
            raise

    def _get_text_pixels(self, text: str, font: ImageFont.FreeTypeFont, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """提取文本的像素级轮廓（每个非透明像素的坐标）"""
        left, top, right, bottom = font.getbbox(text)
        w = right - left
        h = bottom - top

        temp_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        draw.text((-left, -top), text, font=font, fill=(0, 0, 0, 255))  # 临时黑色绘制以提取轮廓

        img_array = np.array(temp_img)
        alpha_mask = img_array[:, :, 3] > 0  # 透明通道>0的为文本像素
        y_coords, x_coords = np.where(alpha_mask)

        text_pixels = set()
        for x, y in zip(x_coords, y_coords):
            abs_x = pos[0] + x
            abs_y = pos[1] + y
            text_pixels.add((abs_x, abs_y))

        return text_pixels

    def _get_shape_pixels(self, shape_type: str, params: Dict) -> Set[Tuple[int, int]]:
        """提取非文本形状（点、线、垂直符号）的像素级轮廓"""
        shape_pixels = set()
        if shape_type == "point":
            # 点的参数：(中心x, 中心y, 半径)
            cx, cy, radius = params["cx"], params["cy"], params["radius"]
            # 生成圆形所有像素
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx**2 + dy**2 <= radius**2:  # 圆内像素
                        shape_pixels.add((cx + dx, cy + dy))
        elif shape_type == "line":
            # 线的参数：(起点x, 起点y, 终点x, 终点y, 线宽)
            x1, y1, x2, y2, width = params["x1"], params["y1"], params["x2"], params["y2"], params["width"]
            # 生成线段所有像素（含线宽）
            # 简化处理：用矩形包围线段，精度足够
            min_x = min(x1, x2) - width
            max_x = max(x1, x2) + width
            min_y = min(y1, y2) - width
            max_y = max(y1, y2) + width
            # 遍历矩形内像素，判断是否在线段附近（简化版）
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # 点到线段的距离公式（简化）
                    line_len_sq = (x2 - x1)**2 + (y2 - y1)** 2
                    if line_len_sq == 0:  # 点线
                        dist_sq = (x - x1)**2 + (y - y1)** 2
                    else:
                        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / line_len_sq))
                        proj_x = x1 + t * (x2 - x1)
                        proj_y = y1 + t * (y2 - y1)
                        dist_sq = (x - proj_x)**2 + (y - proj_y)** 2
                    if dist_sq <= (width / 2)**2:
                        shape_pixels.add((x, y))
        elif shape_type == "perpendicular":
            # 垂直符号参数：(垂足x, 垂足y, 底线端点x, 底线端点y, 垂线端点x, 垂线端点y, 线宽)
            fx, fy, hx, hy, vx, vy, width = params["fx"], params["fy"], params["hx"], params["hy"], params["vx"], params["vy"], params["width"]
            # 生成两条边的像素
            # 底线像素
            line1_pixels = self._get_shape_pixels("line", {
                "x1": fx, "y1": fy, "x2": hx, "y2": hy, "width": width
            })
            # 垂线像素
            line2_pixels = self._get_shape_pixels("line", {
                "x1": fx, "y1": fy, "x2": vx, "y2": vy, "width": width
            })
            shape_pixels = line1_pixels.union(line2_pixels)
        return shape_pixels

    def _check_pixel_collision(self, text_pixels: Set[Tuple[int, int]], 
                              existing_text_pixels: List[Set[Tuple[int, int]]],
                              existing_shape_pixels: List[Set[Tuple[int, int]]],  # 新增：图像已有非文本轮廓像素
                              img_shape: Tuple[int, int], edge_margin: int) -> Tuple[bool, float]:
        """
        修正像素冲突检测：
        1. has_collision：文本像素与以下任何元素冲突则为True
           - 图像边缘（距离 < edge_margin）
           - 现有文本像素（重叠）
           - 现有非文本轮廓像素（点、线、符号等，重叠）
        2. min_dist：文本像素与所有现有元素（文本+非文本）的最小像素距离
        """
        h, w = img_shape[:2]
        min_dist = float('inf')
        all_existing_pixels = []
        # 合并所有现有元素像素（文本+非文本）
        all_existing_pixels.extend(existing_text_pixels)
        all_existing_pixels.extend(existing_shape_pixels)

        # 1. 检查与图像边缘的距离（冲突判断）
        for (x, y) in text_pixels:
            if x < edge_margin or x >= (w - edge_margin) or y < edge_margin or y >= (h - edge_margin):
                return True, 0.0  # 边缘冲突

        # 2. 检查与所有现有元素的像素冲突及距离
        for exist_pixels in all_existing_pixels:
            # 像素重叠即冲突（文本与任何现有元素重叠）
            if text_pixels & exist_pixels:
                return True, 0.0  # 元素重叠冲突

            # 计算与该元素的最小像素距离
            for (x1, y1) in text_pixels:
                for (x2, y2) in exist_pixels:
                    dist = math.hypot(x1 - x2, y1 - y2)
                    if dist < min_dist:
                        min_dist = dist

        # 无任何现有元素时，距离设为极大值
        if not all_existing_pixels:
            min_dist = float('inf')

        return False, min_dist

    def _find_best_text_pos(self, base_px: Tuple[int, int], text: str, 
                           existing_text_pixels: List[Set[Tuple[int, int]]],
                           existing_shape_pixels: List[Set[Tuple[int, int]]],  # 传入非文本轮廓像素
                           base_offset: int, img_shape: Tuple[int, int], text_size: int,
                           edge_margin: int, min_pixel_dist: int) -> Tuple[Tuple[int, int], Set[Tuple[int, int]]]:
        """
        修正排序逻辑：
        1. 绝对优先：has_collision=False（无任何冲突）
        2. 次优先：offset尽量小
        3. 再次：min_dist（与所有元素的距离）尽量大
        4. 最后：方向（正四方向优先级最低）
        """
        try:
            text_font = ImageFont.truetype("arial.ttf", text_size)
        except:
            text_font = ImageFont.load_default()

        candidates = []

        # 正四方向定义（用于标记）
        priority_dirs = [
            ("正右", 1, 0), ("正左", -1, 0),
            ("正上", 0, -1), ("正下", 0, 1)
        ]

        # 生成多offset候选（从小到大为了优先小offset）
        offset_levels = [
            base_offset - 2, base_offset - 1,
            base_offset,
            base_offset + 1, base_offset + 2
        ]
        offset_levels = [o for o in offset_levels if o >= 3]  # 最小offset限制

        # 生成正四方向候选
        for dir_name, dx_coef, dy_coef in priority_dirs:
            for offset in offset_levels:
                dx = dx_coef * offset
                dy = dy_coef * offset
                tx, ty = base_px[0] + dx, base_px[1] + dy

                text_pixels = self._get_text_pixels(text, text_font, (tx, ty))
                # 传入所有现有元素像素（文本+非文本）进行碰撞检测
                has_collision, min_dist = self._check_pixel_collision(
                    text_pixels, existing_text_pixels, existing_shape_pixels,
                    img_shape, edge_margin
                )

                candidates.append({
                    "pos": (tx, ty),
                    "pixels": text_pixels,
                    "has_collision": has_collision,
                    "min_dist": min_dist,
                    "offset": offset,
                    "is_priority_dir": 1,  # 正四方向
                    "dir_name": dir_name
                })

        # 生成其他方向候选（排除正四方向）
        angle_steps = 36  # 每10度一个位置
        for offset in offset_levels:
            for step in range(angle_steps):
                angle = 2 * math.pi * step / angle_steps
                dx = int(offset * math.cos(angle))
                dy = int(offset * math.sin(angle))

                # 跳过正四方向（已单独生成）
                if (abs(dx) > 0 and dy == 0) or (dx == 0 and abs(dy) > 0):
                    continue

                tx, ty = base_px[0] + dx, base_px[1] + dy
                text_pixels = self._get_text_pixels(text, text_font, (tx, ty))
                has_collision, min_dist = self._check_pixel_collision(
                    text_pixels, existing_text_pixels, existing_shape_pixels,
                    img_shape, edge_margin
                )

                candidates.append({
                    "pos": (tx, ty),
                    "pixels": text_pixels,
                    "has_collision": has_collision,
                    "min_dist": min_dist,
                    "offset": offset,
                    "is_priority_dir": 0,  # 非正四方向
                    "dir_name": f"{angle*180/math.pi:.0f}°"
                })

        # 严格按优先级排序（核心修正）
        candidates.sort(
            key=lambda x: (
                x["has_collision"],  # 1. 无冲突（False）绝对优先
                x["offset"],         # 2. 相同冲突状态下，offset小的在前
                -x["min_dist"],      # 3. 相同offset下，与所有元素的距离大的在前
                x["is_priority_dir"] # 4. 最后：非正四方向（0）优先于正四方向（1）
            )
        )

        # 选择最优位置
        best = candidates[0]
        if not best["has_collision"]:
            logger.debug(
                f"文本'{text}'最优位置：{best['dir_name']}，offset={best['offset']:.1f}，"
                f"与所有元素最小距离={best['min_dist']:.1f}px（无冲突）"
            )
        else:
            logger.warning(
                f"文本'{text}'无完全安全位置，选冲突最小位置：{best['dir_name']}，"
                f"offset={best['offset']:.1f}，最小距离={best['min_dist']:.1f}px"
            )

        return best["pos"], best["pixels"]

    def draw_annotations(self, img: np.ndarray, data: Dict, transformer: CoordTransformer) -> np.ndarray:
        if not self.config["enabled"]:
            return img.copy()
            
        img_copy = img.copy()
        existing_text_pixels = []  # 现有文本像素
        existing_shape_pixels = []  # 现有非文本轮廓像素（点、线、符号等）
        points = data.get("points", [])
        lines = data.get("lines", [])
        
        valid_points = [p for p in points if self.point_id_pattern.match(p["id"])]
        points_dict = {p["id"]: p for p in valid_points}
        
        # 绘制点并收集非文本像素
        existing_text_pixels, existing_shape_pixels = self.draw_point_annotations(
            img_copy, valid_points, transformer, existing_text_pixels, existing_shape_pixels
        )
        
        # 绘制线并收集非文本像素
        existing_text_pixels, existing_shape_pixels = self.draw_line_annotations(
            img_copy, lines, points_dict, transformer, existing_text_pixels, existing_shape_pixels
        )
        
        # 绘制垂直符号并收集非文本像素
        existing_shape_pixels = self.draw_perpendicular_annotations(
            img_copy, lines, points_dict, transformer, existing_shape_pixels, data.get("lines", [])
        )
        
        return img_copy

    def draw_point_annotations(self, img: np.ndarray, points: List[Dict], transformer: CoordTransformer,
                              existing_text_pixels: List[Set[Tuple[int, int]]], 
                              existing_shape_pixels: List[Set[Tuple[int, int]]]) -> Tuple[List[Set[Tuple[int, int]]], List[Set[Tuple[int, int]]]]:
        cfg = self.config["point"]
        text_cfg = cfg["text"]
        if not cfg["enabled"]:
            return existing_text_pixels, existing_shape_pixels
        
        img_shape = img.shape
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        base_offset = text_cfg["offset"]

        for point in points:
            try:
                pid = point["id"]
                x = self._parse_math_expr(point["x"]["expr"])
                y = self._parse_math_expr(point["y"]["expr"])
                px, py = transformer.math_to_pixel((x, y))
                
                # 绘制点并提取其像素轮廓
                point_color = cfg["color"]
                radius = cfg["radius"]
                draw.ellipse(
                    [px - radius, py - radius, px + radius, py + radius],
                    fill=point_color, outline=None
                )
                # 收集点的像素到非文本轮廓
                point_pixels = self._get_shape_pixels("point", {
                    "cx": px, "cy": py, "radius": radius
                })
                existing_shape_pixels.append(point_pixels)
                
                # 绘制文本（如果启用）
                if text_cfg["enabled"]:
                    label = point.get("label", pid)
                    if label:
                        best_pos, text_pixels = self._find_best_text_pos(
                            base_px=(px, py),
                            text=label,
                            existing_text_pixels=existing_text_pixels,
                            existing_shape_pixels=existing_shape_pixels,  # 传入非文本像素
                            base_offset=base_offset,
                            img_shape=img_shape,
                            text_size=text_cfg["size"],
                            edge_margin=text_cfg["edge_margin"],
                            min_pixel_dist=text_cfg["min_pixel_dist"]
                        )
                        
                        try:
                            text_font = ImageFont.truetype("arial.ttf", text_cfg["size"])
                        except:
                            text_font = ImageFont.load_default()
                        draw.text(best_pos, label, font=text_font, fill=text_cfg["color"])
                        
                        existing_text_pixels.append(text_pixels)
            except Exception as e:
                logger.warning(f"点标注失败 (id: {pid}): {e}")
                continue
        
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return existing_text_pixels, existing_shape_pixels

    def draw_line_annotations(self, img: np.ndarray, lines: List[Dict], points: Dict, transformer: CoordTransformer,
                             existing_text_pixels: List[Set[Tuple[int, int]]], 
                             existing_shape_pixels: List[Set[Tuple[int, int]]]) -> Tuple[List[Set[Tuple[int, int]]], List[Set[Tuple[int, int]]]]:
        cfg = self.config["line"]
        text_cfg = cfg["text"]
        if not cfg["enabled"]:
            return existing_text_pixels, existing_shape_pixels
        
        img_shape = img.shape
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        base_offset = text_cfg["offset"]

        for line in lines:
            if line.get("type") == "perpendicular":
                continue
            
            try:
                start_id = line["start_point_id"]
                end_id = line["end_point_id"]
                if start_id not in points or end_id not in points:
                    logger.debug(f"线标注跳过（无效端点）: {line.get('id')}")
                    continue
                
                # 解析端点坐标
                start_p = points[start_id]
                end_p = points[end_id]
                s_x = self._parse_math_expr(start_p["x"]["expr"])
                s_y = self._parse_math_expr(start_p["y"]["expr"])
                e_x = self._parse_math_expr(end_p["x"]["expr"])
                e_y = self._parse_math_expr(end_p["y"]["expr"])
                
                s_px, s_py = transformer.math_to_pixel((s_x, s_y))
                e_px, e_py = transformer.math_to_pixel((e_x, e_y))
                
                # 绘制线段并提取其像素轮廓
                line_color = cfg["color"]
                line_width = cfg["width"]
                draw.line(
                    [(s_px, s_py), (e_px, e_py)],
                    fill=line_color,
                    width=line_width,
                    joint="round"
                )
                # 收集线的像素到非文本轮廓
                line_pixels = self._get_shape_pixels("line", {
                    "x1": s_px, "y1": s_py, "x2": e_px, "y2": e_py, "width": line_width
                })
                existing_shape_pixels.append(line_pixels)
                
                # 绘制线文本（如果启用）
                if text_cfg["enabled"]:
                    label = line.get("label", line.get("id"))
                    if label:
                        mid_px, mid_py = (s_px + e_px) // 2, (s_py + e_py) // 2
                        
                        best_pos, text_pixels = self._find_best_text_pos(
                            base_px=(mid_px, mid_py),
                            text=label,
                            existing_text_pixels=existing_text_pixels,
                            existing_shape_pixels=existing_shape_pixels,  # 传入非文本像素
                            base_offset=base_offset,
                            img_shape=img_shape,
                            text_size=text_cfg["size"],
                            edge_margin=text_cfg["edge_margin"],
                            min_pixel_dist=text_cfg["min_pixel_dist"]
                        )
                        
                        try:
                            text_font = ImageFont.truetype("arial.ttf", text_cfg["size"])
                        except:
                            text_font = ImageFont.load_default()
                        draw.text(best_pos, label, font=text_font, fill=text_cfg["color"])
                        
                        existing_text_pixels.append(text_pixels)
            except Exception as e:
                logger.warning(f"线标注失败 (id: {line.get('id')}): {e}")
                continue
        
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return existing_text_pixels, existing_shape_pixels

    def draw_perpendicular_annotations(self, img: np.ndarray, lines: List[Dict], points: Dict, 
                                      transformer: CoordTransformer, 
                                      existing_shape_pixels: List[Set[Tuple[int, int]]],
                                      all_lines: List[Dict]) -> List[Set[Tuple[int, int]]]:
        cfg = self.config["perpendicular"]
        if not cfg["enabled"]:
            return existing_shape_pixels
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        square_side = cfg["square_side"]
        line_id_map = {line["id"]: line for line in all_lines}

        for line in lines:
            if line.get("type") != "perpendicular":
                continue
            
            try:
                foot_id = line["end_point_id"]  # 垂足
                vert_start_id = line["start_point_id"]  # 垂线起点
                desc = line.get("description", "")
                
                if foot_id not in points or vert_start_id not in points:
                    logger.debug(f"垂足/垂线起点无效: 垂足{foot_id}，起点{vert_start_id}")
                    continue
                
                # 解析坐标
                vert_start_p = points[vert_start_id]
                vs_x = self._parse_math_expr(vert_start_p["x"]["expr"])
                vs_y = self._parse_math_expr(vert_start_p["y"]["expr"])
                vs_px, vs_py = transformer.math_to_pixel((vs_x, vs_y))
                
                foot_p = points[foot_id]
                foot_x = self._parse_math_expr(foot_p["x"]["expr"])
                foot_y = self._parse_math_expr(foot_p["y"]["expr"])
                foot_px, foot_py = transformer.math_to_pixel((foot_x, foot_y))

                # 计算方向向量
                line_match = re.search(r'to line (\w+\d*)', desc)
                if not line_match:
                    dir_host = (1, 0)
                    dir_vert = (0, 1)
                else:
                    host_line_id = line_match.group(1)
                    if host_line_id not in line_id_map:
                        dir_host = (1, 0)
                        dir_vert = (0, 1)
                    else:
                        host_line = line_id_map[host_line_id]
                        h_start_id = host_line["start_point_id"]
                        h_end_id = host_line["end_point_id"]
                        if h_start_id not in points or h_end_id not in points:
                            dir_host = (1, 0)
                            dir_vert = (0, 1)
                        else:
                            # 被垂线方向向量
                            h_start_p = points[h_start_id]
                            h_end_p = points[h_end_id]
                            hs_x = self._parse_math_expr(h_start_p["x"]["expr"])
                            hs_y = self._parse_math_expr(h_start_p["y"]["expr"])
                            he_x = self._parse_math_expr(h_end_p["x"]["expr"])
                            he_y = self._parse_math_expr(h_end_p["y"]["expr"])
                            
                            host_vec = (he_x - hs_x, he_y - hs_y)
                            host_len = math.hypot(host_vec[0], host_vec[1])
                            dir_host = (host_vec[0]/host_len, host_vec[1]/host_len) if host_len > 1e-3 else (1, 0)
                            
                            # 垂线方向向量
                            vert_vec = (foot_x - vs_x, foot_y - vs_y)
                            vert_len = math.hypot(vert_vec[0], vert_vec[1])
                            dir_vert = (vert_vec[0]/vert_len, vert_vec[1]/vert_len) if vert_len > 1e-3 else (0, 1)

                # 计算正方形L形端点
                host_end_px = foot_px + dir_host[0] * square_side
                host_end_py = foot_py + dir_host[1] * square_side
                vert_end_px = foot_px + dir_vert[0] * square_side
                vert_end_py = foot_py + dir_vert[1] * square_side

                # 绘制垂直符号并提取其像素轮廓
                perp_color = cfg["color"]
                line_width = cfg["width"]
                draw.line(
                    [(foot_px, foot_py), (int(host_end_px), int(host_end_py))],
                    fill=perp_color,
                    width=line_width
                )
                draw.line(
                    [(foot_px, foot_py), (int(vert_end_px), int(vert_end_py))],
                    fill=perp_color,
                    width=line_width
                )
                # 收集垂直符号的像素到非文本轮廓
                perp_pixels = self._get_shape_pixels("perpendicular", {
                    "fx": foot_px, "fy": foot_py,
                    "hx": host_end_px, "hy": host_end_py,
                    "vx": vert_end_px, "vy": vert_end_py,
                    "width": line_width
                })
                existing_shape_pixels.append(perp_pixels)
                
            except Exception as e:
                logger.warning(f"垂线标注失败 (id: {line.get('id')}): {e}")
                continue
        
        img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return existing_shape_pixels