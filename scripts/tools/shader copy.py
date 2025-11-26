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
from PIL import Image, ImageDraw
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .region import RegionExtractor, RegionExtractConfig
from .shaders import SHADERS

# 共享类型与坐标转换
PointMath = Tuple[float, float]
PointPixel = Tuple[int, int]
system_font = "DejaVu Sans"

# 坐标转换工具类
class CoordTransformer:
    def __init__(self, scale_factor: float, geom_center_x: float, geom_center_y: float,
                 canvas_center_x: float, canvas_center_y: float):
        self.scale_factor = scale_factor
        self.geom_center_x = geom_center_x
        self.geom_center_y = geom_center_y
        self.canvas_center_x = canvas_center_x
        self.canvas_center_y = canvas_center_y

    def math_to_pixel(self, math_coord: Tuple[float, float]) -> Tuple[int, int]:
        """数学坐标→像素坐标"""
        x, y = math_coord
        x_relative = x - self.geom_center_x
        y_relative = y - self.geom_center_y
        x_scaled = x_relative * self.scale_factor
        y_scaled = y_relative * self.scale_factor
        pixel_x = int(round(self.canvas_center_x + x_scaled))
        pixel_y = int(round(self.canvas_center_y - y_scaled))
        return (pixel_x, pixel_y)
    
    def pixel_to_math(self, pixel_coord: Tuple[int, int]) -> Tuple[float, float]:
        """像素坐标→数学坐标（用于轮廓点反向匹配）"""
        x, y = pixel_coord
        x_relative = (x - self.canvas_center_x) / self.scale_factor
        y_relative = (self.canvas_center_y - y) / self.scale_factor
        math_x = self.geom_center_x + x_relative
        math_y = self.geom_center_y + y_relative
        return (round(math_x, 4), round(math_y, 4))


COLOR_NAME_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0), 
    "blue": (255, 0, 0), 
    "white": (255, 255, 255), 
    "black": (0, 0, 0), 
    "yellow": (0, 255, 255) 
}

class GeometryAnnotator:
    """几何标注工具：按优先级优化文本位置（无冲突>小offset>大距离>正四方向）"""
    def __init__(self, annotator_config: Dict):
        self.config = self._validate_annotator_config(annotator_config)
        self.point_id_pattern = re.compile(r'^(?!circle_\d+$).+$')

        self.point_font = self.config["point"]["text"]["font"]
        self.line_font = self.config["line"]["text"]["font"]

    def _validate_annotator_config(self, cfg: Dict) -> Dict:
        default = {
            "point": {
                "enabled": True,
                "radius": 4,
                "color": (0, 0, 255),
                "text": {
                    "enabled": True,
                    "size": 12,
                    "color": (0, 0, 0),
                    "offset": 8,
                    "edge_margin": 5,
                    "min_pixel_dist": 3,
                    "scale": 0.5,
                    "thickness": 1,
                    "font": cv2.FONT_HERSHEY_SIMPLEX
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
                    "min_pixel_dist": 3,
                    "scale": 0.5,
                    "thickness": 1,
                    "font": cv2.FONT_HERSHEY_SIMPLEX
                }
            },
            "perpendicular": {
                "enabled": False,
                "color": (255, 0, 0),
                "width": 2,
                "symbol_size": 10
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

        if not isinstance(config["point"]["text"]["font"], int):
            config["point"]["text"]["font"] = cv2.FONT_HERSHEY_SIMPLEX
        if not isinstance(config["line"]["text"]["font"], int):
            config["line"]["text"]["font"] = cv2.FONT_HERSHEY_SIMPLEX

        return config

    def _parse_math_expr(self, expr_str: str) -> float:
        try:
            return float(sp.sympify(expr_str.replace(" ", "")).evalf())
        except Exception as e:
            logger.warning(f"表达式解析失败: {expr_str}，错误: {e}")
            raise

    def _get_text_pixels(self, text: str, font: int, scale: float, thickness: int, pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx, ty = pos
        
        temp_img = np.zeros((text_height + baseline, text_width, 3), dtype=np.uint8)
        cv2.putText(temp_img, text, (0, text_height), font, scale, (255, 255, 255), thickness)
        
        text_mask = np.any(temp_img > 0, axis=2)
        y_coords, x_coords = np.where(text_mask)
        
        text_pixels = set()
        for x, y in zip(x_coords, y_coords):
            abs_x = tx + x
            abs_y = ty - text_height + y
            text_pixels.add((abs_x, abs_y))
        
        return text_pixels
    
    def get_image_contour_pixels(self, img: np.ndarray, min_contour_area: int = 5) -> Set[Tuple[int, int]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, 
            0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU 
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_NONE
        )
        
        contour_pixels = set()
        for cnt in contours:
            for point in cnt.reshape(-1, 2):
                x, y = int(point[0]), int(point[1])
                contour_pixels.add((x, y))
        
        return contour_pixels


    def _check_pixel_collision(self, text_pixels: Set[Tuple[int, int]], 
                            existing_text_pixels: List[Set[Tuple[int, int]]],
                            contour_pixels: Set[Tuple[int, int]],
                            img_shape: Tuple[int, int], 
                            edge_margin: int,
                            min_pixel_dist: int = 3) -> bool:
        h, w = img_shape[:2]

        for (x, y) in text_pixels:
            if x < edge_margin or x >= (w - edge_margin) or y < edge_margin or y >= (h - edge_margin):
                return True

        if contour_pixels and (text_pixels & contour_pixels):
            return True

        for exist_pixels in existing_text_pixels:
            if text_pixels & exist_pixels:
                return True

        return False

    def _find_best_text_pos(self, img: np.ndarray, base_px: Tuple[int, int], text: str, 
                        existing_text_pixels: List[Set[Tuple[int, int]]], font: int, 
                        scale: float, thickness: int, 
                        base_offset: int, img_shape: Tuple[int, int], edge_margin: int, 
                        min_pixel_dist: int) -> Tuple[Tuple[int, int], Set[Tuple[int, int]]]:
        candidates = []

        priority_dirs = [
            ("正右", 1, 0), ("正左", -2, 0),
            ("正下", 0, -2), ("正上", 0, 1),
            ("右下", 1, -2), ("左下", -2, -2),
            ("右上", 1, 1), ("左上", -2, 1)
        ]

        offset_levels = [
            base_offset - 8, base_offset - 4, base_offset - 2,
            base_offset + 0, base_offset + 2,
            base_offset + 4
        ]
        offset_levels = [o for o in offset_levels if o >= 3] 
        
        min_contour_area = 5
        contour_pixels = self.get_image_contour_pixels(img, min_contour_area) 
        text_cfg = self.config["point"]["text"]
        min_pixel_dist = text_cfg.get("min_pixel_dist", 3)

        for dir_name, dx_coef, dy_coef in priority_dirs:
            for offset in offset_levels:
                dx = dx_coef * offset
                dy = dy_coef * offset
                tx, ty = base_px[0] + dx, base_px[1] + dy

                text_pixels = self._get_text_pixels(text, font, scale, thickness, (tx, ty))
                has_collision = self._check_pixel_collision(
                    text_pixels, existing_text_pixels, contour_pixels, img_shape, edge_margin, min_pixel_dist
                )
                
                v = 2
                if(dx_coef == 0):
                    if(dy_coef == -1):
                        v = 0
                    else:
                        v = 1
                
                candidates.append({
                    "pos": (tx, ty),
                    "pixels": text_pixels,
                    "has_collision": has_collision,
                    "offset": offset,
                    "is_priority_dir": v,
                })

        candidates.sort(
            key=lambda x: (
                x["has_collision"],
                x["is_priority_dir"],
                -x["offset"]
            )
        )

        best = candidates[0]
        (nx, ny) = best["pos"]
        nx = nx + 3
        return (nx, ny), best["pixels"]

    def draw_annotations(self, img: np.ndarray, data: Dict, transformer: Any) -> np.ndarray:
        if not self.config["enabled"]:
            return img.copy()
            
        img_copy = img.copy()
        existing_text_pixels = []
        non_text_bboxes = []
        points = data.get("points", [])
        lines = data.get("lines", [])
        
        valid_points = [p for p in points if self.point_id_pattern.match(p["id"])]
        points_dict = {p["id"]: p for p in valid_points}
        
        existing_text_pixels, non_text_bboxes = self.draw_point_annotations(
            img_copy, valid_points, transformer, existing_text_pixels, non_text_bboxes
        )
        
        existing_text_pixels, non_text_bboxes = self.draw_line_annotations(
            img_copy, lines, points_dict, transformer, existing_text_pixels, non_text_bboxes
        )
        
        self.draw_perpendicular_annotations(
            img_copy, lines, points_dict, transformer, non_text_bboxes, data.get("lines", [])
        )
        
        return img_copy

    def draw_point_annotations(self, img: np.ndarray, points: List[Dict], transformer: Any,
                              existing_text_pixels: List[Set[Tuple[int, int]]], non_text_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[List[Set[Tuple[int, int]]], List[Tuple[int, int, int, int]]]:
        cfg = self.config["point"]
        text_cfg = cfg["text"]
        if not cfg["enabled"] or not text_cfg["enabled"]:
            return existing_text_pixels, non_text_bboxes
        
        img_shape = img.shape
        base_offset = text_cfg["offset"]
        font = self.point_font
        font_scale = text_cfg["scale"]
        font_thickness = text_cfg["thickness"]

        for point in points:
            try:
                pid = point["id"]
                x = self._parse_math_expr(point["x"]["expr"])
                y = self._parse_math_expr(point["y"]["expr"])
                px, py = transformer.math_to_pixel((x, y))
                
                # 用cv2绘制点
                point_color = cfg["color"]
                radius = cfg["radius"]
                cv2.circle(img, (px, py), radius, point_color, -1)
                non_text_bboxes.append((px - radius, py - radius, px + radius, py + radius))
                
                # 用cv2绘制文本
                label = point.get("label", pid)
                if label:
                    best_pos, text_pixels = self._find_best_text_pos(
                        img,
                        base_px=(px, py),
                        text=label,
                        existing_text_pixels=existing_text_pixels,
                        font=font,
                        scale=font_scale,
                        thickness=font_thickness,
                        base_offset=base_offset,
                        img_shape=img_shape,
                        edge_margin=text_cfg["edge_margin"],
                        min_pixel_dist=text_cfg["min_pixel_dist"]
                    )
                    
                    cv2.putText(
                        img, label, best_pos, font, font_scale,
                        text_cfg["color"], font_thickness, cv2.LINE_AA
                    )
                    
                    existing_text_pixels.append(text_pixels)
            except Exception as e:
                logger.warning(f"点标注失败 (id: {pid}): {e}")
                continue
        
        return existing_text_pixels, non_text_bboxes

    def draw_line_annotations(self, img: np.ndarray, lines: List[Dict], points: Dict, transformer: Any,
                             existing_text_pixels: List[Set[Tuple[int, int]]], non_text_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[List[Set[Tuple[int, int]]], List[Tuple[int, int, int, int]]]:
        cfg = self.config["line"]
        text_cfg = cfg["text"]
        if not cfg["enabled"] or not text_cfg["enabled"]:
            return existing_text_pixels, non_text_bboxes
        
        img_shape = img.shape
        base_offset = text_cfg["offset"]
        font = self.line_font
        font_scale = text_cfg["scale"]
        font_thickness = text_cfg["thickness"]

        for line in lines:
            if line.get("type") == "perpendicular":
                continue
            
            try:
                start_id = line["start_point_id"]
                end_id = line["end_point_id"]
                if start_id not in points or end_id not in points:
                    logger.debug(f"线标注跳过（无效端点）: {line.get('id')}")
                    continue
                
                start_p = points[start_id]
                end_p = points[end_id]
                s_x = self._parse_math_expr(start_p["x"]["expr"])
                s_y = self._parse_math_expr(start_p["y"]["expr"])
                e_x = self._parse_math_expr(end_p["x"]["expr"])
                e_y = self._parse_math_expr(end_p["y"]["expr"])
                
                s_px, s_py = transformer.math_to_pixel((s_x, s_y))
                e_px, e_py = transformer.math_to_pixel((e_x, e_y))
                
                # 用cv2绘制线段
                line_color = cfg["color"]
                cv2.line(
                    img, (s_px, s_py), (e_px, e_py),
                    line_color, cfg["width"], cv2.LINE_AA
                )
                line_bbox = (
                    min(s_px, e_px) - cfg["width"],
                    min(s_py, e_py) - cfg["width"],
                    max(s_px, e_px) + cfg["width"],
                    max(s_py, e_py) + cfg["width"]
                )
                non_text_bboxes.append(line_bbox)
                
                # 用cv2绘制线文本
                label = line.get("label", line.get("id"))
                if label:
                    mid_px, mid_py = (s_px + e_px) // 2, (s_py + e_py) // 2
                    
                    best_pos, text_pixels = self._find_best_text_pos(
                        img,
                        base_px=(mid_px, mid_py),
                        text=label,
                        existing_text_pixels=existing_text_pixels,
                        font=font,
                        scale=font_scale,
                        thickness=font_thickness,
                        base_offset=base_offset,
                        img_shape=img_shape,
                        edge_margin=text_cfg["edge_margin"],
                        min_pixel_dist=text_cfg["min_pixel_dist"]
                    )
                    
                    cv2.putText(
                        img, label, best_pos, font, font_scale,
                        text_cfg["color"], font_thickness, cv2.LINE_AA
                    )
                    
                    existing_text_pixels.append(text_pixels)
            except Exception as e:
                logger.warning(f"线标注失败 (id: {line.get('id')}): {e}")
                continue
        
        return existing_text_pixels, non_text_bboxes


    def draw_perpendicular_annotations(self, img: np.ndarray, lines: List[Dict], points: Dict, 
                                    transformer: Any, non_text_bboxes: List[Tuple[int, int, int, int]],
                                    all_lines: List[Dict]) -> None:
        cfg = self.config["perpendicular"]
        if not cfg["enabled"]:
            return
        
        square_side = cfg["symbol_size"] 
        line_id_map = {line["id"]: line for line in all_lines}
        line_color = cfg["color"]
        line_width = cfg["width"]

        for line in lines:
            if line.get("type") != "perpendicular":
                continue
            
            try:
                foot_id = line["end_point_id"]
                vert_start_id = line["start_point_id"]
                desc = line.get("description", "")
                
                if foot_id not in points or vert_start_id not in points:
                    logger.debug(f"垂足/垂线起点无效: 垂足{foot_id}，起点{vert_start_id}")
                    continue
                
                vert_start_p = points[vert_start_id]
                vs_x = self._parse_math_expr(vert_start_p["x"]["expr"])
                vs_y = self._parse_math_expr(vert_start_p["y"]["expr"])
                vs_px, vs_py = transformer.math_to_pixel((vs_x, vs_y))
                
                foot_p = points[foot_id]
                foot_x = self._parse_math_expr(foot_p["x"]["expr"])
                foot_y = self._parse_math_expr(foot_p["y"]["expr"])
                foot_px, foot_py = transformer.math_to_pixel((foot_x, foot_y))

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
                            h_start_p = points[h_start_id]
                            h_end_p = points[h_end_id]
                            hs_x = self._parse_math_expr(h_start_p["x"]["expr"])
                            hs_y = self._parse_math_expr(h_start_p["y"]["expr"])
                            he_x = self._parse_math_expr(h_end_p["x"]["expr"])
                            he_y = self._parse_math_expr(h_end_p["y"]["expr"])
                            
                            host_vec = (he_x - hs_x, he_y - hs_y)
                            host_len = math.hypot(host_vec[0], host_vec[1])
                            if host_len < 1e-3:
                                dir_host = (1, 0)
                            else:
                                dir_host = (host_vec[0]/host_len, host_vec[1]/host_len)
                            
                            foot_to_hstart = math.hypot(foot_x - hs_x, foot_y - hs_y)
                            foot_to_hend = math.hypot(foot_x - he_x, foot_y - he_y)
                            if foot_to_hend < foot_to_hstart:
                                dir_host = (-dir_host[0], -dir_host[1])
                            
                            vert_vec = (foot_x - vs_x, foot_y - vs_y)
                            vert_len = math.hypot(vert_vec[0], vert_vec[1])
                            if vert_len < 1e-3:
                                dir_vert = (0, 1)
                            else:
                                dir_vert = (vert_vec[0]/vert_len, vert_vec[1]/vert_len)
                            
                            dot_product = dir_host[0] * dir_vert[0] + dir_host[1] * dir_vert[1]
                            if abs(dot_product) > 1e-3:
                                dir_vert = (dir_host[1], -dir_host[0])

                host_end_px = foot_px + dir_host[0] * square_side
                host_end_py = foot_py + dir_host[1] * square_side
                
                vert_end_px = foot_px + dir_vert[0] * square_side
                vert_end_py = foot_py + dir_vert[1] * square_side

                # 用cv2绘制L形垂直符号
                cv2.line(
                    img, (int(foot_px), int(foot_py)), (int(host_end_px), int(host_end_py)),
                    line_color, line_width, cv2.LINE_AA
                )
                cv2.line(
                    img, (int(foot_px), int(foot_py)), (int(vert_end_px), int(vert_end_py)),
                    line_color, line_width, cv2.LINE_AA
                )

                sym_min_px = min(foot_px, host_end_px, vert_end_px) - line_width
                sym_min_py = min(foot_py, host_end_py, vert_end_py) - line_width
                sym_max_px = max(foot_px, host_end_px, vert_end_px) + line_width
                sym_max_py = max(foot_py, host_end_py, vert_end_py) + line_width
                non_text_bboxes.append((int(sym_min_px), int(sym_min_py), int(sym_max_px), int(sym_max_py)))
                
            except Exception as e:
                logger.warning(f"垂线标注失败 (id: {line.get('id')}): {e}")
                continue

class EnhancedDrawer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enhanced_jsons: List[Dict] = [] 
        
        # 路径配置
        self.base_output_dir = self.config["global"]["output_root"]
        self.raw_dir = os.path.join(self.base_output_dir, "images/raw")
        self.shaded_dir = os.path.join(self.base_output_dir, "images/shaded")
        self.annotated_dir = os.path.join(self.base_output_dir, "images/annotated")
        self.json_output_dir = os.path.join(self.base_output_dir, "json/shaded")
        self.shaded_path = ''
        
        # 创建必要目录
        for dir_path in [self.raw_dir, self.shaded_dir, self.annotated_dir, self.json_output_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.shaded_jsonl_path = self.config["shader"].get(
            "jsonl_path", 
            os.path.join(self.json_output_dir, f"shaded_{self._get_timestamp()}.jsonl")
        )
        
        # 配置参数
        self.shader_enabled = self.config["shader"].get("enabled", True)
        self.annotator_enabled = self.config.get("annotator", {}).get("enabled", True)
        
        self.point_tolerance = self.config["shader"].get("point_tolerance", 5.0)
        self.line_tolerance = self.config["shader"].get("line_tolerance", 10.0)
        self.arc_tolerance = self.config["shader"].get("arc_tolerance", 15.0)
        self.x_attempts = max(1, self.config["shader"].get("x_attempts", 4))

        self.selected_region_labels = defaultdict(list)
        self.region_extractor = self._init_region_extractor()
        self.drawer = None
        self.processed_results = []  # 存储所有处理结果
        self.annotator = GeometryAnnotator(self.config.get("annotator", {}))
        
    def set_enhanced_data(self, enhanced_jsons: List[Dict]) -> None:
        """
        设置增强JSON数据（用于替代从文件加载的场景）
        Args:
            enhanced_jsons: 增强后的图形JSON列表
        """
        if not isinstance(enhanced_jsons, list):
            raise ValueError("增强JSON数据必须为列表类型")
        self.enhanced_jsons = enhanced_jsons
        logger.info(f"已设置 {len(enhanced_jsons)} 条增强JSON数据")
        
    def _save_jsonl(self) -> None:
        """将收集的所有结果写入JSONL文件"""
        with open(self.shaded_jsonl_path, "w", encoding="utf-8") as f:
            for result in self.processed_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    @staticmethod
    def _get_timestamp() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    def _init_region_extractor(self) -> RegionExtractor:
        """初始化区域提取器"""
        extract_cfg = RegionExtractConfig(
            thresh=self.config["region_extractor"]["thresh"],
            binarize_mode=self.config["region_extractor"]["binarize_mode"],
            min_area=self.config["shader"]["min_pixel_area"],
            fill_holes=self.config["region_extractor"]["fill_holes"],
            arc_detect=self.config["region_extractor"]["arc_detect"],
            curvature_threshold=self.config["region_extractor"]["curvature_threshold"]
        )
        return RegionExtractor(extract_cfg)

    def load_enhanced_jsonl(self, jsonl_path: str) -> None:
        """加载enhanced jsonl文件"""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.enhanced_jsons = [json.loads(line.strip()) for line in f if line.strip()]
        logger.info(f"加载 {len(self.enhanced_jsons)} 条enhanced数据")

    @staticmethod
    def _parse_expr(expr_str: str) -> float:
        try:
            expr_str = expr_str.replace("pi", "sp.pi").replace("sqrt", "sp.sqrt")
            return float(eval(expr_str))
        except Exception as e:
            logger.warning(f"表达式解析失败: {expr_str}，错误: {e}")
            return 0.0

    @staticmethod
    def _get_point_by_id(points: List[Dict], point_id: str) -> Optional[Dict]:
        for p in points:
            if p.get("id") == point_id:
                return p
        logger.warning(f"未找到点ID: {point_id}")
        return None
    
    def _point_in_shadow(self, point: Tuple[int, int], shadow_mask: np.ndarray) -> bool:
        """判断点是否在阴影区域内（基于二值mask）"""
        h, w = shadow_mask.shape[:2]
        x, y = point
        if 0 <= x < w and 0 <= y < h:
            return shadow_mask[y, x] > 0 
        return False 
    
    def _determine_arc_concavity(self, arc_detail: Dict, shadow_mask: np.ndarray, 
                                center_px: Tuple[int, int], transformer: CoordTransformer) -> str:
        """判断圆弧相对于阴影区域是凹陷（inward）还是突出（outward）"""
        try:
            start_math = arc_detail["start_point"]
            end_math = arc_detail["end_point"]
            start_px = transformer.math_to_pixel(start_math)
            end_px = transformer.math_to_pixel(end_math)
        except KeyError as e:
            logger.warning(f"圆弧缺少起点/终点信息：{e}，无法判断凹凸性")
            return "unknown"

        mid_x = (start_px[0] + end_px[0]) // 2
        mid_y = (start_px[1] + end_px[1]) // 2
        mid_point_px = (mid_x, mid_y)

        mid_in_shadow = self._point_in_shadow(mid_point_px, shadow_mask)
        center_in_shadow = self._point_in_shadow(center_px, shadow_mask)

        if mid_in_shadow:
            if not center_in_shadow:
                return "outward"
            else:
                return "inward"
        else:
            logger.debug(f"圆弧中点不在阴影内，无法明确凹凸性（中点：{mid_point_px}）")
            return "unknown"

    def _approximate_contour(self, contour_px: np.ndarray, epsilon_ratio: float = 0.005) -> np.ndarray:
        """对像素轮廓进行多边形近似"""
        if len(contour_px) < 3:
            return contour_px
        perimeter = cv2.arcLength(contour_px, closed=True)
        epsilon = epsilon_ratio * perimeter
        approx_px = cv2.approxPolyDP(contour_px, epsilon, closed=True)
        return approx_px.reshape(-1, 2)

    def _line_error(self, seg1: List[PointPixel], seg2: List[PointPixel]) -> float:
        """计算两条线段的平均端点距离误差"""
        (x1, y1), (x2, y2) = seg1
        (a1, b1), (a2, b2) = seg2
        err1 = (math.hypot(x1 - a1, y1 - b1) + math.hypot(x2 - a2, y2 - b2)) / 2
        err2 = (math.hypot(x1 - a2, y1 - b2) + math.hypot(x2 - a1, y2 - b1)) / 2
        return min(err1, err2)

    def _point_error(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        """计算两点欧氏距离（像素）"""
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def _match_contour_points(self, contour_px: np.ndarray, original_points: List[Dict], 
                             transformer: CoordTransformer) -> Tuple[List[Dict], List[float]]:
        """匹配轮廓上的点并计算坐标差值"""
        matched_points = []
        errors = []
        
        sample_step = max(1, len(contour_px) // 100)
        sampled_px = contour_px[::sample_step]

        for px in sampled_px:
            px_coord = (int(px[0]), int(px[1]))
            math_coord = transformer.pixel_to_math(px_coord)

            min_error = float("inf")
            best_point = None
            for point in original_points:
                try:
                    p_x = self._parse_expr(point["x"]["expr"])
                    p_y = self._parse_expr(point["y"]["expr"])
                    p_math = (p_x, p_y)
                    p_px = transformer.math_to_pixel(p_math)
                    error = self._point_error(px_coord, p_px)
                    
                    if error < min_error:
                        min_error = error
                        best_point = {"id": point["id"]}
                except (KeyError, ValueError) as e:
                    continue

            if best_point and min_error < self.point_tolerance:
                matched_points.append(best_point)
                errors.append(min_error)

        if errors:
            avg_error = sum(errors) / len(errors)
            logger.info(f"轮廓点与原始点的平均坐标差值: {avg_error:.2f}px "
                        f"(配置阈值: {self.point_tolerance}px)")
        else:
            logger.warning("未匹配到任何点，建议降低point_tolerance阈值")

        return matched_points, errors

    def _find_related_lines(self, matched_points: List[Dict], original_lines: List[Dict]) -> List[Dict]:
        """查找与匹配点关联的线条"""
        matched_point_ids = {p["id"] for p in matched_points}
        related_lines = []

        for line in original_lines:
            start_id = line["start_point_id"]
            end_id = line["end_point_id"]
            if start_id in matched_point_ids or end_id in matched_point_ids:
                related_lines.append({
                    "id": line["id"],
                    "start_point_id": start_id,
                    "end_point_id": end_id,
                })

        return related_lines

    def _validate_lines(self, candidate_lines: List[Dict], contour_edges_px: List[List[Tuple[int, int]]],
                       original_points: List[Dict], transformer: CoordTransformer) -> List[Dict]:
        """验证候选线条是否为轮廓的子集"""
        valid_lines = []

        for line in candidate_lines:
            start_point = self._get_point_by_id(original_points, line["start_point_id"])
            end_point = self._get_point_by_id(original_points, line["end_point_id"])
            if not start_point or not end_point:
                continue

            try:
                s_math = (self._parse_expr(start_point["x"]["expr"]), self._parse_expr(start_point["y"]["expr"]))
                e_math = (self._parse_expr(end_point["x"]["expr"]), self._parse_expr(end_point["y"]["expr"]))
                s_px = transformer.math_to_pixel(s_math)
                e_px = transformer.math_to_pixel(e_math)
                line_px = [s_px, e_px]
            except (KeyError, ValueError):
                continue

            min_error = min([self._line_error(edge, line_px) for edge in contour_edges_px])
            if min_error < self.line_tolerance:
                valid_lines.append({**line})

        return valid_lines

    def _find_related_arcs(self, matched_points: List[Dict], original_arcs: List[Dict]) -> List[Dict]:
        """查找与匹配点关联的圆弧"""
        matched_point_ids = {p["id"] for p in matched_points}
        related_arcs = []

        for arc in original_arcs:
            center_id = arc.get("center_point_id")
            if center_id and center_id in matched_point_ids:
                related_arcs.append({
                    "id": arc["id"],
                    "center_point_id": center_id,
                    "radius_expr": arc["radius"]["expr"],
                    "is_related": True
                })

        return related_arcs

    def _arc_error(self, detected_arc: Dict, original_arc: Dict) -> float:
        """计算检测到的圆弧与原始圆弧的误差"""
        center_err = self._point_error(detected_arc["center"], original_arc["center"])
        radius_err = abs(detected_arc["radius"] - original_arc["radius"])
        return (center_err + radius_err) / 2  # 综合误差

    def _process_region_entities(self, region_idx: int, region: Dict, 
                                original_points: List[Dict], original_lines: List[Dict],
                                original_arcs: List[Dict], transformer: CoordTransformer, pixel_contours: np.ndarray) -> Dict:
        """处理单个区域的实体匹配"""
        polygon_info = region["polygon"]
        raw_contour_px = pixel_contours[region_idx].reshape(-1, 2)
        approx_contour_px = self._approximate_contour(raw_contour_px)
        shadow_mask = region["mask"]

        matched_points, _ = self._match_contour_points(approx_contour_px, original_points, transformer)
        if not matched_points:
            logger.warning("未匹配到任何点，跳过该区域的线和弧匹配")
            return {"points": [], "lines": [], "arcs": []}

        contour_edges_px = []
        n_vertices = len(approx_contour_px)
        for i in range(n_vertices):
            p1 = tuple(approx_contour_px[i])
            p2 = tuple(approx_contour_px[(i+1) % n_vertices])
            contour_edges_px.append([p1, p2])

        candidate_lines = self._find_related_lines(matched_points, original_lines)
        valid_lines = self._validate_lines(candidate_lines, contour_edges_px, original_points, transformer)

        candidate_arcs = self._find_related_arcs(matched_points, original_arcs)
        valid_arcs = []
        if polygon_info["has_arc"] and candidate_arcs:
            for arc in candidate_arcs:
                center_point = self._get_point_by_id(original_points, arc["center_point_id"])
                if not center_point:
                    continue
                try:
                    c_math = (self._parse_expr(center_point["x"]["expr"]), self._parse_expr(center_point["y"]["expr"]))
                    radius_math = self._parse_expr(arc["radius_expr"])
                except (KeyError, ValueError):
                    continue

                c_px = transformer.math_to_pixel(c_math)
                radius_px = radius_math * transformer.scale_factor

                for arc_detail in polygon_info["arc_details"]:
                    arc_detail_px = arc_detail.copy()
                    arc_detail_px["center"] = transformer.math_to_pixel(arc_detail["center"])
                    arc_detail_px["radius"] = arc_detail["radius"] * transformer.scale_factor
                    
                    original_arc_px = {
                        "center": c_px,
                        "radius": radius_px,
                        "angle_range": self._parse_expr(arc.get("angle_range", "0"))
                    }
                    error = self._arc_error(arc_detail_px, original_arc_px)
                    if error < self.arc_tolerance:
                        concavity = self._determine_arc_concavity(
                            arc_detail_px, shadow_mask, c_px, transformer
                        )
                        valid_arcs.append({
                            "id": arc["id"],
                            "concavity": concavity
                        })
                        break

        return {
            "points": matched_points,
            "lines": valid_lines,
            "arcs": valid_arcs
        }

    def process(self) -> None:
        if not self.enhanced_jsons:
            raise RuntimeError("未加载enhanced数据，无法处理")

        logger.info("=== 开始处理图像 ===")
        drawer_cfg = self.config['drawer']
        self.drawer = GeometryDrawer(drawer_cfg)
        raw_image_paths = self.drawer.batch_draw()  # 原始图像路径

        self.geometry_params_list = self.drawer.get_geometry_params()
        if len(self.geometry_params_list) != len(raw_image_paths):
            logger.warning(f"图形参数与图像数量不匹配：{len(self.geometry_params_list)} vs {len(raw_image_paths)}")

        # 处理每张原始图像
        for idx, raw_img_path in enumerate(raw_image_paths):
            img_bgr = cv2.imread(raw_img_path)
            if img_bgr is None:
                logger.warning(f"第{idx}张原始图像读取失败，路径：{raw_img_path}，跳过")
                continue
            
            if idx >= len(self.geometry_params_list):
                logger.warning(f"第{idx}张图像无转换参数，跳过")
                continue
            geom_params = self.geometry_params_list[idx]
            
            data = self.enhanced_jsons[idx] if idx < len(self.enhanced_jsons) else {}
            
            # 保存原始图像到指定目录
            raw_filename = os.path.basename(raw_img_path)
            raw_save_path = os.path.join(self.raw_dir, raw_filename)
            cv2.imwrite(raw_save_path, img_bgr)
            
            # 处理原始图像的标注
            transformer = CoordTransformer(
                scale_factor=geom_params["scale_factor"],
                geom_center_x=geom_params["geom_center_x"],
                geom_center_y=geom_params["geom_center_y"],
                canvas_center_x=geom_params["canvas_center_x"],
                canvas_center_y=geom_params["canvas_center_y"]
            )
            
            # 生成原始图像的标注版本
            annotated_raw_img = self.annotator.draw_annotations(img_bgr, data, transformer)
            annotated_raw_filename = f"annotated_raw_{raw_filename}"
            annotated_raw_path = os.path.join(self.annotated_dir, annotated_raw_filename)
            cv2.imwrite(annotated_raw_path, annotated_raw_img)
            
            logger.info(f"成功标注第{idx}张图像，原始版本")
            self.processed_results.append({
                **data,
                "raw_path": raw_save_path,
                "annotated_raw_path": annotated_raw_path,
                "shaded_path": None,
                "annotated_shaded_path": None,
                "shader_enabled": False
            })
            
            # 如果shader禁用，跳过阴影处理
            if not self.shader_enabled:
                continue
            
            # 处理阴影
            for attempt in range(self.x_attempts):
                try:
                    self._process_shaded_image(
                        img_bgr=img_bgr,
                        data=data,
                        idx=idx,
                        attempt=attempt,
                        geom_params=geom_params,
                        raw_save_path=raw_save_path,
                        annotated_raw_path=annotated_raw_path,
                        transformer=transformer
                    )
                except Exception as e:
                    logger.error(f"第{idx}条数据的第{attempt}次阴影尝试失败: {str(e)}")
                    continue

        self._save_jsonl()
        logger.info(f"所有处理结果已保存至：{self.shaded_jsonl_path}")
    
    def process_single(self) -> Dict[str, Any]:
        """
        处理单条增强图形数据的着色与标注（独立运行，线程安全）
        返回：处理后的单条数据字典
        """
        if not self.enhanced_jsons or len(self.enhanced_jsons) != 1:
            raise ValueError("单条处理需且仅需设置1条增强JSON数据（通过set_enhanced_data传入长度为1的列表）")
        
        data = self.enhanced_jsons[0]
        drawer_cfg = self.config['drawer']
        
        # 核心：基于enhance_id作为唯一标识（优先使用data中的enhance_id，确保全流程一致）
        enhance_id = data.get("enhance_id")
        if not enhance_id:
            # 若缺失，基于base_idx和enhance_idx生成规范标识（与其他步骤统一）
            base_idx = data.get("base_idx", 0)
            enhance_idx = data.get("enhance_idx", 0)
            enhance_id = f"base_{base_idx:03d}_enhance_{enhance_idx:03d}"
            data["enhance_id"] = enhance_id  # 补全到数据中，便于后续追踪
            logger.warning(f"数据缺失enhance_id，自动生成：{enhance_id}")
        
        # 初始化绘图器（每个线程独立实例）
        if not self.drawer:
            self.drawer = GeometryDrawer(drawer_cfg)
        
        # 调用draw_single获取图像路径和几何参数（draw_single已基于enhance_id命名）
        raw_img_path, geom_params = self.drawer.draw_single(data)
        if not raw_img_path or not os.path.exists(raw_img_path):
            raise RuntimeError(f"原始图像生成失败，路径无效：{raw_img_path}")
        
        # 验证几何参数
        required_params = ["scale_factor", "geom_center_x", "geom_center_y", "canvas_center_x", "canvas_center_y"]
        if not all(key in geom_params for key in required_params):
            raise RuntimeError(f"几何参数缺失，当前参数：{list(geom_params.keys())}")
        
        # 读取原始图像
        img_bgr = cv2.imread(raw_img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"原始图像读取失败：{raw_img_path}")
        
        # 创建坐标转换器
        transformer = CoordTransformer(
            scale_factor=geom_params["scale_factor"],
            geom_center_x=geom_params["geom_center_x"],
            geom_center_y=geom_params["geom_center_y"],
            canvas_center_x=geom_params["canvas_center_x"],
            canvas_center_y=geom_params["canvas_center_y"]
        )
        
        # 保存原始图像（基于enhance_id命名，与draw_single生成的路径呼应）
        # 原始图像路径格式：{enhance_id}_raw.png（直接使用draw_single生成的路径，无需二次重命名）
        raw_save_path = raw_img_path  # 复用draw_single生成的规范路径，避免重复存储
        # 若需备份到raw_dir，可改为：
        # raw_filename = os.path.basename(raw_img_path)
        # raw_save_path = os.path.join(self.raw_dir, raw_filename)
        # cv2.imwrite(raw_save_path, img_bgr)
        
        # 生成标注后的原始图像（基于enhance_id命名：{enhance_id}_raw_annotated.png）
        annotated_raw_filename = f"{enhance_id}_raw_annotated.png"
        annotated_raw_path = os.path.join(self.annotated_dir, annotated_raw_filename)
        cv2.imwrite(annotated_raw_path, self.annotator.draw_annotations(img_bgr, data, transformer))
        
        # 初始化返回结果（保留enhance_id等核心标识）
        result = {
            **data,
            "enhance_id": enhance_id,  # 明确保留enhance_id
            "base_id": data.get("base_id"),
            "base_idx": data.get("base_idx"),
            "enhance_idx": data.get("enhance_idx"),
            "raw_path": raw_save_path,
            "annotated_raw_path": annotated_raw_path,
            "shaded_path": None,
            "annotated_shaded_path": None,
            "shadow_type": None,
            "shader_enabled": False,
            "entities": data.get("entities", [])
        }
        
        # 阴影处理（传递enhance_id作为唯一标识）
        if self.shader_enabled:
            for attempt in range(self.x_attempts):
                try:
                    shaded_info = self._process_shaded_image_single(
                        img_bgr=img_bgr,
                        data=data,
                        enhance_id=enhance_id,  # 用enhance_id替代原unique_suffix
                        geom_params=geom_params,
                        transformer=transformer
                    )
                    result.update({
                        "shaded_path": shaded_info["shaded_path"],
                        "annotated_shaded_path": shaded_info["annotated_shaded_path"],
                        "shadow_type": shaded_info["shadow_type"],
                        "shader_enabled": True,
                        "entities": result["entities"] + shaded_info["shadow_entities"]
                    })
                    break
                except Exception as e:
                    logger.warning(f"单条数据（enhance_id: {enhance_id}）阴影处理第{attempt+1}次尝试失败：{str(e)}")
                    if attempt == self.x_attempts - 1:
                        logger.error(f"单条数据（enhance_id: {enhance_id}）所有阴影尝试失败，跳过阴影处理")
        
        logger.info(f"单条数据处理完成（enhance_id: {enhance_id}），原始图像：{raw_save_path}")
        return result

    def _process_shaded_image_single(self, img_bgr: np.ndarray, data: Dict, unique_suffix: str,
                                    geom_params: Dict, transformer: CoordTransformer) -> Dict[str, Any]:
        """
        适配单条数据的阴影处理（独立于批量处理，返回关键信息而非修改全局状态）
        返回：阴影处理后的核心信息字典
        """
        # 提取图像元信息
        meta = self._get_meta(data)
        # 提取区域（轮廓、掩码等）
        masks, pixel_contours, _, _, polygons = self.region_extractor.extract(img_bgr, meta)
        if not masks or not polygons or not pixel_contours:
            raise RuntimeError("未提取到有效区域，阴影处理失败")
        
        # 选择阴影类型和参数
        shadow_type = np.random.choice(self.config["shader"]["shadow_types"])
        shader = SHADERS[shadow_type]
        shader_params = self._get_shader_params(shadow_type)
        
        # 选择待着色区域（按面积排序，取最大的N个）
        n_regions = self.config["shader"]["n_regions"]
        select_n = np.random.randint(n_regions[0], n_regions[1]+1)
        # 单条处理无需排除历史选中，直接按面积选择
        available_regions = [{"mask": mask, "polygon": polygon, "label": polygon.get("label")} 
                            for mask, polygon in zip(masks, polygons)]
        if not available_regions:
            raise RuntimeError("无可用着色区域")
        # 按面积降序排序，选择前N个
        available_regions_sorted = sorted(
            available_regions,
            key=lambda x: x["polygon"].get("pixel", {}).get("area_px_est", 0),
            reverse=True
        )
        selected_regions = available_regions_sorted[:select_n]
        
        # 应用阴影效果
        shaded_img = img_bgr.copy()
        for region in selected_regions:
            shaded_img = shader.apply(shaded_img, region["mask"], **shader_params)
        
        # 生成阴影关联的实体信息
        shadow_entities = []
        original_lines = data.get("lines", [])
        original_arcs = data.get("arcs", [])
        original_points = data.get("points", [])
        for region_idx, region in enumerate(selected_regions):
            entity_data = self._process_region_entities(
                region_idx, region, original_points, original_lines, original_arcs, transformer, pixel_contours
            )
            shadow_entities.append({
                "type": "shadow",
                "region_label": region["label"],
                "shadow_type": shadow_type,
                "points": entity_data["points"],
                "lines": entity_data["lines"],
                "arcs": entity_data["arcs"],
                "stats": {
                    "total_points": len(entity_data["points"]),
                    "total_lines": len(entity_data["lines"]),
                    "total_arcs": len(entity_data["arcs"])
                }
            })
        
        # 保存阴影图像（唯一文件名，避免冲突）
        raw_filename = f"shaded_{unique_suffix}.png"
        shaded_path = os.path.join(self.shaded_dir, raw_filename)
        cv2.imwrite(shaded_path, shaded_img)
        
        # 生成并保存阴影图像的标注版本
        annotated_shaded_img = self.annotator.draw_annotations(shaded_img, data, transformer)
        annotated_shaded_path = os.path.join(self.annotated_dir, f"annotated_{raw_filename}")
        cv2.imwrite(annotated_shaded_path, annotated_shaded_img)
        
        # 返回单条阴影处理结果
        return {
            "shaded_path": shaded_path,
            "annotated_shaded_path": annotated_shaded_path,
            "shadow_type": shadow_type,
            "shadow_entities": shadow_entities
        }

    def _process_shaded_image(self, img_bgr: np.ndarray, data: Dict, idx: int, attempt: int,
                             geom_params: Dict, raw_save_path: str, annotated_raw_path: str,
                             transformer: CoordTransformer) -> None:
        """处理阴影图像的生成、标注和保存"""
        meta = self._get_meta(data)
        masks, pixel_contours, _, _, polygons = self.region_extractor.extract(img_bgr, meta)
        if not masks or not polygons or not pixel_contours:
            logger.warning(f"第{idx}条数据的第{attempt}次尝试无有效区域，跳过")
            return

        # 选择阴影类型和参数
        shadow_type = np.random.choice(self.config["shader"]["shadow_types"])
        shader = SHADERS[shadow_type]
        shader_params = self._get_shader_params(shadow_type)

        # 选择区域
        exclude_labels = self.selected_region_labels[idx]
        n_regions = self.config["shader"]["n_regions"]
        select_n = np.random.randint(n_regions[0], n_regions[1]+1)
        selected_regions = self._select_regions(masks, polygons, select_n, exclude_labels)
        if not selected_regions:
            logger.warning(f"第{idx}次尝试无可用未选区域，跳过")
            return

        # 记录选中区域标签
        current_selected_labels = [region["label"] for region in selected_regions]
        self.selected_region_labels[idx].extend(current_selected_labels)

        # 应用阴影
        shaded_img = img_bgr.copy()
        for region in selected_regions:
            shaded_img = shader.apply(shaded_img, region["mask"],** shader_params)
            
        # 生成阴影实体信息
        shadow_entities = []
        original_lines = data.get("lines", [])
        original_arcs = data.get("arcs", [])
        original_points = data.get("points", [])

        for region_idx, region in enumerate(selected_regions):
            entity_data = self._process_region_entities(
                region_idx, region, original_points, original_lines, original_arcs, transformer, pixel_contours
            )

            shadow_entity = {
                "type": "shadow",
                "region_label": region["label"],
                "shadow_type": shadow_type,
                "points": entity_data["points"],
                "lines": entity_data["lines"],
                "arcs": entity_data["arcs"],
                "stats": {
                    "total_points": len(entity_data["points"]),
                    "total_lines": len(entity_data["lines"]),
                    "total_arcs": len(entity_data["arcs"])
                }
            }
            shadow_entities.append(shadow_entity)
            
        # 保存阴影图像
        raw_filename = os.path.basename(raw_save_path)
        shaded_filename = f"shaded_{idx}_attempt_{attempt}_{raw_filename}"
        shaded_path = os.path.join(self.shaded_dir, shaded_filename)
        cv2.imwrite(shaded_path, shaded_img)
        
        # 生成并保存阴影图像的标注版本
        annotated_shaded_img = self.annotator.draw_annotations(shaded_img, data, transformer)
        annotated_shaded_filename = f"annotated_shaded_{idx}_attempt_{attempt}_{raw_filename}"
        annotated_shaded_path = os.path.join(self.annotated_dir, annotated_shaded_filename)
        cv2.imwrite(annotated_shaded_path, annotated_shaded_img)
        
        logger.info(f"成功标注第{idx}张第{attempt}次阴影图像")

        # 更新并保存结果数据
        new_data = data.copy()
        new_data["entities"] = new_data.get("entities", []) + shadow_entities
        new_data.update({
            "raw_path": raw_save_path,
            "annotated_raw_path": annotated_raw_path,
            "shaded_path": shaded_path,
            "annotated_shaded_path": annotated_shaded_path,
            "shadow_type": shadow_type,
            "shader_enabled": True
        })
        self.processed_results.append(new_data)
        self.shaded_path = shaded_path

        logger.info(f"第{idx}条数据第{attempt}次尝试完成，阴影图像路径：{shaded_path}")

    def _get_meta(self, data: Dict) -> Dict:
        """从JSON数据中提取元信息（如几何中心）"""
        points = data.get("points", [])
        if not points:
            return {}
        math_pts = []
        for pt in points:
            try:
                x = float(pt["x"]["expr"])
                y = float(pt["y"]["expr"])
                math_pts.append((x, y))
            except (KeyError, ValueError):
                continue
        if not math_pts:
            return {}
        center_math = (
            sum(p[0] for p in math_pts) / len(math_pts),
            sum(p[1] for p in math_pts) / len(math_pts)
        )
        return {"center_math": center_math}

    def _select_regions(
        self, 
        masks: List[np.ndarray], 
        polygons: List[Dict], 
        n: int, 
        exclude_labels: List[int]
    ) -> List[Dict]:
        """按面积排序选择区域，排除已选过的区域"""
        available_regions = []
        for mask, polygon in zip(masks, polygons):
            region_label = polygon.get("label")
            if region_label is not None and region_label not in exclude_labels:
                available_regions.append({"mask": mask, "polygon": polygon, "label": region_label})
        
        if not available_regions:
            logger.warning(f"无可用未选区域，返回空列表")
            return []
        
        select_n = min(n, len(available_regions))
        available_regions_sorted = sorted(
            available_regions,
            key=lambda x: x["polygon"].get("pixel", {}).get("area_px_est", 0),
            reverse=True
        )
        return available_regions_sorted[:select_n]

    def _get_shader_params(self, shadow_type: str) -> Dict:
        """生成阴影参数（与配置匹配）"""
        intensity = np.random.uniform(*self.config["shader"]["intensity_range"])
        if shadow_type == "hatch":
            return {"spacing": self.config["shader"]["hatch_spacing"], "intensity": intensity}
        elif shadow_type == "crosshatch":
            return {
                "spacing": self.config["shader"]["crosshatch_spacing"],
                "angle1": 45, "angle2": 135,
                "intensity": intensity
            }
        elif shadow_type == "solid":
            return {"color": (128, 128, 128), "intensity": intensity}
        elif shadow_type == "gradient":
            return {
                "start_color": (200, 200, 200),
                "end_color": (100, 100, 100),
                "angle_deg": 45,
                "intensity": intensity
            }
        return {"intensity": intensity}