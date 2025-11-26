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
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from .drawer import GeometryDrawer
from .region import RegionExtractor, RegionExtractConfig
from .shaders import SHADERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        """像素坐标→数学坐标"""
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
        
        # 匹配阈值（可通过配置调整，默认经验值）
        self.line_match_threshold = self.config["shader"].get("line_match_threshold", 0.7)
        self.arc_match_threshold = self.config["shader"].get("arc_match_threshold", 0.7)
        self.x_attempts = max(1, self.config["shader"].get("x_attempts", 4))
        
        
        self.distance_threshold = config.get("distance_threshold", 15)
        self.match_threshold = config.get("match_threshold", 0.97)
        self.min_sample_points = config.get("min_sample_points", 50)

        self.selected_region_labels = defaultdict(list)
        self.region_extractor = self._init_region_extractor()
        self.drawer = None
        self.processed_results = []  # 存储所有处理结果
        self.annotator = GeometryAnnotator(self.config.get("annotator", {}))
        
    def set_enhanced_data(self, enhanced_jsons: List[Dict]) -> None:
        """设置增强JSON数据（用于替代从文件加载的场景）"""
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
        
    def _match_corners_to_original_points(self, corner_points: List[np.ndarray], original_points: List[Dict], transformer: CoordTransformer) -> Dict[tuple, str]:
        """
        将检测到的角点与原始点进行匹配。
        :param corner_points: 检测到的角点像素坐标列表。
        :param original_points: 原始点数据。
        :param transformer: 坐标转换器。
        :return: 一个字典，键是角点的像素坐标元组，值是匹配到的原始点ID。
        """
        corner_to_point_id = {}
        
        if not corner_points:
            return corner_to_point_id

        # 1. 将所有原始点转换为像素坐标，并建立 {像素坐标元组: 点ID} 的映射
        original_points_px = {}
        for p in original_points:
            try:
                px_coord = transformer.math_to_pixel((
                    self._parse_expr(p['x']['expr']),
                    self._parse_expr(p['y']['expr'])
                ))
                # 转换为元组以便作为字典的键
                original_points_px[tuple(px_coord)] = p['id']
            except Exception as e:
                logger.warning(f"转换原始点 {p.get('id')} 失败，跳过: {e}")

        if not original_points_px:
            logger.warning("没有成功转换任何原始点，无法进行角点匹配。")
            return corner_to_point_id

        # 2. 为每个角点寻找最近的原始点
        for corner in corner_points:
            corner_tuple = tuple(corner)
            min_distance = float('inf')
            matched_point_id = None

            for (orig_px_tuple, orig_id) in original_points_px.items():
                # 计算欧氏距离
                distance = np.linalg.norm(np.array(corner) - np.array(orig_px_tuple))
                
                if distance < min_distance:
                    min_distance = distance
                    matched_point_id = orig_id

            # 3. 设置一个距离阈值，只有当距离小于阈值时才认为是有效匹配
            # 这个阈值可以根据图像分辨率和实际情况调整，例如 10 个像素
            match_threshold = self.config.get("shader", {}).get("corner_match_threshold", 10.0)
            if min_distance < match_threshold:
                corner_to_point_id[corner_tuple] = matched_point_id
                logger.debug(f"角点 {corner_tuple} 匹配到原始点 {matched_point_id}，距离: {min_distance:.2f}")
            else:
                logger.debug(f"角点 {corner_tuple} 未找到有效匹配（最近距离: {min_distance:.2f} > 阈值 {match_threshold}）")

        return corner_to_point_id

    # ==================== 新核心逻辑：基于角点+分数的匹配 ====================
    def _process_region_entities(self, region: Dict, 
                                original_points: List[Dict], original_lines: List[Dict],
                                original_arcs: List[Dict], transformer: CoordTransformer) -> Dict:
        """
        简化且鲁棒的实体匹配流程：
        1. 从mask中获取精确轮廓 → 2. 角点检测+轮廓分段 → 3. 片段拟合（直线/圆弧）→ 4. 分数匹配
        """
        # 1. 从mask获取最大轮廓（确保轮廓完整性）
        raw_contour, _ = self._find_contours_from_mask(region['mask'])
        if raw_contour is None:
            logger.warning("未能从mask中找到轮廓，跳过此区域")
            return {"points": [], "lines": [], "arcs": []}

        # 2. 轮廓预处理（简化点数）+ 角点检测（分段依据）
        processed_contour = self._preprocess_contour(raw_contour)
        corner_points = self._detect_corner_points(processed_contour)
        
        
        logger.debug(f"轮廓预处理后，包含 {len(processed_contour)} 个点")
        logger.debug(f"检测到 {len(corner_points)} 个角点")
        
        # ====== 新增步骤：角点与原始点匹配 ======
        corner_to_point_id = self._match_corners_to_original_points(corner_points, original_points, transformer)
        logger.info(f"成功匹配 {len(corner_to_point_id)} 个角点到原始点")
        # print(corner_to_point_id)
        

        # 3. 按角点分段轮廓（无角点则整段为一个片段）
        segments = self._segment_contour_by_corners(processed_contour, corner_points)
        if not segments:
            logger.warning("轮廓分段失败，跳过此区域")
            return {"points": [], "lines": [], "arcs": []}

        # 4. 片段拟合：每个片段同时拟合直线和圆弧，选误差更小的基元
        detected_lines, detected_arcs = self._fit_segments(segments)

        # 5. 原始实体转换为像素坐标（用于计算匹配分数）
        original_lines_px = self._convert_original_lines_to_pixel_simple(original_lines, original_points, transformer)
        original_arcs_px = self._convert_original_arcs_to_pixel_simple(original_arcs, original_points, transformer)

        # 6. 分数匹配：为原始实体找到最佳检测基元
        matched_line_ids = self._match_detected_to_original(
            detected_lines, original_lines_px, 'line', self.line_match_threshold
        )
        matched_arc_ids = self._match_detected_to_original(
            detected_arcs, original_arcs_px, 'arc', self.arc_match_threshold
        )

        # 7. 提取匹配到的点（线的端点、弧的圆心）
        matched_point_ids = self._extract_matched_point_ids(
            matched_line_ids, matched_arc_ids, original_lines, original_arcs
        )
        
        
        corner_matched_point_ids = set(corner_to_point_id.values())
        logger.info(f"通过线段匹配得到的原始点ID: {matched_point_ids}")
        logger.info(f"通过角点直接匹配得到的原始点ID: {corner_matched_point_ids}")

        # 8. 格式化结果（含弧的凹凸性）
        return self._format_result(
            corner_matched_point_ids,
            matched_point_ids, matched_line_ids, matched_arc_ids,
            detected_arcs, original_arcs_px, region['mask'], transformer
        )

    # -------------------- 辅助函数：轮廓处理 --------------------
    @staticmethod
    def _find_contours_from_mask(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从二值mask中提取最大轮廓（确保是目标区域的完整轮廓）"""
        # 只提取最外层轮廓，避免内部噪声干扰
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, None
        # 选择面积最大的轮廓（大概率是目标区域）
        largest_idx = np.argmax([cv2.contourArea(c) for c in contours])
        return contours[largest_idx].reshape(-1, 2), hierarchy

    @staticmethod
    def _preprocess_contour(contour: np.ndarray) -> np.ndarray:
        """简化轮廓点数（减少计算量），保留主要形状"""
        perimeter = cv2.arcLength(contour, True)
        # 宽松的简化阈值（0.005*周长），避免丢失关键角点
        epsilon = 0.001 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx.reshape(-1, 2)

    def _detect_corner_points(self, contour: np.ndarray, min_distance: int = 20) -> List[np.ndarray]:
        """用Harris角点检测找到轮廓上的角点（分段依据）"""
        if len(contour) < 5:  # 轮廓点数太少，无法检测角点
            return []

        # 创建轮廓的灰度图像（Harris检测需要图像输入）
        max_x, max_y = np.max(contour, axis=0).astype(int)
        img = np.zeros((max_y + 2, max_x + 2), dtype=np.uint8)
        cv2.drawContours(img, [contour.astype(int)], -1, 255, 1)  # 绘制轮廓为白色

        # Harris角点检测（参数：块大小2， Sobel核大小3，响应参数0.04）
        img_float = np.float32(img)
        dst = cv2.cornerHarris(img_float, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)  # 膨胀增强角点

        # 筛选角点（阈值：0.01*最大响应值）
        threshold = 0.01 * dst.max()
        y_coords, x_coords = np.where(dst > threshold)

        # 转换为轮廓上的实际点（避免检测到轮廓外的噪声点）
        corner_points = []
        for x, y in zip(x_coords, y_coords):
            # 找到轮廓上离(x,y)最近的点（距离<5才认为是有效角点）
            distances = np.linalg.norm(contour - np.array([x, y]), axis=1)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < 5:
                corner_points.append(contour[closest_idx])

        # 去重（保留距离>min_distance的角点，避免密集重复）
        return self._remove_duplicate_points(corner_points, min_distance)

    @staticmethod
    def _remove_duplicate_points(points: List[np.ndarray], min_distance: int) -> List[np.ndarray]:
        """去除距离过近的重复点"""
        if not points:
            return []
        unique_points = [points[0]]
        for p in points[1:]:
            # 若当前点与所有已保留点的距离都>min_distance，则保留
            if all(np.linalg.norm(p - up) > min_distance for up in unique_points):
                unique_points.append(p)
        return unique_points

    @staticmethod
    def _segment_contour_by_corners(contour: np.ndarray, corner_points: List[np.ndarray]) -> List[np.ndarray]:
        """按角点将轮廓分割为多个片段"""
        if len(corner_points) < 2:  # 角点太少，不分割（整段为一个片段）
            return [contour]

        # 找到角点在轮廓中的索引（用于排序和分段）
        corner_indices = []
        for p in corner_points:
            # 找到轮廓中与角点完全匹配的点的索引
            match_mask = (contour == p).all(axis=1)
            if np.any(match_mask):
                corner_indices.append(np.where(match_mask)[0][0])

        # 排序角点索引（按轮廓顺序）
        corner_indices.sort()
        segments = []
        start_idx = 0

        # 分割为[start_idx, idx]的片段
        for idx in corner_indices:
            if idx > start_idx:  # 避免空片段
                segments.append(contour[start_idx:idx + 1])
            start_idx = idx + 1

        # 补充最后一段：从最后一个角点回到轮廓起点（闭合轮廓）
        if start_idx < len(contour):
            last_segment = np.vstack([contour[start_idx:], contour[:corner_indices[0] + 1]])
            segments.append(last_segment)

        return segments

    # -------------------- 辅助函数：片段拟合 --------------------
    @staticmethod
    def _fit_line_to_segment(segment: np.ndarray) -> Tuple[Dict, float]:
        """拟合直线并计算平均误差（所有点到直线的距离均值）"""
        # 直线拟合（cv2.fitLine：最小二乘，抗噪声）
        vx, vy, x, y = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)
        line_params = {
            'direction': (vx, vy),  # 方向向量
            'point': (x, y),        # 直线上的一个点
            'start': tuple(segment[0]),  # 片段起点（像素坐标）
            'end': tuple(segment[-1]),   # 片段终点（像素坐标）
            'length': np.linalg.norm(segment[-1] - segment[0])  # 片段长度
        }

        # 计算平均误差：所有点到直线的距离
        distances = np.abs(vy * (segment[:, 0] - x) - vx * (segment[:, 1] - y)) / np.sqrt(vx**2 + vy**2)
        avg_error = np.mean(distances) if len(distances) > 0 else float('inf')
        return line_params, avg_error

    @staticmethod
    def _fit_arc_to_segment(segment: np.ndarray) -> Tuple[Dict, float]:
        """拟合圆弧并计算平均误差（所有点到圆弧的距离均值）"""
        if len(segment) < 5:  # 点数太少，无法拟合圆弧
            return {}, float('inf')

        # 用最小外接圆近似圆弧（简化计算，适合大多数场景）
        (x, y), radius = cv2.minEnclosingCircle(segment)
        arc_params = {
            'center': (x, y),        # 圆心（像素坐标）
            'radius': radius,        # 半径（像素）
            'start': tuple(segment[0]),  # 片段起点
            'end': tuple(segment[-1]),   # 片段终点
            'segment': segment       # 保留原始片段（用于凹凸性判断）
        }

        # 计算平均误差：所有点到圆弧的距离
        distances = np.abs(np.linalg.norm(segment - (x, y), axis=1) - radius)
        avg_error = np.mean(distances) if len(distances) > 0 else float('inf')
        return arc_params, avg_error

    def _fit_segments(self, segments: List[np.ndarray]) -> Tuple[List[Dict], List[Dict]]:
        """对所有片段进行直线/圆弧拟合，返回检测到的基元"""
        detected_lines = []
        detected_arcs = []
        # 拟合误差阈值（经验值：直线<5，圆弧<8，可通过配置调整）
        # line_error_thresh = self.config["shader"].get("line_error_thresh", 50.0)
        # arc_error_thresh = self.config["shader"].get("arc_error_thresh", 80.0)
        
        
        line_error_thresh = 50.0
        arc_error_thresh = 80.0

        for seg in segments:
            if len(seg) < 5:  # 过滤太短的片段（噪声）
                continue

            # 同时拟合直线和圆弧
            line_params, line_error = self._fit_line_to_segment(seg)
            arc_params, arc_error = self._fit_arc_to_segment(seg)

            # 选择误差更小的基元（必须低于阈值才有效）
            if line_error < arc_error and line_error < line_error_thresh:
                detected_lines.append({'params': line_params, 'error': line_error})
            elif arc_error < line_error and arc_error < arc_error_thresh:
                detected_arcs.append({'params': arc_params, 'error': arc_error})

        logger.debug(f"片段拟合完成：检测到 {len(detected_lines)} 条直线，{len(detected_arcs)} 段圆弧")
        return detected_lines, detected_arcs

    # -------------------- 辅助函数：坐标转换 --------------------
    def _convert_original_lines_to_pixel_simple(self, original_lines: List[Dict], original_points: List[Dict], transformer: CoordTransformer) -> List[Dict]:
        """将原始线实体转换为像素坐标（用于匹配）"""
        lines_px = []
        for line in original_lines:
            try:
                # 获取线的端点ID
                start_id = line["start_point_id"]
                end_id = line["end_point_id"]
                # 找到端点的数学坐标
                start_p = next(p for p in original_points if p['id'] == start_id)
                end_p = next(p for p in original_points if p['id'] == end_id)
                # 转换为像素坐标
                start_px = transformer.math_to_pixel((
                    self._parse_expr(start_p['x']['expr']),
                    self._parse_expr(start_p['y']['expr'])
                ))
                end_px = transformer.math_to_pixel((
                    self._parse_expr(end_p['x']['expr']),
                    self._parse_expr(end_p['y']['expr'])
                ))
                # 计算线的几何特征（方向、长度）
                vec = (end_px[0] - start_px[0], end_px[1] - start_px[1])
                length = np.linalg.norm(vec)
                dir_unit = vec / length if length > 0 else (0, 0)
                lines_px.append({
                    'id': line['id'],
                    'start': start_px,
                    'end': end_px,
                    'length': length,
                    'dir': dir_unit
                })
            except (StopIteration, KeyError, ValueError, TypeError) as e:
                logger.warning(f"转换原始线 {line.get('id')} 失败：{e}")
        return lines_px

    def _convert_original_arcs_to_pixel_simple(self, original_arcs: List[Dict], original_points: List[Dict], transformer: CoordTransformer) -> List[Dict]:
        """将原始弧实体转换为像素坐标（用于匹配）"""
        arcs_px = []
        for arc in original_arcs:
            try:
                # 获取弧的圆心ID
                center_id = arc["center_point_id"]
                # 找到圆心的数学坐标
                center_p = next(p for p in original_points if p['id'] == center_id)
                # 转换为像素坐标
                center_px = transformer.math_to_pixel((
                    self._parse_expr(center_p['x']['expr']),
                    self._parse_expr(center_p['y']['expr'])
                ))
                # 转换半径为像素（原始半径是数学单位，需乘缩放因子）
                radius_math = self._parse_expr(arc['radius']['expr'])
                radius_px = radius_math * transformer.scale_factor
                arcs_px.append({
                    'id': arc['id'],
                    'center': center_px,
                    'radius': radius_px
                })
            except (StopIteration, KeyError, ValueError, TypeError) as e:
                logger.warning(f"转换原始弧 {arc.get('id')} 失败：{e}")
        return arcs_px

    # -------------------- 辅助函数：分数匹配 --------------------
    def _calculate_match_score(self, detected: Dict, original: Dict, primitive_type: str) -> float:
        """计算检测基元与原始实体的匹配分数（0~1，分数越高越匹配）"""
        if primitive_type == 'line':
            # 线匹配：方向（50%）、位置（30%）、长度（20%）
            det_dir = detected['params']['direction']
            orig_dir = original['dir']
            # 方向相似度：余弦相似度（绝对值，0~1）
            dir_score = max(0.0, np.dot(det_dir, orig_dir))

            # 位置相似度：线段中点距离 / 最小长度（<1/4则为1，否则衰减）
            det_mid = (
                (detected['params']['start'][0] + detected['params']['end'][0]) / 2,
                (detected['params']['start'][1] + detected['params']['end'][1]) / 2
            )
            orig_mid = (
                (original['start'][0] + original['end'][0]) / 2,
                (original['start'][1] + original['end'][1]) / 2
            )
            mid_dist = np.linalg.norm(np.array(det_mid) - np.array(orig_mid))
            min_length = min(detected['params']['length'], original['length'])
            pos_score = 1.0 if mid_dist < (min_length / 4) else max(0.0, 1 - (mid_dist / min_length))

            # 长度相似度：短长度/长长度（0~1）
            len_score = min(detected['params']['length'], original['length']) / max(detected['params']['length'], original['length'])

            # 加权总分
            return 0.5 * dir_score + 0.3 * pos_score + 0.2 * len_score

        elif primitive_type == 'arc':
            # 弧匹配：半径（60%）、圆心位置（40%）
            det_radius = detected['params']['radius']
            orig_radius = original['radius']
            # 半径相似度：1 - |差异|/原始半径（0~1）
            radius_score = max(0.0, 1 - (abs(det_radius - orig_radius) / orig_radius))

            # 圆心位置相似度：距离 / 原始半径（<1/4则为1，否则衰减）
            center_dist = np.linalg.norm(np.array(detected['params']['center']) - np.array(original['center']))
            center_score = 1.0 if center_dist < (orig_radius / 4) else max(0.0, 1 - (center_dist / orig_radius))

            # 加权总分
            return 0.6 * radius_score + 0.4 * center_score

        return 0.0

    def _match_detected_to_original(
        self, detected_list: List[Dict], original_list: List[Dict], 
        primitive_type: str, threshold: float
    ) -> List[str]:
        """为原始实体匹配最佳检测基元（分数>阈值才有效）"""
        matched_ids = []
        if not detected_list or not original_list:
            return matched_ids

        # 记录已匹配的检测基元索引（避免重复匹配）
        used_detected_indices = set()

        for original in original_list:
            best_score = -1.0
            best_detected_idx = -1

            # 遍历所有检测基元，找分数最高的
            for idx, detected in enumerate(detected_list):
                if idx in used_detected_indices:
                    continue  # 跳过已匹配的基元

                score = self._calculate_match_score(detected, original, primitive_type)
                if score > best_score:
                    best_score = score
                    best_detected_idx = idx

            # 分数高于阈值，视为匹配成功
            if best_score >= threshold:
                matched_ids.append(original['id'])
                used_detected_indices.add(best_detected_idx)

        logger.info(f"{primitive_type}匹配成功：{len(matched_ids)}/{len(original_list)}")
        return matched_ids

    # -------------------- 辅助函数：结果处理 --------------------
    @staticmethod
    def _extract_matched_point_ids(
        matched_line_ids: List[str], matched_arc_ids: List[str],
        original_lines: List[Dict], original_arcs: List[Dict]
    ) -> Set[str]:
        """从匹配到的线和弧中提取关联的点ID（线的端点、弧的圆心）"""
        matched_point_ids = set()

        # 提取线的端点ID
        for line_id in matched_line_ids:
            line = next((l for l in original_lines if l['id'] == line_id), None)
            if line:
                matched_point_ids.add(line['start_point_id'])
                matched_point_ids.add(line['end_point_id'])

        # 提取弧的圆心ID
        for arc_id in matched_arc_ids:
            arc = next((a for a in original_arcs if a['id'] == arc_id), None)
            if arc and 'center_point_id' in arc:
                matched_point_ids.add(arc['center_point_id'])

        return matched_point_ids

    def _determine_arc_concavity_simple(
        self, detected_arc: Dict, original_arc: Dict, 
        mask: np.ndarray, transformer: CoordTransformer
    ) -> str:
        """简化的弧凹凸性判断：通过测试点是否在mask内"""
        if not detected_arc or not original_arc:
            return "unknown"

        # 取圆弧片段的中点（代表圆弧的中间位置）
        segment = detected_arc['params']['segment']
        mid_idx = len(segment) // 2
        mid_px = tuple(segment[mid_idx].astype(int))
        # 圆弧圆心
        center_px = tuple(np.array(detected_arc['params']['center']).astype(int))

        # 计算垂直于半径的测试方向（逆时针旋转90度）
        radius_vec = (mid_px[0] - center_px[0], mid_px[1] - center_px[1])
        if np.linalg.norm(radius_vec) < 1e-3:  # 半径为0，无法判断
            return "unknown"
        # 单位化半径向量
        radius_unit = (radius_vec[0] / np.linalg.norm(radius_vec), radius_vec[1] / np.linalg.norm(radius_vec))
        # 垂直方向向量（逆时针旋转90度：(x,y)→(-y,x)）
        perp_unit = (-radius_unit[1], radius_unit[0])

        # 测试点：在垂直方向上距离中点5个像素（避免边界）
        test_px = (
            int(mid_px[0] + perp_unit[0] * 5),
            int(mid_px[1] + perp_unit[1] * 5)
        )

        # 测试点在mask内→凹陷（inward），否则→突出（outward）
        if 0 <= test_px[0] < mask.shape[1] and 0 <= test_px[1] < mask.shape[0]:
            return "inward" if mask[test_px[1], test_px[0]] > 0 else "outward"
        return "unknown"

    def _format_result(
        self, 
        corner_matched_point_ids: Set[str],
        matched_point_ids: Set[str], matched_line_ids: List[str],
        matched_arc_ids: List[str], detected_arcs: List[Dict],
        original_arcs_px: List[Dict], mask: np.ndarray, transformer: CoordTransformer
    ) -> Dict:
        """格式化最终匹配结果"""
        # 格式化角点直接匹配得到的点
        corner_matched_points = [{"id": p_id} for p_id in corner_matched_point_ids]
        
        # 格式化点结果
        points_result = [{"id": p_id} for p_id in matched_point_ids]
        # 格式化线结果
        lines_result = [{"id": line_id} for line_id in matched_line_ids]
        # 格式化弧结果（含凹凸性）
        arcs_result = []
        for arc_id in matched_arc_ids:
            # 找到匹配的检测弧和原始弧
            original_arc = next((a for a in original_arcs_px if a['id'] == arc_id), None)
            detected_arc = next((
                d for d in detected_arcs 
                if self._calculate_match_score(d, original_arc, 'arc') > self.arc_match_threshold
            ), None) if original_arc else None
            # 计算凹凸性
            concavity = self._determine_arc_concavity_simple(
                detected_arc, original_arc, mask, transformer
            ) if detected_arc and original_arc else "unknown"
            arcs_result.append({
                "id": arc_id,
                "concavity": concavity
            })

        return {
            "points_from_corners": corner_matched_points,
            "points": points_result,
            "lines": lines_result,
            "arcs": arcs_result
        }

    def _process_shaded_image(self, img_bgr: np.ndarray, data: Dict, idx: int, attempt: int,
                             geom_params: Dict, raw_save_path: str, annotated_raw_path: str,
                             transformer: CoordTransformer) -> None:
        """处理单张图像的阴影生成与实体匹配"""
        # 1. 提取原始图像的区域（无阴影，确保轮廓清晰）
        meta = self._get_meta(data)
        masks, _, _, _, polygons = self.region_extractor.extract(img_bgr, meta)
        if not masks or not polygons:
            logger.warning(f"第{idx}条数据的第{attempt}次尝试无有效区域，跳过")
            return

        # 2. 选择阴影区域（按面积排序，排除已选）
        exclude_labels = self.selected_region_labels.get(idx, [])
        n_regions = self.config["shader"]["n_regions"]
        select_n = np.random.randint(n_regions[0], n_regions[1]+1)
        selected_regions = self._select_regions(masks, polygons, select_n, exclude_labels)
        if not selected_regions:
            logger.warning(f"第{idx}条数据的第{attempt}次尝试无可用未选区域，跳过")
            return

        # 3. 记录选中区域标签
        current_selected_labels = [region["label"] for region in selected_regions]
        if idx not in self.selected_region_labels:
            self.selected_region_labels[idx] = []
        self.selected_region_labels[idx].extend(current_selected_labels)

        # 4. 匹配实体（调用新的核心匹配方法）
        shadow_entities = []
        original_lines = data.get("lines", [])
        original_arcs = data.get("arcs", [])
        original_points = data.get("points", [])

        for region in selected_regions:
            # 从区域掩码中提取原始轮廓
            raw_contour, _ = self._find_contours_from_mask(region['mask'])
            if len(raw_contour) == 0:
                logger.warning(f"区域 {region['label']} 无有效轮廓，跳过匹配")
                continue
            
            # 调用核心匹配逻辑
            entity_data = self._match_primitives_to_contour(
                raw_contour=raw_contour,
                original_lines=original_lines,
                original_arcs=original_arcs,
                original_points=original_points,
                transformer=transformer
            )

            # 构建阴影实体信息（与原有格式一致）
            shadow_entity = {
                "type": "shadow",
                "region_label": region["label"],
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

        # 5. 应用阴影（对选中区域添加阴影效果）
        shaded_img = img_bgr.copy()
        shadow_type = np.random.choice(self.config["shader"]["shadow_types"])
        shader = SHADERS[shadow_type]
        shader_params = self._get_shader_params(shadow_type)

        for region in selected_regions:
            shaded_img = shader.apply(shaded_img, region["mask"],** shader_params)
            
        # 6. 保存结果（阴影图像、标注、JSON）
        raw_filename = os.path.basename(raw_save_path)
        shaded_filename = f"shaded_{idx}_attempt_{attempt}_{raw_filename}"
        shaded_path = os.path.join(self.shaded_dir, shaded_filename)
        cv2.imwrite(shaded_path, shaded_img)
        
        # 生成标注后的阴影图像
        annotated_shaded_img = self.annotator.draw_annotations(shaded_img, data, transformer)
        annotated_shaded_filename = f"annotated_shaded_{idx}_attempt_{attempt}_{raw_filename}"
        annotated_shaded_path = os.path.join(self.annotated_dir, annotated_shaded_filename)
        cv2.imwrite(annotated_shaded_path, annotated_shaded_img)
        
        logger.info(f"成功标注第{idx}条数据第{attempt}次尝试的阴影图像")

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
        """从JSON数据中提取元信息（几何中心）"""
        points = data.get("points", [])
        if not points:
            return {}
        math_pts = []
        for pt in points:
            try:
                x = float(pt["x"]["expr"])
                y = float(pt["y"]["expr"])
                math_pts.append((x, y))
            except (KeyError, ValueError, TypeError):
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
        """选择指定数量的未选区域（按面积排序）"""
        available_regions = []
        for mask, polygon in zip(masks, polygons):
            region_label = polygon.get("label")
            if region_label is not None and region_label not in exclude_labels:
                # 确保区域字典包含 mask、polygon、label 三个关键字
                available_regions.append({
                    "mask": mask,
                    "polygon": polygon,
                    "label": region_label
                })
        
        if not available_regions:
            logger.warning(f"无可用未选区域，返回空列表")
            return []
        
        # 选择不超过可用数量的区域
        select_n = min(n, len(available_regions))
        # 按面积降序排序（优先选择大面积区域，面积字段从 polygon 中获取）
        available_regions_sorted = sorted(
            available_regions,
            key=lambda x: x["polygon"].get("pixel", {}).get("area_px_est", 0),
            reverse=True
        )
        return available_regions_sorted[:select_n]

    def _get_shader_params(self, shadow_type: str) -> Dict:
        """生成阴影参数（随机化强度等）"""
        intensity = np.random.uniform(*self.config["shader"]["intensity_range"])
        if shadow_type == "hatch":
            return {"spacing": self.config["shader"]["hatch_spacing"], "intensity": intensity}
        elif shadow_type == "crosshatch":
            return {
                "spacing": self.config["shader"]["crosshatch_spacing"],
                "angle1": 45, 
                "angle2": 135,
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
        # 默认参数
        return {"intensity": intensity}

    # ------------------------------
    # 新增的核心匹配逻辑函数（与 _process_shaded_image 对齐）
    # ------------------------------
    def _find_contours_from_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """从区域掩码中提取主轮廓（确保轮廓为 (N, 2) 格式）"""
        # 处理掩码格式：确保为单通道灰度图
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # 二值化：确保掩码只有 0（背景）和 255（前景）
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # 提取轮廓：只保留最外层轮廓，避免内部噪声
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("从掩码中未提取到任何轮廓")
            return np.array([]), []
        
        # 选择最长的轮廓作为主轮廓（默认区域的核心轮廓）
        main_contour = max(contours, key=cv2.contourArea)
        # 转换为 (N, 2) 格式（方便后续坐标计算）
        main_contour = main_contour.reshape(-1, 2)
        logger.info(f"提取主轮廓：共 {len(main_contour)} 个像素点")
        return main_contour, contours

    def _create_distance_map(self, raw_contour: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """为轮廓创建距离图（快速查询任意点到轮廓的最近距离）"""
        if len(raw_contour) == 0:
            logger.error("轮廓为空，无法创建距离图")
            return np.array([]), (0, 0)
        
        # 1. 获取轮廓的边界框（确定距离图的尺寸，留出 10 像素边距避免采样点超出）
        min_x, min_y = np.min(raw_contour, axis=0).astype(int)
        max_x, max_y = np.max(raw_contour, axis=0).astype(int)
        width = max_x - min_x + 20  # 左右各留 10 像素
        height = max_y - min_y + 20  # 上下各留 10 像素
        offset = (min_x - 10, min_y - 10)  # 距离图的坐标偏移（全局坐标 → 局部坐标）
        
        # 2. 创建轮廓掩码（仅轮廓区域为白色，其余为黑色）
        mask = np.zeros((height, width), dtype=np.uint8)
        # 将轮廓坐标转换为距离图的局部坐标
        translated_contour = raw_contour - np.array(offset)
        cv2.drawContours(mask, [translated_contour.astype(int)], -1, 255, thickness=1)
        
        # 3. 计算距离图（DIST_L2 = 欧氏距离，精度最高）
        # 注意：distanceTransform 输入要求「背景为 255，前景为 0」，因此反转掩码
        distance_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
        logger.info(f"创建距离图：尺寸 {width}x{height}，轮廓边界框 [{min_x}, {min_y}] 到 [{max_x}, {max_y}]")
        return distance_map, offset

    @staticmethod
    def _bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """用 Bresenham 算法生成直线上的整数像素采样点"""
        x0, y0 = start
        x1, y1 = end
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def _sample_arc_points(
        self,
        center_px: Tuple[float, float],
        radius_px: float,
        start_px: Tuple[float, float],
        end_px: Tuple[float, float],
        is_complete: bool = False
    ) -> List[Tuple[int, int]]:
        """生成圆弧上的均匀像素采样点（支持完整圆，并兼容短弧）"""
        center = (int(round(center_px[0])), int(round(center_px[1])))
        radius = int(round(radius_px))
        if radius <= 0:
            logger.error("圆弧半径必须大于 0，无法采样")
            return []
        
        # 1. 计算圆弧的起止角度（弧度制）
        if is_complete:
            start_angle = 0.0
            end_angle = 2 * math.pi
            angle_range = end_angle - start_angle
            logger.debug(f"完整圆采样：圆心 {center}，半径 {radius}")
        else:
            start_angle = math.atan2(start_px[1] - center_px[1], start_px[0] - center_px[0])
            end_angle = math.atan2(end_px[1] - center_px[1], end_px[0] - center_px[0])
            if end_angle < start_angle:
                end_angle += 2 * math.pi
            angle_range = end_angle - start_angle
            logger.debug(f"圆弧采样：圆心 {center}，半径 {radius}，角度范围 {angle_range:.2f} 弧度")

        # 2. 【新增逻辑】动态计算采样点数
        # 估算圆弧的像素长度
        arc_length_px = radius * angle_range
        
        # 如果弧长很短，使用一个最小的固定采样点数（例如 5），以确保覆盖整个弧
        min_samples_for_short_arc = 5
        if arc_length_px < self.min_sample_points:
            num_points = min_samples_for_short_arc
            logger.debug(f"圆弧过短（估算长度 {arc_length_px:.2f}px < {self.min_sample_points}px），使用最小采样点数: {num_points}")
        else:
            # 对于正常长度的弧，按原规则采样
            num_points = max(self.min_sample_points, int(round(arc_length_px / 5)))
            
        # 3. 均匀采样圆弧上的点
        points = []
        for i in range(num_points + 1): # +1 确保起点和终点都被包含
            angle = start_angle + (angle_range) * (i / num_points)
            x = int(round(center[0] + radius * math.cos(angle)))
            y = int(round(center[1] + radius * math.sin(angle)))
            points.append((x, y))
        
        logger.debug(f"圆弧采样完成：共 {len(points)} 个采样点")
        return points

    def _parse_point_coords(
        self, 
        point_id: str, 
        original_points: List[Dict], 
        transformer: CoordTransformer
    ) -> Tuple[float, float]:
        """解析点 ID 对应的像素坐标（数学坐标 → 像素坐标）"""
        try:
            # 1. 查找对应的点
            point = next(p for p in original_points if p.get("id") == point_id)
            # 2. 解析点的数学坐标
            x_math = self._parse_expr(point["x"]["expr"])
            y_math = self._parse_expr(point["y"]["expr"])
            # 3. 转换为像素坐标
            x_px, y_px = transformer.math_to_pixel((x_math, y_math))
            # logger.debug(f"解析点 {point_id}：数学坐标 ({x_math:.2f}, {y_math:.2f}) → 像素坐标 ({x_px:.2f}, {y_px:.2f})")
            return (x_px, y_px)
        except StopIteration:
            logger.error(f"未找到点 ID：{point_id}")
            raise
        except KeyError as e:
            logger.error(f"点 {point_id} 缺少必要字段：{str(e)}")
            raise
        except Exception as e:
            logger.error(f"解析点 {point_id} 坐标失败：{str(e)}")
            raise

    def _calculate_match_score(
        self,
        sample_points: List[Tuple[int, int]],
        distance_map: np.ndarray,
        offset: Tuple[int, int]
    ) -> Tuple[float, int, int]:
        """计算采样点的匹配度（有效点占比）"""
        if not sample_points:
            logger.warning("采样点列表为空，匹配度为 0")
            return 0.0, 0, 0
        if distance_map.size == 0:
            logger.error("距离图为空，无法计算匹配度")
            raise ValueError("距离图为空")
        
        total_points = len(sample_points)
        valid_count = 0
        offset_x, offset_y = offset
        map_height, map_width = distance_map.shape

        # 遍历所有采样点，统计有效点（距离 ≤ 阈值）
        for idx, (x_px, y_px) in enumerate(sample_points):
            # 1. 将采样点的全局坐标转换为距离图的局部坐标
            local_x = x_px - offset_x
            local_y = y_px - offset_y
            
            # 2. 检查局部坐标是否在距离图范围内（避免越界）
            if 0 <= local_x < map_width and 0 <= local_y < map_height:
                # 3. 获取该点到轮廓的最近距离
                min_distance = distance_map[int(local_y), int(local_x)]
                # 4. 判断是否为有效点
                if min_distance <= self.distance_threshold:
                    valid_count += 1
                
                # 打印前 5 个点的详细信息（方便调试）
                # if idx < 5:
                #     logger.debug(f"采样点 {idx+1}：全局坐标 ({x_px}, {y_px}) → 局部坐标 ({local_x:.0f}, {local_y:.0f})，最近距离 {min_distance:.2f}")

        # 计算匹配度（避免除零错误）
        match_score = valid_count / total_points if total_points > 0 else 0.0
        # logger.info(f"匹配度计算完成：总采样点 {total_points}，有效点 {valid_count}，匹配度 {match_score:.2%}（阈值：{self.match_threshold:.2%}）")
        return match_score, valid_count, total_points

    def _match_lines(
        self,
        original_lines: List[Dict],
        original_points: List[Dict],
        transformer: CoordTransformer,
        distance_map: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[str]:
        """匹配所有待匹配直线，返回匹配成功的直线 ID 列表"""
        matched_ids = []
        total_lines = len(original_lines)
        if total_lines == 0:
            logger.info("无待匹配的直线")
            return matched_ids
        
        logger.info(f"\n===== 开始匹配直线（共 {total_lines} 条）=====")
        for line in original_lines:
            line_id = line.get("id", f"UnknownLine_{id(line)}")
            logger.info(f"\n--- 匹配直线：{line_id} ---")
            
            try:
                # 1. 解析直线的起止点像素坐标
                start_id = line["start_point_id"]
                end_id = line["end_point_id"]
                start_px = self._parse_point_coords(start_id, original_points, transformer)
                end_px = self._parse_point_coords(end_id, original_points, transformer)
                
                # 2. 生成直线的像素采样点
                sample_points = self._bresenham_line(
                    start=(int(round(start_px[0])), int(round(start_px[1]))),
                    end=(int(round(end_px[0])), int(round(end_px[1])))
                )
                
                # 【新增逻辑】兼容短直线：如果采样点数量少于最小采样点数，则直接使用全部点
                if len(sample_points) == 0:
                    logger.warning(f"直线 {line_id} 无法生成采样点，跳过")
                    continue
                
                # logger.debug(f"直线采样完成：共 {len(sample_points)} 个采样点")

                # 3. 计算匹配度
                match_score, _, _ = self._calculate_match_score(
                    sample_points=sample_points,
                    distance_map=distance_map,
                    offset=offset
                )
                
                # 4. 判断是否匹配成功
                if match_score >= self.match_threshold:
                    matched_ids.append(line_id)
                    logger.info(f"直线 {line_id} 匹配成功！匹配度 {match_score:.2%} ≥ 阈值 {self.match_threshold:.2%}")
                else:
                    logger.info(f"直线 {line_id} 匹配失败！匹配度 {match_score:.2%} < 阈值 {self.match_threshold:.2%}")
            
            except Exception as e:
                logger.error(f"直线 {line_id} 匹配过程出错：{str(e)}，跳过")
                continue
        
        logger.info(f"\n直线匹配完成：共 {len(matched_ids)} 条匹配成功，ID 列表：{matched_ids}")
        return matched_ids

    def _match_arcs(
        self,
        original_arcs: List[Dict],
        original_points: List[Dict],
        transformer: CoordTransformer,
        distance_map: np.ndarray,
        offset: Tuple[int, int]
    ) -> List[str]:
        """匹配所有待匹配圆弧，返回匹配成功的圆弧 ID 列表"""
        matched_ids = []
        total_arcs = len(original_arcs)
        if total_arcs == 0:
            logger.info("无待匹配的圆弧")
            return matched_ids
        
        logger.info(f"\n===== 开始匹配圆弧（共 {total_arcs} 条）=====")
        for arc in original_arcs:
            arc_id = arc.get("id", f"UnknownArc_{id(arc)}")
            logger.info(f"\n--- 匹配圆弧：{arc_id} ---")
            
            try:
                # 1. 解析圆弧的关键参数
                center_id = arc["center_point_id"]
                start_id = arc["start_point_id"]
                end_id = arc["end_point_id"]
                radius_expr = arc["radius"]["expr"]
                is_complete = arc.get("is_complete", False)
                
                # 2. 解析像素坐标和半径
                center_px = self._parse_point_coords(center_id, original_points, transformer)
                start_px = self._parse_point_coords(start_id, original_points, transformer)
                end_px = self._parse_point_coords(end_id, original_points, transformer)
                radius_math = self._parse_expr(radius_expr)
                radius_px = radius_math * transformer.scale_factor  # 数学单位 → 像素单位
                
                # 3. 生成圆弧的像素采样点
                sample_points = self._sample_arc_points(
                    center_px=center_px,
                    radius_px=radius_px,
                    start_px=start_px,
                    end_px=end_px,
                    is_complete=is_complete
                )
                if len(sample_points) < self.min_sample_points:
                    logger.warning(f"圆弧 {arc_id} 采样点过少（{len(sample_points)} 个 < 最小 {self.min_sample_points} 个），跳过")
                    continue
                
                # 4. 计算匹配度
                match_score, _, _ = self._calculate_match_score(
                    sample_points=sample_points,
                    distance_map=distance_map,
                    offset=offset
                )
                
                # 5. 判断是否匹配成功
                if match_score >= self.match_threshold:
                    matched_ids.append(arc_id)
                    logger.info(f"圆弧 {arc_id} 匹配成功！匹配度 {match_score:.2%} ≥ 阈值 {self.match_threshold:.2%}")
                else:
                    logger.info(f"圆弧 {arc_id} 匹配失败！匹配度 {match_score:.2%} < 阈值 {self.match_threshold:.2%}")
            
            except Exception as e:
                logger.error(f"圆弧 {arc_id} 匹配过程出错：{str(e)}，跳过")
                continue
        
        logger.info(f"\n圆弧匹配完成：共 {len(matched_ids)} 条匹配成功，ID 列表：{matched_ids}")
        return matched_ids

    def _extract_matched_point_ids(
        self,
        matched_line_ids: List[str],
        matched_arc_ids: List[str],
        original_lines: List[Dict],
        original_arcs: List[Dict]
    ) -> List[str]:
        """从匹配成功的线/弧中提取关联的点 ID（去重）"""
        matched_point_ids = set()  # 用集合自动去重
        
        # 1. 提取匹配直线关联的点
        for line_id in matched_line_ids:
            try:
                line = next(l for l in original_lines if l.get("id") == line_id)
                matched_point_ids.add(line["start_point_id"])
                matched_point_ids.add(line["end_point_id"])
            except StopIteration:
                logger.warning(f"未找到匹配的直线 {line_id}，跳过其关联点提取")
        
        # 2. 提取匹配圆弧关联的点
        for arc_id in matched_arc_ids:
            try:
                arc = next(a for a in original_arcs if a.get("id") == arc_id)
                matched_point_ids.add(arc["center_point_id"])
                matched_point_ids.add(arc["start_point_id"])
                matched_point_ids.add(arc["end_point_id"])
            except StopIteration:
                logger.warning(f"未找到匹配的圆弧 {arc_id}，跳过其关联点提取")
        
        # 转换为列表并排序（保持一致性）
        matched_point_list = sorted(list(matched_point_ids))
        logger.info(f"\n提取匹配关联点：共 {len(matched_point_list)} 个，ID 列表：{matched_point_list}")
        return matched_point_list

    def _match_primitives_to_contour(
        self,
        raw_contour: np.ndarray,
        original_lines: List[Dict],
        original_arcs: List[Dict],
        original_points: List[Dict],
        transformer: CoordTransformer
    ) -> Dict:
        """核心匹配函数：判断待匹配线/弧是否为轮廓的一部分，返回匹配结果"""
        logger.info("=" * 80)
        logger.info("开始基元匹配流程")
        logger.info("=" * 80)
        
        try:
            # 1. 准备工作：创建距离图
            distance_map, offset = self._create_distance_map(raw_contour)
            if distance_map.size == 0:
                raise ValueError("距离图创建失败，无法继续匹配")
            
            # 2. 分别匹配直线和圆弧
            matched_lines = self._match_lines(
                original_lines=original_lines,
                original_points=original_points,
                transformer=transformer,
                distance_map=distance_map,
                offset=offset
            )
            matched_arcs = self._match_arcs(
                original_arcs=original_arcs,
                original_points=original_points,
                transformer=transformer,
                distance_map=distance_map,
                offset=offset
            )
            
            # 3. 提取关联的点
            matched_points = self._extract_matched_point_ids(
                matched_line_ids=matched_lines,
                matched_arc_ids=matched_arcs,
                original_lines=original_lines,
                original_arcs=original_arcs
            )
            
            # 4. 整理结果（与原有格式一致）
            result = {
                "points": [{"id": p_id} for p_id in matched_points],
                "lines": [{"id": l_id} for l_id in matched_lines],
                "arcs": [{"id": a_id} for a_id in matched_arcs]
            }
            
            logger.info("\n" + "=" * 80)
            logger.info("基元匹配流程结束")
            logger.info(f"最终结果：点 {len(result['points'])} 个，线 {len(result['lines'])} 条，弧 {len(result['arcs'])} 条")
            logger.info("=" * 80)
            return result
        
        except Exception as e:
            logger.error(f"基元匹配流程出错：{str(e)}，返回空结果")
            return {"points": [], "lines": [], "arcs": []}


    # def _process_shaded_image(self, img_bgr: np.ndarray, data: Dict, idx: int, attempt: int,
    #                          geom_params: Dict, raw_save_path: str, annotated_raw_path: str,
    #                          transformer: CoordTransformer) -> None:
    #     """处理单张图像的阴影生成与实体匹配"""
    #     # 1. 提取原始图像的区域（无阴影，确保轮廓清晰）
    #     meta = self._get_meta(data)
    #     masks, _, _, _, polygons = self.region_extractor.extract(img_bgr, meta)
    #     if not masks or not polygons:
    #         logger.warning(f"第{idx}条数据的第{attempt}次尝试无有效区域，跳过")
    #         return

    #     # 2. 选择阴影区域（按面积排序，排除已选）
    #     exclude_labels = self.selected_region_labels[idx]
    #     n_regions = self.config["shader"]["n_regions"]
    #     select_n = np.random.randint(n_regions[0], n_regions[1]+1)
    #     selected_regions = self._select_regions(masks, polygons, select_n, exclude_labels)
    #     if not selected_regions:
    #         logger.warning(f"第{idx}次尝试无可用未选区域，跳过")
    #         return

    #     # 3. 记录选中区域标签
    #     current_selected_labels = [region["label"] for region in selected_regions]
    #     self.selected_region_labels[idx].extend(current_selected_labels)

    #     # 4. 匹配实体（调用新的核心匹配方法）
    #     shadow_entities = []
    #     original_lines = data.get("lines", [])
    #     original_arcs = data.get("arcs", [])
    #     original_points = data.get("points", [])

    #     for region in selected_regions:
    #         # 新逻辑：无需传递pixel_contours，直接从region['mask']提取轮廓
    #         # entity_data = self._process_region_entities(
    #         #     region, original_points, original_lines, original_arcs, transformer
    #         # )
            
    #         raw_contour, _ = self._find_contours_from_mask(region['mask'])
            
    #         # 调用新的匹配逻辑
    #         entity_data = self._match_primitives_to_contour(
    #             raw_contour, original_lines, original_arcs, original_points, transformer
    #         )

    #         # 构建阴影实体信息
    #         # 构建阴影实体信息
    #         shadow_entity = {
    #             "type": "shadow",
    #             "region_label": region["label"],
    #             "points": entity_data["points"],
    #             "lines": entity_data["lines"],
    #             "arcs": entity_data["arcs"],
    #             "stats": {
    #                 "total_points": len(entity_data["points"]),
    #                 "total_lines": len(entity_data["lines"]),
    #                 "total_arcs": len(entity_data["arcs"])
    #             }
    #         }
    #         shadow_entities.append(shadow_entity)

    #     # 5. 应用阴影（对选中区域添加阴影效果）
    #     shaded_img = img_bgr.copy()
    #     shadow_type = np.random.choice(self.config["shader"]["shadow_types"])
    #     shader = SHADERS[shadow_type]
    #     shader_params = self._get_shader_params(shadow_type)

    #     for region in selected_regions:
    #         shaded_img = shader.apply(shaded_img, region["mask"],** shader_params)
            
    #     # 6. 保存结果（阴影图像、标注、JSON）
    #     raw_filename = os.path.basename(raw_save_path)
    #     shaded_filename = f"shaded_{idx}_attempt_{attempt}_{raw_filename}"
    #     shaded_path = os.path.join(self.shaded_dir, shaded_filename)
    #     cv2.imwrite(shaded_path, shaded_img)
        
    #     # 生成标注后的阴影图像
    #     annotated_shaded_img = self.annotator.draw_annotations(shaded_img, data, transformer)
    #     annotated_shaded_filename = f"annotated_shaded_{idx}_attempt_{attempt}_{raw_filename}"
    #     annotated_shaded_path = os.path.join(self.annotated_dir, annotated_shaded_filename)
    #     cv2.imwrite(annotated_shaded_path, annotated_shaded_img)
        
    #     logger.info(f"成功标注第{idx}张第{attempt}次阴影图像")

    #     # 更新并保存结果数据
    #     new_data = data.copy()
    #     new_data["entities"] = new_data.get("entities", []) + shadow_entities
    #     new_data.update({
    #         "raw_path": raw_save_path,
    #         "annotated_raw_path": annotated_raw_path,
    #         "shaded_path": shaded_path,
    #         "annotated_shaded_path": annotated_shaded_path,
    #         "shadow_type": shadow_type,
    #         "shader_enabled": True
    #     })
    #     self.processed_results.append(new_data)
    #     self.shaded_path = shaded_path

    #     logger.info(f"第{idx}条数据第{attempt}次尝试完成，阴影图像路径：{shaded_path}")

    # def _get_meta(self, data: Dict) -> Dict:
    #     """从JSON数据中提取元信息（几何中心）"""
    #     points = data.get("points", [])
    #     if not points:
    #         return {}
    #     math_pts = []
    #     for pt in points:
    #         try:
    #             x = float(pt["x"]["expr"])
    #             y = float(pt["y"]["expr"])
    #             math_pts.append((x, y))
    #         except (KeyError, ValueError):
    #             continue
    #     if not math_pts:
    #         return {}
    #     center_math = (
    #         sum(p[0] for p in math_pts) / len(math_pts),
    #         sum(p[1] for p in math_pts) / len(math_pts)
    #     )
    #     return {"center_math": center_math}

    # def _select_regions(
    #     self, 
    #     masks: List[np.ndarray], 
    #     polygons: List[Dict], 
    #     n: int, 
    #     exclude_labels: List[int]
    # ) -> List[Dict]:
    #     """选择指定数量的未选区域（按面积排序）"""
    #     available_regions = []
    #     for mask, polygon in zip(masks, polygons):
    #         region_label = polygon.get("label")
    #         if region_label is not None and region_label not in exclude_labels:
    #             available_regions.append({"mask": mask, "polygon": polygon, "label": region_label})
        
    #     if not available_regions:
    #         logger.warning(f"无可用未选区域，返回空列表")
    #         return []
        
    #     select_n = min(n, len(available_regions))
    #     # 按面积降序排序（优先选择大面积区域）
    #     available_regions_sorted = sorted(
    #         available_regions,
    #         key=lambda x: x["polygon"].get("pixel", {}).get("area_px_est", 0),
    #         reverse=True
    #     )
    #     return available_regions_sorted[:select_n]

    # def _get_shader_params(self, shadow_type: str) -> Dict:
    #     """生成阴影参数（随机化强度等）"""
    #     intensity = np.random.uniform(*self.config["shader"]["intensity_range"])
    #     if shadow_type == "hatch":
    #         return {"spacing": self.config["shader"]["hatch_spacing"], "intensity": intensity}
    #     elif shadow_type == "crosshatch":
    #         return {
    #             "spacing": self.config["shader"]["crosshatch_spacing"],
    #             "angle1": 45, "angle2": 135,
    #             "intensity": intensity
    #         }
    #     elif shadow_type == "solid":
    #         return {"color": (128, 128, 128), "intensity": intensity}
    #     elif shadow_type == "gradient":
    #         return {
    #             "start_color": (200, 200, 200),
    #             "end_color": (100, 100, 100),
    #             "angle_deg": 45,
    #             "intensity": intensity
    #         }
    #     return {"intensity": intensity}

    def process(self) -> None:
        """处理所有增强JSON数据，生成阴影图像和标注"""
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
            
            # 处理阴影（多次尝试）
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
                    logger.error(f"第{idx}条数据的第{attempt}次阴影尝试失败: {str(e)}", exc_info=True)
                    continue

        # 保存所有处理结果到JSONL
        self._save_jsonl()
        logger.info(f"=== 所有图像处理完成 ===")
        logger.info(f"处理结果已保存至：{self.shaded_jsonl_path}")
        logger.info(f"原始图像目录：{self.raw_dir}")
        logger.info(f"阴影图像目录：{self.shaded_dir}")
        logger.info(f"标注图像目录：{self.annotated_dir}")