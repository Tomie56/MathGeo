import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import sympy as sp
from sympy import symbols, simplify, sqrt, pi, expand, sign, Rational, atan
import timeout_decorator

# 配置日志（与主程序日志兼容）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GeometryCalculator')

class GeometryCalculator:
    """几何参数计算器，修复弓形面积凹凸性判断逻辑，精准控制加减"""
    PROCESS_TIMEOUT = 30 

    def __init__(self):
        self.PROCESS_TIMEOUT = 30 
        self.EXPR_LENGTH_THRESHOLD = 150
        self.math_functions = {
            'sqrt': sqrt,
            'pi': pi,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'log': sp.log,
            'Rational': Rational,
            'atan': atan 
        }
    
    def _preprocess_expression(self, expr_str: str) -> str:
        """预处理表达式：统一格式，不额外修改符号"""
        expr_str = expr_str.replace(' ', '')
        expr_str = expr_str.replace('×', '*').replace('÷', '/')
        expr_str = expr_str.replace('+-', '-').replace('-+', '-')
        return expr_str
    
    def _parse_expression(self, expr_str: str) -> Optional[Any]:
        """解析数学表达式（用sympify，支持atan/pi等符号）"""
        try:
            if not expr_str:
                logger.warning("空表达式，解析失败")
                return None
            processed_expr = self._preprocess_expression(expr_str)
            return sp.sympify(processed_expr, locals=self.math_functions)
        except Exception as e:
            logger.warning(f"解析表达式失败: 原始表达式='{expr_str}', 处理后='{processed_expr}', 错误: {str(e)}")
            return None
    
    def _get_point_coords(self, points: List[Dict[str, Any]], point_id: str) -> Optional[Tuple[Any, Any]]:
        """获取点的坐标表达式（不强制Rational转换）"""
        for point in points:
            if point.get('id') == point_id:
                x_expr_str = point.get('x', {}).get('expr')
                y_expr_str = point.get('y', {}).get('expr')
                if not x_expr_str or not y_expr_str:
                    logger.warning(f"点 {point_id} 缺少x/y坐标表达式")
                    return None
                x_expr = self._parse_expression(x_expr_str)
                y_expr = self._parse_expression(y_expr_str)
                if x_expr is None or y_expr is None:
                    logger.warning(f"点 {point_id} 坐标表达式解析失败")
                    return None
                return (x_expr, y_expr)
        logger.warning(f"未找到点: {point_id}")
        return None
    
    def _calculate_distance(self, p1: Tuple[Any, Any], p2: Tuple[Any, Any]) -> Optional[Any]:
        """计算两点距离（用sp.simplify）"""
        x1, y1 = p1
        x2, y2 = p2
        try:
            dx = x2 - x1
            dy = y2 - y1
            distance_sq = dx**2 + dy**2
            distance = sqrt(distance_sq)
            return simplify(expand(distance))
        except Exception as e:
            logger.warning(f"两点距离计算失败: {str(e)}")
            return None
    
    def _calculate_line_length(self, line: Dict[str, Any], points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算线段长度（核心用sp.simplify）"""
        try:
            line_id = line.get('id', 'unknown_line')
            start_id = line.get('start_point_id')
            end_id = line.get('end_point_id')
            if not start_id or not end_id:
                logger.warning(f"线段 {line_id} 缺少起止点ID")
                return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
            
            start_coords = self._get_point_coords(points, start_id)
            end_coords = self._get_point_coords(points, end_id)
            if not start_coords or not end_coords:
                logger.warning(f"线段 {line_id} 起止点坐标无效")
                return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
            
            length_expr = self._calculate_distance(start_coords, end_coords)
            if length_expr is None:
                return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
            
            try:
                length_value = float(sp.N(length_expr, 8))
            except Exception as e:
                logger.warning(f"线段 {line_id} 长度表达式转数值失败: {str(e)}")
                length_value = None
            
            return {
                'expr': str(length_expr),
                'latex': sp.latex(length_expr),
                'value': round(length_value, 8) if length_value is not None else None
            }
        except Exception as e:
            logger.error(f"计算线段 {line.get('id')} 长度失败: {str(e)}")
            return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
    
    def _calculate_arc_length(self, arc: Dict[str, Any], points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算圆弧长度（核心用sp.simplify）"""
        try:
            arc_id = arc.get('id', 'unknown_arc')
            radius_expr_str = arc.get('radius', {}).get('expr')
            if not radius_expr_str:
                logger.warning(f"圆弧 {arc_id} 缺少半径表达式")
                return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
            radius_expr = self._parse_expression(radius_expr_str)
            if radius_expr is None:
                return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
            
            # 计算圆心角表达式
            if arc.get('is_complete', False):
                angle_diff = 2 * pi
                logger.debug(f"完整圆弧 {arc_id}：半径={radius_expr}")
            else:
                if 'angle' in arc:
                    angle_str = arc['angle'].get('expr')
                    if not angle_str:
                        logger.warning(f"圆弧 {arc_id} 缺少angle表达式")
                        return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
                    angle_diff = self._parse_expression(angle_str)
                    if angle_diff is None:
                        return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
                else:
                    logger.warning(f"圆弧 {arc_id} 缺少角度信息")
                    return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
            
            # 计算弧长
            length_expr = simplify(radius_expr * angle_diff)
            try:
                length_value = float(sp.N(length_expr, 8))
            except Exception as e:
                logger.warning(f"圆弧 {arc_id} 长度表达式转数值失败: {str(e)}")
                length_value = None
            
            return {
                'expr': str(length_expr),
                'latex': sp.latex(length_expr),
                'value': round(length_value, 8) if length_value is not None else None
            }
        except Exception as e:
            logger.error(f"计算圆弧 {arc.get('id')} 长度失败: {str(e)}")
            return {'expr': 'unknown', 'latex': 'unknown', 'value': None}
    
    def _symbolic_shoelace_formula(self, point_coords: List[Tuple[Any, Any]]) -> Optional[Any]:
        """符号版鞋带公式"""
        n = len(point_coords)
        if n < 3:
            logger.warning(f"多边形点数不足（需≥3），当前点数：{n}")
            return None
        
        area_expr = 0
        for i in range(n):
            x_i, y_i = point_coords[i]
            x_j, y_j = point_coords[(i+1) % n]
            area_expr += (x_i * y_j) - (x_j * y_i)
        
        return simplify(abs(area_expr) * Rational(1, 2))
    
    def _judge_loop_winding_direction_symbolic(self, point_coords: List[Tuple[Any, Any]]) -> int:
        """判断环的环绕方向（1=逆时针，-1=顺时针）"""
        n = len(point_coords)
        if n < 3:
            return 1  # 点数不足时默认逆时针
        
        cross_sum = 0
        for i in range(n):
            x_i, y_i = point_coords[i]
            x_j, y_j = point_coords[(i+1) % n]
            cross_sum += (x_j - x_i) * (y_j + y_i)
        
        simplified_sum = simplify(cross_sum)
        return 1 if sp.sign(simplified_sum) > 0 else -1
    
    def _calculate_arc_segment_area_symbolic(self, 
                                            arc: Dict[str, Any], 
                                            points: List[Dict[str, Any]],
                                            loop_direction: int) -> Optional[Any]:
        """符号版弓形面积（核心修复：精准判断弧相对于环的凹凸性，控制加减）"""
        try:
            arc_id = arc.get('id', 'unknown_arc')
            # 1. 获取关键信息
            center_id = arc.get('center_point_id')
            start_id = arc.get('start_point_id')
            end_id = arc.get('end_point_id')
            radius_expr_str = arc.get('radius', {}).get('expr')
            angle_expr_str = arc.get('angle', {}).get('expr')
            start_angle_expr_str = arc.get('start_angle', {}).get('expr')
            end_angle_expr_str = arc.get('end_angle', {}).get('expr')
            
            # 2. 校验必填字段
            required_ids = [center_id, start_id, end_id]
            if not all(required_ids) or not radius_expr_str:
                logger.warning(f"圆弧 {arc_id} 缺少关键信息（圆心ID/起点ID/终点ID/半径expr）")
                return None
            
            # 3. 解析关键表达式
            radius_expr = self._parse_expression(radius_expr_str)
            center_coords = self._get_point_coords(points, center_id)
            start_coords = self._get_point_coords(points, start_id)
            end_coords = self._get_point_coords(points, end_id)
            if None in [radius_expr, center_coords, start_coords, end_coords]:
                logger.warning(f"圆弧 {arc_id} 关键表达式解析失败")
                return None
            cx, cy = center_coords
            sx, sy = start_coords
            ex, ey = end_coords
            
            # 4. 解析圆心角表达式（确保为正）
            theta_expr = None
            if angle_expr_str:
                theta_expr = self._parse_expression(angle_expr_str)
                theta_expr = simplify(abs(theta_expr))  # 强制圆心角为正
            elif start_angle_expr_str and end_angle_expr_str:
                start_angle = self._parse_expression(start_angle_expr_str)
                end_angle = self._parse_expression(end_angle_expr_str)
                if start_angle and end_angle:
                    theta_expr = simplify(abs(end_angle - start_angle))  # 强制圆心角为正
            if theta_expr is None:
                logger.warning(f"圆弧 {arc_id} 缺少有效角度expr")
                return None
            
            # 5. 计算扇形面积和三角形面积
            sector_area_expr = simplify(Rational(1, 2) * radius_expr**2 * theta_expr)
            triangle_area_expr = simplify(
                Rational(1, 2) * abs((sx - cx) * (ey - cy) - (ex - cx) * (sy - cy))
            )
            # 弓形面积（扇形 - 三角形，恒为正，后续根据凹凸性调整符号）
            segment_area_expr = simplify(sector_area_expr - triangle_area_expr)
            
            # 6. 核心修复：判断弧相对于环的凹凸性（关键逻辑）
            # 向量1：环的前进方向（弧起点→弧终点）
            vec_se = (ex - sx, ey - sy)
            # 向量2：弧起点→圆心
            vec_sc = (cx - sx, cy - sy)
            # 叉乘：判断圆心在前进方向的左侧（正）/右侧（负）
            cross_expr = simplify(vec_se[0] * vec_sc[1] - vec_se[1] * vec_sc[0])
            cross_sign = sp.sign(cross_expr)
            
            # 7. 结合环方向，判断凹凸性并控制弓形面积的加减
            # 逻辑规则：
            # - 环逆时针（loop_direction=1）：
            #   圆心在左侧（cross_sign>0）→ 弧凸向环外 → 面积需减去弓形
            #   圆心在右侧（cross_sign<0）→ 弧凸向环内 → 面积需加上弓形
            # - 环顺时针（loop_direction=-1）：
            #   圆心在左侧（cross_sign>0）→ 弧凸向环内 → 面积需加上弓形
            #   圆心在右侧（cross_sign<0）→ 弧凸向环外 → 面积需减去弓形
            if loop_direction == 1:  # 逆时针环
                if cross_sign > 0:  # 凸向环外 → 减弓形
                    final_area = simplify(-segment_area_expr)
                    logger.debug(f"圆弧 {arc_id}：逆时针环+凸向环外 → 弓形面积（减）={str(final_area)}")
                else:  # 凸向环内 → 加弓形
                    final_area = segment_area_expr
                    logger.debug(f"圆弧 {arc_id}：逆时针环+凸向环内 → 弓形面积（加）={str(final_area)}")
            else:  # 顺时针环
                if cross_sign > 0:  # 凸向环内 → 加弓形
                    final_area = segment_area_expr
                    logger.debug(f"圆弧 {arc_id}：顺时针环+凸向环内 → 弓形面积（加）={str(final_area)}")
                else:  # 凸向环外 → 减弓形
                    final_area = simplify(-segment_area_expr)
                    logger.debug(f"圆弧 {arc_id}：顺时针环+凸向环外 → 弓形面积（减）={str(final_area)}")
            
            return final_area
        
        except Exception as e:
            logger.error(f"计算圆弧 {arc_id} 弓形面积失败: {str(e)}")
            return None
    
    def _calculate_shadow_perimeter_symbolic(self, 
                                           shadow_entity: Dict[str, Any],
                                           lines: List[Dict[str, Any]],
                                           arcs: List[Dict[str, Any]]) -> Optional[Any]:
        """符号版周长（累加线/弧长度）"""
        perimeter_expr = 0
        # 累加线段长度
        line_ids = {l['id'] for l in shadow_entity.get('lines', [])}
        for line in lines:
            line_id = line.get('id')
            if line_id not in line_ids:
                continue
            line_length_expr = self._parse_expression(line['length'].get('expr'))
            if line_length_expr and line['length']['expr'] != 'unknown':
                perimeter_expr += line_length_expr
        
        # 累加圆弧长度
        arc_ids = {a['id'] for a in shadow_entity.get('arcs', [])}
        for arc in arcs:
            arc_id = arc.get('id')
            if arc_id not in arc_ids:
                continue
            arc_length_expr = self._parse_expression(arc['length'].get('expr'))
            if arc_length_expr and arc['length']['expr'] != 'unknown':
                perimeter_expr += arc_length_expr
        
        return simplify(perimeter_expr) if perimeter_expr != 0 else None
    
    def _calculate_shadow_area_symbolic(self, 
                                      shadow_entity: Dict[str, Any],
                                      points: List[Dict[str, Any]],
                                      lines: List[Dict[str, Any]],
                                      arcs: List[Dict[str, Any]]) -> Optional[Any]:
        """符号版面积（多边形面积±弓形面积，基于凹凸性判断）"""
        try:
            # 仅处理单个环
            ordered_loops = shadow_entity.get('ordered_loops', [])
            if len(ordered_loops) != 1:
                logger.warning(f"shadow entity（region_label={shadow_entity.get('region_label')}）仅支持单个环，当前环数：{len(ordered_loops)}")
                return None
            
            ordered_point_ids = ordered_loops[0].get('ordered_points', [])
            if len(ordered_point_ids) < 2:
                logger.warning(f"环的点数不足（需≥2），当前点数：{len(ordered_point_ids)}")
                return None
            
            # 1. 获取环的有序点坐标
            point_coords = []
            for p_id in ordered_point_ids:
                coords = self._get_point_coords(points, p_id)
                if not coords:
                    logger.warning(f"环中点 {p_id} 坐标无效")
                    return None
                point_coords.append(coords)
            
            # 2. 判断环的环绕方向
            loop_direction = self._judge_loop_winding_direction_symbolic(point_coords)
            logger.debug(f"shadow entity（region_label={shadow_entity.get('region_label')}）：环绕方向={'逆时针' if loop_direction==1 else '顺时针'}")
            
            # 3. 计算面积
            has_arcs = len(shadow_entity.get('arcs', [])) > 0
            if not has_arcs:
                polygon_area = self._symbolic_shoelace_formula(point_coords)
                logger.debug(f"shadow entity（region_label={shadow_entity.get('region_label')}）：纯多边形面积={str(polygon_area)}")
                return polygon_area
            else:
                # 多边形面积（鞋带公式）
                polygon_area_expr = self._symbolic_shoelace_formula(point_coords)
                if polygon_area_expr is None:
                    return None
                
                # 累加所有弓形面积（已根据凹凸性处理正负）
                total_segment_area_expr = 0
                arc_ids = {a['id'] for a in shadow_entity['arcs']}
                for arc in arcs:
                    if arc.get('id') in arc_ids:
                        seg_area_expr = self._calculate_arc_segment_area_symbolic(arc, points, loop_direction)
                        if seg_area_expr is not None:
                            total_segment_area_expr += seg_area_expr
                
                # 总面=多边形面积+弓形面积（弓形面积已含加减符号）
                total_area_expr = simplify(polygon_area_expr + total_segment_area_expr)
                logger.debug(f"shadow entity（region_label={shadow_entity.get('region_label')}）：多边形面积={str(polygon_area_expr)}，总弓形面积={str(total_segment_area_expr)}，总面积={str(total_area_expr)}")
                return total_area_expr
        
        except Exception as e:
            logger.error(f"计算shadow entity（region_label={shadow_entity.get('region_label')}）面积失败: {str(e)}")
            return None
    
    def _process_shadow_entities(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """处理shadow entity，添加周长和面积字段"""
        points = result.get('points', [])
        lines = result.get('lines', [])
        arcs = result.get('arcs', [])
        entities = result.get('entities', [])
        count = 0
        
        for entity in entities:
            if entity.get('type') == 'shadow' and entity.get('validity'):
                perimeter_expr = self._calculate_shadow_perimeter_symbolic(entity, lines, arcs)
                area_expr = self._calculate_shadow_area_symbolic(entity, points, lines, arcs)
                
                entity['perimeter'] = {
                    'expr': str(perimeter_expr) if perimeter_expr is not None else 'unknown',
                    'latex': sp.latex(perimeter_expr) if perimeter_expr is not None else 'unknown'
                }
                entity['area'] = {
                    'expr': str(area_expr) if area_expr is not None else 'unknown',
                    'latex': sp.latex(area_expr) if area_expr is not None else 'unknown'
                }

                count += 1
                
                # logger.info(f"shadow entity（region_label={entity.get('region_label')}）："
                #              f"周长表达式={str(perimeter_expr) if perimeter_expr else 'unknown'}, "
                #              f"面积表达式={str(area_expr) if area_expr else 'unknown'}")
        
        result["shadow_regions"] = count

        return result
    
    def calculate(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算所有几何参数（核心用sp.simplify）"""
        # 深拷贝避免修改原始数据
        result = json.loads(json.dumps(json_data))
        points = result.get('points', [])

        # 辅助函数：根据ID找点
        def get_point(point_id: str) -> Optional[Dict[str, Any]]:
            return next((p for p in points if p.get('id') == point_id), None)
        
        # 辅助函数：检查表达式长度
        def check_expr_length(expr: Any, desc: str) -> None:
            if expr is None:
                return
            if not isinstance(expr, str):
                if isinstance(expr, dict):
                    expr_str = json.dumps(expr, ensure_ascii=False)
                else:
                    expr_str = str(expr)
            else:
                expr_str = expr
            simplified = expr_str.replace(" ", "").replace("\n", "")
            if len(simplified) > self.EXPR_LENGTH_THRESHOLD:
                raise ValueError(f"表达式过于复杂（{desc}，长度={len(simplified)}）：{simplified[:50]}...")

        # 处理线条
        if 'lines' in result:
            for line in result['lines']:
                start_id = line.get('start_point_id')
                end_id = line.get('end_point_id')
                start_point = get_point(start_id)
                end_point = get_point(end_id)

                if start_point:
                    check_expr_length(start_point.get('x'), f"线条起点({start_id})的x坐标")
                    check_expr_length(start_point.get('y'), f"线条起点({start_id})的y坐标")
                if end_point:
                    check_expr_length(end_point.get('x'), f"线条终点({end_id})的x坐标")
                    check_expr_length(end_point.get('y'), f"线条终点({end_id})的y坐标")

                line['length'] = self._calculate_line_length(line, points)

        # 处理圆弧
        if 'arcs' in result:
            for arc in result['arcs']:
                center_id = arc.get('center_point_id')
                start_id = arc.get('start_point_id')
                end_id = arc.get('end_point_id')
                center_point = get_point(center_id)
                start_point = get_point(start_id)
                end_point = get_point(end_id)

                if center_point:
                    check_expr_length(center_point.get('x'), f"圆弧圆心({center_id})的x坐标")
                    check_expr_length(center_point.get('y'), f"圆弧圆心({center_id})的y坐标")
                if start_point:
                    check_expr_length(start_point.get('x'), f"圆弧起点({start_id})的x坐标")
                    check_expr_length(start_point.get('y'), f"圆弧起点({start_id})的y坐标")
                if end_point:
                    check_expr_length(end_point.get('x'), f"圆弧终点({end_id})的x坐标")
                    check_expr_length(end_point.get('y'), f"圆弧终点({end_id})的y坐标")

                arc['length'] = self._calculate_arc_length(arc, points)

        # 处理shadow entity
        result = self._process_shadow_entities(result)

        return result
    
    def calculate_single(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """单条几何数据专用接口"""
        try:
            if not isinstance(json_data, dict):
                raise ValueError("输入必须是字典类型的单条几何数据")
            if 'points' not in json_data:
                raise ValueError("单条几何数据缺少 'points' 字段，无法计算长度")
            
            logger.debug(f"开始处理单条几何数据（ID: {json_data.get('id', 'unknown')}）")
            result = self.calculate(json_data)
            logger.debug(f"单条几何数据处理完成（ID: {json_data.get('id', 'unknown')}）")
            return result
        
        except Exception as e:
            logger.error(f"单条几何数据计算失败: {str(e)}")
            error_data = json_data.copy()
            error_data['calculation_error'] = str(e)
            return error_data
    
    @timeout_decorator.timeout(PROCESS_TIMEOUT, timeout_exception=TimeoutError)
    def _calculate_with_timeout(self, data):
        """带超时控制的计算方法"""
        return self.calculate(data)

    def process_jsonl(self, input_path: str, output_path: str) -> None:
        """批量处理JSONL文件（适配main.py的run_gt调用）"""
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(input_path, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            line_num = 0
            for line in f_in:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    processed_data = self._calculate_with_timeout(data)
                    json.dump(processed_data, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    
                    if line_num % 10 == 0:
                        logger.info(f"已处理 {line_num} 条数据）")
                except TimeoutError:
                    logger.error(f"第 {line_num} 行数据处理超时（超过{self.PROCESS_TIMEOUT}秒），跳过")
                    continue
                except ValueError as ve:
                    if "表达式过于复杂" in str(ve):
                        logger.error(f"第 {line_num} 行 {str(ve)}，跳过")
                    else:
                        logger.error(f"第 {line_num} 行数据格式错误: {str(ve)}，跳过")
                    continue
                except Exception as e:
                    logger.error(f"第 {line_num} 行数据处理失败: {str(e)}，跳过", exc_info=True)
                    continue
        
        logger.info(f"JSONL文件处理完成！结果保存至: {output_path}")