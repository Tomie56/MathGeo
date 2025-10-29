import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import sympy as sp
from sympy import symbols, simplify, sqrt, pi, N, expand
import timeout_decorator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GeometryCalculator')

class GeometryCalculator:
    """几何参数计算器，基于表达式精确计算线条和圆弧长度"""
    PROCESS_TIMEOUT = 30 
    def __init__(self):
        self.PROCESS_TIMEOUT = 30 
        self.EXPR_LENGTH_THRESHOLD = 150
        # 预定义sympy符号和常用函数
        self.math_functions = {
            'sqrt': sqrt,
            'pi': pi,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'log': sp.log
        }
        # 确保sympy使用精确符号计算
        sp.init_printing(use_unicode=True)
    
    def _preprocess_expression(self, expr_str: str) -> str:
        """预处理表达式：去除空格、统一格式"""
        # 去除所有空格（避免"sqrt(3) / 2"等带空格的表达式解析错误）
        expr_str = expr_str.replace(' ', '')
        # 替换中文符号为英文（如果有）
        expr_str = expr_str.replace('×', '*').replace('÷', '/')
        # 处理负号表达式（如"-sqrt(3)/3"）
        expr_str = expr_str.replace('+-', '-').replace('-+', '-')
        return expr_str
    
    def _parse_expression(self, expr_str: str) -> Any:
        """解析数学表达式并返回sympy表达式（严格基于输入表达式）"""
        try:
            # 预处理表达式
            processed_expr = self._preprocess_expression(expr_str)
            # 使用sympy的parse_expr更安全地解析表达式（替代eval）
            return sp.parse_expr(
                processed_expr,
                local_dict=self.math_functions,
                evaluate=False  # 不立即求值，保留原始表达式结构
            )
        except Exception as e:
            logger.error(f"解析表达式失败: 原始表达式='{expr_str}', 处理后='{processed_expr}', 错误: {str(e)}")
            raise
    
    def _get_point_coords(self, points: List[Dict[str, Any]], point_id: str) -> Tuple[Any, Any]:
        """获取点的坐标表达式（严格基于point.x.expr和point.y.expr）"""
        for point in points:
            if point['id'] == point_id:
                try:
                    x_expr = self._parse_expression(point['x']['expr'])
                    y_expr = self._parse_expression(point['y']['expr'])
                    return (x_expr, y_expr)
                except KeyError as e:
                    raise ValueError(f"点 {point_id} 缺少坐标信息: {e}")
        raise ValueError(f"未找到点: {point_id}")
    
    def _calculate_distance(self, p1: Tuple[Any, Any], p2: Tuple[Any, Any]) -> Any:
        """计算两点之间的距离（基于坐标表达式，保留精确符号）"""
        x1, y1 = p1
        x2, y2 = p2
        
        # 严格按照距离公式计算：√[(x2-x1)² + (y2-y1)²]
        dx = x2 - x1
        dy = y2 - y1
        distance_sq = dx**2 + dy**2
        distance = sqrt(distance_sq)
        
        # 先展开再化简，确保复杂表达式正确处理
        return simplify(expand(distance))
    
    def _calculate_line_length(self, line: Dict[str, Any], points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算线段长度（完全基于端点坐标表达式）"""
        try:
            line_id = line.get('id', 'unknown_line')
            start_id = line['start_point_id']
            end_id = line['end_point_id']
            
            # 获取端点坐标表达式
            start_coords = self._get_point_coords(points, start_id)
            end_coords = self._get_point_coords(points, end_id)
            
            # 计算长度表达式
            length_expr = self._calculate_distance(start_coords, end_coords)
            
            # 计算数值结果（基于符号表达式求值，保留8位小数）
            try:
                length_value = float(N(length_expr, 8))  # N()用于符号转数值
            except Exception as e:
                logger.warning(f"线段 {line_id} 无法计算数值结果: {str(e)}")
                length_value = None
            
            return {
                'expr': str(length_expr),          # 符号表达式字符串
                'latex': sp.latex(length_expr),    # LaTeX格式
                'value': round(length_value, 8) if length_value is not None else None  # 数值结果
            }
        except Exception as e:
            logger.error(f"计算线段 {line.get('id')} 长度失败: {str(e)}")
            return {
                'expr': 'unknown',
                'latex': 'unknown',
                'value': None
            }
    
    def _calculate_arc_length(self, arc: Dict[str, Any], points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算圆弧长度（基于半径、角度和完整性标识）"""
        try:
            arc_id = arc.get('id', 'unknown_arc')
            
            # 解析半径表达式（严格基于输入）
            radius_expr = self._parse_expression(arc['radius']['expr'])
            
            # 1. 优先处理完整圆弧（is_complete: true）
            if arc.get('is_complete', False):
                # 完整圆的弧长 = 2πr
                angle_diff = 2 * pi
                length_expr = simplify(radius_expr * angle_diff)
                logger.debug(f"完整圆弧 {arc_id}：半径={radius_expr}，弧长={length_expr}")
            
            # 2. 非完整圆弧：从角度信息计算
            else:
                # 解析角度范围（优先 start_angle/end_angle，其次 angle 字段）
                if 'start_angle' in arc and 'end_angle' in arc:
                    start_angle = self._parse_expression(arc['start_angle']['expr'])
                    end_angle = self._parse_expression(arc['end_angle']['expr'])
                    angle_diff = end_angle - start_angle
                elif 'angle' in arc:
                    # angle 字段直接表示圆心角（弧度）
                    angle_diff = self._parse_expression(arc['angle']['expr'])
                else:
                    raise ValueError(f"圆弧 {arc_id} 缺少角度信息（需 start_angle/end_angle 或 angle）")
                
                # 确保角度差在 [0, 2π] 范围内（处理负角度或超范围角度）
                angle_diff = angle_diff % (2 * pi)
                # 弧长公式：半径 × 圆心角（弧度）
                length_expr = simplify(radius_expr * angle_diff)
                logger.debug(f"非完整圆弧 {arc_id}：半径={radius_expr}，圆心角={angle_diff}，弧长={length_expr}")
            
            # 计算数值结果（保留8位小数）
            try:
                length_value = float(N(length_expr, 8))  # N() 用于符号转数值
            except Exception as e:
                logger.warning(f"圆弧 {arc_id} 无法计算数值结果: {str(e)}")
                length_value = None
            
            return {
                'expr': str(length_expr),
                'latex': sp.latex(length_expr),
                'value': round(length_value, 8) if length_value is not None else None
            }
        except Exception as e:
            logger.error(f"计算圆弧 {arc.get('id')} 长度失败: {str(e)}")
            return {
                'expr': 'unknown',
                'latex': 'unknown',
                'value': None
            }
    
    def calculate(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
            """计算所有线条和圆弧的长度（新增：检查表达式长度）"""
            # 深拷贝避免修改原始数据
            result = json.loads(json.dumps(json_data))
            points = result.get('points', [])

            # 辅助函数：根据ID找点
            def get_point(point_id: str) -> Optional[Dict[str, Any]]:
                return next((p for p in points if p.get('id') == point_id), None)
            
            # 辅助函数：检查表达式长度（可选项：先化简再检查）
            def check_expr_length(expr: Any, desc: str) -> None:
                # 1. 处理None值
                if expr is None:
                    return
                # 2. 非字符串类型（如dict、int、float）转为字符串
                if not isinstance(expr, str):
                    # 字典转JSON字符串，其他类型直接转字符串
                    if isinstance(expr, dict):
                        expr_str = json.dumps(expr, ensure_ascii=False)
                    else:
                        expr_str = str(expr)
                else:
                    expr_str = expr
                # 3. 清洗表达式（去空格、换行）
                simplified = expr_str.replace(" ", "").replace("\n", "")
                # 4. 检查长度
                if len(simplified) > self.EXPR_LENGTH_THRESHOLD:
                    raise ValueError(f"表达式过于复杂（{desc}，长度={len(simplified)}）：{simplified[:50]}...")


            # 处理线条：先检查点坐标表达式长度
            if 'lines' in result:
                for line in result['lines']:
                    start_id = line.get('start_point_id')
                    end_id = line.get('end_point_id')
                    start_point = get_point(start_id)
                    end_point = get_point(end_id)

                    # 检查起点坐标
                    if start_point:
                        check_expr_length(start_point.get('x'), f"线条起点({start_id})的x坐标")
                        check_expr_length(start_point.get('y'), f"线条起点({start_id})的y坐标")
                    # 检查终点坐标
                    if end_point:
                        check_expr_length(end_point.get('x'), f"线条终点({end_id})的x坐标")
                        check_expr_length(end_point.get('y'), f"线条终点({end_id})的y坐标")

                    # 原有计算逻辑
                    line['length'] = self._calculate_line_length(line, points)

            # 处理圆弧：先检查点坐标表达式长度
            if 'arcs' in result:
                for arc in result['arcs']:
                    center_id = arc.get('center_point_id')
                    start_id = arc.get('start_point_id')
                    end_id = arc.get('end_point_id')
                    center_point = get_point(center_id)
                    start_point = get_point(start_id)
                    end_point = get_point(end_id)

                    # 检查圆心坐标
                    if center_point:
                        check_expr_length(center_point.get('x'), f"圆弧圆心({center_id})的x坐标")
                        check_expr_length(center_point.get('y'), f"圆弧圆心({center_id})的y坐标")
                    # 检查圆弧起点/终点坐标
                    if start_point:
                        check_expr_length(start_point.get('x'), f"圆弧起点({start_id})的x坐标")
                        check_expr_length(start_point.get('y'), f"圆弧起点({start_id})的y坐标")
                    if end_point:
                        check_expr_length(end_point.get('x'), f"圆弧终点({end_id})的x坐标")
                        check_expr_length(end_point.get('y'), f"圆弧终点({end_id})的y坐标")

                    # 原有计算逻辑
                    arc['length'] = self._calculate_arc_length(arc, points)

            return result
    
    def calculate_single(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算单条几何数据中所有线条和圆弧的长度（完全基于输入表达式，单条数据专用接口）
        Args:
            json_data: 单条几何数据JSON字典，需包含 'points'、'lines'（可选）、'arcs'（可选）字段
        Returns:
            处理后的JSON字典，线条和圆弧新增 'length' 字段（含表达式、LaTeX格式和数值结果）
        """
        try:
            # 验证输入数据结构
            if not isinstance(json_data, dict):
                raise ValueError("输入必须是字典类型的单条几何数据")
            if 'points' not in json_data:
                raise ValueError("单条几何数据缺少 'points' 字段，无法计算长度")
            
            logger.debug(f"开始处理单条几何数据（ID: {json_data.get('id', 'unknown')}）")
            
            # 复用现有calculate方法的核心逻辑（单条数据处理）
            result = self.calculate(json_data)
            
            logger.debug(f"单条几何数据处理完成（ID: {json_data.get('id', 'unknown')}）")
            return result
        
        except Exception as e:
            logger.error(f"单条几何数据计算失败: {str(e)}")
            # 失败时返回原始数据，附加错误信息
            error_data = json_data.copy()
            error_data['calculation_error'] = str(e)
            return error_data
    
    @timeout_decorator.timeout(PROCESS_TIMEOUT, timeout_exception=TimeoutError)
    def _calculate_with_timeout(self, data):
        """带超时控制的计算方法（内部调用原有calculate）"""
        return self.calculate(data)

    def process_jsonl(self, input_path: str, output_path: str) -> None:
        if not os.path.exists(input_path):
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
                    processed_data = self.calculate(data)
                    json.dump(processed_data, f_out, ensure_ascii=False)
                    f_out.write('\n')
                    
                    if line_num % 10 == 0:
                        logger.info(f"已处理 {line_num} 条数据")
                except ValueError as ve:
                    if "表达式过于复杂" in str(ve):
                        logger.error(f"第 {line_num} 行 {str(ve)}，跳过")
                    else:
                        logger.error(f"第 {line_num} 行数据错误: {str(ve)}，跳过")
                    continue
                except Exception as e:
                    logger.error(f"第 {line_num} 行处理失败: {str(e)}，跳过")
                    continue
        
        logger.info(f"处理完成，结果保存至: {output_path}")