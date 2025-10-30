import random
import logging
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp
from sympy import simplify, acos, pi, sqrt, cos

logger = logging.getLogger('QAGenerator')

class QAGenerator:
    """几何问题生成器，支持基于level的元素选择和综合难度计算"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 长度问题模板
        self.length_templates = [
            "What is the length of {edge_name}?",
            "Calculate the length of {edge_name}.",
            "Determine the length of {edge_name} in the figure.",
            "Find the length of {edge_name}.",
            "What is the measurement of {edge_name}?"
        ]
        
        # 角度问题模板
        self.angle_templates = [
            "What is the measure of angle {angle_name} in radians?",
            "Express the angle at point {vertex_id} between {edge1_name} and {edge2_name} as a radian expression.",
            "Determine the radian measure of the angle formed by {edge1_name} and {edge2_name} at {vertex_id}.",
            "Find the radian expression for angle {angle_name}.",
            "What is the radian value of the angle between {edge1_name} and {edge2_name} at vertex {vertex_id}?"
        ]
        
        # 面积问题模板
        self.area_templates = [
            "What is the area of the figure?",
            "Calculate the area of the shape."
        ]
        
        # 表达式复杂度分段阈值（支持后续修改）
        self.EXPR_COMPLEX_THRESHOLD1 = 15  # 简单表达式阈值
        self.EXPR_COMPLEX_THRESHOLD2 = 30  # 中等表达式阈值
        self.EXPR_COMPLEX_THRESHOLD3 = 100  # 中等表达式阈值
        
        # 复杂度权重分配（支持后续修改）
        self.GRAPH_WEIGHT = 0.2    # 图复杂度权重
        self.QUESTION_WEIGHT = 0.5  # 问题复杂度权重
        self.ANSWER_WEIGHT = 0.3   # 答案复杂度权重

    # ------------------------------ 核心辅助方法（基于level的复杂度计算） ------------------------------
    def _get_element_level(self, element: Dict[str, Any], element_type: str) -> int:
        """获取元素的level值，默认为1"""
        if element_type == 'point':
            return element.get('level', 1)
        elif element_type in ['line', 'edge']:
            return element.get('level', 1)
        return 1

    def _calculate_graph_complexity(self, geo_data: Dict[str, Any]) -> float:
        """计算图的复杂度：基于点、边的数量和平均level"""
        entities_count = len(geo_data.get('entities', []))
        points = geo_data.get('points', [])
        lines = geo_data.get('lines', [])
        arcs = geo_data.get('arcs', [])
        
        # 基础复杂度：元素数量
        points_count = len(points)
        edges_count = len(lines) + len(arcs)
        count_complexity = points_count * 0.5 + edges_count * 0.5
        
        # 等级复杂度：平均level值
        if points_count > 0:
            avg_point_level = sum(self._get_element_level(p, 'point') for p in points) / points_count
        else:
            avg_point_level = 1
            
        if edges_count > 0:
            avg_edge_level = (sum(self._get_element_level(l, 'line') for l in lines) + 
                             sum(self._get_element_level(a, 'edge') for a in arcs)) / edges_count
        else:
            avg_edge_level = 1
            
        level_complexity = avg_point_level * 0.3 + avg_edge_level * 0.7
        
        # 综合图复杂度
        return count_complexity * 0.6 + level_complexity * 4.0  # 缩放因子使量级合理

    def _calculate_question_complexity(self, qt_type: str, **kwargs) -> float:
        """计算问题本身的复杂度：基于涉及元素的level"""
        if qt_type == 'length':
            # 长度问题：基于边的level
            edge_level = kwargs.get('edge_level', 1)
            return edge_level * 2.0  # 边level直接影响问题复杂度
            
        elif qt_type == 'angle':
            # 角度问题：基于点和两条边的平均level
            point_level = kwargs.get('point_level', 1)
            edge1_level = kwargs.get('edge1_level', 1)
            edge2_level = kwargs.get('edge2_level', 1)
            avg_level = (point_level + edge1_level + edge2_level) / 3.0
            return avg_level * 2.0  # 平均level影响问题复杂度
            
        elif qt_type == 'area':
            # 面积问题：基于图形复杂度（默认中等）
            return 3.0
        return 1.0

    def _calculate_answer_complexity(self, expr_length: int) -> float:
        """计算答案的复杂度：基于表达式长度的分段计算"""
        if expr_length <= self.EXPR_COMPLEX_THRESHOLD1:
            return 5.0  # 简单
        elif expr_length <= self.EXPR_COMPLEX_THRESHOLD2:
            return 10.0  # 中等
        elif expr_length <= self.EXPR_COMPLEX_THRESHOLD3:
            return 20.0  # 较复杂
        else:
            return 50.0  # 复杂

    def _calculate_overall_difficulty(self, graph_complex: float, question_complex: float, answer_complex: float) -> float:
        """计算总体难度：加权求和"""
        return (graph_complex * self.GRAPH_WEIGHT +
                question_complex * self.QUESTION_WEIGHT +
                answer_complex * self.ANSWER_WEIGHT)

    # ------------------------------ 原有辅助方法（保持或修改） ------------------------------
    def _get_edge_name(self, edge_type: str, edge_data: Dict[str, Any]) -> str:
        if edge_type == 'line':
            start = edge_data['start_point_id']
            end = edge_data['end_point_id']
            return f"Line_{start}_{end}"
        elif edge_type == 'arc':
            if edge_data.get('is_complete', False):
                center = edge_data.get('center_point_id', edge_data['id'])
                return f"Circle_{center}"
            else:
                start = edge_data['start_point_id']
                end = edge_data['end_point_id']
                return f"Arc_{start}_{end}"
        return edge_data['id']

    def _get_point_coords(self, points: List[Dict[str, Any]], point_id: str) -> Tuple[sp.Expr, sp.Expr]:
        for point in points:
            if point['id'] == point_id:
                x_expr = sp.sympify(point['x']['expr'].replace('^', '**'))
                y_expr = sp.sympify(point['y']['expr'].replace('^', '**'))
                return (x_expr, y_expr)
        raise ValueError(f"未找到点 {point_id} 的坐标信息")

    def _get_edge_other_vertex(self, edge_data: Dict[str, Any], vertex_id: str) -> str:
        if edge_data['start_point_id'] == vertex_id:
            return edge_data['end_point_id']
        elif edge_data['end_point_id'] == vertex_id:
            return edge_data['start_point_id']
        else:
            raise ValueError(f"边 {self._get_edge_name('line', edge_data)} 不包含顶点 {vertex_id}")

    def _calculate_vector(self, start_point: Tuple[sp.Expr, sp.Expr], end_point: Tuple[sp.Expr, sp.Expr]) -> Tuple[sp.Expr, sp.Expr]:
        return (end_point[0] - start_point[0], end_point[1] - start_point[1])

    def _is_valid_angle(self, angle_rad: sp.Expr) -> bool:
        is_zero = simplify(angle_rad) == 0
        is_straight = simplify(angle_rad - pi) == 0
        return not (is_zero or is_straight)

    def _is_radian_expression(self, angle_rad: sp.Expr) -> bool:
        simplified = simplify(angle_rad)
        if simplified == 0:
            return False
        pi_free = simplify(simplified / pi)
        return pi_free.is_Rational

    # ------------------------------ 优化：元素选择逻辑（优先高level元素） ------------------------------
    def _select_edge(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择level值高的边"""
        edges = []
        # 处理线段（包含level信息）
        if 'lines' in geo_data:
            for line in geo_data['lines']:
                edge_name = self._get_edge_name('line', line)
                edge_level = self._get_element_level(line, 'line')
                edges.append({
                    'type': 'line',
                    'data': line,
                    'name': edge_name,
                    'level': edge_level
                })
        # 处理圆弧
        if 'arcs' in geo_data:
            for arc in geo_data['arcs']:
                edge_name = self._get_edge_name('arc', arc)
                edge_level = self._get_element_level(arc, 'edge')
                edges.append({
                    'type': 'arc',
                    'data': arc,
                    'name': edge_name,
                    'level': edge_level
                })
        
        if not edges:
            logger.warning("未找到任何线条或圆弧数据")
            return None
        
        # 筛选有长度信息的边
        edges_with_length = [e for e in edges if 'length' in e['data']]
        if not edges_with_length:
            logger.warning("未找到有长度信息的边")
            return None
        
        # 按level降序排序，优先选择高level边
        edges_with_length.sort(key=lambda x: -x['level'])
        
        # 从高level边中随机选择（增加一点随机性）
        # 前30%的高level边有更高概率被选中
        top_percent = int(len(edges_with_length) * 0.3) or 1
        top_candidates = edges_with_length[:top_percent]
        return random.choice(top_candidates)

    def _select_angle_edges(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择包含高level点和边的角度组合"""
        points = geo_data.get('points', [])
        lines = geo_data.get('lines', [])
        line_map = {
            line['id']: {
                'data': line,
                'name': self._get_edge_name('line', line),
                'level': self._get_element_level(line, 'line')
            } for line in lines
        }
        
        valid_combinations = []  # 存储所有有效角度组合
        
        for point in points:
            point_id = point['id']
            point_level = self._get_element_level(point, 'point')
            related_edges = point.get('related_edges', [])
            line_edges = [e for e in related_edges if not e.startswith('Arc') and e in line_map]
            if len(line_edges) < 2:
                continue
            
            from itertools import combinations
            for edge1_id, edge2_id in combinations(line_edges, 2):
                try:
                    edge1 = line_map[edge1_id]
                    edge2 = line_map[edge2_id]
                    edge1_data = edge1['data']
                    edge2_data = edge2['data']
                    
                    # 获取端点坐标
                    vertex1 = self._get_edge_other_vertex(edge1_data, point_id)
                    vertex2 = self._get_edge_other_vertex(edge2_data, point_id)
                    p_coords = self._get_point_coords(points, point_id)
                    v1_coords = self._get_point_coords(points, vertex1)
                    v2_coords = self._get_point_coords(points, vertex2)
                    
                    # 计算角度
                    vec1 = self._calculate_vector(p_coords, v1_coords)
                    vec2 = self._calculate_vector(p_coords, v2_coords)
                    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                    mod1 = sqrt(vec1[0]**2 + vec1[1]** 2)
                    mod2 = sqrt(vec2[0]**2 + vec2[1]** 2)
                    cos_theta = simplify(dot_product / (mod1 * mod2))
                    cos_theta = sp.Piecewise((1, cos_theta > 1), (-1, cos_theta < -1), (cos_theta, True))
                    angle_rad = simplify(acos(cos_theta))
                    
                    if self._is_valid_angle(angle_rad):
                        # 计算角度组合的综合level（点level + 两边level的平均）
                        combo_level = point_level * 0.4 + (edge1['level'] + edge2['level']) * 0.3
                        angle_str = str(angle_rad)
                        valid_combinations.append({
                            'vertex_id': point_id,
                            'vertex_level': point_level,
                            'edge1_id': edge1_id,
                            'edge1_name': edge1['name'],
                            'edge1_level': edge1['level'],
                            'edge2_id': edge2_id,
                            'edge2_name': edge2['name'],
                            'edge2_level': edge2['level'],
                            'vertex1': vertex1,
                            'vertex2': vertex2,
                            'angle_rad': angle_rad,
                            'angle_str': angle_str,
                            'expr_length': len(angle_str),
                            'combo_level': combo_level  # 用于排序的综合level
                        })
                except Exception as e:
                    edge1_name = line_map.get(edge1_id, {}).get('name', edge1_id)
                    edge2_name = line_map.get(edge2_id, {}).get('name', edge2_id)
                    logger.debug(f"计算点 {point_id} 的边 {edge1_name}-{edge2_name} 夹角失败: {str(e)}")
                    continue
        
        if not valid_combinations:
            logger.debug("未找到符合条件的角度")
            return None
        
        # 按综合level降序排序，优先选择高level组合
        valid_combinations.sort(key=lambda x: -x['combo_level'])
        
        # 从高level组合中随机选择
        top_percent = int(len(valid_combinations) * 0.3) or 1
        top_candidates = valid_combinations[:top_percent]
        return random.choice(top_candidates)

    # ------------------------------ 优化：问题生成（计算综合难度） ------------------------------
    def _generate_length_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        selected = self._select_edge(geo_data)
        if not selected:
            return None
            
        edge_type = selected['type']
        edge_data = selected['data']
        edge_name = selected['name']
        edge_level = selected['level']
        
        if 'length' not in edge_data:
            logger.warning(f"{edge_name} 没有长度信息，跳过")
            return None
        
        length_info = edge_data['length']
        # 计算长度表达式长度
        expr_str = length_info['expr']
        expr_length = len(expr_str)
        
        # 生成描述
        qt_description = f"{edge_type.capitalize()} {edge_name} (Level: {edge_level}, Expr Length: {expr_length})"
        if 'description' in edge_data:
            qt_description += f" (Desc: {edge_data['description']})"
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='length',
            edge_level=edge_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        question = random.choice(self.length_templates).format(edge_name=edge_name)
        
        return {
            "question": question,
            "gt": length_info,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff  # 综合难度值
        }

    def _generate_angle_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        angle_data = self._select_angle_edges(geo_data)
        if not angle_data:
            logger.debug("未找到符合条件的角度")
            return None
        
        # 解析角度数据
        vertex_id = angle_data['vertex_id']
        vertex_level = angle_data['vertex_level']
        edge1_name = angle_data['edge1_name']
        edge1_level = angle_data['edge1_level']
        edge2_name = angle_data['edge2_name']
        edge2_level = angle_data['edge2_level']
        vertex1 = angle_data['vertex1']
        vertex2 = angle_data['vertex2']
        angle_rad = angle_data['angle_rad']
        angle_str = angle_data['angle_str']
        expr_length = angle_data['expr_length']
        
        angle_name = f"{vertex1}-{vertex_id}-{vertex2}"
        gt = {
            'expr': angle_str,
            'latex': sp.latex(angle_rad),
            'is_standard_radian': self._is_radian_expression(angle_rad)
        }
        
        # 生成描述
        level_note = f" (Levels: Point={vertex_level}, Edge1={edge1_level}, Edge2={edge2_level})"
        expr_note = f" (Expr Length: {expr_length})"
        qt_description = (
            f"Angle {angle_name} by {edge1_name} & {edge2_name} at {vertex_id}{level_note}{expr_note}"
        )
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(
            qt_type='angle',
            point_level=vertex_level,
            edge1_level=edge1_level,
            edge2_level=edge2_level
        )
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        return {
            "question": random.choice(self.angle_templates).format(
                angle_name=angle_name, vertex_id=vertex_id, edge1_name=edge1_name, edge2_name=edge2_name
            ),
            "gt": gt,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff  # 综合难度值
        }

    def _generate_area_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """面积问题（基于图复杂度计算）"""
        # 面积表达式默认中等长度
        expr_length = 15
        
        # 计算各部分复杂度
        graph_complex = self._calculate_graph_complexity(geo_data)
        question_complex = self._calculate_question_complexity(qt_type='area')
        answer_complex = self._calculate_answer_complexity(expr_length)
        
        # 计算总体难度
        diff = self._calculate_overall_difficulty(graph_complex, question_complex, answer_complex)
        
        return {
            "question": random.choice(self.area_templates),
            "gt": {"expr": "placeholder", "latex": "placeholder", "is_standard_radian": False},
            "qt_description": "Area calculation",
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "graph_complexity": graph_complex,
            "question_complexity": question_complex,
            "answer_complexity": answer_complex,
            "diff": diff  # 综合难度值
        }

    # ------------------------------ 生成入口（保持权重逻辑不变） ------------------------------
    def generate(self, geo_data: Dict[str, Any], num_questions: int, question_types: List[str], type_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        qa_pairs = []
        generated_keys = set()  # 存储已生成问题的唯一标识（用于去重）
        MAX_RETRY = 3  # 单类型问题生成的最大重试次数

        valid_types = [qt for qt in question_types if qt in ['length', 'area', 'angle']]
        if not valid_types:
            logger.warning("未指定有效问题类型，使用默认长度问题")
            valid_types = ['length']
        
        # 处理权重配置，生成目标问题类型列表
        if not type_weights or not isinstance(type_weights, dict):
            target_types = random.choices(valid_types, k=num_questions)
        else:
            weighted_types = []
            weights = []
            for qt in valid_types:
                if qt in type_weights and isinstance(type_weights[qt], (int, float)) and type_weights[qt] > 0:
                    weighted_types.append(qt)
                    weights.append(type_weights[qt])
            if not weighted_types:
                logger.warning("权重配置无效，退化为均等概率")
                target_types = random.choices(valid_types, k=num_questions)
            else:
                total_weight = sum(weights)
                normalized_weights = [w / total_weight for w in weights]
                target_types = random.choices(weighted_types, weights=normalized_weights, k=num_questions)
        
        # 生成问题
        for qt in target_types:
            retry = 0
            while retry < MAX_RETRY:
                try:
                    # 生成对应类型的问题
                    if qt == 'length':
                        qa = self._generate_length_question(geo_data)
                    elif qt == 'angle':
                        qa = self._generate_angle_question(geo_data)
                    elif qt == 'area':
                        qa = self._generate_area_question(geo_data)
                    else:
                        logger.warning(f"不支持的问题类型: {qt}")
                        qa = None

                    if not qa:
                        retry += 1
                        logger.debug(f"第{retry}次重试生成{qt}类型问题（生成失败）")
                        continue

                    # 提取问题的唯一标识
                    if qt == 'length':
                        edge_id = qa['gt'].get('edge_id') or qa['qt_description'].split()[1]
                        unique_key = f"length_{edge_id}"
                    elif qt == 'angle':
                        angle_name = qa['question'].split('angle ')[1].split()[0]
                        unique_key = f"angle_{angle_name}"
                    elif qt == 'area':
                        unique_key = "area"

                    # 检查是否重复
                    if unique_key in generated_keys:
                        retry += 1
                        logger.debug(f"第{retry}次重试生成{qt}类型问题（重复：{unique_key}）")
                        continue

                    # 添加到结果
                    qa_pairs.append(qa)
                    generated_keys.add(unique_key)
                    logger.debug(f"生成{qt}类型问题（唯一标识：{unique_key}，难度：{qa['diff']:.2f}）")
                    break

                except Exception as e:
                    retry += 1
                    logger.error(f"第{retry}次重试生成{qt}类型问题失败: {str(e)}")
                    continue

            if retry >= MAX_RETRY:
                logger.debug(f"生成{qt}类型问题超过最大重试次数（{MAX_RETRY}次），跳过")

        return qa_pairs