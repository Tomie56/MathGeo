import random
import logging
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp
from sympy import simplify, acos, pi, sqrt, cos

logger = logging.getLogger('QAGenerator')

class QAGenerator:
    """几何问题生成器，支持难度等级计算+优先新产生元素（表达式复杂度按长度计算）"""
    
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
        # 表达式长度阈值（可按需调整）
        self.EXPR_COMPLEX_THRESHOLD = 15  # 超过20字符视为复杂表达式

    # ------------------------------ 核心辅助方法（修改表达式复杂度计算） ------------------------------
    def _is_newly_generated_edge(self, edge_id: str) -> bool:
        """判断线段是否为增强操作新产生的（基于ID前缀）"""
        new_edge_prefixes = ['ConnP', 'MidL', 'VMidL', 'PerpL', 'ExtL', 'DiamL']
        return any(edge_id.startswith(prefix) for prefix in new_edge_prefixes)

    def _calculate_graph_complexity(self, geo_data: Dict[str, Any]) -> int:
        """计算图的复杂度（1-3分）：基于实体数和点数"""
        entities_count = len(geo_data.get('entities', []))
        points_count = len(geo_data.get('points', []))
        
        if entities_count <= 3 and points_count <= 5:
            return 1  # 简单
        elif 4 <= entities_count <= 6 and 6 <= points_count <= 10:
            return 2  # 中等
        else:
            return 3  # 复杂

    def _calculate_question_complexity(self, qt_type: str, **kwargs) -> int:
        """计算问题本身的复杂度（1-3分）：新元素+表达式长度"""
        complexity = 1  # 基础分
        
        # 1. 新产生元素加分（核心权重）
        if qt_type == 'length':
            if kwargs.get('is_new_edge', False):
                complexity += 1  # 新边+1分
        elif qt_type == 'angle':
            new_edge_count = sum([kwargs.get('is_new_edge1', False), kwargs.get('is_new_edge2', False)])
            complexity += new_edge_count  # 每条新边+1分（最多+2分）
        
        # 2. 表达式长度加分（超过阈值视为复杂）
        if kwargs.get('expr_length', 0) > self.EXPR_COMPLEX_THRESHOLD:
            complexity += 1  # 长表达式+1分
        
        return min(complexity, 3)  # 限制最大3分

    def _merge_complexity_levels(self, graph_level: int, question_level: int) -> int:
        """合并图复杂度和问题复杂度，得到最终难度等级（1-5分）"""
        total = graph_level * 0.4 + question_level * 0.6  # 图占40%，问题占60%
        level = int(round(total * 2 - 1))  # 线性映射到1-5分
        return max(1, min(level, 5))  # 边界限制

    # ------------------------------ 原有辅助方法（保持不变） ------------------------------
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

    # ------------------------------ 优化：元素选择逻辑（优先新产生元素） ------------------------------
    def _select_edge(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择新产生的、有长度的边"""
        edges = []
        # 处理线段（含是否新边标记）
        if 'lines' in geo_data:
            for line in geo_data['lines']:
                edge_name = self._get_edge_name('line', line)
                edges.append({
                    'type': 'line',
                    'data': line,
                    'name': edge_name,
                    'is_new': self._is_newly_generated_edge(line['id'])
                })
        # 处理圆弧（暂不考虑新产生）
        if 'arcs' in geo_data:
            for arc in geo_data['arcs']:
                edge_name = self._get_edge_name('arc', arc)
                edges.append({
                    'type': 'arc',
                    'data': arc,
                    'name': edge_name,
                    'is_new': False
                })
        
        if not edges:
            logger.warning("未找到任何线条或圆弧数据")
            return None
        
        # 筛选有长度信息的边
        edges_with_length = [e for e in edges if 'length' in e['data']]
        if not edges_with_length:
            logger.warning("未找到有长度信息的边")
            return None
        
        # 优先选择新产生的边（有描述更佳）
        new_edges = [e for e in edges_with_length if e['is_new']]
        if new_edges:
            new_edges_with_desc = [e for e in new_edges if 'description' in e['data']]
            return random.choice(new_edges_with_desc) if new_edges_with_desc else random.choice(new_edges)
        
        # 无新边时选择原始边
        edges_with_desc = [e for e in edges_with_length if 'description' in e['data']]
        return random.choice(edges_with_desc) if edges_with_desc else random.choice(edges_with_length)

    def _select_angle_edges(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """优先选择包含新产生边的角度组合"""
        points = geo_data.get('points', [])
        lines = geo_data.get('lines', [])
        line_map = {
            line['id']: {
                'data': line,
                'name': self._get_edge_name('line', line),
                'is_new': self._is_newly_generated_edge(line['id'])
            } for line in lines
        }
        
        valid_combinations = []  # 存储所有有效角度组合
        
        for point in points:
            point_id = point['id']
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
                        # 记录组合信息（含是否新边、表达式长度）
                        angle_str = str(angle_rad)
                        valid_combinations.append({
                            'vertex_id': point_id,
                            'edge1_id': edge1_id,
                            'edge1_name': edge1['name'],
                            'is_new_edge1': edge1['is_new'],
                            'edge2_id': edge2_id,
                            'edge2_name': edge2['name'],
                            'is_new_edge2': edge2['is_new'],
                            'vertex1': vertex1,
                            'vertex2': vertex2,
                            'angle_rad': angle_rad,
                            'angle_str': angle_str,
                            'expr_length': len(angle_str)  # 计算角度表达式长度
                        })
                except Exception as e:
                    edge1_name = line_map.get(edge1_id, {}).get('name', edge1_id)
                    edge2_name = line_map.get(edge2_id, {}).get('name', edge2_id)
                    logger.debug(f"计算点 {point_id} 的边 {edge1_name}-{edge2_name} 夹角失败: {str(e)}")
                    continue
        
        if not valid_combinations:
            logger.debug("未找到符合条件的角度")
            return None
        
        # 优先选择含新边的组合（排序：两条新边 > 一条新边 > 无新边）
        def sort_key(comb):
            return -(comb['is_new_edge1'] + comb['is_new_edge2'])
        valid_combinations.sort(key=sort_key)
        
        return valid_combinations[0]

    # ------------------------------ 优化：问题生成（按长度计算表达式复杂度） ------------------------------
    def _generate_length_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        selected = self._select_edge(geo_data)
        if not selected:
            return None
            
        edge_type = selected['type']
        edge_data = selected['data']
        edge_name = selected['name']
        is_new_edge = selected['is_new']
        
        if 'length' not in edge_data:
            logger.warning(f"{edge_name} 没有长度信息，跳过")
            return None
        
        length_info = edge_data['length']
        # 计算长度表达式长度（用原始expr字符串）
        expr_str = length_info['expr']
        expr_length = len(expr_str)
        
        # 生成描述
        qt_description = f"{edge_type.capitalize()} {edge_name} (New: {is_new_edge}, Expr Length: {expr_length})"
        if 'description' in edge_data:
            qt_description += f" (Desc: {edge_data['description']})"
        
        # 计算难度等级
        graph_level = self._calculate_graph_complexity(geo_data)
        question_level = self._calculate_question_complexity(
            qt_type='length',
            is_new_edge=is_new_edge,
            expr_length=expr_length
        )
        level = self._merge_complexity_levels(graph_level, question_level)
        
        question = random.choice(self.length_templates).format(edge_name=edge_name)
        
        return {
            "question": question,
            "gt": length_info,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "level": level,  # 新增难度等级
        }

    def _generate_angle_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        angle_data = self._select_angle_edges(geo_data)
        if not angle_data:
            logger.debug("未找到符合条件的角度")
            return None
        
        # 解析角度数据
        vertex_id = angle_data['vertex_id']
        edge1_name = angle_data['edge1_name']
        edge2_name = angle_data['edge2_name']
        is_new_edge1 = angle_data['is_new_edge1']
        is_new_edge2 = angle_data['is_new_edge2']
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
        new_edge_note = f" (New edges: 1={is_new_edge1}, 2={is_new_edge2})"
        expr_note = f" (Expr Length: {expr_length})"
        qt_description = (
            f"Angle {angle_name} by {edge1_name} & {edge2_name} at {vertex_id}{new_edge_note}{expr_note}"
        )
        
        # 计算难度等级
        graph_level = self._calculate_graph_complexity(geo_data)
        question_level = self._calculate_question_complexity(
            qt_type='angle',
            is_new_edge1=is_new_edge1,
            is_new_edge2=is_new_edge2,
            expr_length=expr_length
        )
        level = self._merge_complexity_levels(graph_level, question_level)
        
        return {
            "question": random.choice(self.angle_templates).format(
                angle_name=angle_name, vertex_id=vertex_id, edge1_name=edge1_name, edge2_name=edge2_name
            ),
            "gt": gt,
            "qt_description": qt_description,
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "level": level,  # 新增难度等级
            "expr_length": expr_length  # 可选：输出表达式长度，便于调试
        }

    def _generate_area_question(self, geo_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """面积问题（占位，默认难度3分）"""
        graph_level = self._calculate_graph_complexity(geo_data)
        # 面积表达式默认按中等长度计算
        question_level = self._calculate_question_complexity(qt_type='area', expr_length=15)
        level = self._merge_complexity_levels(graph_level, question_level)
        return {
            "question": random.choice(self.area_templates),
            "gt": {"expr": "placeholder", "latex": "placeholder", "is_standard_radian": False},
            "qt_description": "Area calculation (placeholder)",
            "description": geo_data.get('description', ''),
            "raw_path": geo_data.get('raw_path', ''),
            "annotated_raw_path": geo_data.get('annotated_raw_path', ''),
            "level": level,
            "expr_length": 15  # 占位长度
        }

    # ------------------------------ 生成入口（保持权重逻辑不变） ------------------------------
    def generate(self, geo_data: Dict[str, Any], num_questions: int, question_types: List[str], type_weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        qa_pairs = []
        generated_keys = set()  # 存储已生成问题的唯一标识（用于去重）
        MAX_RETRY = 3  # 单类型问题生成的最大重试次数（避免重复时无限循环）

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
        
        # 生成问题（核心：去重逻辑）
        for qt in target_types:
            retry = 0
            while retry < MAX_RETRY:  # 重复时重试，最多MAX_RETRY次
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

                    # 提取问题的唯一标识（关键逻辑：定义“相同问题”的判断标准）
                    if qt == 'length':
                        # 长度问题：唯一标识为线段/圆弧的ID（避免重复问同一线段长度）
                        edge_id = qa['gt'].get('edge_id') or qa['qt_description'].split()[1]  # 从描述中提取边名
                        unique_key = f"length_{edge_id}"
                    elif qt == 'angle':
                        # 角度问题：唯一标识为顶点+两条边的组合（避免重复问同一角度）
                        angle_name = qa['question'].split('angle ')[1].split()[0]  # 从问题中提取角度名（如A-B-C）
                        unique_key = f"angle_{angle_name}"
                    elif qt == 'area':
                        # 面积问题：同一图形只问一次
                        unique_key = "area"

                    # 检查是否重复
                    if unique_key in generated_keys:
                        retry += 1
                        logger.debug(f"第{retry}次重试生成{qt}类型问题（重复：{unique_key}）")
                        continue

                    # 不重复则添加到结果
                    qa_pairs.append(qa)
                    generated_keys.add(unique_key)
                    logger.debug(f"生成{qt}类型问题（唯一标识：{unique_key}）")
                    break  # 跳出重试循环

                except Exception as e:
                    retry += 1
                    logger.error(f"第{retry}次重试生成{qt}类型问题失败: {str(e)}")
                    continue

            # 超过最大重试次数仍失败/重复，则放弃该类型问题
            if retry >= MAX_RETRY:
                logger.debug(f"生成{qt}类型问题超过最大重试次数（{MAX_RETRY}次），跳过")

        return qa_pairs