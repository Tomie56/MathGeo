import asyncio
import json
import os
import sys
import re
from datetime import datetime
import traceback
import base64
from collections import defaultdict
from tqdm import tqdm
import argparse
from mimetypes import guess_type
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import sympy as sp

# ------------------------------ 全局配置 ------------------------------
API_KEY = "495e7f4ae82ddc5ccdb928b1bb686375"
BASE_URL = "https://dl.yunstorm.com/v1"
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"  

# 并发配置
MAX_WORKERS = 16
max_retry = 3
req_per_question = 1

# sympy初始化
sp.init_printing(use_latex=True)
PI = sp.pi
SQRT = sp.sqrt
ATAN = sp.atan
ASIN = sp.asin
ACOS = sp.acos

# ------------------------------ 工具函数 ------------------------------
def log_message(message):
    """增强型日志记录"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

def extract_geometry_ids(json_data):
    """提取JSON中所有几何元素ID，用于LLM生成带ID的问题"""
    ids = {
        "points": [p["id"] for p in json_data.get("points", [])],
        "lines": [l["id"] for l in json_data.get("lines", [])],
        "arcs": [a["id"] for a in json_data.get("arcs", [])],
        "entities": [e["id"] for e in json_data.get("entities", []) if "id" in e],
        "entity_types": [e["type"] for e in json_data.get("entities", []) if "type" in e]
    }
    # 补充阴影区域标签（如果有）
    shadow_labels = [e.get("region_label") for e in json_data.get("entities", []) if e.get("type") == "shadow"]
    ids["shadow_regions"] = [f"shadow_{l}" for l in shadow_labels if l is not None]
    return ids

# ------------------------------ LLM问题生成核心类 ------------------------------
class LLMQuestionGenerator:
    def __init__(self, n_questions=5):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
        )
        self.request_queue = asyncio.Queue()
        self.start_time = datetime.now()
        self.success_count = 0
        self.failure_count = 0
        self.total_processed = 0
        self.permission_error_count = 0
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.n_questions = n_questions  # 生成的问题数量

    async def get_image_url(self, image_path):
        """适配OSS链接/本地图片转base64"""
        if image_path.startswith(("http://", "https://", "s3://")):
            return image_path
        else:
            try:
                mime_type, _ = guess_type(image_path)
                if mime_type is None:
                    file_ext = image_path.split('.')[-1].lower()
                    mime_type = f'image/{file_ext}' if file_ext in ['png', 'jpg', 'jpeg', 'webp'] else 'application/octet-stream'
                
                with open(image_path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime_type};base64,{base64_data}"
            except Exception as e:
                log_message(f"图片处理失败 -> {image_path} | 错误：{str(e)}")
                return None

    async def construct_messages(self, image_paths, item):
        """构造LLM调用消息（参考Description生成完整题干+带ID的问题）"""
        # 提取当前样本的所有几何元素ID
        geom_ids = extract_geometry_ids(item)
        has_shadow = len(geom_ids["shadow_regions"]) > 0
        has_arcs = len(geom_ids["arcs"]) > 0

        image_contents = []
        for img_path in image_paths:
            img_url = await self.get_image_url(img_path)
            if img_url:
                image_contents.append({
                    'type': 'image_url',
                    'image_url': {'url': img_url}
                })
            else:
                log_message(f"跳过无效图片 -> {img_path}")
                return None

        # 核心提示词：参考Description生成完整题干，保留ID+难度+类型要求
        text_content = {
            'type': 'text',
            'text': (
                f'Based on the provided geometric description and image information, generate {self.n_questions} math problem stems in English. \n'
                f'=== STYLE REQUIREMENTS ===\n'
                f'1. Adhere to the language style of common math problem formats: fully, rigorously, and concisely integrate the geometric context from the description, then pose a clear question.\n'
                f'2. MUST include specific geometric element IDs (from the list below) in each problem stem:\n'
                f"   - Points: {', '.join(geom_ids['points'])}\n"
                f"   - Lines: {', '.join(geom_ids['lines'])}\n"
                f"   - Entities: {', '.join(geom_ids['entities'])}\n"
                f"   - Shadow regions: {', '.join(geom_ids['shadow_regions']) if has_shadow else 'none'}\n"
                f"   - Arcs: {', '.join(geom_ids['arcs']) if has_arcs else 'none'}\n"
                f'3. Question types (cover ALL applicable types):\n'
                f'   - Length: line segments (single/multiple minimal lines combined), arc length\n'
                f'   - Area: rectangle/circle/shadow region (use entity ID or shadow label)\n'
                f'   - Perimeter: rectangle/circle (use entity ID)\n'
                f'   - Angle: arc central angle (use arc ID)\n'
                f'4. Support IMPLICIT lines (not in JSON but composed of minimal lines, e.g., "the line from point A to E (composed of L1 and L4)")\n'
                f'5. Assign a difficulty label (1 = easiest, 5 = hardest) to each question based on reasoning complexity.\n'
                f'=== GEOMETRIC CONTEXT ===\n'
                f'Description of Geometric Context: {item.get("qt_description", item["description"])}\n'
                f'Full Geometric Description: {item["description"]}\n'
                f'=== OUTPUT FORMAT ===\n'
                f'Output ONLY a JSON array (no extra text) in this strict format:\n'

                f'"question": "[Complete English problem stem integrating description + clear question]", '
                f'"difficulty_label": [1-5], '
                f'"question_type": "[length/area/angle/perimeter]", '
                f'"target_ids": ["list of target element IDs (e.g., ["L1", "base_rectangle_1:4"])"]'
            )
        }

        return [
            {
                'role': 'user',
                'content': image_contents + [text_content]
            }
        ]

    def call_openai_api_sync(self, messages):
        """同步调用OpenAI风格API"""
        try:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_tokens=32768,
                top_p=0.95,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            log_message(f"❌ API调用错误：{str(e)}")
            return None

    async def call_openai_api_async(self, idx, image_paths, item):
        """异步包装API调用"""
        messages = await self.construct_messages(image_paths, item)
        if not messages:
            log_message(f"样本{idx}：消息构造失败")
            return None

        if self.permission_error_count >= 3:
            log_message(f"⚠️  累计{self.permission_error_count}次权限错误，停止样本{idx}重试")
            return None

        for retry in range(max_retry):
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self.call_openai_api_sync,
                    messages
                )
                if result:
                    # 验证返回是否为合法JSON
                    try:
                        json.loads(result)
                        return result
                    except json.JSONDecodeError:
                        log_message(f"样本{idx}：第{retry+1}次返回非JSON格式，重试中...")
                        await asyncio.sleep((2 + retry)**2)
                else:
                    log_message(f"样本{idx}：第{retry+1}次调用返回空，重试中...")
                    await asyncio.sleep((2 + retry)**2)
            except Exception as e:
                log_message(f"样本{idx}：第{retry+1}次调用异常 -> {str(e)}")
                await asyncio.sleep((2 + retry)**2)
        return None

    async def worker(self, pbar, tmp_dir):
        """异步工作线程"""
        while True:
            task_retrieved = False
            try:
                idx, item = await self.request_queue.get()
                task_retrieved = True
                
                output_json = os.path.join(tmp_dir, f"{idx}.json")
                start_time = datetime.now()
                status = "失败"

                # 跳过已完成任务
                if os.path.exists(output_json):
                    try:
                        async with aiofiles.open(output_json, 'r') as f:
                            existing_data = json.loads(await f.read())
                            if len(existing_data.get("generated_question", [])) >= req_per_question:
                                self.total_processed += 1
                                pbar.update(1)
                                self.request_queue.task_done()
                                continue
                    except Exception as e:
                        log_message(f"样本{idx}：中间文件校验失败 -> {str(e)}")

                # 处理图片路径
                image_paths = []
                if 'annotated_shaded_path' is n:
                    image_paths = [item['annotated_raw_path']]
                elif isinstance(item.get('image'), list):
                    image_paths = item['image']
                elif item.get('image'):
                    image_paths = [item['image']]
                else:
                    log_message(f"样本{idx}：无有效图片路径")
                    self.total_processed += 1
                    pbar.update(1)
                    self.request_queue.task_done()
                    continue

                # 生成问题
                results = []
                for attempt in range(req_per_question + max_retry):
                    result = await self.call_openai_api_async(idx, image_paths, item)
                    success = bool(result)
                    if success:
                        results.append(result)
                        if len(results) >= req_per_question:
                            break

                # 写入中间文件
                tmp_path = f"{output_json}.tmp"
                async with aiofiles.open(tmp_path, 'w', encoding='utf-8') as f:
                    item['generated_question'] = results[:req_per_question]
                    await f.write(json.dumps(item, ensure_ascii=False, indent=2))
                os.replace(tmp_path, output_json)

                # 统计
                elapsed = (datetime.now() - start_time).total_seconds()
                if len(results) >= req_per_question:
                    status = "成功"
                    self.success_count += 1
                else:
                    self.failure_count += 1
                
                self.total_processed += 1
                log_message(
                    f"样本{idx} | 状态:{status} | 耗时:{elapsed:.2f}s | 有效结果:{len(results)}/{req_per_question}"
                )
                
                # 更新进度条
                processing_speed = self.total_processed / (datetime.now() - self.start_time).total_seconds()
                pbar.set_postfix({
                    '成功': self.success_count,
                    '失败': self.failure_count,
                    '速度': f"{processing_speed:.2f}条/s"
                })

            except asyncio.CancelledError:
                log_message(f"工作线程被取消 | 样本{idx if task_retrieved else '未知'}")
                break
            except Exception as e:
                log_message(f"样本{idx if task_retrieved else '未知'} | 业务异常 -> {str(e)}")
                traceback.print_exc()
            finally:
                if task_retrieved:
                    self.request_queue.task_done()

    async def generate_questions(self, input_path, output_path):
        """生成LLM问题主流程"""
        # 读取输入JSONL
        log_message(f"读取输入文件：{input_path}")
        items = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    items.append(data)
                except json.JSONDecodeError:
                    log_message(f"跳过行{line_num}：JSON解析失败")
        
        total_items = len(items)
        if total_items == 0:
            log_message("无有效条目，退出")
            return

        # 创建临时目录
        tmp_dir = os.path.join(os.path.dirname(output_path), "tmp_question")
        os.makedirs(tmp_dir, exist_ok=True)

        # 填充任务队列
        for idx, item in enumerate(items):
            await self.request_queue.put((idx, item))

        # 并发处理
        with tqdm(
            total=total_items,
            desc="生成带完整题干的几何问题",
            file=sys.stdout,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余{remaining}] {postfix}"
        ) as pbar:
            worker_count = min(MAX_WORKERS, total_items)
            workers = [asyncio.create_task(self.worker(pbar, tmp_dir)) for _ in range(worker_count)]
            
            await self.request_queue.join()
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # 汇总结果
        log_message(f"汇总结果到：{output_path}")
        success_count = 0
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as outf:
            for idx in range(total_items):
                json_file = os.path.join(tmp_dir, f"{idx}.json")
                if os.path.exists(json_file):
                    try:
                        async with aiofiles.open(json_file, 'r') as f:
                            item = json.loads(await f.read())
                            if len(item.get("generated_question", [])) >= req_per_question:
                                await outf.write(json.dumps(item, ensure_ascii=False) + '\n')
                                success_count += 1
                    except Exception as e:
                        log_message(f"跳过异常文件（序号{idx}）：{str(e)}")

        # 最终统计
        log_message(
            f"生成完成 | 总条目{total_items} | 成功{success_count} | 成功率：{success_count/total_items:.1%}"
        )
        return output_path

# ------------------------------ 几何求解器核心类（基于sympy） ------------------------------
class GeometrySolver:
    def __init__(self):
        # sympy符号定义
        self.x, self.y, self.r, self.theta = sp.symbols('x y r theta')
        # 表达式解析映射（转换为sympy语法）
        self.expr_mapping = {
            'π': 'PI',
            '√': 'SQRT',
            '½': 'sp.Rational(1,2)',
            '⅓': 'sp.Rational(1,3)',
            '¼': 'sp.Rational(1,4)',
            'arctan': 'ATAN',
            'arcsin': 'ASIN',
            'arccos': 'ACOS',
            '×': '*',
            '÷': '/',
            '^': '**',
            '²': '**2',
            '³': '**3'
        }

    def parse_expr_to_sympy(self, expr):
        """将字符串表达式解析为sympy符号表达式"""
        if not expr or expr.strip() == "":
            return sp.Integer(0)
        
        parsed_expr = expr.strip()
        # 替换特殊符号为sympy语法
        for k, v in self.expr_mapping.items():
            parsed_expr = parsed_expr.replace(k, v)
        
        # 处理分数（如3/2 → sp.Rational(3,2)）
        parsed_expr = re.sub(r'(\d+)/(\d+)', r'sp.Rational(\1,\2)', parsed_expr)
        
        try:
            # 执行表达式计算（sympy符号计算）
            sympy_expr = eval(parsed_expr, {'sp': sp, 'PI': PI, 'SQRT': SQRT, 'ATAN': ATAN, 'ASIN': ASIN, 'ACOS': ACOS})
            return sympy_expr
        except Exception as e:
            log_message(f"Sympy表达式解析失败：{expr} → {parsed_expr} | 错误：{str(e)}")
            return sp.Integer(0)

    def sympy_to_latex(self, sympy_expr):
        """将sympy表达式转换为LaTeX格式"""
        try:
            return sp.latex(sympy_expr)
        except:
            return str(sympy_expr)

    def sympy_to_str(self, sympy_expr):
        """将sympy表达式转换为普通字符串"""
        try:
            expr_str = str(sympy_expr)
            # 还原特殊符号
            reverse_mapping = {
                'PI': 'π',
                'sqrt': '√',
                'Rational': '',
                'atan': 'arctan',
                'asin': 'arcsin',
                'acos': 'arccos'
            }
            for k, v in reverse_mapping.items():
                expr_str = expr_str.replace(k, v)
            # 清理冗余括号（如Rational(1,2) → 1/2）
            expr_str = re.sub(r'\((\d+)/(\d+)\)', r'\1/\2', expr_str)
            expr_str = re.sub(r'sp\.', '', expr_str)
            return expr_str
        except:
            return str(sympy_expr)

    def get_line_length(self, json_data, line_id):
        """获取单条线段长度（sympy符号计算）"""
        # 1. 先查直接定义的线段
        for line in json_data.get('lines', []):
            if line['id'] == line_id:
                expr_str = line['length']['expr']
                latex_str = line['length']['latex']
                sympy_expr = self.parse_expr_to_sympy(expr_str)
                return self.sympy_to_str(sympy_expr), latex_str, sympy_expr
        
        # 2. 处理隐含线段（按点拼接最小线段）
        # 提取所有点的坐标（sympy符号）
        point_coords = {}
        for p in json_data.get('points', []):
            x_sym = self.parse_expr_to_sympy(p['x']['expr'])
            y_sym = self.parse_expr_to_sympy(p['y']['expr'])
            point_coords[p['id']] = (x_sym, y_sym)
        
        # 若line_id是隐含线段（如"A-E"），解析端点并计算距离
        if '-' in line_id and all(p in point_coords for p in line_id.split('-')):
            p1, p2 = line_id.split('-')
            x1, y1 = point_coords[p1]
            x2, y2 = point_coords[p2]
            # sympy计算欧几里得距离
            dx = x2 - x1
            dy = y2 - y1
            length_sym = SQRT(dx**2 + dy**2)
            length_str = self.sympy_to_str(length_sym)
            length_latex = self.sympy_to_latex(length_sym)
            return length_str, length_latex, length_sym
        
        # 3. 拼接多个最小线段（如"L1+L4"）
        if '+' in line_id:
            total_sym = sp.Integer(0)
            total_str = ""
            total_latex = ""
            sub_lines = line_id.split('+')
            for sub_line in sub_lines:
                sub_str, sub_latex, sub_sym = self.get_line_length(json_data, sub_line.strip())
                total_sym += sub_sym
                total_str += f"{sub_str} + " if total_str else sub_str
                total_latex += f"{sub_latex} + " if total_latex else sub_latex
            total_str = total_str.rstrip(' + ')
            total_latex = total_latex.rstrip(' + ')
            return total_str, total_latex, total_sym
        
        # 无匹配线段
        return "", "", sp.Integer(0)

    def get_entity_attr(self, json_data, entity_id, attr):
        """获取实体属性（area/perimeter/radius等）"""
        for entity in json_data.get('entities', []):
            if entity.get('id') == entity_id and attr in entity:
                expr_str = entity[attr]['expr']
                latex_str = entity[attr]['latex']
                sympy_expr = self.parse_expr_to_sympy(expr_str)
                return self.sympy_to_str(sympy_expr), latex_str, sympy_expr
        return "", "", sp.Integer(0)

    def get_shadow_attr(self, json_data, shadow_label, attr):
        """获取阴影区域属性（area/perimeter）"""
        for entity in json_data.get('entities', []):
            if entity.get('type') == 'shadow' and entity.get('region_label') == int(shadow_label.split('_')[-1]):
                if attr in entity:
                    expr_str = entity[attr]['expr']
                    latex_str = entity[attr]['latex']
                    sympy_expr = self.parse_expr_to_sympy(expr_str)
                    return self.sympy_to_str(sympy_expr), latex_str, sympy_expr
        return "", "", sp.Integer(0)

    def get_arc_attr(self, json_data, arc_id, attr):
        """获取弧属性（length/angle/radius）"""
        for arc in json_data.get('arcs', []):
            if arc['id'] == arc_id and attr in arc:
                expr_str = arc[attr]['expr']
                latex_str = arc[attr].get('latex', '')
                sympy_expr = self.parse_expr_to_sympy(expr_str)
                return self.sympy_to_str(sympy_expr), latex_str, sympy_expr
        return "", "", sp.Integer(0)

    def calculate_diff(self, gt_expr, point_count, line_count, arc_count, difficulty_label):
        """计算diff值（基于sympy表达式复杂度）"""
        # 1. 表达式复杂度（符号数量+运算次数）
        sympy_expr = self.parse_expr_to_sympy(gt_expr)
        # 统计符号数量
        symbol_count = len(list(sympy_expr.free_symbols)) if gt_expr else 0
        # 统计运算次数（通过字符串解析）
        op_count = len(re.findall(r'\+|-|\*|/|√|π|arctan|arcsin|arccos', gt_expr)) if gt_expr else 0
        expr_complexity = (symbol_count * 2 + op_count) / 20

        # 2. 图形复杂度（点+线+弧数量）
        graph_complexity = (point_count + line_count + arc_count) / 30

        # 3. 难度权重
        label_weight = difficulty_label / 5

        # 4. 综合diff（加权求和+放大系数）
        diff = float((expr_complexity * 0.4) + (graph_complexity * 0.3) + (label_weight * 0.3)) * 5
        return diff

    def solve_question(self, json_data, question):
        """核心求解：根据问题类型+目标ID计算GT（sympy符号计算）"""
        question_text = question['question']
        question_type = question['question_type']
        difficulty_label = question['difficulty_label']
        target_ids = question.get('target_ids', [])
        
        # 提取图形元素数量
        point_count = len(json_data.get('points', []))
        line_count = len(json_data.get('lines', []))
        arc_count = len(json_data.get('arcs', []))
        
        # 初始化GT
        gt_expr = ""
        gt_latex = ""

        # 按问题类型求解
        if question_type == "length":
            # 长度：线段/弧
            for target_id in target_ids:
                if target_id.startswith('L') or '-' in target_id or '+' in target_id:
                    # 线段长度（直接/隐含/拼接）
                    expr, latex, _ = self.get_line_length(json_data, target_id)
                elif target_id.startswith('arc_'):
                    # 弧长
                    expr, latex, _ = self.get_arc_attr(json_data, target_id, 'length')
                if expr:
                    gt_expr = expr
                    gt_latex = latex
                    break

        elif question_type == "area":
            # 面积：实体/阴影
            for target_id in target_ids:
                if target_id.startswith('shadow_'):
                    # 阴影区域面积
                    expr, latex, _ = self.get_shadow_attr(json_data, target_id, 'area')
                else:
                    # 实体（矩形/圆）面积
                    expr, latex, _ = self.get_entity_attr(json_data, target_id, 'area')
                if expr:
                    gt_expr = expr
                    gt_latex = latex
                    break

        elif question_type == "perimeter":
            # 周长：实体（矩形/圆）
            for target_id in target_ids:
                expr, latex, _ = self.get_entity_attr(json_data, target_id, 'perimeter')
                if expr:
                    gt_expr = expr
                    gt_latex = latex
                    break

        elif question_type == "angle":
            # 角度：弧的圆心角
            for target_id in target_ids:
                if target_id.startswith('arc_'):
                    expr, latex, _ = self.get_arc_attr(json_data, target_id, 'angle')
                    if expr:
                        gt_expr = expr
                        gt_latex = latex
                        break

        # 计算diff
        diff = self.calculate_diff(gt_expr, point_count, line_count, arc_count, difficulty_label)

        # 构造结果
        return {
            "gt": {
                "expr": gt_expr,
                "latex": gt_latex
            },
            "diff": diff,
            "question_type": question_type,
            "generated_question": [question_text]
        }

    def process_solutions(self, input_jsonl_path, llm_questions_path, output_path):
        """批量求解并生成最终格式"""
        # 读取原始JSONL
        log_message(f"读取原始几何数据：{input_jsonl_path}")
        raw_data = []
        with open(input_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append(json.loads(line.strip()))
        
        # 读取LLM生成的问题
        log_message(f"读取LLM生成的问题：{llm_questions_path}")
        llm_data = []
        with open(llm_questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                llm_data.append(json.loads(line.strip()))
        
        # 逐样本处理
        log_message(f"开始求解并生成最终结果：{output_path}")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for raw_item, llm_item in zip(raw_data, llm_data):
                try:
                    # 解析LLM生成的问题列表
                    questions = json.loads(llm_item['generated_question'][0])
                    for q in questions:
                        # 求解每个问题
                        solve_res = self.solve_question(raw_item, q)
                        # 构造最终输出
                        final_output = {
                            "image": raw_item.get('annotated_raw_path', ''),
                            "description": raw_item.get('description', ''),
                            "gt": solve_res['gt'],
                            "diff": solve_res['diff'],
                            "question_type": solve_res['question_type'],
                            "generated_question": solve_res['generated_question']
                        }
                        out_f.write(json.dumps(final_output, ensure_ascii=False) + '\n')
                except Exception as e:
                    log_message(f"处理样本失败：{str(e)}")
                    traceback.print_exc()
        log_message(f"最终结果已写入：{output_path}")

# ------------------------------ 主函数 ------------------------------
async def main():
    parser = argparse.ArgumentParser(description="几何问题生成+求解全流程")
    parser.add_argument("--input_path", default = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/results_n500_v4/json/final/shaded_with_gt_20251203_071430_009.jsonl", help="原始几何数据JSONL路径")
    parser.add_argument("--output", default = "/mnt/afs/jingjinhao/project/GeoChain/MathGeo/A500_v4/api/gen_question.jsonl", help="最终结果输出路径")
    parser.add_argument("--n", type=int, default=1, help="生成的问题数量（默认5）")
    parser.add_argument("--tmp_llm_output", default="./tmp_llm_questions.jsonl", help="LLM问题临时输出路径")
    args = parser.parse_args()

    # 校验输入文件
    if not os.path.exists(args.input_path):
        log_message(f"错误：输入文件不存在 → {args.input_path}")
        sys.exit(1)

    # 步骤1：生成带ID的LLM问题
    generator = LLMQuestionGenerator(n_questions=args.n)
    await generator.generate_questions(args.input_path, args.tmp_llm_output)

    # 步骤2：Sympy求解并生成最终结果
    solver = GeometrySolver()
    solver.process_solutions(args.input_path, args.tmp_llm_output, args.output)

    log_message("全流程执行完成！")

if __name__ == '__main__':
    asyncio.run(main())
