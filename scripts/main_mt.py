import os
import json
import random
import logging
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed  # 移至顶部导入
import cv2
import sympy as sp
from sympy import symbols, simplify

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geochain.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('GeoChain')

# 导入工具模块（确保各模块路径正确）
from tools.template import TemplateGenerator  # 基础图形生成器
from tools.builder import RandomGeometryBuilder  # 图形增强器
from tools.drawer import GeometryDrawer  # 图形绘制器
from tools.shader import EnhancedDrawer # 区域着色器
from tools.gt import GeometryCalculator  # 几何参数计算器（保留接口）
from tools.qa_template import QAGenerator  # 问答生成器（保留接口）


class GeoChainPipeline:
    """图形生成与处理全流程管道（所有流程内部多线程，指定步骤超时）"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._init_workspace()
        self._set_random_seeds()
        
        # 并行配置（所有流程共用线程数）
        self.thread_num = self.config['global'].get('thread_num', 4)  # 多线程数量（默认4）
        self.template_builder_timeout = 3600  # 单次生成超时（1小时，与值统一）
        
        # 中间数据存储
        self.base_jsonl_path: Optional[str] = None  # 基础图形JSONL文件路径
        self.enhanced_jsonl_path: Optional[str] = None  # 增强图形JSONL路径
        self.enhanced_jsons: List[Dict[str, Any]] = []  # 增强图形JSON列表（含base_idx和enhance_idx）
        self.shaded_jsonl_path: Optional[str] = None  # 着色图形JSONL路径
        self.gt_jsonl_path: Optional[str] = None  # 带GT参数的最终JSONL路径
        self.qa_jsonl_path: Optional[str] = None  # 问答数据JSONL路径
        self.raw_image_paths: List[str] = []  # 原始图像路径列表
        self.shaded_image_paths: List[str] = []  # 着色图像路径列表（保留接口）

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载并验证总配置文件（补充缺失默认值）"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 验证核心配置项
            required_sections = ['template', 'builder', 'drawer', 'shader', 'gt', 'annotator', 'qa']
            for section in required_sections:
                if section not in config:
                    logger.warning(f"配置文件缺少模块: {section}，将使用默认配置")
                    config[section] = {}
            # 补充builder必要默认值（避免KeyError）
            builder_cfg = config['builder']
            builder_cfg['num_enhancements'] = builder_cfg.get('num_enhancements', 3)
            builder_cfg['single_enhance_timeout'] = builder_cfg.get('single_enhance_timeout', 30)
            builder_cfg['rounds_distribution'] = builder_cfg.get('rounds_distribution', {1: 1})
            builder_cfg['min_operations_per_round'] = builder_cfg.get('min_operations_per_round', 1)
            builder_cfg['max_operations_per_round'] = builder_cfg.get('max_operations_per_round', 1)
            builder_cfg['allowed_operation_types'] = builder_cfg.get('allowed_operation_types', 
                                                                   ['connect_points', 'connect_midpoints', 'draw_perpendicular', 'draw_diameter'])
            builder_cfg['operation_probs'] = builder_cfg.get('operation_probs', {})
            builder_cfg['operation_constraints'] = builder_cfg.get('operation_constraints', {})
            return config
        
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {str(e)}")

    def _init_workspace(self) -> None:
        """初始化工作目录结构"""
        out_root = self.config['global']['output_root']
        subdirs = [
            'json/base',          # 基础图形JSON
            'json/enhanced',      # 增强图形JSON
            'json/shaded',        # 着色图形JSON
            'json/final',         # 带GT参数的最终JSON
            'images/raw',         # 原始图像
            'images/shaded',      # 着色图像
            'images/annotated',   # 标注图像
            'qa'                  # 问答数据
        ]
        
        for subdir in subdirs:
            dir_path = os.path.join(out_root, subdir)
            os.makedirs(dir_path, exist_ok=True)
        logger.info(f"工作目录初始化完成: {out_root}")

    def _set_random_seeds(self) -> None:
        main_seed = self.config['global'].get('main_seed', 42)
        random.seed(main_seed)
        logger.info(f"主随机种子: {main_seed}")
        
        self.config['template']['seed'] = main_seed + 1
        self.config['builder']['seed'] = main_seed + 2
        self.config['shader']['seed'] = main_seed + 3
        self.config['qa']['seed'] = main_seed + 4
        logger.info("各模块随机种子初始化完成")

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def _run_task_with_timeout(self, task_func: Callable, *args, **kwargs) -> Tuple[str, Any]:
        """执行单个任务，添加超时限制（仅用于run_template/run_builder）"""
        result_queue = queue.Queue(maxsize=1)

        def task_wrapper():
            try:
                result = task_func(*args, **kwargs)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))

        task_thread = threading.Thread(target=task_wrapper, daemon=True)
        task_thread.start()
        task_thread.join(timeout=self.template_builder_timeout)

        if task_thread.is_alive():
            return ('timeout', None)
        else:
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return ('error', RuntimeError("任务无返回结果"))

    def _run_task_no_timeout(self, task_func: Callable, *args, **kwargs) -> Tuple[str, Any]:
        """执行单个任务，无超时限制（用于shader/gt/qa流程）"""
        try:
            result = task_func(*args, **kwargs)
            return ('success', result)
        except Exception as e:
            return ('error', e)

    def _parallel_process(self,
                        task_list: List[Any],
                        task_func: Callable,
                        output_path: str,
                        has_timeout: bool = False,
                        is_multi_result: bool = False) -> Tuple[List[Any], int]:
        """修复版本：先收集结果，再统一写入文件，避免多线程直接操作文件句柄"""
        success_results = []
        success_count = 0  # 成功的任务数（非结果数）
        total_tasks = len(task_list)

        # 多线程执行任务，先收集所有结果
        with ThreadPoolExecutor(max_workers=self.thread_num) as executor:
            # 提交所有任务
            futures = {executor.submit(self._run_task, task_func, task, has_timeout): task for task in task_list}

            # 处理任务结果
            for future in as_completed(futures):
                task = futures[future]
                task_id = task[0] if task else "unknown"
                try:
                    result = future.result()
                    if result is not None:
                        # 处理多结果场景（如增强图形生成）
                        if is_multi_result and isinstance(result, list):
                            success_results.extend(result)
                            logger.debug(f"任务 {task_id} 执行成功（生成{len(result)}个结果）")
                        else:
                            success_results.append(result)
                            logger.debug(f"任务 {task_id} 执行成功")
                        success_count += 1  # 任务数+1（无论结果数量）
                    else:
                        logger.warning(f"任务 {task_id} 未返回有效结果")
                except Exception as e:
                    logger.error(f"任务 {task_id} 执行失败: {str(e)}", exc_info=True)

        # 所有任务完成后，在主线程统一写入文件（避免自由变量问题）
        with open(output_path, 'a', encoding='utf-8') as f:
            for item in success_results:
                # 支持单结果和多结果写入
                if is_multi_result and isinstance(item, list):
                    for sub_item in item:
                        json.dump(sub_item, f, ensure_ascii=False)
                        f.write("\n")
                else:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

        logger.info(f"并行任务完成：成功 {success_count}/{total_tasks} 个任务，共生成 {len(success_results)} 个结果")
        return success_results, success_count

    def _run_task(self, task_func: Callable, task: Any, has_timeout: bool) -> Any:
        """辅助函数：执行单个任务（含超时控制）"""
        if has_timeout:
            result_queue = queue.Queue(maxsize=1)

            def wrapper():
                try:
                    result = task_func(task)
                    result_queue.put(('success', result))
                except Exception as e:
                    result_queue.put(('error', e))

            thread = threading.Thread(target=wrapper)
            thread.start()
            thread.join(timeout=self.template_builder_timeout)  # 超时时间

            if thread.is_alive():
                raise TimeoutError(f"任务 {task[0]} 超时（{self.template_builder_timeout}秒）")

            status, result = result_queue.get()
            if status == 'error':
                raise result
            return result
        else:
            return task_func(task)

    # ------------------------------ 步骤1：生成基础图形 ------------------------------
    def run_template(self) -> None:
        logger.info(f"=== 开始生成基础图形 ===")
        template_cfg = self.config['template']
        seed = template_cfg.get('seed')
        n = template_cfg.get('n', 10)

        self.base_jsonl_path = os.path.join(
            template_cfg.get('output_dir', os.path.join(self.config['global']['output_root'], 'json/base')),
            f'base_{self._get_timestamp()}.jsonl'
        )
        # 确保目录存在（避免写入失败）
        os.makedirs(os.path.dirname(self.base_jsonl_path), exist_ok=True)
        with open(self.base_jsonl_path, 'w', encoding='utf-8') as f:
            pass  # 清空文件

        task_list = [(f"template-{i+1}", seed + 42 * i if seed else None, i+1) for i in range(n)]

        def template_task(task):
            task_id, sample_seed, base_idx = task
            generator = TemplateGenerator(template_cfg, seed=sample_seed)
            generator.generate_base_shape()
            generator.generate_derivations()
            base_json = generator.export_json()
            base_json['base_idx'] = base_idx
            base_json['base_id'] = f"base_{base_idx:03d}"
            return base_json

        # 执行任务并强制校验写入结果
        self.enhanced_jsons, success_count = self._parallel_process(
            task_list=task_list,
            task_func=template_task,
            output_path=self.base_jsonl_path,
            has_timeout=True
        )

        # 二次校验base/jsonl完整性
        if not os.path.exists(self.base_jsonl_path) or os.path.getsize(self.base_jsonl_path) == 0:
            raise RuntimeError("基础图形JSONL生成失败，文件为空")
        
        with open(self.base_jsonl_path, 'r', encoding='utf-8') as f:
            actual_lines = sum(1 for line in f if line.strip())
            
        if actual_lines != len(self.enhanced_jsons):
            logger.error(f"基础图形记录不完整：预期{len(self.enhanced_jsons)}条，实际{actual_lines}条")
        logger.info(f"基础图形JSONL保存至: {self.base_jsonl_path}（共{len(self.enhanced_jsons)}个基础图）")

    # ------------------------------ 步骤2：增强图形（核心修复：参数解析+超时保留结果） ------------------------------
    def run_builder(self) -> None:
        """步骤2：生成增强图形（支持单轮超时，保留已完成结果）"""
        if not self.base_jsonl_path or not os.path.exists(self.base_jsonl_path):
            raise RuntimeError("未找到基础图形JSONL文件，无法执行增强步骤")
        
        builder_cfg = self.config['builder']
        logger.info(f"=== 开始生成增强图形（多线程：{self.thread_num}个，总超时：{self.template_builder_timeout}秒，单轮超时：{builder_cfg['single_enhance_timeout']}秒） ===")

        self.enhanced_jsonl_path = os.path.join(
            builder_cfg.get('output_dir', os.path.join(self.config['global']['output_root'], 'json/enhanced')),
            f'enhanced_{self._get_timestamp()}.jsonl'
        )
        os.makedirs(os.path.dirname(self.enhanced_jsonl_path), exist_ok=True)
        with open(self.enhanced_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 构建任务列表：(任务ID, 基础JSON, 任务种子, 基础图索引, builder配置)
        task_list = []
        with open(self.base_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    base_json = json.loads(line)
                    base_idx = base_json.get('base_idx', line_num)
                    task_seed = builder_cfg['seed'] + line_num * 42
                    task_list.append((f"builder-{base_idx}", base_json, task_seed, base_idx, builder_cfg))

        if not task_list:
            raise RuntimeError("基础图形JSONL文件为空，无法执行增强步骤")

        # 单个任务处理函数（修复参数解析+复用现有增强方法）
        def builder_task(task):
            # 正确解析5个任务参数
            task_id, base_json, task_seed, base_idx, builder_cfg = task
            num_enhancements = builder_cfg['num_enhancements']
            single_enhance_timeout = builder_cfg['single_enhance_timeout']
            
            completed_enhancements = []
            start_time = time.time()

            for enh_idx in range(1, num_enhancements + 1):
                # 总超时判断
                if time.time() - start_time > self.template_builder_timeout:
                    logger.warning(f"任务 {task_id} 总超时，已生成{len(completed_enhancements)}/{num_enhancements}个结果")
                    break

                try:
                    # 复用RandomGeometryBuilder的generate_enhancements，生成单个增强结果
                    def generate_single():
                        builder = RandomGeometryBuilder(base_json)
                        # 配置为生成1个增强结果（1轮操作）
                        config = {
                            "rounds_distribution": builder_cfg['rounds_distribution'],
                            "min_operations_per_round": builder_cfg['min_operations_per_round'],
                            "max_operations_per_round": builder_cfg['max_operations_per_round'],
                            "allowed_operation_types": builder_cfg['allowed_operation_types'],
                            "operation_probs": builder_cfg['operation_probs'],
                            "operation_constraints": builder_cfg['operation_constraints'],
                            "seed": task_seed + enh_idx * 100,
                            "max_points": 500,
                            "max_lines": 300
                        }
                        # 生成1个增强结果（取第一个）
                        enhancements = builder.generate_enhancements(config)
                        return enhancements[0] if enhancements else None

                    # 单轮增强超时控制
                    enh_queue = queue.Queue(maxsize=1)
                    enh_thread = threading.Thread(target=lambda q: q.put(generate_single()), args=(enh_queue,), daemon=True)
                    enh_thread.start()
                    enh_thread.join(timeout=single_enhance_timeout)

                    if enh_thread.is_alive():
                        logger.warning(f"任务 {task_id} 第{enh_idx}轮超时，跳过")
                        continue

                    # 获取结果（避免直接调用导致重复执行）
                    enh_json = enh_queue.get_nowait() if not enh_queue.empty() else None
                    if not enh_json:
                        logger.warning(f"任务 {task_id} 第{enh_idx}轮生成失败，跳过")
                        continue

                    # 补充标识信息（确保与全流程一致）
                    enh_json['base_idx'] = base_idx
                    enh_json['enhance_idx'] = enh_idx
                    enh_json['enhance_id'] = f"base_{base_idx:03d}_enhance_{enh_idx:03d}"
                    enh_json['base_id'] = base_json.get('base_id', f"base_{base_idx:03d}")
                    completed_enhancements.append(enh_json)
                    logger.info(f"任务 {task_id} 第{enh_idx}轮增强成功")

                except Exception as e:
                    logger.error(f"任务 {task_id} 第{enh_idx}轮异常：{str(e)}，跳过")
                    continue

            return completed_enhancements

        # 多线程处理（is_multi_result=True：单个任务返回多个增强结果）
        self.enhanced_jsons, success_count = self._parallel_process(
            task_list=task_list,
            task_func=builder_task,
            output_path=self.enhanced_jsonl_path,
            has_timeout=True,
            is_multi_result=True
        )

        if not self.enhanced_jsons:
            raise RuntimeError("未生成任何有效的增强图形")
        
        self.config['drawer']['jsonl_path'] = self.enhanced_jsonl_path
        logger.info(f"增强图形JSONL保存至: {self.enhanced_jsonl_path}（共{len(self.enhanced_jsons)}个增强图）")

    # ------------------------------ 步骤3：绘制原始图像（规范命名） ------------------------------
    def run_drawer(self) -> None:
        """步骤3：绘制原始图像（多线程，按“base_001_enhance_002_raw.png”命名）"""
        if not self.enhanced_jsons:
            raise RuntimeError("未生成增强图形，无法执行绘图步骤")
        
        logger.info(f"=== 开始绘制原始图像（多线程：{self.thread_num}个） ===")
        drawer_cfg = self.config['drawer']
        output_dir = drawer_cfg.get('output_dir', os.path.join(self.config['global']['output_root'], 'images/raw'))
        os.makedirs(output_dir, exist_ok=True)

        # 构建任务列表：(任务ID, 增强图形JSON)
        task_list = [(f"drawer-{i+1}", json_data) for i, json_data in enumerate(self.enhanced_jsons)]

        def drawer_task(task):
            task_id, json_data = task
            # 获取增强标识（确保非空）
            enhance_id = json_data.get('enhance_id', f"base_{json_data.get('base_idx',0):03d}_enhance_{json_data.get('enhance_idx',0):03d}")
            img_ext = '.png'

            # 绘制图像
            drawer = GeometryDrawer(drawer_cfg)
            img_path, _ = drawer.draw_single(json_data)

            # 规范命名并保存
            new_img_path = os.path.join(output_dir, f"{enhance_id}_raw{img_ext}")
            if img_path and os.path.exists(img_path):
                cv2.imwrite(new_img_path, cv2.imread(img_path))
                os.remove(img_path)

            return new_img_path

        # 多线程处理
        self.raw_image_paths, success_count = self._parallel_process(
            task_list=task_list,
            task_func=drawer_task,
            output_path=os.path.join(output_dir, 'drawer_paths.jsonl'),
            has_timeout=False
        )

        logger.info(f"原始图像保存完成（共 {len(self.raw_image_paths)} 张，成功 {success_count} 个任务）")

    # ------------------------------ 步骤4：区域着色与标注（规范命名） ------------------------------
    def run_shader(self) -> None:
        logger.info(f"=== 开始区域着色与标注 ===")
        shader_config = self.config['shader']
        output_root = self.config['global']['output_root']
        shader_enabled = shader_config.get('enabled', False)

        # 输出路径配置
        json_output_dir = os.path.join(output_root, 'json/shaded')
        self.shaded_jsonl_path = os.path.join(json_output_dir, f'shaded_{self._get_timestamp()}.jsonl')
        os.makedirs(json_output_dir, exist_ok=True)
        with open(self.shaded_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 图像输出目录
        shaded_img_dir = os.path.join(output_root, 'images/shaded')
        annotated_img_dir = os.path.join(output_root, 'images/annotated')
        os.makedirs(shaded_img_dir, exist_ok=True)
        os.makedirs(annotated_img_dir, exist_ok=True)

        # 构建任务列表
        task_list = []
        with open(self.enhanced_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_data = json.loads(line)
                    enhance_id = json_data.get('enhance_id')
                    if not enhance_id:
                        base_idx = json_data.get('base_idx', 0)
                        enhance_idx = json_data.get('enhance_idx', 0)
                        enhance_id = f"base_{base_idx:03d}_enhance_{enhance_idx:03d}"
                        json_data['enhance_id'] = enhance_id
                        logger.warning(f"增强数据缺失enhance_id，自动生成：{enhance_id}（行号：{line_num}）")
                    task_list.append((f"shader-{line_num}", json_data))
                except json.JSONDecodeError:
                    logger.error(f"增强JSONL第{line_num}行格式错误，跳过")
                    continue

        def shader_task(task):
            task_id, json_data = task
            enhance_id = json_data['enhance_id']
            base_id = json_data.get('base_id', f"base_{json_data.get('base_idx', 0):03d}")
            base_idx = json_data.get('base_idx', 0)
            enhance_idx = json_data.get('enhance_idx', 0)

            # 初始化着色工具
            tool_config = {
                "global": {"output_root": output_root},
                "drawer": {**self.config['drawer'], "output_dir": shaded_img_dir},
                "shader": {** shader_config, "enabled": shader_enabled},
                "annotator": {**self.config.get('annotator', {}), "output_dir": annotated_img_dir},
                "region_extractor": self.config.get("region_extractor", {})
            }
            enhanced_drawer = EnhancedDrawer(tool_config)
            enhanced_drawer.set_enhanced_data([json_data])
            result = enhanced_drawer.process_single()

            # 处理着色图像路径
            img_ext = '.png'
            shaded_path = result.get('shaded_path')
            new_shaded_path = os.path.join(shaded_img_dir, f"{enhance_id}_shaded{img_ext}")
            if shaded_path and os.path.exists(shaded_path):
                os.rename(shaded_path, new_shaded_path)
            result['shaded_path'] = new_shaded_path

            # 处理标注图像路径
            annotated_shaded_path = result.get('annotated_shaded_path')
            new_annotated_path = os.path.join(annotated_img_dir, f"{enhance_id}_annotated{img_ext}")
            if annotated_shaded_path and os.path.exists(annotated_shaded_path):
                os.rename(annotated_shaded_path, new_annotated_path)
            result['annotated_shaded_path'] = new_annotated_path

            # 保留核心标识
            result['enhance_id'] = enhance_id
            result['base_id'] = base_id
            result['base_idx'] = base_idx
            result['enhance_idx'] = enhance_idx

            return result

        # 多线程处理
        shaded_results, success_count = self._parallel_process(
            task_list=task_list,
            task_func=shader_task,
            output_path=self.shaded_jsonl_path,
            has_timeout=False
        )
        logger.info(f"着色与标注完成：成功 {success_count}/{len(task_list)} 个图形，结果保存至: {self.shaded_jsonl_path}")

    # ------------------------------ 步骤5：计算几何参数 ------------------------------
    def run_gt(self) -> None:
        """步骤5：计算几何参数（多线程，无超时）"""
        if not self.shaded_jsonl_path or not os.path.exists(self.shaded_jsonl_path):
            raise RuntimeError("未找到着色图形JSONL文件，无法执行GT计算")
        
        logger.info(f"=== 开始计算几何参数（多线程：{self.thread_num}个） ===")
        output_root = self.config['global']['output_root']

        gt_output_dir = os.path.join(output_root, 'json/final')
        self.gt_jsonl_path = os.path.join(gt_output_dir, f'shaded_with_gt_{self._get_timestamp()}.jsonl')
        os.makedirs(gt_output_dir, exist_ok=True)
        with open(self.gt_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 构建任务列表
        task_list = []
        with open(self.shaded_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    json_data = json.loads(line)
                    # 确保enhance_id存在
                    if not json_data.get('enhance_id'):
                        base_idx = json_data.get('base_idx', 0)
                        enhance_idx = json_data.get('enhance_idx', 0)
                        json_data['enhance_id'] = f"base_{base_idx:03d}_enhance_{enhance_idx:03d}"
                    task_list.append((f"gt-{line_num}", json_data))

        def gt_task(task):
            task_id, json_data = task
            calculator = GeometryCalculator()
            result = calculator.calculate_single(json_data)
            # 保留核心标识
            result['enhance_id'] = json_data['enhance_id']
            result['base_id'] = json_data.get('base_id')
            result['base_idx'] = json_data.get('base_idx')
            result['enhance_idx'] = json_data.get('enhance_idx')
            return result

        # 多线程处理
        gt_results, success_count = self._parallel_process(
            task_list=task_list,
            task_func=gt_task,
            output_path=self.gt_jsonl_path,
            has_timeout=False
        )

        self.enhanced_jsons = gt_results
        logger.info(f"几何参数计算完成：成功 {success_count}/{len(task_list)} 个图形，结果保存至: {self.gt_jsonl_path}")

    # ------------------------------ 步骤6：生成问答数据 ------------------------------
    def run_qa(self) -> None:
        """步骤6：生成问答数据（多线程，无超时）"""
        if not self.gt_jsonl_path or not os.path.exists(self.gt_jsonl_path):
            raise RuntimeError("未找到GT图形JSONL文件，无法执行问答生成")
        
        logger.info(f"=== 开始生成问答数据（多线程：{self.thread_num}个） ===")
        qa_config = self.config['qa']
        output_root = self.config['global']['output_root']

        qa_output_dir = os.path.join(output_root, 'qa')
        self.qa_jsonl_path = os.path.join(qa_output_dir, f'qa_{self._get_timestamp()}.jsonl')
        os.makedirs(qa_output_dir, exist_ok=True)
        with open(self.qa_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 构建任务列表
        task_list = []
        with open(self.gt_jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    json_data = json.loads(line)
                    task_list.append((f"qa-{line_num}", json_data))

        def qa_task(task):
            task_id, json_data = task
            qa_generator = QAGenerator(qa_config)
            qa_pairs = qa_generator.generate(
                geo_data=json_data,
                num_questions=qa_config.get('num_questions_per_geo', 3),
                question_types=qa_config.get('question_types', ['length', 'comparison', 'sum', 'ratio'])
            )
            # 关联核心标识
            for qa in qa_pairs:
                qa['enhance_id'] = json_data['enhance_id']
                qa['base_id'] = json_data.get('base_id')
                qa['base_idx'] = json_data.get('base_idx')
                qa['enhance_idx'] = json_data.get('enhance_idx')
            return qa_pairs[0]

        # 多线程处理
        qa_results, success_count = self._parallel_process(
            task_list=task_list,
            task_func=qa_task,
            output_path=self.qa_jsonl_path,
            has_timeout=False
        )

        logger.info(f"问答生成完成：成功 {success_count}/{len(task_list)} 个图形，生成 {len(qa_results)} 个问答对，结果保存至: {self.qa_jsonl_path}")

    # ------------------------------ 全流程执行 ------------------------------
    def run(self) -> None:
        """执行全流程（步骤间串行，步骤内并行）"""
        try:
            self.run_template()      # 步骤1：基础图形（多线程+超时）
            self.run_builder()       # 步骤2：增强图形（多线程+超时，保留部分结果）
            self.run_drawer()        # 步骤3：绘制图像（多线程，规范命名）
            self.run_shader()        # 步骤4：着色标注（多线程，规范命名）
            self.run_gt()            # 步骤5：GT计算（多线程）
            self.run_qa()            # 步骤6：问答生成（多线程）
            logger.info("=== 全流程执行完成 ===")
        except Exception as e:
            logger.error(f"流程执行失败: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoChain图形生成与处理全流程（所有流程多线程+指定步骤超时）")
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/afs/jingjinhao/project/GeoChain/MathGeo/scripts/config.json",
        help="总配置文件路径"
    )
    args = parser.parse_args()

    # 启动全流程
    pipeline = GeoChainPipeline(args.config)
    pipeline.run()