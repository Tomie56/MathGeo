import os
import json
import random
import logging
import threading
import queue
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
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
from tools.drawer import GeometryDrawer  # 图像绘制器
from tools.shader import EnhancedDrawer # 区域着色器
from tools.gt import GeometryCalculator  # 几何参数计算器（保留接口）
from tools.qa_template import QAGenerator  # 问答生成器（保留接口）


class GeoChainPipeline:
    """图形生成与处理全流程管道（支持多线程+超时限制）"""
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._init_workspace()
        self._set_random_seeds()
        
        # 并行与超时配置
        self.thread_num = self.config['global'].get('thread_num', 4)  # 多线程数量（默认4）
        self.task_timeout = 500  # 单次生成超时时间（20秒）
        
        # 中间数据存储
        self.base_jsonl_path: Optional[str] = None  # 基础图形JSONL文件路径
        self.enhanced_jsons: List[Dict[str, Any]] = []  # 增强图形JSON列表
        self.raw_image_paths: List[str] = []  # 原始图像路径列表
        self.shaded_image_paths: List[str] = []  # 着色图像路径列表（保留接口）
        self.final_image_paths: List[str] = []  # 最终标注图像路径列表（保留接口）
        self.shaded_jsonl_path: Optional[str] = None  # 着色图形JSONL路径（保留接口）
        self.gt_jsonl_path: Optional[str] = None  # 带GT参数的最终JSONL路径

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载并验证总配置文件（补充线程数默认值）"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 验证核心配置项（包含所有可能模块，即使暂时不启用）
            required_sections = ['template', 'builder', 'drawer', 'shader', 'gt', 'annotator', 'qa']
            for section in required_sections:
                if section not in config:
                    logger.warning(f"配置文件缺少模块: {section}，将使用默认配置")
                    config[section] = {}
            # 补充全局线程数默认值
            if 'thread_num' not in config.get('global', {}):
                config['global']['thread_num'] = 4
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {str(e)}")

    def _init_workspace(self) -> None:
        """初始化工作目录结构（包含所有模块所需目录）"""
        out_root = self.config['global']['output_root']
        subdirs = [
            'json/base',          # 基础图形JSON
            'json/enhanced',      # 增强图形JSON
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

    def _run_task_with_timeout(self, task_func, *args, timeout: float = None, **kwargs) -> Tuple[str, Any]:
        """
        执行单个任务，添加超时限制
        返回格式：(状态, 结果) -> 状态：success/timeout/error
        """
        result_queue = queue.Queue(maxsize=1)

        def task_wrapper():
            """任务包装器（捕获异常并返回结果）"""
            try:
                result = task_func(*args, **kwargs)
                result_queue.put(('success', result))
            except Exception as e:
                result_queue.put(('error', e))

        # 启动任务线程
        task_thread = threading.Thread(target=task_wrapper, daemon=True)
        task_thread.start()
        # 等待任务完成或超时
        task_thread.join(timeout=timeout)

        if task_thread.is_alive():
            # 任务超时
            return ('timeout', None)
        else:
            # 任务完成，获取结果
            try:
                return result_queue.get_nowait()
            except queue.Empty:
                return ('error', RuntimeError("任务无返回结果"))

    def run_template(self) -> None:
        """步骤1：生成基础图形JSONL（多线程+单次20秒超时）"""
        logger.info(f"=== 开始生成基础图形（多线程：{self.thread_num}个，超时：{self.task_timeout}秒） ===")
        template_cfg = self.config['template']
        seed = template_cfg.get('seed')
        n = template_cfg.get('n', 10)

        self.base_jsonl_path = os.path.join(
            template_cfg.get('output_dir', os.path.join(self.config['global']['output_root'], 'json/base')),
            f'base_{self._get_timestamp()}.jsonl'
        )

        # 任务队列：存储 (任务索引, 样本种子)
        task_queue = queue.Queue()
        for i in range(n):
            sample_seed = seed + 42 * i if seed is not None else None
            task_queue.put((i + 1, sample_seed))  # i+1 为样本序号

        # 文件写入锁（确保线程安全）
        write_lock = threading.Lock()
        success_count = 0

        def worker():
            """线程工作函数：处理队列中的任务"""
            nonlocal success_count
            while not task_queue.empty():
                try:
                    sample_idx, sample_seed = task_queue.get_nowait()
                    logger.info(f"线程[{threading.current_thread().name}] 处理基础图形样本 {sample_idx}/{n}（种子：{sample_seed}）")

                    # 定义单个样本生成任务
                    def generate_sample():
                        generator = TemplateGenerator(template_cfg, seed=sample_seed)
                        generator.generate_base_shape()
                        generator.generate_derivations()
                        return generator.export_json()

                    # 执行任务（带超时）
                    status, result = self._run_task_with_timeout(generate_sample, timeout=self.task_timeout + 40)

                    if status == 'success':
                        # 线程安全写入文件
                        with write_lock:
                            with open(self.base_jsonl_path, "a", encoding="utf-8") as f:
                                json.dump(result, f, ensure_ascii=False)
                                f.write("\n")
                        logger.info(f"线程[{threading.current_thread().name}] 样本 {sample_idx} 生成成功")
                        success_count += 1
                    elif status == 'timeout':
                        logger.error(f"线程[{threading.current_thread().name}] 样本 {sample_idx} 生成超时（{self.task_timeout}秒），跳过")
                    else:
                        logger.error(f"线程[{threading.current_thread().name}] 样本 {sample_idx} 生成失败：{str(result)}，跳过")
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"线程[{threading.current_thread().name}] 处理任务异常：{str(e)}，跳过")
                finally:
                    task_queue.task_done()

        # 启动线程池
        threads = []
        for i in range(self.thread_num):
            t = threading.Thread(target=worker, name=f"Template-Worker-{i+1}")
            t.start()
            threads.append(t)

        # 等待所有任务完成
        task_queue.join()
        logger.info(f"基础图形生成完成：成功 {success_count}/{n} 个样本")

        if not os.path.exists(self.base_jsonl_path) or os.path.getsize(self.base_jsonl_path) == 0:
            raise RuntimeError("未生成任何有效的基础图形JSONL文件")
        logger.info(f"基础图形JSONL保存至: {self.base_jsonl_path}")

    def run_builder(self) -> None:
        """步骤2：读取基础图形JSONL，生成增强图形（多线程+单次20秒超时）"""
        if not self.base_jsonl_path or not os.path.exists(self.base_jsonl_path):
            raise RuntimeError("未找到基础图形JSONL文件，无法执行增强步骤")
        
        logger.info(f"=== 开始生成增强图形（多线程：{self.thread_num}个，超时：{self.task_timeout}秒） ===")
        builder_cfg = self.config['builder']

        # 构建增强图形输出路径
        output_filename = f'enhanced_{self._get_timestamp()}.jsonl'
        enhanced_jsonl_path = os.path.join(
            builder_cfg.get('output_dir'),
            output_filename
        )
        # 清空输出文件（避免追加旧数据）
        with open(enhanced_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 读取所有基础图形（先加载到内存，避免多线程读取文件冲突）
        base_tasks = []
        with open(self.base_jsonl_path, 'r', encoding='utf-8') as base_f:
            for line_num, line in enumerate(base_f, 1):
                line = line.strip()
                if line:
                    task_seed = builder_cfg['seed'] + line_num * 42  # 每个任务独立种子
                    base_tasks.append((line_num, line, task_seed))

        if not base_tasks:
            raise RuntimeError("基础图形JSONL文件为空，无法执行增强步骤")

        # 任务队列：存储 (行号, 基础图形JSON字符串, 任务种子)
        task_queue = queue.Queue()
        for task in base_tasks:
            task_queue.put(task)

        # 线程安全锁（文件写入+结果收集）
        write_lock = threading.Lock()
        self.enhanced_jsons = []
        success_count = 0
        total_tasks = len(base_tasks)

        def worker():
            """线程工作函数：处理单个基础图形的增强"""
            nonlocal success_count
            while not task_queue.empty():
                try:
                    line_num, line, task_seed = task_queue.get_nowait()
                    logger.info(f"线程[{threading.current_thread().name}] 处理基础图形 #{line_num}")

                    # 定义单个增强任务
                    def enhance_task():
                        base_json = json.loads(line)
                        base_builder = RandomGeometryBuilder(base_json)
                        return base_builder.generate_enhancements({
                            "rounds_distribution": builder_cfg.get('rounds_distribution', {1: 1}),
                            "min_operations_per_round": builder_cfg.get('min_operations_per_round', 1),
                            "max_operations_per_round": builder_cfg.get('max_operations_per_round', 1),
                            "allowed_operation_types": builder_cfg.get('allowed_operation_types', 
                                                                    ['connect_points', 'connect_midpoints', 'draw_perpendicular', 'draw_diameter']),
                            "operation_probs": builder_cfg.get('operation_probs', {}), 
                            "operation_constraints": builder_cfg.get('operation_constraints', {}),
                            "seed": task_seed
                        })

                    # 执行任务（带超时）
                    status, result = self._run_task_with_timeout(enhance_task, timeout=self.task_timeout)

                    if status == 'success':
                        enhancements = result
                        # 线程安全写入文件和收集结果
                        with write_lock:
                            with open(enhanced_jsonl_path, 'a', encoding='utf-8') as enhanced_f:
                                for enh_json in enhancements:
                                    json.dump(enh_json, enhanced_f, ensure_ascii=False)
                                    enhanced_f.write("\n")
                                    self.enhanced_jsons.append(enh_json)
                        logger.info(f"线程[{threading.current_thread().name}] 基础图形 #{line_num} 增强成功（{len(enhancements)}个结果）")
                        success_count += 1
                    elif status == 'timeout':
                        logger.error(f"线程[{threading.current_thread().name}] 基础图形 #{line_num} 增强超时（{self.task_timeout}秒），跳过")
                    else:
                        logger.error(f"线程[{threading.current_thread().name}] 基础图形 #{line_num} 增强失败：{str(result)}，跳过")
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"线程[{threading.current_thread().name}] 处理基础图形 #{line_num} 异常：{str(e)}，跳过")
                finally:
                    task_queue.task_done()

        # 启动线程池
        threads = []
        for i in range(self.thread_num):
            t = threading.Thread(target=worker, name=f"Builder-Worker-{i+1}")
            t.start()
            threads.append(t)

        # 等待所有任务完成
        task_queue.join()
        logger.info(f"增强图形生成完成：成功处理 {success_count}/{total_tasks} 个基础图形")

        if not self.enhanced_jsons:
            raise RuntimeError("未生成任何有效的增强图形")
        
        # 更新drawer配置中的输入路径
        self.config['drawer']['jsonl_path'] = enhanced_jsonl_path
        logger.info(f"增强图形JSONL保存至: {enhanced_jsonl_path}（共{len(self.enhanced_jsons)}个增强图形）")

    def run_drawer(self) -> None:
        """步骤3：绘制原始图像"""
        if not self.enhanced_jsons:
            raise RuntimeError("未生成增强图形，无法执行绘图步骤")
        
        logger.info("=== 开始绘制原始图像 ===")
        drawer_cfg = self.config['drawer']
        # annotator_cfg = self.config['annotator']

        drawer = GeometryDrawer(drawer_cfg)
        self.raw_image_paths = drawer.batch_draw()
        logger.info(f"原始图像保存完成（共 {len(self.raw_image_paths)} 张）")

    def run_shader(self) -> None:
        """步骤4：区域着色与标注（多线程实现，无超时，基于process_single处理单条数据）"""
        # 1. 提取核心配置参数
        shader_config = self.config['shader']
        annotator_config = self.config.get('annotator', {})
        output_root = self.config['global']['output_root']
        shader_enabled = shader_config.get('enabled', False)
        thread_num = self.config['global'].get('thread_num', 4)  # 复用全局线程数配置

        # 2. 输出路径配置
        json_output_dir = os.path.join(output_root, 'json/shaded')
        os.makedirs(json_output_dir, exist_ok=True)
        self.shaded_jsonl_path = os.path.join(
            json_output_dir, 
            f'shaded_{self._get_timestamp()}.jsonl'
        )
        # 清空输出文件（避免追加旧数据）
        with open(self.shaded_jsonl_path, 'w', encoding='utf-8') as f:
            pass

        # 3. 图像输出目录（确保存在）
        shaded_img_dir = os.path.join(output_root, 'images/shaded')
        annotated_img_dir = os.path.join(output_root, 'images/annotated')
        os.makedirs(shaded_img_dir, exist_ok=True)
        os.makedirs(annotated_img_dir, exist_ok=True)

        # 4. 准备任务数据（每条增强数据作为独立任务）
        if not self.enhanced_jsons:
            raise RuntimeError("未找到增强图形数据（enhanced_jsons为空），无法执行着色步骤")
        total_tasks = len(self.enhanced_jsons)
        logger.info(f"=== 开始区域着色与标注（多线程：{thread_num}个，总任务数：{total_tasks}） ===")

        # 任务队列：存储（任务索引，单条增强数据）
        task_queue = queue.Queue()
        for idx, data in enumerate(self.enhanced_jsons):
            task_queue.put((idx, data))

        # 5. 线程安全工具（结果收集+文件写入锁）
        write_lock = threading.Lock()  # 确保JSONL写入线程安全
        success_count = 0  # 成功处理的任务数
        failed_indices = []  # 失败的任务索引

        # 6. 线程工作函数（处理单条数据）
        def worker():
            nonlocal success_count, failed_indices
            # 每个线程创建独立的EnhancedDrawer实例（避免状态共享）
            tool_config = {
                "global": {"output_root": output_root},
                "drawer": self.config['drawer'],
                "shader": {** shader_config, 
                    "enabled": shader_enabled,
                    "jsonl_path": self.shaded_jsonl_path
                },
                "annotator": annotator_config,
                "region_extractor": self.config.get("region_extractor", {})
            }
            # 初始化当前线程的EnhancedDrawer
            thread_drawer = EnhancedDrawer(tool_config)

            while not task_queue.empty():
                try:
                    task_idx, data = task_queue.get_nowait()
                    enhance_id = data.get("enhance_id", f"task_{task_idx}")
                    logger.debug(f"线程[{threading.current_thread().name}] 处理任务 {task_idx+1}/{total_tasks}（enhance_id: {enhance_id}）")

                    # 设置单条数据并调用process_single
                    thread_drawer.set_enhanced_data([data])  # 传入长度为1的列表（符合process_single要求）
                    result = thread_drawer.process_single()  # 调用单条处理方法

                    # 线程安全写入结果到JSONL
                    with write_lock:
                        with open(self.shaded_jsonl_path, 'a', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False)
                            f.write("\n")

                    success_count += 1
                    logger.debug(f"线程[{threading.current_thread().name}] 任务 {task_idx+1} 处理成功（enhance_id: {enhance_id}）")

                except queue.Empty:
                    break  # 队列空则退出
                except Exception as e:
                    # 记录失败任务索引，不中断其他任务
                    failed_indices.append(task_idx)
                    logger.error(
                        f"线程[{threading.current_thread().name}] 任务 {task_idx+1} 处理失败（enhance_id: {enhance_id}）：{str(e)}",
                        exc_info=True
                    )
                finally:
                    task_queue.task_done()  # 标记任务完成

        # 7. 启动线程池
        threads = []
        for i in range(thread_num):
            t = threading.Thread(target=worker, name=f"Shader-Worker-{i+1}")
            t.start()
            threads.append(t)

        # 8. 等待所有任务完成
        task_queue.join()
        # 确认所有线程已结束
        for t in threads:
            t.join()

        # 9. 结果校验与日志
        if success_count == 0:
            raise RuntimeError("所有着色与标注任务均失败，未生成任何有效结果")
        
        logger.info(
            f"=== 区域着色与标注完成 ==="
            f"\n- 总任务数：{total_tasks}"
            f"\n- 成功：{success_count}"
            f"\n- 失败：{len(failed_indices)}（索引：{failed_indices if failed_indices else '无'}）"
            f"\n- 结果JSONL路径：{self.shaded_jsonl_path}"
        )
          
    def run_gt(self) -> None:
        """步骤5：计算几何参数（基于符号表达式，结果为化简后的表达式）"""
        # 检查 shader 处理结果是否存在
        if not self.shaded_jsonl_path or not os.path.exists(self.shaded_jsonl_path):
            raise RuntimeError("未找到 shader 处理后的 JSONL 文件，无法执行几何参数计算")
        
        logger.info("=== 开始计算几何参数（符号表达式化简） ===")
        calculator = GeometryCalculator()
        
        # 生成输出路径（在原有路径基础上添加 _with_gt 标识）
        gt_output_dir = os.path.join(self.config['global']['output_root'], 'json/final')
        os.makedirs(gt_output_dir, exist_ok=True)
        
        # 提取原始文件名并添加 GT 标识
        shaded_filename = os.path.basename(self.shaded_jsonl_path)
        gt_filename = shaded_filename.replace('shaded_', 'shaded_with_gt_')
        self.gt_jsonl_path = os.path.join(gt_output_dir, gt_filename)
        
        try:
            # 调用计算器批量处理 JSONL 文件
            calculator.process_jsonl(
                input_path=self.shaded_jsonl_path,
                output_path=self.gt_jsonl_path
            )
            
            # 读取处理后的结果更新到内存（可选，根据后续流程需求）
            self.enhanced_jsons = []
            with open(self.gt_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.enhanced_jsons.append(json.loads(line))
            
            logger.info(f"几何参数计算完成，结果保存至: {self.gt_jsonl_path}")
            logger.info(f"共处理 {len(self.enhanced_jsons)} 条数据，每条数据已添加线条/圆弧长度信息")
            
        except Exception as e:
            logger.error(f"几何参数计算流程失败: {str(e)}", exc_info=True)
            raise

    def run_qa(self) -> None:
        """步骤6：基于GT计算结果生成问答数据（直接读取self.gt_jsonl_path）"""
        # 检查GT处理结果路径是否有效
        if not hasattr(self, 'gt_jsonl_path') or not self.gt_jsonl_path or not os.path.exists(self.gt_jsonl_path):
            raise RuntimeError("GT计算结果路径无效或不存在，请先执行run_gt生成结果")
        
        logger.info(f"使用GT结果文件: {self.gt_jsonl_path}")

        # 解析qa配置参数
        qa_config = self.config['qa']
        output_dir = qa_config.get('output_dir', os.path.join(self.config['global']['output_root'], 'qa'))
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        
        # 生成输出文件名（基于GT文件名和时间戳）
        gt_basename = os.path.splitext(os.path.basename(self.gt_jsonl_path))[0]
        timestamp = self._get_timestamp()
        qa_output_path = os.path.join(output_dir, f"qa_{gt_basename}.jsonl")

        try:
            # 初始化问答生成器
            qa_generator = QAGenerator(qa_config)
            logger.info(f"=== 开始生成问答数据 ===")
            logger.info(f"QA配置参数: {json.dumps(qa_config, ensure_ascii=False, indent=2)}")

            # 处理GT结果JSONL，逐行生成问答
            processed_count = 0
            total_qa = 0
            with open(self.gt_jsonl_path, 'r', encoding='utf-8') as f_in, \
                open(qa_output_path, 'w', encoding='utf-8') as f_out:

                for line_num, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # 解析单条带GT参数的几何数据
                        geo_data = json.loads(line)
                        
                        # 生成问答对（数量和类型由qa_config控制）
                        qa_pairs = qa_generator.generate(
                            geo_data=geo_data,
                            num_questions=qa_config.get('num_questions_per_geo', 3),
                            question_types=qa_config.get('question_types', ['length', 'comparison', 'sum', 'ratio']) 
                        )
                        
                        # 写入输出文件（每条问答对一行）
                        for qa in qa_pairs:
                            
                            json.dump(qa, f_out, ensure_ascii=False)
                            f_out.write('\n')
                        
                        processed_count += 1
                        total_qa += len(qa_pairs)
                        if line_num % 10 == 0:
                            logger.info(f"已处理 {line_num} 条几何数据，累计生成 {total_qa} 个问答对")

                    except json.JSONDecodeError:
                        logger.error(f"GT文件第 {line_num} 行JSON格式错误，跳过")
                        continue
                    except Exception as e:
                        logger.error(f"处理第 {line_num} 条数据时生成问答失败: {str(e)}，跳过")
                        continue

            logger.info(f"=== 问答生成完成 ===")
            logger.info(f"有效处理几何数据: {processed_count} 条")
            logger.info(f"生成问答对总数: {total_qa} 个")
            logger.info(f"问答结果保存至: {qa_output_path}")

        except Exception as e:
            logger.error(f"问答生成流程失败: {str(e)}", exc_info=True)
            raise

    def run(self) -> None:
        """执行全流程"""
        try:
            self.run_template()      # 生成基础图形（多线程+超时）
            self.run_builder()       # 增强图形（多线程+超时）
            # self.run_drawer()        # 绘制原始图像
            self.run_shader()        # 区域阴影与标注
            self.run_gt()            # 计算参数
            self.run_qa()            # 生成问答
            logger.info("=== 全流程执行完成 ===")
        except Exception as e:
            logger.error(f"流程执行失败: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoChain图形生成与处理全流程（支持多线程+超时）")
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