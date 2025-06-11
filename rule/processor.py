# processor.py
import pandas as pd
import httpx
import time
import json
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from config import CSV_FILE_PATH, TIMEOUT, MAX_RETRIES, RETRY_DELAY, MAX_WORKERS, RESULTS_FILE_PATH

def process_row(row, agent, logger):
    title = row['标题']
    content = row['正文']
    human_result = row['人工结果']
    predicted_result = row['预测结果']

    # Validate title and content
    if pd.isna(title) or pd.isna(content):
        logger.error(f"跳过无效行，标题: {title}, 正文: {content}，原因: 标题或正文为空或NaN")
        return {
            'title': str(title) if not pd.isna(title) else '未知标题',
            'verdict': predicted_result if not pd.isna(predicted_result) else '未知',
            'analysis': "无效行：标题或正文为空或NaN"
        }

    logger.debug(f"处理标题数据: {title}, 正文长度: {len(str(content))} 字符")

    # 构造分析提示词
    prompt = f"""
    分析标题为“{title}”的戏剧脚本内容：

    **正文**：
    {content}

    **预测结果**：
    - 人工结果：{human_result}
    - 预测结果：{predicted_result}


    请提供详细分析，涵盖：
    1. 内容合规性（根据审核风险标准）
    2. 剧情与人物（主线、矛盾、人设、关系）
    3. 爽点与投流（羞辱点、爽点、名场面、开篇、一卡）
    4. 核心元素与情感代餐

    输出审核结论（必须为“{predicted_result}”），逐维度说明通过或不通过的原因，引用脚本具体内容，并提供改进建议。输出格式为JSON结构。
    """
    logger.debug(f"提示词构造完成，标题: {title}, 长度: {len(prompt)} 字符")

    # 运行代理
    start_time = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                result = agent.run_sync(prompt)
                execution_time = time.time() - start_time
                logger.info(f"分析完成，标题: {title}, 执行时间: {execution_time:.2f} 秒")
                logger.debug(f"分析输出: {result.output[:200]}...")
                return {
                    'title': title,
                    'verdict': predicted_result,
                    'analysis': result.output
                }
        except Exception as e:
            logger.error(f"分析失败，标题: {title}, 尝试 {attempt}/{MAX_RETRIES}, 错误: {str(e)}")
            if attempt == MAX_RETRIES:
                logger.error(f"达到最大重试次数，标题: {title}")
                return {
                    'title': title,
                    'verdict': predicted_result,
                    'analysis': f"分析失败: {str(e)}"
                }
            time.sleep(RETRY_DELAY)

def json_to_csv(json_data, csv_file_path, logger):
    try:
        # Flatten JSON data for CSV with Chinese field names, combining status and reason
        flattened_data = []
        for item in json_data:
            row = {
                '标题': item['title'],
                '审核结论': item['verdict'],
                '合规性_分析': '',
                '剧情_分析': '',
                '人物_分析': '',
                '爽点_分析': '',
                '市场适应性_分析': '',
                '核心元素_分析': '',
                '情感代餐_分析': '',
                '改进建议': ''
            }
            if isinstance(item['analysis'], str) and (item['analysis'].startswith("分析失败") or item['analysis'].startswith("无效行")):
                row['合规性_分析'] = item['analysis']
                flattened_data.append(row)
                logger.warning(f"跳过无效分析，标题: {item['title']}，原因: {item['analysis']}")
                continue
            try:
                analysis = json.loads(item['analysis'].strip('```json\n').strip('\n```')) if isinstance(item['analysis'], str) else item['analysis']
                row.update({
                    '合规性_分析': f"状态: {analysis['analysis']['compliance']['status']}; 原因: {analysis['analysis']['compliance']['reason']};示例: {analysis['analysis']['compliance']['example']}",
                    '剧情_分析': f"状态: {analysis['analysis']['plot']['status']}; 原因: {analysis['analysis']['plot']['reason']};示例: {analysis['analysis']['plot']['example']}",
                    '人物_分析': f"状态: {analysis['analysis']['characters']['status']}; 原因: {analysis['analysis']['characters']['reason']};示例:{analysis['analysis']['characters']['example']}",
                    '爽点_分析': f"状态: {analysis['analysis']['refreshment']['status']}; 原因: {analysis['analysis']['refreshment']['reason']};示例:{analysis['analysis']['refreshment']['example']}",
                    '市场适应性_分析': f"状态: {analysis['analysis']['marketability']['status']}; 原因: {analysis['analysis']['marketability']['reason']};示例：{analysis['analysis']['marketability']['example']}",
                    '核心元素_分析': f"状态: {analysis['analysis']['core_elements']['status']}; 原因: {analysis['analysis']['core_elements']['reason']};示例：{analysis['analysis']['core_elements']['example']}",
                    '情感代餐_分析': f"状态: {analysis['analysis']['emotional_fulfillment']['status']}; 原因: {analysis['analysis']['emotional_fulfillment']['reason']};示例: {analysis['analysis']['emotional_fulfillment']['example']}",
                    '改进建议': '; '.join(analysis['suggestions'])
                })
                flattened_data.append(row)
            except json.JSONDecodeError as e:
                row['合规性_分析'] = f"JSON 解析失败: {str(e)}"
                flattened_data.append(row)
                logger.error(f"JSON 解析失败，标题: {item['title']}，错误: {str(e)}")

        # Write to CSV
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
            writer.writeheader()
            writer.writerows(flattened_data)
        logger.info(f"JSON 结果已转换为 CSV 文件: {csv_file_path}")
    except Exception as e:
        logger.error(f"转换 JSON 到 CSV 失败: {str(e)}")

def analyze_content(agent, logger):
    logger.debug(f"开始读取 CSV 文件: {CSV_FILE_PATH}")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        logger.info(f"成功读取 CSV 文件，包含 {len(df)} 行数据")
    except Exception as e:
        logger.error(f"读取 CSV 文件失败: {str(e)}")
        return []

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_row = {executor.submit(process_row, row, agent, logger): row for _, row in df.iterrows()}
        for future in future_to_row:
            try:
                result = future.result()
                if result:  # Ensure result is not None
                    results.append(result)
                    logger.debug(f"完成处理标题: {result['title']}")
            except Exception as e:
                logger.error(f"线程处理失败，标题: {future_to_row[future]['标题']}, 错误: {str(e)}")

    # 保存结果到 JSON 文件
    try:
        with open(RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"分析结果已保存到: {RESULTS_FILE_PATH}")
    except Exception as e:
        logger.error(f"保存结果到 {RESULTS_FILE_PATH} 失败: {str(e)}")

    # 转换为 CSV 文件
    csv_file_path = RESULTS_FILE_PATH.replace('.json', '.csv')
    json_to_csv(results, csv_file_path, logger)

    logger.info(f"分析完成，共处理 {len(results)} 条记录")
    return results