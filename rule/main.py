# main.py
import json
from logger import setup_logger
from agent import create_agent
from processor import analyze_content
from config import RESULTS_FILE_PATH

if __name__ == "__main__":
    logger = setup_logger()
    logger.debug("程序启动")
    try:
        agent = create_agent()
        analysis_results = analyze_content(agent, logger)
        for result in analysis_results:
            logger.debug(f"输出分析结果，标题: {result['title']}")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        csv_file_path = RESULTS_FILE_PATH.replace('.json', '.csv')
        logger.info(f"所有分析结果已输出并保存到 JSON 文件: {RESULTS_FILE_PATH} 和 CSV 文件: {csv_file_path}")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
    logger.debug("程序结束")