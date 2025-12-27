"""
任务状态检查器
查询MongoDB中的任务状态
"""
import logging
from common.mongodb_connector import MongoDBConnector
from common.config import MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASSWORD

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('literature_review_service.log')
    ]
)
logger = logging.getLogger("state_checker")


def get_task_state_service(user_id: str, task_id: str, task_type: str = "review") -> dict:
    """
    查询任务的状态和结果，支持不同类型的任务

    Args:
        user_id: 用户ID
        task_id: 任务ID
        task_type: 任务类型，可选值包括：
                  "review", "research_data", "hypothesis", "introduction", 
                  "ai_data", "full_paper", "papers", "batch_download",
                  "risk_diagnose", "quote_match", "info_extraction"

    Returns:
        dict: 包含任务状态和结果信息的字典
    """
    logger.info(f"查询任务状态: 用户 {user_id}, 任务 {task_id}, 类型 {task_type}")

    try:
        # 创建MongoDB连接器
        mongo_connector = MongoDBConnector()

        # 切换到Newbit数据库
        db = mongo_connector.client["Newbit"]

        # 根据任务类型选择不同的集合
        collection_map = {
            "research_data": "research_data",
            "hypothesis": "hypothesis",
            "introduction": "introduction",
            "ai_data": "ai_data",
            "full_paper": "full_paper",
            "papers": "papers",
            "batch_download": "download_tasks",
            "risk_diagnose": "risk_diagnose",
            "quote_match": "quote_match",
            "info_extraction": "info_extraction",
            "review": "review"  # 默认
        }

        collection_name = collection_map.get(task_type, "review")
        collection = db[collection_name]

        # 设置投影字段
        if task_type == "ai_data":
            # ai_data类型额外排除questionnaire和results字段
            projection = {
                "_id": 0,
                "user_id": 0,
                "task_id": 0,
                "questionnaire": 0,
                "results": 0
            }
        else:
            projection = {"_id": 0, "user_id": 0, "task_id": 0}

        # 查询任务
        task = collection.find_one(
            {"user_id": user_id, "task_id": task_id},
            projection
        )

        if not task:
            logger.warning(f"未找到任务: 用户 {user_id}, 任务 {task_id}, 类型 {task_type}")
            return {
                "status": "未找到",
                "status_code": -1,
                "task_type": task_type
            }

        # 统一的状态映射
        status_map = {0: "失败", 1: "成功", 2: "进行中"}
        status_code = task.get("state", -1)

        # 根据任务类型获取查询字段名称
        query_field_map = {
            "hypothesis": "hypothesis",
            "papers": "paper_title"
        }
        query_field = query_field_map.get(task_type, "query")

        # 构建基本返回结果
        result = {
            "status": status_map.get(status_code, "未知"),
            "status_code": status_code,
            "query": task.get(query_field, ""),
            "update_time": task.get("update_time", ""),
            "task_type": task_type
        }

        # 根据状态添加不同信息
        if status_code == 0:  # 失败
            result["error"] = task.get("error", "")
            result["message"] = task.get("message", "")
            logger.info(f"任务失败: {task_id}, 错误: {result['error']}")

        elif status_code == 1:  # 成功
            # 根据不同任务类型添加特定字段
            if task_type == "research_data":
                if "research_data" in task and task["research_data"] != "{}":
                    result["research_data"] = task.get("research_data", "{}")
                    result["meta_data"] = task.get("meta_data", [])
                else:
                    result["message"] = task.get("message", "未找到相关研究数据")
                    result["meta_data"] = task.get("meta_data", [])

            elif task_type == "hypothesis":
                result.update({
                    "cot": task.get("cot", ""),
                    "draft": task.get("draft", ""),
                    "main_text": task.get("main_text", ""),
                    "references": task.get("references", []),
                    "hypothesis_text": task.get("hypothesis_text", ""),
                    "pdf_urls": task.get("pdf_urls", [])
                })

            elif task_type == "introduction":
                result.update({
                    "query": task.get("research_topic", ""),
                    "paragraph1": task.get("paragraph1", ""),
                    "paragraph2": task.get("paragraph2", ""),
                    "paragraph3": task.get("paragraph3", ""),
                    "references2": task.get("references2", []),
                    "references3": task.get("references3", []),
                    "pdf_urls2": task.get("pdf_urls2", []),
                    "pdf_urls3": task.get("pdf_urls3", []),
                    "main_text": task.get("main_text", ""),
                    "complete_text": task.get("complete_text", "")
                })

            elif task_type == "ai_data":
                result.update({
                    "excel_url": task.get("excel_url", ""),
                    "markdown_table": task.get("markdown_table", ""),
                    "successful_count": task.get("successful_count", [])
                })

            elif task_type == "full_paper":
                result.update({
                    "hypotheses": task.get("hypotheses", {}),
                    "introduction": task.get("introduction", {}),
                    "literature_review": task.get("literature_review", {}),
                    "content": task.get("content", []),
                    "pdf_urls": task.get("pdf_urls", [])
                })

            elif task_type == "papers":
                result.update({
                    "full_paper_text": task.get("full_paper_text", {}),
                    "all_references": task.get("all_references", {}),
                    "all_pdf_urls": task.get("all_pdf_urls", {})
                })

            elif task_type == "batch_download":
                result.update({
                    "download_url": task.get("download_url", ""),
                    "total_files": task.get("total_files", 0),
                    "message": task.get("message", "")
                })

            elif task_type == "risk_diagnose":
                result.update({
                    "errors": task.get("errors", ""),
                    "error_paragraphs": task.get("error_paragraphs", 0)
                })

            elif task_type == "quote_match":
                result.update({
                    "results": task.get("results", ""),
                    "matched_quotes": task.get("matched_quotes", 0)
                })

            elif task_type == "info_extraction":
                result.update({
                    "result": task.get("result", "")
                })

            else:  # review
                result["review_text"] = task.get("review_text", {})
                if "processed_review_text" in task:
                    result["processed_review_text"] = task.get("processed_review_text", "")

            logger.info(f"任务成功: {task_id}, 类型: {task_type}")

        return result

    except Exception as e:
        logger.error(f"查询任务状态出错: 用户 {user_id}, 任务 {task_id}, 类型 {task_type}, 错误: {str(e)}")
        return {
            "status": "查询出错",
            "error": str(e),
            "status_code": -2,
            "task_type": task_type
        }