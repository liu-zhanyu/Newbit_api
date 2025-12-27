"""
研究数据导出器
异步生成Excel文件并上传到OSS
"""
import pandas as pd
import asyncio
from typing import Dict, Any
from datetime import datetime
from common.oss_handler import OSSHandler
from common.mongodb_connector import MongoDBConnector


async def async_generate_research_data(
        query: str,
        user_id: str,
        task_id: str
) -> Dict[str, Any]:
    """
    异步生成研究数据Excel文件

    工作流程：
    1. 检索数据（通过ResearchDataRetriever）
    2. 生成Excel文件
    3. 上传到OSS
    4. 更新MongoDB中的任务状态

    Args:
        query: 研究查询
        user_id: 用户ID
        task_id: 任务ID

    Returns:
        任务结果字典
    """
    from chat_with_data.retrieval.data_retriever import ResearchDataRetriever

    # 初始化组件
    data_retriever = ResearchDataRetriever()
    oss_handler = OSSHandler()
    mongo_connector = MongoDBConnector()

    try:
        # 更新任务状态为处理中
        mongo_connector.update_one(
            query={"task_id": task_id, "user_id": user_id},
            update={
                "$set": {
                    "status": "processing",
                    "update_time": datetime.now()
                }
            },
            upsert=True
        )

        # 第一步：检索数据
        print(f"任务 {task_id}: 开始检索数据")
        df, metadata = data_retriever.retrieve_data_for_query(query, top_k_per_variable=3)

        if df.empty:
            # 未找到数据
            mongo_connector.update_one(
                query={"task_id": task_id},
                update={
                    "$set": {
                        "status": "completed",
                        "error": "未找到相关数据",
                        "update_time": datetime.now()
                    }
                }
            )
            return {
                "status": "failed",
                "error": "未找到相关数据"
            }

        # 第二步：生成Excel文件
        print(f"任务 {task_id}: 生成Excel文件")

        # 创建临时文件
        import tempfile
        import os

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            # 写入Excel
            with pd.ExcelWriter(temp_file_path, engine='openpyxl') as writer:
                # 写入数据
                df.to_excel(writer, sheet_name='数据', index=False)

                # 写入元数据
                metadata_df = pd.DataFrame([
                    {"项目": "核心变量", "内容": ", ".join(metadata.get("core_variables", []))},
                    {"项目": "控制变量", "内容": ", ".join(metadata.get("control_variables", []))},
                    {"项目": "指标数量", "内容": metadata.get("total_indicators", 0)},
                    {"项目": "数据行数", "内容": df.shape[0]},
                    {"项目": "数据列数", "内容": df.shape[1]},
                    {"项目": "生成时间", "内容": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                ])
                metadata_df.to_excel(writer, sheet_name='元数据', index=False)

            # 第三步：上传到OSS
            print(f"任务 {task_id}: 上传文件到OSS")

            oss_file_path = f"research_data/{user_id}/{task_id}.xlsx"
            oss_url = oss_handler.upload_file(temp_file_path, oss_file_path)

            if not oss_url:
                raise Exception("文件上传OSS失败")

            # 第四步：更新MongoDB状态
            mongo_connector.update_one(
                query={"task_id": task_id},
                update={
                    "$set": {
                        "status": "completed",
                        "oss_url": oss_url,
                        "metadata": metadata,
                        "data_shape": list(df.shape),
                        "update_time": datetime.now()
                    }
                }
            )

            print(f"任务 {task_id}: 完成")

            return {
                "status": "success",
                "task_id": task_id,
                "oss_url": oss_url,
                "metadata": metadata
            }

        finally:
            # 删除临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        print(f"任务 {task_id} 失败: {e}")

        # 更新失败状态
        mongo_connector.update_one(
            query={"task_id": task_id},
            update={
                "$set": {
                    "status": "failed",
                    "error": str(e),
                    "update_time": datetime.now()
                }
            }
        )

        return {
            "status": "failed",
            "error": str(e)
        }