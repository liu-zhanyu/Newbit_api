"""
数据检索器
协调整个数据检索流程：变量提取 → 指标搜索 → 数据检索
"""
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from chat_with_data.preprocessing.variable_extractor import extract_variables
from chat_with_data.preprocessing.indicator_suggester import suggest_measurement_indicators
from chat_with_data.retrieval.indicator_searcher import search_indicators, extract_indicator_names
from common.mongodb_connector import MongoDBConnector


class ResearchDataRetriever:
    """研究数据检索器类"""

    def __init__(self):
        """初始化数据检索器"""
        self.mongo_connector = MongoDBConnector()

    def retrieve_data_for_query(
            self,
            query: str,
            top_k_per_variable: int = 3
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        基于研究查询检索相关数据

        工作流程：
        1. 提取核心变量和控制变量
        2. 为每个变量推荐测量指标
        3. 在ES中搜索指标
        4. 从MongoDB批量检索数据

        Args:
            query: 研究查询
            top_k_per_variable: 每个变量检索的指标数量

        Returns:
            (数据DataFrame, 元数据字典)
            元数据包含：
            - core_variables: 核心变量列表
            - control_variables: 控制变量列表
            - suggested_indicators: 推荐的指标
            - found_indicators: 实际找到的指标
        """
        print(f"开始为查询检索数据: {query}")

        # 第一步：提取变量
        print("\n=== 第一步：提取变量 ===")
        core_variables, control_variables = extract_variables(query)

        if not core_variables:
            print("未能提取到核心变量，终止检索")
            return pd.DataFrame(), {}

        all_variables = core_variables + control_variables

        # 第二步：推荐指标
        print("\n=== 第二步：推荐测量指标 ===")
        suggested_indicators = suggest_measurement_indicators(all_variables, context=query)

        # 第三步：搜索指标
        print("\n=== 第三步：在ES中搜索指标 ===")
        found_indicators = {}
        all_indicator_names = []

        # 使用多线程并行搜索
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_variable = {
                executor.submit(search_indicators, var, top_k_per_variable): var
                for var in all_variables
            }

            for future in as_completed(future_to_variable):
                variable = future_to_variable[future]
                try:
                    search_results = future.result()
                    indicator_names = extract_indicator_names(search_results)
                    found_indicators[variable] = {
                        "indicator_names": indicator_names,
                        "search_results": search_results
                    }
                    all_indicator_names.extend(indicator_names)
                    print(f"变量 '{variable}' 找到 {len(indicator_names)} 个指标")
                except Exception as e:
                    print(f"搜索变量 '{variable}' 的指标时出错: {e}")
                    found_indicators[variable] = {
                        "indicator_names": [],
                        "search_results": []
                    }

        # 去重指标名称
        all_indicator_names = list(set(all_indicator_names))
        print(f"\n总共找到 {len(all_indicator_names)} 个唯一指标")

        if not all_indicator_names:
            print("未找到任何指标，返回空数据")
            return pd.DataFrame(), {
                "core_variables": core_variables,
                "control_variables": control_variables,
                "suggested_indicators": suggested_indicators,
                "found_indicators": found_indicators
            }

        # 第四步：从MongoDB检索数据
        print("\n=== 第四步：从MongoDB检索数据 ===")
        df = self.mongo_connector.get_data_for_all_indicators(all_indicator_names)

        # 构建元数据
        metadata = {
            "core_variables": core_variables,
            "control_variables": control_variables,
            "suggested_indicators": suggested_indicators,
            "found_indicators": found_indicators,
            "total_indicators": len(all_indicator_names),
            "data_shape": df.shape if not df.empty else (0, 0)
        }

        print(f"\n检索完成: 数据形状 {df.shape}")

        return df, metadata

    def retrieve_data_for_indicators(
            self,
            indicator_names: List[str]
    ) -> pd.DataFrame:
        """
        直接根据指标名称列表检索数据

        Args:
            indicator_names: 指标名称列表

        Returns:
            数据DataFrame
        """
        if not indicator_names:
            return pd.DataFrame()

        print(f"检索 {len(indicator_names)} 个指标的数据")
        df = self.mongo_connector.get_data_for_all_indicators(indicator_names)

        return df