"""
MongoDB连接器
处理与MongoDB的连接和基本查询操作
"""
from pymongo import MongoClient
from typing import Optional, Dict, Any, List
import pandas as pd
from common.config import (
    MONGO_HOST,
    MONGO_PORT,
    MONGO_USER,
    MONGO_PASSWORD,
    MONGO_DB,
    MONGO_COLLECTION
)


class MongoDBConnector:
    """MongoDB连接器类"""

    def __init__(self, db_name: Optional[str] = None, collection_name: Optional[str] = None):
        """
        初始化MongoDB连接器

        Args:
            db_name: 数据库名称，默认使用配置中的MONGO_DB
            collection_name: 集合名称，默认使用配置中的MONGO_COLLECTION
        """
        self.client = None
        self.db = None
        self.collection = None
        self.db_name = db_name or MONGO_DB
        self.collection_name = collection_name or MONGO_COLLECTION
        self.connect()

    def connect(self) -> bool:
        """建立MongoDB连接"""
        try:
            connection_string = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}"
            self.client = MongoClient(connection_string)

            # 测试连接
            self.client.server_info()

            # 选择数据库和集合
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]

            print(f"✓ 成功连接到MongoDB: {self.db_name}.{self.collection_name}")
            return True
        except Exception as e:
            print(f"✗ 连接MongoDB时出错: {e}")
            return False

    def get_data_for_indicator(self, indicator_name: str) -> pd.DataFrame:
        """
        从MongoDB检索给定指标的数据

        Args:
            indicator_name: 要检索的指标名称

        Returns:
            包含指标数据的DataFrame
        """
        try:
            query = {indicator_name: {"$exists": True}}
            projection = {"_id": 0, indicator_name: 1, "id": 1}

            cursor = self.collection.find(query, projection)
            data = list(cursor)

            if not data:
                print(f"未找到指标的数据: {indicator_name}")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            print(f"检索到指标 {indicator_name} 的 {len(df)} 条记录")
            return df

        except Exception as e:
            print(f"检索指标 {indicator_name} 的数据时出错: {e}")
            return pd.DataFrame()

    def get_data_for_all_indicators(self, indicator_names: List[str]) -> pd.DataFrame:
        """
        从MongoDB一次性检索多个指标的数据，仅获取year字段大于等于2010的文档

        Args:
            indicator_names: 要检索的指标名称列表

        Returns:
            包含所有指标数据的合并DataFrame
        """
        print(f"正在从MongoDB一次性检索 {len(indicator_names)} 个指标的2010年及之后的数据...")

        if not indicator_names:
            print("没有提供指标名称")
            return pd.DataFrame()

        try:
            # 构建查询条件
            indicators_conditions = [
                {name: {"$exists": True}} for name in indicator_names
            ]

            query = {
                "$and": [
                    {"$or": indicators_conditions},
                    {"year": {"$gte": 2010}}
                ]
            }

            # 构建投影
            projection = {"_id": 0, "id": 1, "year": 1, "city": 1}
            for name in indicator_names:
                projection[name] = 1

            # 执行查询
            cursor = self.collection.find(query, projection)
            data = list(cursor)

            if not data:
                print("未找到2010年及之后的任何指标数据")
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame(data)

            # 调整列顺序：优先显示id, city, year
            priority_columns = ['id', 'city', 'year']
            all_columns = list(df.columns)
            for col in priority_columns:
                if col in all_columns:
                    all_columns.remove(col)

            new_column_order = [col for col in priority_columns if col in df.columns] + all_columns
            df = df[new_column_order]

            print(f"一次性检索到 {len(df)} 条2010年及之后的记录")

            # 输出各年份的记录数
            if 'year' in df.columns:
                year_counts = df['year'].value_counts().sort_index()
                print("\n各年份记录数:")
                for year, count in year_counts.items():
                    print(f"  - {year}: {count} 条记录")

            # 输出各指标的记录数
            print("\n各指标的记录数:")
            for name in indicator_names:
                if name in df.columns:
                    count = df[name].notna().sum()
                    print(f"  - {name}: {count} 条记录")
                else:
                    print(f"  - {name}: 0 条记录")

            return df

        except Exception as e:
            print(f"一次性检索多个指标数据时出错: {e}")
            return pd.DataFrame()

    def find_one(self, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """
        查询单个文档

        Args:
            query: 查询条件
            projection: 投影条件（可选）

        Returns:
            查询结果文档，如果未找到则返回None
        """
        try:
            return self.collection.find_one(query, projection)
        except Exception as e:
            print(f"查询文档时出错: {e}")
            return None

    def find_many(self, query: Dict[str, Any], projection: Optional[Dict[str, Any]] = None, limit: int = 0) -> List[Dict]:
        """
        查询多个文档

        Args:
            query: 查询条件
            projection: 投影条件（可选）
            limit: 限制返回数量（0表示不限制）

        Returns:
            查询结果文档列表
        """
        try:
            cursor = self.collection.find(query, projection)
            if limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception as e:
            print(f"查询文档时出错: {e}")
            return []

    def insert_one(self, document: Dict[str, Any]) -> Optional[str]:
        """
        插入单个文档

        Args:
            document: 要插入的文档

        Returns:
            插入文档的ID，失败则返回None
        """
        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            print(f"插入文档时出错: {e}")
            return None

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> bool:
        """
        更新单个文档

        Args:
            query: 查询条件
            update: 更新内容
            upsert: 如果不存在是否插入

        Returns:
            是否更新成功
        """
        try:
            result = self.collection.update_one(query, update, upsert=upsert)
            return result.modified_count > 0 or (upsert and result.upserted_id is not None)
        except Exception as e:
            print(f"更新文档时出错: {e}")
            return False

    def delete_one(self, query: Dict[str, Any]) -> bool:
        """
        删除单个文档

        Args:
            query: 查询条件

        Returns:
            是否删除成功
        """
        try:
            result = self.collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            print(f"删除文档时出错: {e}")
            return False

    def close(self):
        """关闭MongoDB连接"""
        if self.client:
            self.client.close()
            print("MongoDB连接已关闭")