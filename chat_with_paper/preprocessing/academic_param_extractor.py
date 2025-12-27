"""
学术参数提取器
从用户的学术查询中提取年份、期刊名称、作者信息和语言信息
"""
from datetime import datetime
import re
import json
from typing import Dict, List, Union, Optional
import concurrent.futures
from common.llm_call import handler


class AcademicParamExtractor:
    """学术参数提取器类"""

    def __init__(self):
        """初始化学术参数提取器"""
        self.current_year = datetime.now().year
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def _call_llm_api(self, prompt: str) -> str:
        """调用大模型API获取结果"""
        try:
            try:
                response = handler.call_llm(
                    provider="zhipuai",
                    prompt=prompt,
                    model="glm-4-airx",
                    max_tokens=20,
                    temperature=0.7
                )
                if "API call failed" in str(response):
                    raise Exception("API call failed detected in response")
            except:
                response = handler.call_llm(
                    provider="openai",
                    prompt=prompt,
                    model="gpt-4o-mini",
                    max_tokens=20,
                    temperature=0.7
                )
            return response
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""

    def _create_year_prompt(self, query: str) -> str:
        """创建年份提取的提示词"""
        return f"""
<TASK>
你是一个学术搜索引擎的查询解析助手。你的任务是从用户的自然语言查询中准确提取年份(year)信息，以便后续进行结构化搜索。
</TASK>

<CURRENT_DATE>
当前日期：{self.current_date}
当前年份：{self.current_year}
</CURRENT_DATE>

<EXTRACTION_RULES>
## 年份(year)提取规则
1. 识别显性时间：
   - 直接提取数字年份(如"2020")
   - 处理带世纪的年份：
     * "19世纪初" → [1800, 1820]
     * "19世纪中" → [1840, 1860]
     * "19世纪末" → [1880, 1900]
     * "20世纪30年代" → [1930, 1939]
2. 解析相对时间：将相对表述转换为具体年份，当前年份为{self.current_year}
   - "去年" → [{self.current_year - 1}]
   - "近五年" → [{self.current_year - 4}, {self.current_year}]
   - "十年前" → [{self.current_year - 10}]
3. 年份处理：
   - 单一年份也使用数组格式，如[2020]
   - 年份范围使用[start_year, end_year]格式
4. 异常值校验：排除未来年份(>{self.current_year})
</EXTRACTION_RULES>

<OUTPUT_STRUCTURE>
提取结果必须严格按照以下JSON数组格式返回，不要包含任何键名：

[QUERY_BEGIN]
number[] | []  // 年份必须是数字数组，单一年份如[2020]，范围如[2015, 2020]，无年份则为[]
[QUERY_END]
</OUTPUT_STRUCTURE>

<EXAMPLES>
用户查询1: "我想找王明和李华2021年发表在《自然》杂志上关于量子计算的文章"
输出：
[QUERY_BEGIN]
[2021]
[QUERY_END]

用户查询2: "近五年Science上有关气候变化的综述文章都有哪些？"
输出：
[QUERY_BEGIN]
[{self.current_year - 4}, {self.current_year}]
[QUERY_END]
</EXAMPLES>

<INPUT>
{query}
</INPUT>

<OUTPUT_INSTRUCTION>
仔细分析用户查询文本<INPUT>，提取所有可能的年份信息。
按<OUTPUT_STRUCTURE>中定义的格式构建JSON数组（不要包含键名）。
将结果包含在[QUERY_BEGIN]和[QUERY_END]标记之间。
只返回结果数组，不要返回任何额外解释或分析过程。
</OUTPUT_INSTRUCTION>

你的输出：
"""

    def _create_journal_prompt(self, query: str) -> str:
        """创建期刊提取的提示词"""
        return f"""
<TASK>
你是一个学术搜索引擎的查询解析助手。你的任务是从用户的自然语言查询中准确提取期刊名称(journal)信息。
</TASK>

<EXTRACTION_RULES>
## 期刊名称(journal)提取规则
1. 识别标记：书名号《》中的出版物名称
2. 关键词识别：与"期刊"、"杂志"、"journal"等词相关的名称
3. 处理缩写：保留原始输入形式，包括英文缩写和中文全称
4. 处理多期刊情况：
   - 如果查询中明确提到多个期刊，将所有期刊名称作为数组返回
   - 使用连词("和"、"或"、"与"、"and"等)识别多个期刊
5. 输出格式：
   - 单一期刊：字符串数组，如["Nature"]
   - 多个期刊：字符串数组，如["Nature", "Science"]
   - 无期刊：[]
</EXTRACTION_RULES>

<INPUT>
{query}
</INPUT>

你的输出：
"""

    def _create_authors_prompt(self, query: str) -> str:
        """创建作者提取的提示词"""
        return f"""
<TASK>
你是一个学术搜索引擎的查询解析助手。你的任务是从用户的自然语言查询中准确提取作者(authors)信息。
</TASK>

<EXTRACTION_RULES>
## 作者(authors)提取规则
1. 识别人名：定位查询中的姓名(中文或外文)
2. 分隔符识别：通过逗号、顿号等分隔多个作者
3. 过滤非人名：排除"教授"、"团队"、"实验室"等非人名词汇
4. 中英文名称对应：
   - 对于中文名称，同时提供中文原名和对应的英文名称
   - 所有作者名称放在同一个数组中，先中文后英文
   - 例如："张娟娟" → ["张娟娟", "Zhang, Juanjuan"]
5. 姓名格式：
   - 中文名称保持原样
   - 英文名称按"姓, 名"的格式输出
</EXTRACTION_RULES>

<INPUT>
{query}
</INPUT>

你的输出：
"""

    def _create_language_prompt(self, query: str) -> str:
        """创建语言识别的提示词"""
        return f"""
<TASK>
你是一个学术搜索引擎的查询解析助手。你的任务是从用户的自然语言查询中识别查询使用的语言。
</TASK>

<EXTRACTION_RULES>
## 语言(language)识别规则
1. 如果查询中明确提到希望查询特定语言的论文（如"英文论文、英文文献"），则提取该语言
2. 输出格式：语言的中文名称，如"中文"、"英文"等
3. 如果查询混合了多种语言，则返回null
4. 如果无法确定语言或没有特定语言要求，则返回null
</EXTRACTION_RULES>

<INPUT>
{query}
</INPUT>

你的输出：
"""

    def _parse_year_result(self, result_text: str) -> List[int]:
        """解析年份结果"""
        try:
            result_match = re.search(r'\[QUERY_BEGIN\](.*?)\[QUERY_END\]', result_text, re.DOTALL)
            if result_match:
                json_str = result_match.group(1).strip()
                years = json.loads(json_str)
                if isinstance(years, list):
                    return years
            return []
        except Exception as e:
            print(f"解析年份结果失败: {e}")
            return []

    def _parse_journal_result(self, result_text: str) -> List[str]:
        """解析期刊结果"""
        try:
            result_match = re.search(r'\[QUERY_BEGIN\](.*?)\[QUERY_END\]', result_text, re.DOTALL)
            if result_match:
                json_str = result_match.group(1).strip()
                journals = json.loads(json_str)
                if isinstance(journals, list):
                    return journals
            return []
        except Exception as e:
            print(f"解析期刊结果失败: {e}")
            return []

    def _parse_authors_result(self, result_text: str) -> List[str]:
        """解析作者结果"""
        try:
            result_match = re.search(r'\[QUERY_BEGIN\](.*?)\[QUERY_END\]', result_text, re.DOTALL)
            if result_match:
                json_str = result_match.group(1).strip()
                authors = json.loads(json_str)
                if isinstance(authors, list):
                    return authors
            return []
        except Exception as e:
            print(f"解析作者结果失败: {e}")
            return []

    def _parse_language_result(self, result_text: str) -> Optional[str]:
        """解析语言结果"""
        try:
            result_match = re.search(r'\[QUERY_BEGIN\](.*?)\[QUERY_END\]', result_text, re.DOTALL)
            if result_match:
                json_str = result_match.group(1).strip()
                if json_str.lower() == "null":
                    return None
                language = json_str.strip('"\'')
                return language if language else None
            return None
        except Exception as e:
            print(f"解析语言结果失败: {e}")
            return None

    def extract_params(self, query: str) -> Dict[str, Union[List[int], List[str], str, None]]:
        """
        从用户的学术查询中提取参数，采用多线程并行处理

        Returns:
            {
                "year": List[int],
                "journal": List[str],
                "authors": List[str],
                "language": str或None
            }
        """
        try:
            year_prompt = self._create_year_prompt(query)
            journal_prompt = self._create_journal_prompt(query)
            authors_prompt = self._create_authors_prompt(query)
            language_prompt = self._create_language_prompt(query)

            # 多线程并行调用API
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_year = executor.submit(self._call_llm_api, year_prompt)
                future_journal = executor.submit(self._call_llm_api, journal_prompt)
                future_authors = executor.submit(self._call_llm_api, authors_prompt)
                future_language = executor.submit(self._call_llm_api, language_prompt)

                year_result = future_year.result()
                journal_result = future_journal.result()
                authors_result = future_authors.result()
                language_result = future_language.result()

            years = self._parse_year_result(year_result)
            journals = self._parse_journal_result(journal_result)
            authors = self._parse_authors_result(authors_result)
            language = self._parse_language_result(language_result)

            return {
                "year": years,
                "journal": journals,
                "authors": authors,
                "language": language
            }

        except Exception as e:
            print(f"从学术查询中提取参数失败: {e}")
            return {"year": [], "journal": [], "authors": [], "language": None}