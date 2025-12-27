"""
数据分析器
使用OpenAI Function Call进行智能数据分析
包含多种统计分析工具：线性回归、相关性分析、描述性统计、分组统计、亚组分析
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
from tabulate import tabulate
from statsmodels.formula.api import ols
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


class OrganizeFunctions:
    """辅助函数组织类，包含各种工具函数"""

    @staticmethod
    def _get_regression_significance_stars(p_value: float) -> str:
        """
        根据p值返回显著性星号

        Args:
            p_value: p值

        Returns:
            显著性星号字符串
        """
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        elif p_value < 0.1:
            return "+"
        else:
            return ""

    @staticmethod
    def _generate_regression_markdown_table(
        model,
        dependent_var: str,
        independent_vars: List[str],
        control_vars: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        生成Markdown格式的回归结果表格

        Args:
            model: statsmodels回归模型结果
            dependent_var: 因变量列名
            independent_vars: 自变量列名列表
            control_vars: 控制变量列名列表
            metadata: 变量元数据字典

        Returns:
            Markdown格式的表格字符串
        """
        # 准备表头
        headers = ["变量", "系数", "标准误", "t值", "p值", "显著性"]

        # 准备表格数据
        rows = []

        # 先添加截距
        if 'Intercept' in model.params:
            intercept_row = [
                "截距",
                f"{model.params['Intercept']:.4f}",
                f"{model.bse['Intercept']:.4f}",
                f"{model.tvalues['Intercept']:.4f}",
                f"{model.pvalues['Intercept']:.4f}",
                OrganizeFunctions._get_regression_significance_stars(model.pvalues['Intercept'])
            ]
            rows.append(intercept_row)

        # 添加自变量
        for var in independent_vars:
            if var in model.params:
                var_row = [
                    var,
                    f"{model.params[var]:.4f}",
                    f"{model.bse[var]:.4f}",
                    f"{model.tvalues[var]:.4f}",
                    f"{model.pvalues[var]:.4f}",
                    OrganizeFunctions._get_regression_significance_stars(model.pvalues[var])
                ]
                rows.append(var_row)

        # 添加控制变量
        if control_vars:
            # 添加分隔行
            rows.append(["---", "---", "---", "---", "---", "---"])

            for var in control_vars:
                if var in model.params:
                    var_row = [
                        var,
                        f"{model.params[var]:.4f}",
                        f"{model.bse[var]:.4f}",
                        f"{model.tvalues[var]:.4f}",
                        f"{model.pvalues[var]:.4f}",
                        OrganizeFunctions._get_regression_significance_stars(model.pvalues[var])
                    ]
                    rows.append(var_row)

        # 添加模型统计量
        rows.append(["---", "---", "---", "---", "---", "---"])
        rows.append(["观测数", f"{model.nobs:.0f}", "", "", "", ""])
        rows.append(["R²", f"{model.rsquared:.4f}", "", "", "", ""])
        rows.append(["调整后R²", f"{model.rsquared_adj:.4f}", "", "", "", ""])
        rows.append(["F统计量", f"{model.fvalue:.4f}", "", "", f"{model.f_pvalue:.4f}",
                    OrganizeFunctions._get_regression_significance_stars(model.f_pvalue)])

        # 创建Markdown表格
        markdown_table = tabulate(rows, headers, tablefmt="pipe")

        table_title = f"### 线性回归分析结果: {dependent_var}"
        table_notes = """
**注:**
* 显著性: *** p<0.001, ** p<0.01, * p<0.05, + p<0.1
"""

        return f"{table_title}\n\n{markdown_table}\n{table_notes}"

    @staticmethod
    def _generate_correlation_markdown_table(
        corr_matrix: pd.DataFrame,
        p_values: Optional[pd.DataFrame] = None
    ) -> str:
        """
        生成Markdown格式的相关性分析表格

        Args:
            corr_matrix: 相关系数矩阵
            p_values: 对应的p值矩阵

        Returns:
            Markdown格式的表格字符串
        """
        # 准备表头
        headers = ["变量"] + list(corr_matrix.columns)

        # 准备表格数据
        rows = []

        for var_row in corr_matrix.index:
            row = [var_row]
            for var_col in corr_matrix.columns:
                corr_value = corr_matrix.loc[var_row, var_col]

                if p_values is not None and var_row != var_col:
                    p_value = p_values.loc[var_row, var_col]
                    stars = OrganizeFunctions._get_regression_significance_stars(p_value)
                    cell = f"{corr_value:.3f}{stars}"
                else:
                    cell = f"{corr_value:.3f}"

                row.append(cell)

            rows.append(row)

        # 创建Markdown表格
        markdown_table = tabulate(rows, headers, tablefmt="pipe")

        table_title = "### 相关性分析结果"
        table_notes = """
**注:**
* 显著性: *** p<0.001, ** p<0.01, * p<0.05, + p<0.1
* 对角线为变量自身相关(值为1)
"""

        return f"{table_title}\n\n{markdown_table}\n{table_notes}"

    @staticmethod
    def _generate_descriptive_markdown_table(stats_df: pd.DataFrame) -> str:
        """
        生成Markdown格式的描述性统计表格，以统计量为列，变量为行

        Args:
            stats_df: 包含描述性统计量的DataFrame

        Returns:
            Markdown格式的表格字符串
        """
        # 转置DataFrame，使统计量成为列，变量成为行
        transposed_stats = stats_df.transpose()

        # 创建Markdown表格
        markdown_table = transposed_stats.to_markdown(tablefmt="pipe", floatfmt=".4f")

        table_title = "### 描述性统计分析结果"
        table_notes = """
**注:**
* count: 有效观测数
* mean: 均值
* std: 标准差
* min: 最小值
* 25%: 第一四分位数
* 50%: 中位数
* 75%: 第三四分位数
* max: 最大值
"""

        return f"{table_title}\n\n{markdown_table}\n{table_notes}"


class Tools:
    """分析工具类，包含各种统计分析方法"""

    def __init__(self, df: pd.DataFrame, metadata: List[Dict[str, Any]]):
        """
        初始化工具类

        Args:
            df: pandas DataFrame, 需要分析的数据
            metadata: List[Dict], 每一列的元数据描述
        """
        self.df = df
        self.metadata = {item['indicator_name']: item for item in metadata}
        self.numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        self.categorical_cols = [
            col for col in df.columns
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 10
        ]

    def linear_regression(
        self,
        dependent_var: str,
        independent_vars: List[str],
        control_vars: Optional[List[str]] = None,
        robust: bool = False
    ) -> Dict[str, Any]:
        """
        执行线性回归分析

        Args:
            dependent_var: 因变量列名
            independent_vars: 自变量列名列表
            control_vars: 控制变量列名列表
            robust: 是否使用稳健标准误

        Returns:
            包含线性回归结果的字典
        """
        if dependent_var not in self.numeric_cols:
            return {"error": f"因变量 '{dependent_var}' 必须是数值型"}

        # 检查自变量和控制变量
        all_indep_vars = independent_vars.copy()
        if control_vars:
            all_indep_vars.extend(control_vars)

        all_vars = [dependent_var] + all_indep_vars
        missing_cols = [col for col in all_vars if col not in self.df.columns]

        if missing_cols:
            return {"error": f"以下列在数据框中不存在: {', '.join(missing_cols)}"}

        # 创建公式
        formula = f"{dependent_var} ~ {' + '.join(all_indep_vars)}"

        try:
            # 拟合模型
            model = ols(formula, data=self.df).fit()

            if robust:
                # 使用稳健标准误
                cov_type = 'HC3'  # 使用 HC3 估计量，适用于异方差情况
                model = model.get_robustcov_results(cov_type=cov_type)

            # 统一输出格式
            statistics = {
                "formula": formula,
                "r_squared": model.rsquared,
                "adjusted_r_squared": model.rsquared_adj,
                "f_statistic": model.fvalue,
                "f_pvalue": model.f_pvalue,
                "aic": model.aic,
                "bic": model.bic,
                "nobs": model.nobs,
                "df_model": model.df_model,
                "df_resid": model.df_resid,
                "coefficients": {}
            }

            # 收集系数
            for var_name in model.params.index:
                statistics["coefficients"][var_name] = {
                    "coef": float(model.params[var_name]),
                    "std_err": float(model.bse[var_name]),
                    "t_value": float(model.tvalues[var_name]),
                    "p_value": float(model.pvalues[var_name]),
                    "conf_int_lower": float(model.conf_int().loc[var_name, 0]),
                    "conf_int_upper": float(model.conf_int().loc[var_name, 1]),
                    "significance": OrganizeFunctions._get_regression_significance_stars(
                        model.pvalues[var_name]
                    )
                }

            # 生成Markdown格式的回归表格
            markdown_table = OrganizeFunctions._generate_regression_markdown_table(
                model, dependent_var, independent_vars, control_vars, self.metadata
            )

            return {
                "statistics": statistics,
                "markdown_table": markdown_table
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"线性回归分析失败: {str(e)}"}

    def correlation_analysis(
        self,
        variables: List[str],
        method: str = 'pearson',
        show_p_values: bool = True
    ) -> Dict[str, Any]:
        """
        执行相关性分析

        Args:
            variables: 需要分析相关性的变量列名列表
            method: 相关系数计算方法，可选 'pearson'、'spearman'、'kendall'
            show_p_values: 是否计算并显示p值

        Returns:
            包含相关性分析结果的字典
        """
        # 检查变量是否存在
        missing_cols = [col for col in variables if col not in self.df.columns]
        if missing_cols:
            return {"error": f"以下列在数据框中不存在: {', '.join(missing_cols)}"}

        # 检查变量是否为数值型
        non_numeric = [col for col in variables if col not in self.numeric_cols]
        if non_numeric:
            return {"error": f"以下列不是数值型，无法进行相关性分析: {', '.join(non_numeric)}"}

        try:
            # 只选择指定的变量并移除所有缺失值
            df_clean = self.df[variables].dropna()

            # 计算相关系数
            corr_matrix = df_clean.corr(method=method)

            # 统一输出格式
            statistics = {
                "method": method,
                "correlation_matrix": corr_matrix.to_dict(),
                "variables": variables
            }

            # 计算p值（如果需要）
            if show_p_values and method == 'pearson':
                import scipy.stats as stats
                p_values = pd.DataFrame(
                    np.zeros(corr_matrix.shape),
                    index=corr_matrix.index,
                    columns=corr_matrix.columns
                )

                for i, var1 in enumerate(variables):
                    for j, var2 in enumerate(variables):
                        if i != j:  # 非对角线元素
                            corr, p = stats.pearsonr(df_clean[var1], df_clean[var2])
                            p_values.loc[var1, var2] = p

                statistics["p_values"] = p_values.to_dict()

                # 生成Markdown表格
                markdown_table = OrganizeFunctions._generate_correlation_markdown_table(
                    corr_matrix, p_values
                )
            else:
                # 生成不带p值的Markdown表格
                markdown_table = OrganizeFunctions._generate_correlation_markdown_table(
                    corr_matrix
                )

            return {
                "statistics": statistics,
                "markdown_table": markdown_table
            }

        except Exception as e:
            return {"error": f"相关性分析失败: {str(e)}"}

    def descriptive_statistics(
        self,
        variables: Optional[List[str]] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行描述性统计分析

        Args:
            variables: 需要分析的变量列名列表，如果为None则分析所有数值型变量
            include: 需要包含的统计量列表，可选项包括：
                    'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'

        Returns:
            包含描述性统计结果的字典
        """
        if variables is None:
            variables = self.numeric_cols
        else:
            # 检查变量是否存在
            missing_cols = [col for col in variables if col not in self.df.columns]
            if missing_cols:
                return {"error": f"以下列在数据框中不存在: {', '.join(missing_cols)}"}

            # 检查变量是否为数值型
            non_numeric = [col for col in variables if col not in self.numeric_cols]
            if non_numeric:
                return {"error": f"以下列不是数值型，无法进行描述性统计分析: {', '.join(non_numeric)}"}

        try:
            # 计算描述性统计量
            stats_df = self.df[variables].describe()

            # 如果指定了需要包含的统计量，则只保留这些行
            if include:
                valid_includes = [inc for inc in include if inc in stats_df.index]
                if valid_includes:
                    stats_df = stats_df.loc[valid_includes]

            # 添加额外的统计量：偏度和峰度
            skew = self.df[variables].skew()
            kurtosis = self.df[variables].kurtosis()

            # 统一输出格式
            statistics = {
                "variables": variables,
                "basic_statistics": stats_df.to_dict(),
                "additional_statistics": {
                    "skewness": skew.to_dict(),
                    "kurtosis": kurtosis.to_dict()
                }
            }

            # 生成Markdown表格
            markdown_table = OrganizeFunctions._generate_descriptive_markdown_table(stats_df)

            return {
                "statistics": statistics,
                "markdown_table": markdown_table
            }

        except Exception as e:
            return {"error": f"描述性统计分析失败: {str(e)}"}

    def group_statistics(
        self,
        group_vars: List[str],
        indicators: List[str],
        statistics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行分组统计分析，计算不同亚组的统计特征

        Args:
            group_vars: 用于分组的变量列名列表
            indicators: 需要分析的指标列名列表
            statistics: 需要计算的统计量列表，可选项包括：
                       'mean', 'std', 'min', 'max', 'count', 'median', 'sum', 'var'

        Returns:
            包含分组统计结果的字典
        """
        # 检查变量是否存在
        missing_cols = [col for col in group_vars + indicators if col not in self.df.columns]
        if missing_cols:
            return {"error": f"以下列在数据框中不存在: {', '.join(missing_cols)}"}

        # 检查统计量是否有效
        valid_stats = ['mean', 'std', 'min', 'max', 'count', 'median', 'sum', 'var']
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max', 'count']
        else:
            invalid_stats = [stat for stat in statistics if stat not in valid_stats]
            if invalid_stats:
                return {
                    "error": f"以下统计量无效: {', '.join(invalid_stats)}，"
                            f"有效选项为: {', '.join(valid_stats)}"
                }

        try:
            # 执行分组统计
            grouped = self.df.groupby(group_vars)
            result_dfs = {}

            for stat in statistics:
                if stat == 'mean':
                    result_dfs[stat] = grouped[indicators].mean()
                elif stat == 'std':
                    result_dfs[stat] = grouped[indicators].std()
                elif stat == 'min':
                    result_dfs[stat] = grouped[indicators].min()
                elif stat == 'max':
                    result_dfs[stat] = grouped[indicators].max()
                elif stat == 'count':
                    result_dfs[stat] = grouped[indicators].count()
                elif stat == 'median':
                    result_dfs[stat] = grouped[indicators].median()
                elif stat == 'sum':
                    result_dfs[stat] = grouped[indicators].sum()
                elif stat == 'var':
                    result_dfs[stat] = grouped[indicators].var()

            # 统一输出格式
            output_stats = {}
            for stat, df in result_dfs.items():
                output_stats[stat] = df.reset_index().to_dict(orient='records')

            # 生成Markdown表格
            markdown_tables = {}
            for stat, df in result_dfs.items():
                table_df = df.reset_index()
                markdown_tables[stat] = (
                    f"### {stat.capitalize()} by {', '.join(group_vars)}\n\n"
                    f"{table_df.to_markdown(tablefmt='pipe', floatfmt='.4f')}"
                )

            return {
                "statistics": {
                    "group_vars": group_vars,
                    "value_vars": indicators,
                    "statistics_calculated": statistics,
                    "results": output_stats
                },
                "markdown_tables": markdown_tables
            }

        except Exception as e:
            return {"error": f"分组统计分析失败: {str(e)}"}

    def subgroup_analysis(
        self,
        filters: List[Dict[str, Any]],
        indicators: Optional[List[str]] = None,
        statistics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行针对特定亚组的分析，先筛选数据，再进行描述性统计

        Args:
            filters: 筛选条件列表，每个条件是一个字典，格式为
                    {"column": 列名, "operator": 运算符, "value": 值}
                    运算符可选: "==", "!=", ">", "<", ">=", "<=", "in", "not in"
            indicators: 需要分析的指标列名列表，如果为None则分析所有数值型指标
            statistics: 需要计算的统计量列表，
                       默认为 ['count', 'mean', 'std', 'min', 'max', 'median']

        Returns:
            包含亚组分析结果的字典
        """
        # 设置默认统计量
        if statistics is None:
            statistics = ['count', 'mean', 'std', 'min', 'max', 'median']

        # 设置默认分析指标
        if indicators is None:
            indicators = self.numeric_cols

        # 验证筛选条件
        for condition in filters:
            if 'column' not in condition or 'operator' not in condition or 'value' not in condition:
                return {"error": "筛选条件格式错误，必须包含 'column', 'operator' 和 'value' 字段"}

            column = condition['column']
            operator = condition['operator']

            if column not in self.df.columns:
                return {"error": f"筛选条件中的列 '{column}' 在数据框中不存在"}

            valid_operators = ["==", "!=", ">", "<", ">=", "<=", "in", "not in"]
            if operator not in valid_operators:
                return {"error": f"无效的运算符 '{operator}'，有效选项为: {', '.join(valid_operators)}"}

        # 验证分析指标
        missing_cols = [col for col in indicators if col not in self.df.columns]
        if missing_cols:
            return {"error": f"以下列在数据框中不存在: {', '.join(missing_cols)}"}

        try:
            # 应用筛选条件
            filtered_df = self.df.copy()
            for condition in filters:
                column = condition['column']
                operator = condition['operator']
                value = condition['value']

                if operator == "==":
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif operator == "!=":
                    filtered_df = filtered_df[filtered_df[column] != value]
                elif operator == ">":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif operator == "<":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif operator == ">=":
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif operator == "<=":
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif operator == "in":
                    if not isinstance(value, list):
                        return {"error": "'in' 运算符要求值必须是列表"}
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                elif operator == "not in":
                    if not isinstance(value, list):
                        return {"error": "'not in' 运算符要求值必须是列表"}
                    filtered_df = filtered_df[~filtered_df[column].isin(value)]

            # 获取筛选后的样本量
            sample_size = len(filtered_df)

            # 处理样本量为0的情况
            if sample_size == 0:
                return {
                    "error": "筛选后没有符合条件的数据",
                    "filters_applied": filters
                }
            # 处理样本量为1的情况 - 直接返回该样本的值
            elif sample_size == 1:
                single_record = {var: filtered_df[var].iloc[0] for var in indicators}

                return {
                    "statistics": {
                        "sample_size": 1,
                        "filters_applied": filters,
                        "indicators": indicators,
                        "single_record": single_record
                    },
                    "markdown_table": (
                        f"### 亚组分析结果 (单个样本)\n\n"
                        f"筛选条件: {json.dumps(filters, ensure_ascii=False)}\n\n"
                        f"样本数: 1\n\n" +
                        pd.DataFrame([single_record]).T.rename(columns={0: "值"}).to_markdown(
                            tablefmt="pipe"
                        )
                    )
                }

            # 处理多样本情况 - 执行描述性统计
            stats_dict = {}
            df_stats = filtered_df[indicators]

            for stat in statistics:
                if stat == 'count':
                    stats_dict[stat] = df_stats.count()
                elif stat == 'mean':
                    stats_dict[stat] = df_stats.mean()
                elif stat == 'std':
                    stats_dict[stat] = df_stats.std()
                elif stat == 'min':
                    stats_dict[stat] = df_stats.min()
                elif stat == 'max':
                    stats_dict[stat] = df_stats.max()
                elif stat == 'median':
                    stats_dict[stat] = df_stats.median()
                elif stat == '25%':
                    stats_dict[stat] = df_stats.quantile(0.25)
                elif stat == '75%':
                    stats_dict[stat] = df_stats.quantile(0.75)

            # 创建结果DataFrame
            stats_df = pd.DataFrame(stats_dict)

            # 生成Markdown表格
            markdown_table = (
                f"### 亚组分析结果\n\n"
                f"筛选条件: {json.dumps(filters, ensure_ascii=False)}\n\n"
                f"样本数: {sample_size}\n\n" +
                stats_df.T.to_markdown(tablefmt="pipe", floatfmt='.4f')
            )

            return {
                "statistics": {
                    "sample_size": sample_size,
                    "filters_applied": filters,
                    "indicators": indicators,
                    "statistics_calculated": statistics,
                    "results": stats_df.to_dict()
                },
                "markdown_table": markdown_table
            }

        except Exception as e:
            return {"error": f"亚组分析失败: {str(e)}"}


class ChatWithData:
    """数据分析类，整合工具和API调用"""

    def __init__(self, df: pd.DataFrame, metadata: List[Dict[str, Any]]):
        """
        初始化数据分析类

        Args:
            df: pandas DataFrame, 需要分析的数据
            metadata: List[Dict], 每一列的元数据描述
        """
        self.df = df
        self.metadata = metadata
        self.tools = Tools(df, metadata)
        self.numeric_cols = self.tools.numeric_cols
        self.categorical_cols = self.tools.categorical_cols

    def get_function_descriptions(self) -> List[Dict[str, Any]]:
        """返回OpenAI Function Call所需的函数描述"""
        return [
            {
                "name": "linear_regression",
                "description": "执行线性回归分析，分析自变量对因变量的影响，可以包含控制变量",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dependent_var": {
                            "type": "string",
                            "description": "因变量（被解释变量）列名"
                        },
                        "independent_vars": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "自变量（解释变量）列名列表"
                        },
                        "control_vars": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "控制变量列名列表"
                        },
                        "robust": {
                            "type": "boolean",
                            "description": "是否使用稳健标准误（处理异方差性）"
                        }
                    },
                    "required": ["dependent_var", "independent_vars"]
                }
            },
            {
                "name": "correlation_analysis",
                "description": "执行相关性分析，计算多个变量之间的相关系数和显著性",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "需要分析相关性的变量列名列表"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman", "kendall"],
                            "description": "相关系数计算方法"
                        },
                        "show_p_values": {
                            "type": "boolean",
                            "description": "是否计算并显示p值"
                        }
                    },
                    "required": ["variables"]
                }
            },
            {
                "name": "descriptive_statistics",
                "description": "执行描述性统计分析，计算变量的统计特征",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "需要分析的变量列名列表"
                        },
                        "include": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
                            },
                            "description": "需要包含的统计量列表"
                        }
                    }
                }
            },
            {
                "name": "group_statistics",
                "description": "执行分组统计分析，计算不同亚组的统计特征",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_vars": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "用于分组的变量列名列表"
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "需要计算统计量的指标列名列表"
                        },
                        "statistics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["mean", "std", "min", "max", "count", "median", "sum", "var"]
                            },
                            "description": "需要计算的统计量列表"
                        }
                    },
                    "required": ["group_vars", "indicators"]
                }
            },
            {
                "name": "subgroup_analysis",
                "description": "筛选特定亚组的样本，获取指定指标的值或描述性统计",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string", "description": "需要筛选的列名"},
                                    "operator": {
                                        "type": "string",
                                        "enum": ["==", "!=", ">", "<", ">=", "<=", "in", "not in"],
                                        "description": "筛选操作符"
                                    },
                                    "value": {"description": "筛选值"}
                                },
                                "required": ["column", "operator", "value"]
                            },
                            "description": "筛选条件列表"
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "需要分析的指标名列表"
                        },
                        "statistics": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["count", "mean", "std", "min", "max", "median", "25%", "75%"]
                            },
                            "description": "需要计算的统计量列表"
                        }
                    },
                    "required": ["filters", "indicators"]
                }
            }
        ]

    def classify_query_type(self, query: str) -> int:
        """
        使用大模型判断用户查询属于哪种类型

        Args:
            query: 用户查询内容

        Returns:
            0表示元数据查询型(可直接回答)，1表示数据分析型(需要执行分析)
        """
        prompt = f"""<TASK>
你是一个精确的Excel表格查询分类器。请根据<CLASSIFICATION_FRAMEWORK>中定义的标准，
将用户查询分类为以下两种类型之一：元数据查询型(0)或数据分析型(1)。
</TASK>

<QUERY>
{query}
</QUERY>

<CLASSIFICATION_FRAMEWORK>
## 类型定义
0. 元数据查询型（可直接回答）：询问表格的元数据信息，如指标的定义、描述、单位、数据来源等
1. 数据分析型（需要执行分析）：需要对数据进行实际计算、统计或分析

## 判断标准
### 元数据查询型(0)核心特征：
- 询问指标的概念、定义或含义
- 询问数据的来源、收集方法或更新时间
- 询问计量单位或统计口径

### 数据分析型(1)核心特征：
- 询问具体数值或统计量
- 要求对数据进行筛选、分组或排序
- 探究指标之间的关系
- 需要对数据进行计算处理
</CLASSIFICATION_FRAMEWORK>

<OUTPUT_INSTRUCTION>
只返回分类结果数字：0或1
</OUTPUT_INSTRUCTION>

分类结果："""

        try:
            response = llm_handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=1
            )

            result = response.strip()
            return 1 if result == "1" else 0

        except Exception as e:
            print(f"调用大模型分类查询失败: {e}")
            return 0

    def get_analysis_plan(self, query: str) -> Dict[str, Any]:
        """
        根据用户查询和DataFrame信息，使用LLM API生成分析计划和回应

        Args:
            query: 用户的自然语言查询

        Returns:
            包含content、function和parameters的字典
        """
        # 构造DataFrame信息
        df_info = {
            'columns': {col: str(self.df[col].dtype) for col in self.df.columns},
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'rows': len(self.df)
        }

        function_descriptions = self.get_function_descriptions()

        # 处理元数据，确保它是JSON可序列化的，并转换为指定格式
        metadata_for_api = []
        if hasattr(self, 'metadata') and isinstance(self.metadata, list):
            # 过滤出数据框中存在的列的元数据，并转换为指定格式
            for item in self.metadata:
                if isinstance(item, dict) and 'indicator_name' in item and item['indicator_name'] in self.df.columns:
                    formatted_item = {
                        "指标": item.get('indicator_name', ''),
                        "指标用于测量的变量": item.get('variable', ''),
                        "指标描述": item.get('description', ''),
                        "统计单位": item.get('statistical_unit', ''),
                        "指标数据来源": item.get('source', '')
                    }
                    metadata_for_api.append(formatted_item)

        try:
            # 导入必要的模块
            from common.llm_call import handler
            from openai import OpenAI
            from common.config import OPENAI_API_KEY

            # 创建OpenAI客户端（用于Function Call）
            openai_client = OpenAI(api_key=OPENAI_API_KEY)

            # 构造API请求
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
    你是一个专业的数据分析助手。你的唯一任务是将用户的自然语言数据分析请求转换为相应的函数调用，不生成任何其他文本内容。

    严格要求：
    1. 不要生成任何解释、分析或建议
    2. 不要回复任何文本
    3. 只返回函数调用
    4. 强制使用函数调用，不允许普通文本回复

    关于数据结构：
    1. DataFrame的列名代表的是"指标"(indicator)，而非直接的变量
    2. 元数据包含：指标、指标用于测量的变量、指标描述、统计单位、指标数据来源

    请只使用DataFrame中实际存在的列名(指标)进行分析。
                    """},
                    {"role": "user", "content": f"""
    以下是DataFrame的信息：

    列名和数据类型: {json.dumps(df_info['columns'], ensure_ascii=False)}
    数值型指标: {df_info['numeric_cols']}
    分类型指标: {df_info['categorical_cols']}
    行数: {df_info['rows']}

    指标元数据信息: {json.dumps(metadata_for_api, ensure_ascii=False, indent=2)}

    用户的查询是: "{query}"

    只返回函数调用，不要返回任何文本内容。
                    """}
                ],
                functions=function_descriptions,
                function_call="auto"
            )

            # 解析响应
            message = response.choices[0].message
            print(message)

            # 准备返回值，始终包含content
            result = {
                "content": message.content or "我已分析了您的请求",
                "function": None,
                "parameters": {}
            }

            # 如果有函数调用，则添加到结果中
            if message.function_call:
                function_name = message.function_call.name
                try:
                    function_args = json.loads(message.function_call.arguments)

                    # 获取当前函数的有效参数
                    valid_params = []
                    for func in function_descriptions:
                        if func['name'] == function_name:
                            # 不包含explanation
                            valid_params = [
                                prop for prop in func['parameters']['properties'].keys()
                                if prop != 'explanation'
                            ]
                            break

                    # 清理参数，只保留有效参数
                    cleaned_args = {k: v for k, v in function_args.items() if k in valid_params}

                    # 更新结果
                    result["function"] = function_name
                    result["parameters"] = cleaned_args

                except json.JSONDecodeError:
                    # 处理JSON解析错误，保持content
                    result["content"] += " (无法解析函数参数)"

            return result

        except Exception as e:
            return {
                "content": f"处理您的请求时出现错误: {str(e)}",
                "function": None,
                "parameters": {}
            }
    def execute_analysis(self, analysis_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行分析计划

        Args:
            analysis_plan: 分析计划字典

        Returns:
            分析结果
        """
        if 'error' in analysis_plan:
            return analysis_plan

        function_name = analysis_plan.get('function')
        parameters = analysis_plan.get('parameters', {}).copy()

        if not function_name or not hasattr(self.tools, function_name):
            return {
                "error": f"未知的分析方法: {function_name}",
                "explanation": "请检查分析计划中的函数名是否正确"
            }

        # 执行分析函数
        try:
            result = getattr(self.tools, function_name)(**parameters)
            return result
        except Exception as e:
            return {
                "error": f"执行分析失败: {str(e)}",
                "explanation": str(e)
            }

    def run(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询：判断查询类型，执行相应流程，返回结果

        Args:
            query: 用户的自然语言查询

        Returns:
            包含df_info, metadata和markdown_table的字典
        """
        # 构造DataFrame信息
        df_info = {
            'columns': {col: str(self.df[col].dtype) for col in self.df.columns},
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'rows': len(self.df)
        }

        # 处理元数据
        metadata_for_api = []
        if hasattr(self, 'metadata') and isinstance(self.metadata, list):
            for item in self.metadata:
                if isinstance(item, dict) and 'indicator_name' in item:
                    if item['indicator_name'] in self.df.columns:
                        formatted_item = {
                            "指标": item.get('indicator_name', ''),
                            "指标用于测量的变量": item.get('variable', ''),
                            "指标描述": item.get('description', ''),
                            "统计单位": item.get('statistical_unit', ''),
                            "指标数据来源": item.get('source', '')
                        }
                        metadata_for_api.append(formatted_item)

        try:
            # 判断查询类型
            query_type = self.classify_query_type(query)

            # 普通查询 - 只返回基础信息
            if query_type == 0:
                return {
                    "df_info": df_info,
                    "metadata": metadata_for_api,
                    "markdown_table": ""
                }

            # 数据分析查询 - 制定并执行分析计划
            else:
                # 获取分析计划
                analysis_plan = self.get_analysis_plan(query)

                # 如果没有指定分析函数，返回基础信息
                if analysis_plan.get('function') is None:
                    return {
                        "df_info": df_info,
                        "metadata": metadata_for_api,
                        "markdown_table": "",
                        "content": "未在方法库中找到合适的数据分析方法"
                    }

                # 执行分析
                analysis_result = self.execute_analysis({
                    'function': analysis_plan.get('function'),
                    'parameters': analysis_plan.get('parameters', {})
                })

                # 检查分析是否成功
                if 'error' in analysis_result:
                    return {
                        "df_info": df_info,
                        "metadata": metadata_for_api,
                        "markdown_table": "",
                        "content": analysis_result['error']
                    }

                # 提取Markdown表格
                markdown_table = analysis_result.get('markdown_table', "")

                return {
                    "df_info": df_info,
                    "metadata": metadata_for_api,
                    "markdown_table": markdown_table,
                    "content": "分析执行成功"
                }

        except Exception as e:
            # 处理异常情况
            return {
                "df_info": df_info,
                "metadata": metadata_for_api,
                "markdown_table": "",
                "content": f"处理查询时发生错误: {str(e)}"
            }