import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataProcessor:
    """处理数据加载、预处理和分割"""

    def __init__(self, label_mapping=None):
        """
        初始化数据处理器

        参数:
            label_mapping: 标签映射字典，用于将文本标签转换为数值
        """
        self.label_mapping = label_mapping or {'不通过': 0, '通过': 1,'T': 1, 'F': 0}

    def load_data(self, file_path):
        """加载数据文件"""
        print(f"加载数据: {file_path}")
        df = pd.read_csv(file_path)
        rf = df.dropna(how = 'all')
        return rf

    def preprocess_train_data(self, df):
        """预处理训练数据"""
        # 提取特征和标签
        if '人工审核结果' in df.columns:
            X = df.drop(['人工审核结果'], axis=1)
            y = df['人工审核结果'].map(self.label_mapping)
            # 检查并处理 NaN 值
            nan_count = y.isna().sum()
            if nan_count > 0:
                print(f"发现 {nan_count} 个 NaN 值，将其移除。")
                valid_indices = ~y.isna()
                X = X[valid_indices]
                y = y[valid_indices]
            return X, y
        return df, None

    def preprocess_test_data(self, df):
        """预处理测试数据"""
        X = df.drop(['人工审核结果'], axis=1) if '人工审核结果' in df.columns else df
        y_real = df['人工审核结果'] if '人工审核结果' in df.columns else None
        title = df['标题'] if '标题' in df.columns else None
        content = df['正文'] if '正文' in df.columns else None

        return X, y_real, title, content

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """分割训练集和验证集"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def process_categorical_features(self, df, inplace=True):
        """处理分类特征，确保它们是字符串类型"""
        if not inplace:
            df = df.copy()

        cat_features = [col for col in df.columns
                        if df[col].dtype == 'object' or df[col].dtype.name == 'category']

        for col in cat_features:
            df[col] = df[col].astype(str)

        return df
