import pandas as pd
import torch
from deeplotx import RecursiveSequential, LongTextEncoder

long_text_encoder = LongTextEncoder(
    max_length=4096,
    chunk_size=448,
    overlapping=32,
    model_name_or_path='hfl/chinese-bert-wwm-ext',
    device='cpu'
)

_test_embedding_dim = long_text_encoder.encode('test', flatten=False).shape[-1]
model = RecursiveSequential(input_dim=_test_embedding_dim, output_dim=1, recursive_layers=4, hidden_dim=512, model_name=r'E:\project\Catboost\RecursiveSequential', device='cpu').load()


class ModelPredictor:
    """处理模型预测和结果分析"""

    def __init__(self, model, class_labels=None, threshold=0.5):
        """
        初始化模型预测器

        参数:
            model: 训练好的模型
            class_labels: 类别标签映射
            threshold: 分类阈值
        """
        self.model = model
        self.class_labels = class_labels or ['不通过', '通过']
        self.threshold = threshold


    def process_categorical_features(self, X_test):
        """处理测试数据中的分类特征，确保它们与训练数据一致"""
        X_test_processed = X_test.copy()

        # 获取模型训练时的分类特征索引和名称
        cat_feature_indices = self.model.get_cat_feature_indices()
        cat_feature_names = [self.model.feature_names_[i] for i in cat_feature_indices]

        # 处理每个分类特征的缺失值
        for col in cat_feature_names:
            if col not in X_test_processed.columns:
                continue

            # 检查是否存在NaN或空值
            has_na = X_test_processed[col].isna().any()
            has_empty = X_test_processed[col].apply(lambda x: isinstance(x, str) and x.strip() == '').any()

            if has_na or has_empty:
                X_test_processed[col] = X_test_processed[col].fillna('nan').astype(str)
                X_test_processed[col] = X_test_processed[col].apply(
                    lambda x: 'nan' if isinstance(x, str) and x.strip() == '' else x)

        return X_test_processed


    def predict(self, X_test):
        """使用训练好的模型进行预测"""
        # 处理分类特征
        X_test_processed = self.process_categorical_features(X_test)

        # 预测概率
        y_proba = self.model.predict_proba(X_test_processed)[:, 1]  # 获取类别1的概率
        y_pred = (y_proba >= self.threshold).astype(int)
        y_pred_labels = [self.class_labels[label] for label in y_pred]

        return y_pred_labels, y_proba


    def generate_results(self, predictions, probabilities, accuracy,
                         title=None, content=None, y_real=None,theme=None):
        """生成结果DataFrame"""
        result_data = {
            '预测结果': predictions,
            '预测概率': probabilities,
            '模型准确率': accuracy
        }

        # 添加原始文本和真实标签(如果有)
        if title is not None:
            result_data['标题'] = title
        if content is not None:
            result_data['正文'] = content
        if theme is not None:
            result_data['题材'] = theme
        if y_real is not None:
            result_data['人工结果'] = y_real

        return pd.DataFrame(result_data)
