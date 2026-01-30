import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    """处理模型训练和评估"""

    def __init__(self, model_params=None):
        """
        初始化模型训练器

        参数:
            model_params: CatBoost模型参数
        """
        self.model_params =   {
            'iterations': 5000,
            'learning_rate': 0.05,
            'depth': 4,
            'l2_leaf_reg': 5,
            'random_seed': 42,
            'eval_metric': 'Accuracy',
            'verbose': 100,
            'early_stopping_rounds':20,

        }
        self.model = None

    def get_cat_features(self, X):
        """确定分类特征"""
        cat_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                cat_features.append(col)
        return cat_features

    def train(self, X_train, y_train, X_val, y_val):
        """训练CatBoost模型"""
        # 确定分类特征
        cat_features = self.get_cat_features(X_train)
        print(f"分类特征: {cat_features}")

        # 获取分类特征索引
        cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]

        # 不再需要文本特征，因为已经转换为嵌入向量
        text_features = []

        # 创建CatBoost池
        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices, text_features=text_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices, text_features=text_features)

        # 初始化模型
        self.model = CatBoostClassifier(**self.model_params)

        # 训练模型
        print("开始训练模型...")
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            plot=False
        )

        # 评估模型
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"验证集准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_val, y_pred, target_names=['不通过', '通过']))

        # 分析特征重要性
        feature_importance = self.model.get_feature_importance(train_pool)
        feature_names = X_train.columns.tolist()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })

        importance_df = importance_df.sort_values('Importance', ascending=False)

        print("\n特征重要性:")
        print(importance_df.head(10))

        return self.model, accuracy

    def save_model(self, model_path):
        """保存训练好的模型"""
        if self.model:
            self.model.save_model(model_path)
            print(f"模型已保存至 {model_path}")
            return True
        return False
