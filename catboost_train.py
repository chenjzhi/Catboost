import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from csv_save import MLResultsSaver


def load_and_preprocess_data(train_file, test_file=None):
    """加载数据并进行简单预处理"""
    # 加载训练数据
    train_data = pd.read_csv(train_file)
    # 提取特征和标签
    X = train_data.drop(['人工审核结果'], axis=1)

    y = train_data['人工审核结果']
    y = y.map({'不通过': 0, '通过': 1})

    if test_file:
        test_data = pd.read_csv(test_file)
        X_test = test_data.drop(['人工审核结果'], axis=1) if '人工审核结果' in test_data.columns else test_data
        X_real = test_data['人工审核结果']
        title = test_data['标题']
        return X, y, X_test, X_real,title

    return X, y


def train_catboost_model(X_train, y_train,X_val,y_val):
    """训练CatBoost分类模型"""

    # 确定分类特征
    cat_features = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
            cat_features.append(col)

    print(f"分类特征: {cat_features}")

    # 获取分类特征索引
    cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]
    text_features = ['正文'] if '正文' in X_train.columns else []
    # 创建CatBoost池
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices, text_features=text_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices, text_features=text_features)

    # 初始化模型
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='Accuracy',
        verbose=100,
        early_stopping_rounds=50
    )

    # 训练模型
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )

    # 评估模型
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"验证集准确率: {accuracy:.4f}")
    print("分类报告:")
    print(classification_report(y_val, y_pred, target_names=['不通过', '通过']))

    # 分析特征重要性
    feature_importance = model.get_feature_importance(train_pool)
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })

    importance_df = importance_df.sort_values('Importance', ascending=False)

    print("\n特征重要性:")
    print(importance_df.head(10))
    return model,accuracy


def predict_with_model(model, X_test, class_labels=['不通过', '通过']):
    """使用训练好的模型进行预测，强制处理测试数据的分类特征缺失值"""
    X_test_processed = X_test.copy()

    # 获取模型训练时的分类特征索引和名称
    cat_feature_indices = model.get_cat_feature_indices()
    cat_feature_names = [model.feature_names_[i] for i in cat_feature_indices]

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

    # 预测概率
    y_proba = model.predict_proba(X_test_processed)[:, 1]  # 获取类别1的概率
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)
    y_pred_labels = [class_labels[label] for label in y_pred]

    return y_pred_labels, y_proba


if __name__ == "__main__":
    train_file = "dataset/processed_data.csv"
    test_file = "dataset/test.csv"

    print("加载数据...")
    X, y, X_test,x_real,title = load_and_preprocess_data(train_file, test_file)

    print("\n数据类型检查:")
    X.info()

    cat_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    for col in cat_features:
        X[col] = X[col].astype(str)

    print("\n训练CatBoost模型...")
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    np.random.seed(100)
    mask = np.random.random(len(y_val)) < 0.2
    y_val_n = y_val.copy()
    y_val_n[mask] = 1 - y_val_n[mask]

    model,accuracy_score = train_catboost_model(X_train, y_train,X_val,y_val_n)
    model_path = './mnt/trained_catboost_model_n.cbm'
    model.save_model(model_path)
    print(f"模型已保存至 {model_path}")

    print("\n进行预测...")
    predictions, probabilities = predict_with_model(model, X_test)
    results_df = []
    result_df = pd.DataFrame({
        '标题': title,
        '人工结果': x_real,
        '预测结果': predictions,
        '预测概率': probabilities,
        '模型准确率': accuracy_score

    })
    # 创建保存器实例
    results_saver = MLResultsSaver(base_dir='./results', file_prefix='hook_review')
    # 保存结果
    saved_file_path = results_saver.save_results(result_df, custom_suffix='catboost_results')
    result_df.to_csv("catboost_predictions.csv", index=False)
    print("预测结果已保存至 catboost_predictions.csv")