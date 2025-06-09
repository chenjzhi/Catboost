# seed_testing.py
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from csv_save import MLResultsSaver
import os
from datetime import datetime


def load_and_preprocess_data(train_file, test_file=None):
    """加载数据并进行简单预处理"""
    train_data = pd.read_csv(train_file)
    X = train_data.drop(['人工审核结果'], axis=1)
    y = train_data['人工审核结果']
    y = y.map({'不通过': 0, '通过': 1})

    if test_file:
        test_data = pd.read_csv(test_file)
        X_test = test_data.drop(['人工审核结果'], axis=1) if '人工审核结果' in test_data.columns else test_data
        return X, y, X_test

    return X, y


def train_catboost_model(X_train, y_train, X_val, y_val):
    """使用指定的随机种子训练CatBoost分类模型"""
    # 确定分类特征
    cat_features = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
            cat_features.append(col)

    # 获取分类特征索引
    cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features]

    text_features = ['正文'] if '正文' in X_train.columns else []

    # # 添加标签噪声，使用与seed相关的随机状态
    # rng = np.random.RandomState(random_seed + 100)  # 使用不同的seed偏移，确保噪声不同
    # mask = rng.random(len(y_val)) < 0.2
    # y_val_n = y_val.copy()
    # y_val_n[mask] = 1 - y_val_n[mask]

    # 创建CatBoost池
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices, text_features=text_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices, text_features=text_features)

    # 初始化模型，使用指定的random_seed
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        eval_metric='Accuracy',
        verbose=0,
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
    accuracy = accuracy_score(y_val_n, y_pred)

    return model, accuracy


def test_multiple_seeds(X, y, seeds, output_dir='./seed_test_results'):
    """测试多个随机种子并保存结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    # 固定的训练集/验证集划分，只改变模型的seed
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for seed in seeds:
        print(f"测试 seed = {seed}...")
        model, accuracy = train_catboost_model(X_train, y_train, X_val, y_val, seed)

        # 保存模型
        model_path = os.path.join(output_dir, f"model_seed_{seed}.cbm")
        model.save_model(model_path)

        results.append({
            'seed': seed,
            'accuracy': accuracy,
            'model_path': model_path
        })

    # 保存结果到CSV
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(output_dir, f"seed_test_results_{timestamp}.csv")
    results_df.to_csv(results_csv, index=False)

    # 使用MLResultsSaver也保存一份
    saver = MLResultsSaver(base_dir=output_dir, file_prefix='seed_test')
    saver.save_results(results_df, custom_suffix='accuracy')

    print(f"已完成所有seed测试，结果保存在: {results_csv}")
    return results_df


if __name__ == "__main__":
    train_file = "dataset/processed_data.csv"
    test_file = "dataset/test.csv"  # 可选，用于最终预测