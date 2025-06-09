import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from catboost_train import predict_with_model, train_catboost_model, load_and_preprocess_data
from csv_save import MLResultsSaver

for seeds in range(0,500,10):
    train_file = "../dataset/processed_data.csv"
    test_file = "../dataset/test.csv"

    print("加载数据...")
    X, y, X_test = load_and_preprocess_data(train_file, test_file)

    print("\n数据类型检查:")
    X.info()

    cat_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    for col in cat_features:
        X[col] = X[col].astype(str)

    print("\n训练CatBoost模型...")
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    np.random.seed(seeds)
    mask = np.random.random(len(y_val)) < 0.2
    y_val_n = y_val.copy()
    y_val_n[mask] = 1 - y_val_n[mask]

    model, accuracy_score = train_catboost_model(X_train, y_train, X_val, y_val_n)

    model_path = '../mnt/trained_catboost_model_n.cbm'
    model.save_model(model_path)
    print(f"模型已保存至 {model_path}")

    print("\n进行预测...")
    predictions, probabilities = predict_with_model(model, X_test)

    result_df = pd.DataFrame({
        '预测结果': predictions,
        '预测概率': probabilities,
        '模型准确率': accuracy_score,
        "种子值": seeds
    })
    # 创建保存器实例
    results_saver = MLResultsSaver(base_dir='../results/test', file_prefix='hook_review')
    # 保存结果
    saved_file_path = results_saver.save_results(result_df, custom_suffix='catboost_results')
    result_df.to_csv("catboost_predictions.csv", index=False)
    print("预测结果已保存至 catboost_predictions.csv")