import os

from sklearn.metrics import accuracy_score

from config import train_file, text_columns, force_recompute_embeddings, catboost_model_path, knn_model_path, test_file, \
    results_dir, results_prefix, results_suffix
from csv_save import MLResultsSaver
from data_processor import DataProcessor
from knn_model import KNNModel  # 导入新的KNN模块
from model_predictor import ModelPredictor
from model_trainer import ModelTrainer
from text_embedder import TextEmbedder


def main():
    # 初始化处理器
    embedder = TextEmbedder(embeddings_dir="embeddings")
    data_processor = DataProcessor()

    # 加载和预处理训练数据
    train_df = data_processor.load_data(train_file)
    X, y = data_processor.preprocess_train_data(train_df)

    # 处理文本嵌入
    X = embedder.process_dataframe(
        X,
        text_columns,
        train_file,
        force_recompute_embeddings
    )

    # 处理分类特征
    X = data_processor.process_categorical_features(X)

    print("\n数据类型检查:")
    X.info()

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = data_processor.split_data(X, y)

    # 训练 CatBoost 模型
    model_trainer = ModelTrainer()
    catboost_model, catboost_accuracy = model_trainer.train(X_train, y_train, X_val, y_val)
    model_trainer.save_model(catboost_model_path)

    # 训练 K 近邻模型
    knn = KNNModel(n_neighbors=6, weights='uniform', metric='manhattan')
    knn_model, knn_accuracy = knn.train(X_train, y_train, X_val, y_val)
    knn.save_model(knn_model_path)
    print(f"K 近邻模型验证集准确率: {knn_accuracy:.4f}")

    # 加载和预处理测试数据
    if test_file and os.path.exists(test_file):
        test_df = data_processor.load_data(test_file)

        # 保存原始测试数据用于结果展示
        original_test_df = test_df.copy()

        # 处理文本嵌入
        test_df = embedder.process_dataframe(
            test_df,
            text_columns,
            test_file,
            force_recompute_embeddings
        )

        # 预处理测试数据
        X_test, X_real, title, content = data_processor.preprocess_test_data(test_df)

        # 进行预测
        catboost_predictor = ModelPredictor(catboost_model)
        catboost_predictions, catboost_probabilities = catboost_predictor.predict(X_test)

        # 使用KNN模型进行预测
        knn_predictions, knn_probabilities = knn.predict(X_test)

        # 集成预测结果（调整权重）
        final_probabilities = 0.6 * catboost_probabilities + 0.4 * knn_probabilities
        final_predictions = (final_probabilities >= 0.5).astype(int)
        final_pred_labels = ['不通过' if label == 0 else '通过' for label in final_predictions]
        label_mapping = {'F': '不通过', 'T': '通过', '不通过': '不通过', '通过': '通过'}
        X_real = X_real.map(label_mapping)
        test_accuracy = accuracy_score(X_real, final_pred_labels)
        print('测试集准确率', test_accuracy)

        # 生成结果
        # 生成结果（包含所有样本，包括未知标签的样本）
        result_df = catboost_predictor.generate_results(
            final_pred_labels,
            final_probabilities,
            test_accuracy,
            title=original_test_df['标题'] if '标题' in original_test_df.columns else None,
            content=original_test_df['正文'] if '正文' in original_test_df.columns else None,
            theme=original_test_df['题材'] if '题材' in original_test_df.columns else None,
            y_real=X_real
        )

        # 保存结果
        result_df.to_csv("catboost_predictions.csv", index=False)
        print("预测结果已保存至 catboost_predictions.csv")

        # 使用 MLResultsSaver 保存结果
        results_saver = MLResultsSaver(
            base_dir=results_dir,
            file_prefix=results_prefix
        )
        saved_file_path = results_saver.save_results(
            result_df,
            custom_suffix=results_suffix
        )
        print(f"结果已保存至 {saved_file_path}")


if __name__ == "__main__":
    main()
