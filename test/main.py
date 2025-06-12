import os


from text_embedder import TextEmbedder
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from model_predictor import ModelPredictor
from csv_save import MLResultsSaver
from sklearn.metrics import accuracy_score

def main():
    # 配置参数
    config = {
        'train_file': "../dataset/processed_data.csv",
        'test_file': "../dataset/海剧.csv",
        'model_path': '../mnt/trained_catboost_model_n1.cbm',
        'text_columns': ['正文', '标题'],
        'force_recompute_embeddings': False,
        'results_dir': '../results',
        'results_prefix': 'hook_review',
        'results_suffix': 'catboost_results'
    }

    # 初始化处理器
    embedder = TextEmbedder(embeddings_dir="embeddings")
    data_processor = DataProcessor()

    # 加载和预处理训练数据
    train_df = data_processor.load_data(config['train_file'])
    X, y = data_processor.preprocess_train_data(train_df)

    # 处理文本嵌入
    X = embedder.process_dataframe(
        X,
        config['text_columns'],
        config['train_file'],
        config['force_recompute_embeddings']
    )

    # 处理分类特征
    X = data_processor.process_categorical_features(X)

    print("\n数据类型检查:")
    X.info()

    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = data_processor.split_data(X, y)

    # 训练模型
    model_trainer = ModelTrainer()
    model, accuracy = model_trainer.train(X_train, y_train, X_val, y_val)
    model_trainer.save_model(config['model_path'])

    # 加载和预处理测试数据
    if config['test_file'] and os.path.exists(config['test_file']):
        test_df = data_processor.load_data(config['test_file'])

        # 保存原始测试数据用于结果展示
        original_test_df = test_df.copy()

        # 处理文本嵌入
        test_df = embedder.process_dataframe(
            test_df,
            config['text_columns'],
            config['test_file'],
            config['force_recompute_embeddings']
        )

        # 预处理测试数据
        X_test, X_real, title, content = data_processor.preprocess_test_data(test_df)

        # 进行预测
        predictor = ModelPredictor(model)
        predictions, probabilities = predictor.predict(X_test)

        X_real = X_real.map({'F': '不通过', 'T': '通过'})
        test_accuracy = accuracy_score(X_real, predictions)
        print('测试集准确率',test_accuracy)
        # 生成结果
        result_df = predictor.generate_results(
            predictions,
            probabilities,
            test_accuracy,
            title=original_test_df['标题'] if '标题' in original_test_df.columns else None,
            content=original_test_df['正文'] if '正文' in original_test_df.columns else None,
            y_real=X_real
        )

        # 保存结果
        result_df.to_csv("catboost_predictions.csv", index=False)
        print("预测结果已保存至 catboost_predictions.csv")

        # 使用MLResultsSaver保存结果
        results_saver = MLResultsSaver(
            base_dir=config['results_dir'],
            file_prefix=config['results_prefix']
        )
        saved_file_path = results_saver.save_results(
            result_df,
            custom_suffix=config['results_suffix']
        )
        print(f"结果已保存至 {saved_file_path}")


if __name__ == "__main__":
    main()
