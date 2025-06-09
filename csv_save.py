import os
import pandas as pd
from datetime import datetime


class MLResultsSaver:
    def __init__(self, base_dir='./ml_results', file_prefix='ml_results'):
        self.base_dir = base_dir
        self.file_prefix = file_prefix
        os.makedirs(self.base_dir, exist_ok=True)

    def _generate_filename(self, custom_suffix=None):
        # 获取当前时间戳，格式化为字符串
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 构建文件名
        if custom_suffix:
            filename = f"{self.file_prefix}_{timestamp}_{custom_suffix}.csv"
        else:
            filename = f"{self.file_prefix}_{timestamp}.csv"
        # 返回完整的文件路径
        return os.path.join(self.base_dir, filename)

    def save_results(self, results, custom_suffix=None):
        # 生成文件路径
        file_path = self._generate_filename(custom_suffix)
        # 保存为 CSV 文件
        results.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")
        # 返回保存的文件路径，方便后续使用
        return file_path

# 示例用法
if __name__ == "__main__":
    # 创建保存器实例
    results_saver = MLResultsSaver(base_dir='./my_ml_results', file_prefix='my_ml_experiment')

    data = {
        'model': ['Logistic Regression', 'Random Forest', 'SVM'],
        'accuracy': [0.85, 0.92, 0.88],
        'precision': [0.83, 0.90, 0.87],
        'recall': [0.87, 0.94, 0.89]
    }
    results_df = pd.DataFrame(data)

    # 保存结果
    saved_file_path = results_saver.save_results(results_df, custom_suffix='classification_metrics')
    print(f"Saved to: {saved_file_path}")