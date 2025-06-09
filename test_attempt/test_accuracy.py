import os
import pandas as pd
from glob import glob
from csv_save import MLResultsSaver


def extract_accuracy_and_seed(folder_path):
    """提取指定文件夹中所有结果文件的准确率和种子值"""
    # 获取所有结果文件
    result_files = glob(os.path.join(folder_path, "*.csv"))

    # 存储提取的数据
    accuracy_data = []

    # 遍历所有结果文件
    for file_path in result_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 确保文件包含所需的列
            if '模型准确率' in df.columns and '种子值' in df.columns:
                # 提取第一行的准确率和种子值（假设所有行的这些值相同）
                accuracy = df['模型准确率'].iloc[0]
                seed = df['种子值'].iloc[0]

                # 添加到结果列表
                accuracy_data.append({
                    '种子值': seed,
                    '模型准确率': accuracy,
                    '来源文件': os.path.basename(file_path)
                })
            else:
                print(f"警告: 文件 {file_path} 不包含'模型准确率'或'种子值'列")
        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {e}")

    return accuracy_data


def main():
    # 结果文件夹路径
    test_folder = '../results/test'

    # 提取准确率和种子值
    print(f"从 {test_folder} 提取数据...")
    accuracy_data = extract_accuracy_and_seed(test_folder)

    # 检查是否提取到数据
    if not accuracy_data:
        print("未找到包含准确率和种子值的文件！")
        return

    # 创建DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)

    # 按准确率降序排序
    sorted_df = accuracy_df.sort_values(by='模型准确率', ascending=False)

    # 重置索引
    sorted_df = sorted_df.reset_index(drop=True)

    # 添加排名列
    sorted_df['排名'] = sorted_df.index + 1

    # 重命名列
    sorted_df = sorted_df.rename(columns={
        '模型准确率': '准确率',
        '来源文件': '结果文件'
    })

    # 选择列顺序
    sorted_df = sorted_df[['排名', '种子值', '准确率', '结果文件']]

    # 保存结果
    results_saver = MLResultsSaver(base_dir='../results/test', file_prefix='accuracy_summary')
    saved_file_path = results_saver.save_results(sorted_df, custom_suffix='all_seeds')

    print(f"\n准确率汇总已保存至: {saved_file_path}")
    print("\n前5个最佳模型:")
    print(sorted_df.head().to_string(index=False))


if __name__ == "__main__":
    main()