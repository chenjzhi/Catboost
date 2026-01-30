import pandas as pd
import torch


def main_plus():
    from deeplotx import RecursiveSequential, LongTextEncoder

    long_text_encoder = LongTextEncoder(
        max_length=4096,
        chunk_size=448,
        overlapping=32,
        model_name_or_path='hfl/chinese-bert-wwm-ext',
        device='cpu'
    )

    _test_embedding_dim = long_text_encoder.encode('test', flatten=False).shape[-1]
    model = RecursiveSequential(input_dim=_test_embedding_dim, output_dim=1, recursive_layers=4, hidden_dim=512,
                                model_name=r'E:\project\Catboost\RecursiveSequential2', device='cpu').load()
    # 测试
    test_set = pd.read_csv("../dataset/海剧.csv").to_dict(orient='records') + pd.read_csv("../dataset/processed_data.csv").to_dict(orient='records')
    acc_count = 0
    for test_item in test_set:
        score = torch.sigmoid(model.predict(long_text_encoder.encode(test_item['正文'], flatten=False)))
        print(f'{test_item['标题']}\tScore={score}')
        if score >= 0.5:
            if test_item['人工审核结果'] == 'T':
                acc_count += 1
        else:
            if test_item['人工审核结果'] == 'F':
                acc_count += 1
    print(f'ACC={acc_count / (len(test_set) + 1e-10)}')


if __name__ == '__main__':
    main_plus()