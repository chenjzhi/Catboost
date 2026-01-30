import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


class TextEmbedder:
    """处理文本特征的嵌入转换"""

    def __init__(self, model_name="BAAI/bge-m3", embeddings_dir="embeddings"):
        self.model_name = model_name
        self.cache_dir = os.path.join(embeddings_dir, 'cache')
        self.embeddings_dir = embeddings_dir
        self.model = None

        # 创建嵌入向量保存目录
        for dir_path in [embeddings_dir, self.cache_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


    def load_model(self):
        """加载文本嵌入模型"""
        if self.model is None:
            print(f"加载{self.model_name}模型...")
            # self.model = SentenceTransformer(self.model_name,cache_folder=self.cache_dir)
            self.model = SentenceTransformer("./cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")

        return self.model


    def get_embeddings(self, texts, text_column, data_file, force_recompute=False):
        # 生成嵌入文件路径
        file_basename = os.path.basename(data_file)
        embeddings_file = os.path.join(
            self.embeddings_dir,
            f"{file_basename}_{text_column}_embeddings.npy"
        )

        # 检查是否可以加载预计算的嵌入向量
        if not force_recompute and os.path.exists(embeddings_file):
            print(f"从 {embeddings_file} 加载 {text_column} 的嵌入向量...")
            embeddings = np.load(embeddings_file)
        else:
            # 获取文本嵌入
            model = self.load_model()
            print(f"处理{text_column}列的嵌入...")
            embeddings = model.encode(texts, convert_to_numpy=True)

            # 保存嵌入向量
            np.save(embeddings_file, embeddings)
            print(f"{text_column}的嵌入向量已保存至 {embeddings_file}")

        return embeddings


    def process_dataframe(self, df, text_columns, data_file, force_recompute=False):

        # 确保所有文本列都存在且填充空值
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # 处理每个文本列
        embedding_features = {}
        for col in text_columns:
            if col in df.columns:
                embeddings = self.get_embeddings(
                    df[col].tolist(),
                    col,
                    data_file,
                    force_recompute
                )

                # 为每个嵌入维度创建新特征
                for i in range(embeddings.shape[1]):
                    embedding_features[f"{col}_emb_{i}"] = embeddings[:, i]

        # 使用pd.concat一次性添加所有嵌入特征
        if embedding_features:
            embeddings_df = pd.DataFrame(embedding_features, index=df.index)
            df = pd.concat([df, embeddings_df], axis=1)

            # 删除原始文本列
            df = df.drop(columns=text_columns)

        return df
