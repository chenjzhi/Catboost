from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNModel:
    """K近邻模型封装类，用于训练、预测和评估KNN模型"""

    def __init__(self, n_neighbors=5, weights='uniform', metric='manhattan'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = None
        self.scaler = None

    def train(self, X_train, y_train, X_val, y_val, preprocess=True):
        # 初始化KNN模型
        self.model = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        return self.model, accuracy

    def predict(self, X_test):
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        y_pred_labels = ['不通过' if label == 0 else '通过' for label in y_pred]
        return y_pred_labels, y_proba

    def save_model(self, model_path):
        if self.model:
            import joblib
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'n_neighbors': self.n_neighbors,
                'weights': self.weights,
                'metric': self.metric
            }
            joblib.dump(model_data, model_path)
            print(f"KNN模型已保存至 {model_path}")
            return True
        return False

    @classmethod
    def load_model(cls, model_path):
        import joblib
        model_data = joblib.load(model_path)
        knn_model = cls(
            n_neighbors=model_data['n_neighbors'],
            weights=model_data['weights'],
            metric=model_data['metric']
        )
        knn_model.model = model_data['model']
        knn_model.scaler = model_data['scaler']
        return knn_model
