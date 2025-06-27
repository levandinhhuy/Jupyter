# Cài đặt thư viện cần thiết
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Tải dữ liệu Iris
iris = load_iris()
X = iris.data  # Các đặc trưng
y = iris.target  # Nhãn

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình cây quyết định
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)  # Huấn luyện mô hình

# Dự đoán trên tập kiểm tra
y_pred = clf.predict(X_test)

# Tính độ chính xác
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')

# Vẽ cây quyết định
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Decision Tree for Iris Dataset')
plt.show()
