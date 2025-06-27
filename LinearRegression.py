# Cài đặt thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Tạo dữ liệu mẫu
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Biến độc lập
y = 4 + 3 * X + np.random.randn(100, 1)  # Biến phụ thuộc với chút nhiễu

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)  # Huấn luyện mô hình

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Vẽ biểu đồ
plt.scatter(X_test, y_test, color='blue', label='Dữ liệu thực tế')
plt.plot(X_test, y_pred, color='red', label='Dự đoán')
plt.title('Hồi Quy Tuyến Tính')
plt.xlabel('Biến độc lập (X)')
plt.ylabel('Biến phụ thuộc (y)')
plt.legend()
plt.grid()
plt.show()

# In các tham số hồi quy
print(f'Hệ số chặn (b0): {model.intercept_[0]:.2f}')
print(f'Hệ số góc (b1): {model.coef_[0][0]:.2f}')
