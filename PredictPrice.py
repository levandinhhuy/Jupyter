import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

area = np.array([50, 80, 100, 150, 200]).reshape(-1, 1)
price = np.array([1.2, 2.0, 2.5, 3.0, 5.0])

model = LinearRegression()
model.fit(area, price)

beta_1 = model.coef_[0]
beta_0 = model.intercept_

area_to_predict = np.array([[120]])
predicted_price = model.predict(area_to_predict)

plt.scatter(area, price, color='blue', label='Dữ liệu thực tế')
plt.plot(area, model.predict(area), color='red', label=f'Đường hồi quy: y = {beta_1:.3f}x + {beta_0:.3f}')
plt.scatter(area_to_predict, predicted_price, color='green', label=f'Dự đoán (120m2): {predicted_price[0]:.2f} triệu đồng')

plt.title('Dự đoán giá nhà dựa trên diện tích')
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (triệu VND)')
plt.legend()

plt.show()

print(f'Dự đoán giá nhà cho diện tích 120m2: {predicted_price[0]:.2f} triệu VND')