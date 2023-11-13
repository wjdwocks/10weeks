import numpy as np
import matplotlib.pyplot as plt
from sklearn import (linear_model, metrics)

true_line = lambda x : -2/3 * x + 14/3
data_range = np.array([-4,12])
data_num = 100
noise_std = 0.5

x = np.random.uniform(data_range[0], data_range[1], size=data_num)
y = true_line(x)

xn = x + np.random.normal(scale=noise_std, size= x.shape)
yn = y + np.random.normal(scale=noise_std, size = y.shape)

model = linear_model.LinearRegression()
model.fit(xn.reshape(-1,1), yn.reshape(-1,1))
score = model.score(xn.reshape(-1,1), yn.reshape(-1,1))

plt.title(f'Line: y={model.coef_[0][0]:.3f}*x + {model.intercept_[0]:.3f} (score={score:.3f})')
plt.plot(data_range, true_line(data_range), 'r-', label = 'The true line')
plt.plot(xn, yn, 'b.', label='Noisy data')
plt.plot(data_range, model.coef_[0]*data_range + model.intercept_, 'g-', label = 'Estimate')
plt.xlim(data_range)
plt.legend()
plt.show()
        