import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt

df1 = pd.read_excel('数据.xlsx', sheet_name='sheet1')


date_1 = [df1[col].to_list() for col in df1.columns]


a = date_1[0][0:1000]
c = []
for i in range(1000):
    b = a[i]
    if i == 0:
        c.append(b)
    else:
        c.append(c[-1] * 0.9 + b * 0.1)
fig, ax = plt.subplots()
ax.set_xlabel('Episodes')
ax.set_ylabel('Rewards')
plt.plot(c)
#plt.plot(a)
plt.show()


