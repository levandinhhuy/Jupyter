import matplotlib.pyplot as plt
import numpy as np

categories = {'A', 'B', 'C', 'D', 'E'}
values = [4, 3 ,2, 5, 4]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()

values += values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

ax.fill(angles, values, color='skyblue', alpha=0.4)
ax.plot(angles, values, color='skyblue', linewidth=2)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

plt.title('Radar Chart')
plt.show()