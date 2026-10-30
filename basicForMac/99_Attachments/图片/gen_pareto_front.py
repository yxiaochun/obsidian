import matplotlib.pyplot as plt
import matplotlib

# 中文字体配置
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

M1 = [(100, 80), (96, 92), (92, 97), (85, 99), (78, 100)]   # 旧模型 (广告收入, PGMV)
M2 = [(100, 88), (97, 95), (94, 99), (90, 101), (86, 102)]  # 新模型


def pareto_front(points):
    pts = sorted(set(points), reverse=True)
    front, best_g = [], float('-inf')
    for rev, g in pts:
        if g > best_g:
            front.append((rev, g))
            best_g = g
    return front


fig, ax = plt.subplots(figsize=(8, 6))

for pts, name, color in [(M1, 'M1 旧模型', '#5B8FF9'), (M2, 'M2 新模型', '#F6A23B')]:
    f = pareto_front(pts)
    x, y = zip(*f)
    ax.plot(x, y, '-o', label=name, color=color, linewidth=2, markersize=8)
    for rev, g in f:
        ax.annotate(f'({rev}, {g})', (rev, g), textcoords='offset points',
                    xytext=(6, -12), fontsize=9, color=color)

# 理想点与参考点（用于说明 Hypervolume / 膝盖点）
ax.scatter([100], [102], marker='*', s=200, color='#7BE188', zorder=5, label='理想点')
ax.scatter([76], [88], marker='x', s=80, color='#999999', zorder=5, label='参考点(坏点)')

ax.set_xlabel('广告收入（相对值）', fontsize=12)
ax.set_ylabel('PGMV（相对值）', fontsize=12)
ax.set_title('帕累托前沿对比：M1 vs M2（扫权重 k 采样）', fontsize=13)
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')
ax.set_xlim(73, 103)
ax.set_ylim(75, 104)

plt.tight_layout()
plt.savefig('/Users/yangxiaochun/Documents/obsidian/basicForMac/99_Attachments/图片/帕累托前沿对比示例-M1vsM2.png',
            dpi=150, bbox_inches='tight')
print('saved')
