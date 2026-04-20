# taobao_portfolio.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("淘宝用户行为分析 - 作品集项目")
print("=" * 60)

# ========== 1. 读取数据 ==========
print("\n[1/6] 读取数据...")
df = pd.read_csv('淘宝用户行为.csv')
df['invoice_date'] = pd.to_datetime(df['invoice_date'])
df['total_amount'] = df['quantity'] * df['price']
print(f"数据量: {len(df)} 行, {df['customer_id'].nunique()} 个用户")

# ========== 2. 模块一：用户画像分析 ==========
print("\n[2/6] 模块一：用户画像分析...")

# 2.1 性别分析
gender_stats = df.groupby('gender').agg({
    'total_amount': 'sum',
    'quantity': 'sum',
    'customer_id': 'nunique'
}).round(2)
gender_stats['客单价'] = gender_stats['total_amount'] / gender_stats['customer_id']

# 2.2 年龄分段
bins = [18, 25, 35, 45, 55, 100]
labels = ['18-24岁', '25-34岁', '35-44岁', '45-54岁', '55岁以上']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

age_stats = df.groupby('age_group', observed=True).agg({
    'total_amount': 'sum',
    'customer_id': 'nunique',
    'quantity': 'sum'
}).round(2)
age_stats['人均消费'] = age_stats['total_amount'] / age_stats['customer_id']

# 2.3 交叉分析：年龄+性别
cross_stats = df.groupby(['age_group', 'gender'], observed=True)['total_amount'].sum().unstack().fillna(0)

# ========== 3. 模块二：品类分析 ==========
print("\n[3/6] 模块二：品类分析...")

category_stats = df.groupby('category').agg({
    'total_amount': 'sum',
    'quantity': 'sum',
    'customer_id': 'nunique'
}).round(2).sort_values('total_amount', ascending=False)

category_stats['客单价'] = category_stats['total_amount'] / category_stats['quantity']
category_stats['销量占比'] = (category_stats['quantity'] / category_stats['quantity'].sum() * 100).round(1)

# 品类-性别偏好
category_gender = df.groupby(['category', 'gender'])['quantity'].sum().unstack().fillna(0)
category_gender['女性占比'] = (category_gender['Female'] / (category_gender['Female'] + category_gender['Male']) * 100).round(1)

# ========== 4. 模块三：RFM用户分层 ==========
print("\n[4/6] 模块三：RFM用户分层...")

current_date = df['invoice_date'].max()

# 修复：reset_index() 保留customer_id作为普通列
rfm = df.groupby('customer_id').agg({
    'invoice_date': lambda x: (current_date - x.max()).days,
    'invoice_no': 'nunique',
    'total_amount': 'sum'
}).rename(columns={
    'invoice_date': 'recency',
    'invoice_no': 'frequency',
    'total_amount': 'monetary'
}).reset_index()  # 这行很重要！把customer_id变回普通列

# RFM打分
rfm['R_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])

rfm['RFM_score'] = rfm['R_score'].astype(int) + rfm['F_score'].astype(int) + rfm['M_score'].astype(int)

def rfm_segment(row):
    if row['RFM_score'] >= 13:
        return '高价值用户'
    elif row['RFM_score'] >= 10:
        return '潜力用户'
    elif row['RFM_score'] >= 7:
        return '一般用户'
    else:
        return '低活跃用户'

rfm['segment'] = rfm.apply(rfm_segment, axis=1)

# 修复：用customer_id计数，而不是用列名
segment_stats = rfm.groupby('segment').agg({
    'monetary': 'sum',
    'customer_id': 'count'  # 这里用customer_id
}).round(2)
segment_stats = segment_stats.rename(columns={'customer_id': '用户数', 'monetary': '消费总额'})
segment_stats['消费占比'] = (segment_stats['消费总额'] / segment_stats['消费总额'].sum() * 100).round(1)
segment_stats['人数占比'] = (segment_stats['用户数'] / segment_stats['用户数'].sum() * 100).round(1)

# ========== 5. 模块四：时间趋势分析 ==========
print("\n[5/6] 模块四：时间趋势分析...")

df['year_month'] = df['invoice_date'].dt.to_period('M')
monthly_sales = df.groupby('year_month')['total_amount'].sum()

df['weekday'] = df['invoice_date'].dt.dayofweek
weekday_map = {0:'周一',1:'周二',2:'周三',3:'周四',4:'周五',5:'周六',6:'周日'}
df['weekday_name'] = df['weekday'].map(weekday_map)
weekday_sales = df.groupby('weekday_name')['total_amount'].sum()
weekday_sales = weekday_sales.reindex(['周一','周二','周三','周四','周五','周六','周日'])

# 找出销售高峰月
peak_months = monthly_sales.sort_values(ascending=False).head(3)

# ========== 6. 生成所有图表 ==========
print("\n[6/6] 生成图表...")

fig = plt.figure(figsize=(16, 14))

# 图1：性别销售额对比
ax1 = fig.add_subplot(3, 3, 1)
colors = ['#FF6B6B', '#4ECDC4']
bars = ax1.bar(gender_stats.index, gender_stats['total_amount'], color=colors)
ax1.set_title('图1: 不同性别销售额对比', fontsize=12, fontweight='bold')
ax1.set_ylabel('销售额(元)')
for bar, val in zip(bars, gender_stats['total_amount']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val/10000:.1f}万', ha='center', va='bottom')

# 图2：年龄段人均消费
ax2 = fig.add_subplot(3, 3, 2)
bars = ax2.bar(age_stats.index, age_stats['人均消费'], color='#95E1D3')
ax2.set_title('图2: 不同年龄段人均消费', fontsize=12, fontweight='bold')
ax2.set_ylabel('人均消费(元)')
ax2.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, age_stats['人均消费']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}', ha='center', va='bottom')

# 图3：年龄+性别交叉分析（堆叠柱状图）
ax3 = fig.add_subplot(3, 3, 3)
cross_stats.plot(kind='bar', stacked=True, ax=ax3, color=['#FF6B6B', '#4ECDC4'])
ax3.set_title('图3: 各年龄段性别消费对比', fontsize=12, fontweight='bold')
ax3.set_ylabel('销售额(元)')
ax3.tick_params(axis='x', rotation=45)
ax3.legend(title='性别')

# 图4：品类销售额TOP8
ax4 = fig.add_subplot(3, 3, 4)
top8 = category_stats.head(8)
bars = ax4.barh(range(len(top8)), top8['total_amount'].values/10000)
ax4.set_yticks(range(len(top8)))
ax4.set_yticklabels(top8.index)
ax4.set_title('图4: 品类销售额TOP8', fontsize=12, fontweight='bold')
ax4.set_xlabel('销售额(万元)')
for i, (bar, val) in enumerate(zip(bars, top8['total_amount'].values/10000)):
    ax4.text(bar.get_width(), i, f'{val:.1f}万', va='center')

# 图5：品类客单价对比
ax5 = fig.add_subplot(3, 3, 5)
top8_price = top8.sort_values('客单价', ascending=True)
bars = ax5.barh(range(len(top8_price)), top8_price['客单价'].values)
ax5.set_yticks(range(len(top8_price)))
ax5.set_yticklabels(top8_price.index)
ax5.set_title('图5: 品类客单价对比', fontsize=12, fontweight='bold')
ax5.set_xlabel('客单价(元)')

# 图6：品类性别偏好
ax6 = fig.add_subplot(3, 3, 6)
top8_gender = category_gender.loc[top8.index]
top8_gender[['Female', 'Male']].plot(kind='bar', ax=ax6, color=['#FF6B6B', '#4ECDC4'])
ax6.set_title('图6: TOP8品类性别偏好', fontsize=12, fontweight='bold')
ax6.set_ylabel('销量')
ax6.tick_params(axis='x', rotation=45)
ax6.legend(title='性别')

# 图7：RFM用户分层
ax7 = fig.add_subplot(3, 3, 7)
segment_colors = ['#FF6B6B', '#FFB347', '#95E1D3', '#4ECDC4']
bars = ax7.bar(segment_stats.index, segment_stats['消费占比'], color=segment_colors)
ax7.set_title('图7: 不同层级用户消费贡献', fontsize=12, fontweight='bold')
ax7.set_ylabel('消费占比(%)')
for bar, val in zip(bars, segment_stats['消费占比']):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val}%', ha='center', va='bottom')

# 图8：月度销售额趋势
ax8 = fig.add_subplot(3, 3, 8)
months = monthly_sales.index.astype(str)
ax8.plot(months, monthly_sales.values/10000, marker='o', linewidth=2, color='#FF6B6B')
ax8.set_title('图8: 月度销售额趋势', fontsize=12, fontweight='bold')
ax8.set_ylabel('销售额(万元)')
ax8.tick_params(axis='x', rotation=45)

# 标注高峰月
for month in peak_months.index:
    month_str = str(month)
    value = monthly_sales[month]/10000
    ax8.annotate(f'峰值:{value:.0f}万', xy=(month_str, value), xytext=(10, 5), 
                 textcoords='offset points', fontsize=8, color='red')

# 图9：星期销售额分布
ax9 = fig.add_subplot(3, 3, 9)
bars = ax9.bar(weekday_sales.index, weekday_sales.values/10000, color='#95E1D3')
ax9.set_title('图9: 星期销售额分布', fontsize=12, fontweight='bold')
ax9.set_ylabel('销售额(万元)')
for bar, val in zip(bars, weekday_sales.values/10000):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}万', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('taobao_analysis_all_charts.png', dpi=150, bbox_inches='tight')
print("图表已保存: taobao_analysis_all_charts.png")
plt.close()

# ========== 7. 导出Excel ==========
print("\n导出Excel文件...")

with pd.ExcelWriter('taobao_analysis_portfolio.xlsx', engine='openpyxl') as writer:
    # 用户画像
    gender_stats.to_excel(writer, sheet_name='1_性别分析')
    age_stats.to_excel(writer, sheet_name='2_年龄段分析')
    cross_stats.to_excel(writer, sheet_name='3_性别年龄交叉')
    
    # 品类分析
    category_stats.to_excel(writer, sheet_name='4_品类分析')
    category_gender.to_excel(writer, sheet_name='5_品类性别偏好')
    
    # RFM分层
    segment_stats.to_excel(writer, sheet_name='6_RFM分层统计')
    rfm.head(100).to_excel(writer, sheet_name='7_RFM用户明细TOP100')
    
    # 时间趋势
    monthly_sales.to_frame('销售额').to_excel(writer, sheet_name='8_月度趋势')
    weekday_sales.to_frame('销售额').to_excel(writer, sheet_name='9_星期趋势')

print("Excel已保存: taobao_analysis_portfolio.xlsx")

# ========== 8. 打印核心结论 ==========
print("\n" + "=" * 60)
print("【核心分析结论】")
print("=" * 60)

print("\n1. 用户画像:")
print(f"   - 女性用户贡献销售额: {gender_stats.loc['Female', 'total_amount']/10000:.1f}万")
print(f"   - 男性用户贡献销售额: {gender_stats.loc['Male', 'total_amount']/10000:.1f}万")
top_age = age_stats['人均消费'].idxmax()
print(f"   - {top_age}人均消费最高: {age_stats.loc[top_age, '人均消费']:.0f}元")

print("\n2. 品类表现:")
top_category = category_stats.index[0]
print(f"   - 销售额最高品类: {top_category}, {category_stats.loc[top_category, 'total_amount']/10000:.1f}万")
high_price_cat = category_stats['客单价'].idxmax()
print(f"   - 客单价最高品类: {high_price_cat}, {category_stats.loc[high_price_cat, '客单价']:.0f}元")

print("\n3. 用户分层:")
high_value_pct = segment_stats.loc['高价值用户', '人数占比']
high_value_sales = segment_stats.loc['高价值用户', '消费占比']
print(f"   - 高价值用户占比: {high_value_pct}%, 贡献销售额: {high_value_sales}%")

print("\n4. 时间规律:")
peak_month = peak_months.index[0]
print(f"   - 销售最高峰: {peak_month}")
best_weekday = weekday_sales.idxmax()
print(f"   - 一周中销售额最高: {best_weekday}")

print("\n" + "=" * 60)
print("【业务建议】")
print("=" * 60)
print("""
1. 精准营销: 重点针对25-34岁女性用户推送Technology品类优惠
2. 组合销售: 购买Food & Beverage的用户推荐搭配Technology产品
3. VIP运营: 对高价值用户建立专属社群，发放专属优惠券
4. 活动节奏: 大促安排在12月，周末加大投放力度
""")

print("\n✅ 分析完成! 生成的文件:")
print("   - taobao_analysis_all_charts.png (9张组合图表)")
print("   - taobao_analysis_portfolio.xlsx (详细数据)")