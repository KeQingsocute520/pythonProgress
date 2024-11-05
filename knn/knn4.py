import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 原始数据
data = [
    {"city": "南京", "weather": "4日（今天）:多云,10,东北风3-4级\n5日（明天）:晴,17/8,东北风3-4级转<3级\n6日（后天）:晴,17/7,东北风3-4级转<3级\n7日（周四）:晴转多云,17/8,东风3-4级转<3级\n8日（周五）:阴转多云,16/10,东北风<3级\n9日（周六）:多云转晴,21/11,东风<3级\n10日（周日）:晴,21/10,东北风<3级\n"},
    {"city": "南通", "weather": "4日（今天）:多云,11,东北风<3级\n5日（明天）:晴,17/7,北风<3级\n6日（后天）:晴转多云,17/6,北风<3级\n7日（周四）:多云,17/10,东北风<3级\n8日（周五）:阴转多云,18/12,东北风<3级\n9日（周六）:晴,21/12,东风<3级\n10日（周日）:晴,20/10,东北风<3级\n"},
    {"city": "扬州", "weather": "4日（今天）:多云,9,北风<3级\n5日（明天）:晴,18/5,北风<3级\n6日（后天）:晴,17/6,东北风<3级\n7日（周四）:晴转多云,18/8,东风<3级\n8日（周五）:阴转多云,17/11,东北风<3级\n9日（周六）:多云转晴,21/12,东风<3级\n10日（周日）:晴,21/9,东风<3级\n"},
    {"city": "无锡", "weather": "4日（今天）:多云,11,东北风<3级\n5日（明天）:晴,18/8,北风<3级\n6日（后天）:晴,18/8,东北风<3级\n7日（周四）:多云转晴,17/9,东北风<3级\n8日（周五）:阴,18/13,东北风<3级\n9日（周六）:多云转晴,21/12,东北风<3级\n10日（周日）:晴,21/10,东北风<3级\n"},
    {"city": "镇江", "weather": "4日（今天）:多云,10,北风<3级\n5日（明天）:晴,17/6,北风<3级\n6日（后天）:晴,16/7,东北风<3级\n7日（周四）:晴转阴,17/8,东风<3级\n8日（周五）:阴转多云,16/10,东北风<3级\n9日（周六）:晴,20/12,东风<3级\n10日（周日）:晴,20/10,东风<3级\n"},
    {"city": "徐州", "weather": "4日（今天）:多云,6,东北风<3级\n5日（明天）:晴,16/4,东风<3级\n6日（后天）:晴,17/5,东南风<3级\n7日（周四）:阴,17/8,东南风<3级\n8日（周五）:多云转晴,14/7,东风<3级\n9日（周六）:多云,18/9,东风<3级\n10日（周日）:多云转晴,20/10,南风<3级\n"},
    {"city": "盐城", "weather": "4日（今天）:多云,9,北风<3级\n5日（明天）:晴,18/5,北风<3级\n6日（后天）:晴,17/5,东北风<3级\n7日（周四）:晴转阴,16/8,东风<3级\n8日（周五）:阴,16/10,东风<3级\n9日（周六）:多云转晴,21/10,东风<3级\n10日（周日）:晴转多云,20/10,东北风<3级\n"},
    {"city": "苏州", "weather": "4日（今天）:多云,13,北风<3级\n5日（明天）:晴,18/10,北风<3级\n6日（后天）:晴,17/10,东北风<3级\n7日（周四）:多云转晴,17/11,东北风<3级\n8日（周五）:阴转小雨,19/13,东北风<3级\n9日（周六）:阴转晴,22/13,东北风<3级\n10日（周日）:晴,21/12,东北风<3级\n"},
    {"city": "淮安", "weather": "4日（今天）:多云,7,北风4-5级\n5日（明天）:晴,14/4,北风<3级\n6日（后天）:晴,17/4,东风<3级\n7日（周四）:晴转阴,16/7,东风<3级\n8日（周五）:阴转多云,16/10,东风<3级\n9日（周六）:多云,22/11,东风<3级\n10日（周日）:晴转多云,21/9,东风<3级\n"},
    {"city": "连云港", "weather": "4日（今天）:多云,6,北风<3级\n5日（明天）:晴,15/3,北风<3级\n6日（后天）:晴,17/5,东北风<3级\n7日（周四）:多云,18/7,东南风<3级\n8日（周五）:阴,17/8,东风<3级\n9日（周六）:多云,20/10,东风<3级\n10日（周日）:多云,20/9,东风<3级\n"},
    {"city": "泰州", "weather": "4日（今天）:多云,10,北风<3级\n5日（明天）:晴,16/6,北风<3级\n6日（后天）:晴,16/5,东北风<3级\n7日（周四）:晴转阴,17/7,东风<3级\n8日（周五）:阴转多云,16/10,东北风<3级\n9日（周六）:晴,21/11,东风<3级\n10日（周日）:晴,20/9,东风<3级\n"},
    {"city": "常州", "weather": "4日（今天）:多云,12,北风<3级\n5日（明天）:晴,17/8,北风<3级\n6日（后天）:晴,17/8,东北风<3级\n7日（周四）:晴转多云,17/9,东风<3级\n8日（周五）:阴转多云,18/12,东北风<3级\n9日（周六）:晴,21/11,东风<3级\n10日（周日）:晴,20/11,东风<3级\n"},
    {"city": "宿迁", "weather": "4日（今天）:多云,6,东北风<3级\n5日（明天）:晴,16/5,东北风<3级\n6日（后天）:晴,17/5,东风<3级\n7日（周四）:阴,17/8,东南风<3级\n8日（周五）:多云,15/8,东风<3级\n9日（周六）:晴转多云,20/11,东南风<3级\n10日（周日）:晴转多云,20/10,东南风<3级\n"}
]

# 解析数据
parsed_data = []
for item in data:
    city = item['city']
    weather_lines = item['weather'].split('\n')
    for line in weather_lines:
        parts = line.split(':')
        if len(parts) != 2:
            print(f"警告: 数据格式不正确 - {line}")
            continue
        date = parts[0]
        details = parts[1].split(',')
        if len(details) < 2:
            print(f"警告: 数据格式不正确 - {line}")
            continue
        temp_range = details[1].split('/')
        if len(temp_range) != 2:
            print(f"警告: 温度范围格式不正确 - {line}")
            continue
        high_temp = int(temp_range[0])
        low_temp = int(temp_range[1])
        parsed_data.append({
            'city': city,
            'date': date,
            'high_temp': high_temp,
            'low_temp': low_temp
        })

df = pd.DataFrame(parsed_data)

# 计算平均温度
average_temps = df.groupby('city').agg({'high_temp': 'mean', 'low_temp': 'mean'}).reset_index()
average_temps.columns = ['city', 'avg_high_temp', 'avg_low_temp']

# 绘制平均温度折线图
plt.figure(figsize=(12, 6))
plt.plot(average_temps['city'], average_temps['avg_high_temp'], marker='o', label='平均最高温度')
plt.plot(average_temps['city'], average_temps['avg_low_temp'], marker='o', label='平均最低温度')
plt.xlabel('城市')
plt.ylabel('温度 (°C)')
plt.title('各城市平均温度')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 绘制最高温度箱型图
plt.figure(figsize=(12, 6))
plt.boxplot([df[df['city'] == city]['high_temp'] for city in average_temps['city']], positions=range(len(average_temps)), widths=0.6, patch_artist=True, labels=average_temps['city'])
plt.xlabel('城市')
plt.ylabel('温度 (°C)')
plt.title('各城市最高温度箱型图')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 绘制最低温度箱型图
plt.figure(figsize=(12, 6))
plt.boxplot([df[df['city'] == city]['low_temp'] for city in average_temps['city']], positions=range(len(average_temps)), widths=0.6, patch_artist=True, labels=average_temps['city'])
plt.xlabel('城市')
plt.ylabel('温度 (°C)')
plt.title('各城市最低温度箱型图')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()