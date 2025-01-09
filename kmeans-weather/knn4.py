import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 原始数据
data = [
{"city": "南京", "weather": "9日（今天）:多云转晴,4/-3℃,北风4-5级转3-4级\n10日（明天）:多云,4/-2℃,北风3-4级转<3级\n11日（后天）:多云,7/0℃,西南风3-4级转<3级\n12日（周日）:晴,10/-1℃,西北风<3级\n13日（周一）:晴转多云,13/2℃,东南风<3级\n14日（周二）:阴,11/3℃,东北风<3级\n15日（周三）:多云转晴,9/-2℃,东北风<3级\n"},
{"city": "南通", "weather": "9日（今天）:阴转多云,4/-3℃,西北风3-4级转<3级\n10日（明天）:晴转多云,5/-3℃,西北风<3级\n11日（后天）:多云,6/-2℃,西南风<3级\n12日（周日）:晴,9/-2℃,西北风<3级\n13日（周一）:晴,10/0℃,西南风<3级\n14日（周二）:多云转阴,12/1℃,东北风<3级\n15日（周三）:晴,8/-2℃,北风3-4级转<3级\n"},
{"city": "盐城", "weather": "9日（今天）:多云转晴,2/-5℃,西北风3-4级转<3级\n10日（明天）:晴转多云,3/-4℃,西北风<3级\n11日（后天）:多云,6/-2℃,西南风<3级\n12日（周日）:晴,8/-2℃,西北风<3级\n13日（周一）:晴转阴,10/-1℃,西南风<3级\n14日（周二）:阴转多云,9/1℃,东北风<3级\n15日（周三）:晴,7/-2℃,北风<3级\n"},
{"city": "苏州", "weather": "9日（今天）:多云,4/-2℃,西北风4-5级\n10日（明天）:多云,3/-1℃,西北风<3级\n11日（后天）:多云,5/0℃,西南风<3级\n12日（周日）:晴,10/-1℃,西风<3级\n13日（周一）:晴,11/2℃,东南风<3级\n14日（周二）:多云转阴,14/5℃,东北风<3级\n15日（周三）:多云转晴,9/0℃,北风<3级\n"},
{"city": "镇江", "weather": "9日（今天）:阴转晴,3/-4℃,北风<3级\n10日（明天）:晴转阴,3/-3℃,西北风<3级\n11日（后天）:多云,6/-1℃,西南风<3级\n12日（周日）:晴,9/-1℃,西北风<3级\n13日（周一）:晴转多云,11/-1℃,南风<3级\n14日（周二）:阴,10/1℃,东北风<3级\n15日（周三）:晴转多云,8/-1℃,北风<3级\n"},
{"city": "泰州", "weather": "9日（今天）:阴转晴,4/-6℃,北风3-4级转<3级\n10日（明天）:晴转阴,4/-5℃,西北风<3级\n11日（后天）:多云,6/-2℃,西南风<3级\n12日（周日）:晴,9/-2℃,西北风<3级\n13日（周一）:晴转多云,10/0℃,南风<3级\n14日（周二）:阴,12/0℃,东北风<3级\n15日（周三）:晴转多云,8/-2℃,北风<3级\n"},
{"city": "扬州", "weather": "9日（今天）:阴转晴,4/-2℃,北风<3级\n10日（明天）:晴转阴,4/-2℃,西北风<3级\n11日（后天）:晴转多云,7/0℃,西南风<3级\n12日（周日）:晴转多云,9/0℃,西北风<3级\n13日（周一）:晴转多云,11/0℃,南风<3级\n14日（周二）:阴,12/2℃,东北风<3级\n15日（周三）:晴转多云,8/0℃,北风<3级\n"},
{"city": "徐州", "weather": "9日（今天）:多云转晴,3/-4℃,北风<3级\n10日（明天）:晴转多云,4/-4℃,西风<3级\n11日（后天）:晴,7/-2℃,西南风<3级\n12日（周日）:晴,6/-1℃,西北风<3级\n13日（周一）:晴转阴,10/0℃,西南风<3级\n14日（周二）:阴转晴,8/-2℃,东北风<3级\n15日（周三）:晴,5/-2℃,北风<3级\n"},
{"city": "无锡", "weather": "9日（今天）:阴转晴,4/-3℃,西北风<3级\n10日（明天）:晴转阴,4/-2℃,西北风<3级\n11日（后天）:多云,6/0℃,西南风<3级\n12日（周日）:晴,9/0℃,西北风<3级\n13日（周一）:晴,11/1℃,东南风<3级\n14日（周二）:阴,13/3℃,东北风<3级\n15日（周三）:晴转多云,9/0℃,北风<3级\n"},
{"city": "连云港", "weather": "9日（今天）:晴,2/-7℃,北风3-4级转<3级\n10日（明天）:晴,4/-5℃,西风<3级\n11日（后天）:多云,7/-3℃,西南风<3级\n12日（周日）:晴,7/-2℃,西北风<3级\n13日（周一）:晴转阴,9/-1℃,西南风<3级\n14日（周二）:阴转多云,6/-1℃,东北风<3级\n15日（周三）:晴,5/-4℃,北风<3级\n"},
{"city": "宿迁", "weather": "9日（今天）:多云转晴,3/-6℃,北风<3级\n10日（明天）:晴转多云,4/-4℃,西风<3级\n11日（后天）:晴,8/-2℃,西南风<3级\n12日（周日）:晴,7/-2℃,西北风<3级\n13日（周一）:多云转阴,10/0℃,南风<3级\n14日（周二）:阴转多云,7/-1℃,东北风<3级\n15日（周三）:晴,5/-2℃,北风<3级\n"},
{"city": "常州", "weather": "9日（今天）:多云,3/-3℃,北风<3级\n10日（明天）:多云,4/-2℃,西北风<3级\n11日（后天）:阴转多云,6/-1℃,西南风<3级\n12日（周日）:多云转晴,9/-1℃,西北风<3级\n13日（周一）:多云,11/1℃,南风<3级\n14日（周二）:阴,11/2℃,东北风<3级\n15日（周三）:阴转多云,9/-1℃,北风<3级\n"},
{"city": "淮安", "weather": "9日（今天）:阴转晴,2/-6℃,北风3-4级\n10日（明天）:晴转多云,4/-5℃,西北风<3级\n11日（后天）:多云,6/-3℃,西南风<3级\n12日（周日）:晴,8/-1℃,西北风<3级\n13日（周一）:多云转阴,10/0℃,南风<3级\n14日（周二）:阴转多云,8/0℃,东北风<3级\n15日（周三）:晴,7/-2℃,北风<3级\n"}
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
        high_temp = int(temp_range[0].strip('℃'))
        low_temp = int(temp_range[1].strip('℃'))
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