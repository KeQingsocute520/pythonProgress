import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

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
def parse_weather_data(data):
    parsed_data = []
    for item in data:
        city = item['city']
        weather_lines = item['weather'].split('\n')
        for line in weather_lines:
            if not line.strip():
                continue
            parts = line.split(':')
            date = parts[0]
            details = parts[1].split(',')
            condition = details[0]
            temp_high_low = details[1].split('/')
            wind = details[2]
            parsed_data.append({
                'city': city,
                'date': date,
                'condition': condition,
                'temp_high': int(temp_high_low[0]),
                'temp_low': int(temp_high_low[1]) if len(temp_high_low) > 1 else None,
                'wind': wind
            })
    return parsed_data

# 转换数据
def encode_features(df):
    le_condition = LabelEncoder()
    le_wind = LabelEncoder()
    df['condition_encoded'] = le_condition.fit_transform(df['condition'])
    df['wind_encoded'] = le_wind.fit_transform(df['wind'])
    return df, le_condition, le_wind

# 解析数据
parsed_data = parse_weather_data(data)
df = pd.DataFrame(parsed_data)

# 编码特征
df, le_condition, le_wind = encode_features(df)

# 处理缺失值
df.fillna(df.mean(), inplace=True)

# 处理无穷大值
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# 标准化数据
scaler = StandardScaler()
df[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']] = scaler.fit_transform(df[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']])

# 特征和标签
X = df[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']]
y = df['city']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练KNN模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)

# 输出结果
print("预测结果:")
for i in range(len(y_test)):
    print(f"实际城市: {y_test.iloc[i]}, 预测城市: {y_pred[i]}")