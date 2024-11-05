import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import sys
# 设置 Pandas 显示选项
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 自动调整列宽
pd.set_option('display.max_colwidth', None)  # 显示完整列内容

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


def parse_weather_data(data):
    parsed_data = []
    for entry in data:
        city = entry['city']
        weather_lines = entry['weather'].split('\n')
        for line in weather_lines:
            if line:
                parts = line.split(':')
                date = parts[0]
                weather_info = parts[1].split(',')
                condition = weather_info[0]
                temp = weather_info[1].split('/')
                wind = weather_info[2].strip()
                parsed_data.append({
                    'city': city,
                    'date': date,
                    'condition': condition,
                    'temp_high': int(temp[0]),
                    'temp_low': int(temp[1]) if '/' in temp else int(temp[0]),
                    'wind': wind
                })
    return parsed_data


# 数据预处理
def preprocess_data(parsed_data):
    df = pd.DataFrame(parsed_data)

    # 编码分类变量
    le_condition = LabelEncoder()
    le_wind = LabelEncoder()
    df['condition_encoded'] = le_condition.fit_transform(df['condition'])
    df['wind_encoded'] = le_wind.fit_transform(df['wind'])

    # 提取特征和标签
    features = df[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']]
    labels = df[['city', 'date', 'condition', 'temp_high', 'temp_low', 'wind', 'condition_encoded', 'wind_encoded']]

    return features, labels, le_condition, le_wind


# 训练模型
def train_model(features, labels):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels['condition_encoded'])
    return model


# 预测未来天气
def predict_future_weather(model, le_condition, le_wind, future_dates, current_features):
    future_features = current_features.copy()
    future_features['date'] = future_dates
    future_features['condition_encoded'] = 0  # 初始值，会被预测结果覆盖
    future_features['wind_encoded'] = 0  # 初始值，会被预测结果覆盖

    predictions = []
    for i in range(len(future_dates)):
        future_feature = future_features.iloc[i:i + 1]
        predicted_condition_encoded = model.predict(
            future_feature[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']])
        predicted_condition = le_condition.inverse_transform(predicted_condition_encoded)[0]

        # 假设风向不变
        predicted_wind = le_wind.inverse_transform([future_feature['wind_encoded'].values[0]])[0]

        predictions.append({
            'city': future_feature['city'].values[0],
            'date': future_feature['date'].values[0],
            'condition': predicted_condition,
            'temp_high': future_feature['temp_high'].values[0],
            'temp_low': future_feature['temp_low'].values[0],
            'wind': predicted_wind
        })

    return predictions


# 主函数
def main():
    parsed_data = parse_weather_data(data)
    features, labels, le_condition, le_wind = preprocess_data(parsed_data)
    model = train_model(features, labels)

    cities = labels['city'].unique()
    future_dates = ['11日（周一）', '12日（周二）', '13日（周三）', '14日（周四）', '15日（周五）']

    future_predictions = []
    for city in cities:
        city_data = labels[labels['city'] == city]
        last_five_days = city_data.tail(5)
        average_temp_high = last_five_days['temp_high'].mean()
        average_temp_low = last_five_days['temp_low'].mean()
        last_wind = last_five_days['wind'].iloc[-1]

        future_features = pd.DataFrame({
            'city': [city] * len(future_dates),
            'temp_high': [average_temp_high] * len(future_dates),
            'temp_low': [average_temp_low] * len(future_dates),
            'condition_encoded': [0] * len(future_dates),
            'wind_encoded': [le_wind.transform([last_wind])[0]] * len(future_dates)
        })

        city_predictions = predict_future_weather(model, le_condition, le_wind, future_dates, future_features)
        future_predictions.extend(city_predictions)

    future_df = pd.DataFrame(future_predictions)
    print(future_df)


if __name__ == "__main__":
    main()