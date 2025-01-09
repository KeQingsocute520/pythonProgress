import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

# 设置 Pandas 显示选项（仅在调试时启用）
DEBUG = True
if DEBUG:
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 自动调整列宽
    pd.set_option('display.max_colwidth', None)  # 显示完整列内容

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

def parse_weather_data(data):
    parsed_data = []
    for entry in data:
        city = entry['city']
        weather_lines = entry['weather'].split('\n')
        for line in weather_lines:
            if line:
                try:
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
                except Exception as e:
                    print(f"Error parsing line '{line}': {e}")
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

    # 保存编码器
    joblib.dump(le_condition, 'le_condition.pkl')
    joblib.dump(le_wind, 'le_wind.pkl')

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

    predictions = []
    for i in range(len(future_dates)):
        future_feature = future_features.iloc[i:i + 1]
        predicted_condition_encoded = model.predict(
            future_feature[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']])
        predicted_condition = le_condition.inverse_transform(predicted_condition_encoded)[0]

        # 使用当前数据中的最后一个风向作为预测风向
        predicted_wind = le_wind.inverse_transform([future_feature['wind_encoded'].values[0]])[0]

        predictions.append({
            'city': future_feature['city'].values[0],
            'date': future_feature['date'].values[0],
            'condition': predicted_condition,
            'temp_high': future_feature['temp_high'].values[0],
            'temp_low': future_feature['temp_low'].values[0],
            'wind': predicted_wind
        })

        # 更新下一个预测的特征值
        if i < len(future_dates) - 1:
            next_future_feature = future_features.iloc[i + 1:i + 2]
            next_future_feature['temp_high'] = future_feature['temp_high'].values[0] + (i + 1) * 0.5  # 示例：逐步增加温度
            next_future_feature['temp_low'] = future_feature['temp_low'].values[0] + (i + 1) * 0.5  # 示例：逐步增加温度
            next_future_feature['condition_encoded'] = predicted_condition_encoded
            next_future_feature['wind_encoded'] = le_wind.transform([predicted_wind])[0]

    return predictions

# 主函数
def main():
    parsed_data = parse_weather_data(data)
    features, labels, le_condition, le_wind = preprocess_data(parsed_data)
    model = train_model(features, labels)

    cities = labels['city'].unique()
    future_dates = ['16日（周四）', '17日（周五）', '18日（周六）', '19日（周日）', '20日（周一）']

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
            'condition_encoded': [last_five_days['condition_encoded'].iloc[-1]] * len(future_dates),
            'wind_encoded': [le_wind.transform([last_wind])[0]] * len(future_dates)
        })

        city_predictions = predict_future_weather(model, le_condition, le_wind, future_dates, future_features)
        future_predictions.extend(city_predictions)

    future_df = pd.DataFrame(future_predictions)
    print(future_df)

if __name__ == "__main__":
    main()
