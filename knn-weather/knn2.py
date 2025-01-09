
data = [
    {"city": "南京", "weather": "9日（今天）:多云转晴,4/-3,北风4-5级转3-4级\n10日（明天）:多云,4/-2,北风3-4级转<3级\n11日（后天）:多云,7/0,西南风3-4级转<3级\n12日（周日）:晴,10/-1,西北风<3级\n13日（周一）:晴转多云,13/2,东南风<3级\n14日（周二）:阴,11/3,东北风<3级\n15日（周三）:多云转晴,9/-2,东北风<3级\n"},
    {"city": "南通", "weather": "9日（今天）:阴转多云,4/-3,西北风3-4级转<3级\n10日（明天）:晴转多云,5/-3,西北风<3级\n11日（后天）:多云,6/-2,西南风<3级\n12日（周日）:晴,9/-2,西北风<3级\n13日（周一）:晴,10/0,西南风<3级\n14日（周二）:多云转阴,12/1,东北风<3级\n15日（周三）:晴,8/-2,北风3-4级转<3级\n"},
    {"city": "盐城", "weather": "9日（今天）:多云转晴,2/-5,西北风3-4级转<3级\n10日（明天）:晴转多云,3/-4,西北风<3级\n11日（后天）:多云,6/-2,西南风<3级\n12日（周日）:晴,8/-2,西北风<3级\n13日（周一）:晴转阴,10/-1,西南风<3级\n14日（周二）:阴转多云,9/1,东北风<3级\n15日（周三）:晴,7/-2,北风<3级\n"},
    {"city": "苏州", "weather": "9日（今天）:多云,4/-2,西北风4-5级\n10日（明天）:多云,3/-1,西北风<3级\n11日（后天）:多云,5/0,西南风<3级\n12日（周日）:晴,10/-1,西风<3级\n13日（周一）:晴,11/2,东南风<3级\n14日（周二）:多云转阴,14/5,东北风<3级\n15日（周三）:多云转晴,9/0,北风<3级\n"},
    {"city": "镇江", "weather": "9日（今天）:阴转晴,3/-4,北风<3级\n10日（明天）:晴转阴,3/-3,西北风<3级\n11日（后天）:多云,6/-1,西南风<3级\n12日（周日）:晴,9/-1,西北风<3级\n13日（周一）:晴转多云,11/-1,南风<3级\n14日（周二）:阴,10/1,东北风<3级\n15日（周三）:晴转多云,8/-1,北风<3级\n"},
    {"city": "泰州", "weather": "9日（今天）:阴转晴,4/-6,北风3-4级转<3级\n10日（明天）:晴转阴,4/-5,西北风<3级\n11日（后天）:多云,6/-2,西南风<3级\n12日（周日）:晴,9/-2,西北风<3级\n13日（周一）:晴转多云,10/0,南风<3级\n14日（周二）:阴,12/0,东北风<3级\n15日（周三）:晴转多云,8/-2,北风<3级\n"},
    {"city": "扬州", "weather": "9日（今天）:阴转晴,4/-2,北风<3级\n10日（明天）:晴转阴,4/-2,西北风<3级\n11日（后天）:晴转多云,7/0,西南风<3级\n12日（周日）:晴转多云,9/0,西北风<3级\n13日（周一）:晴转多云,11/0,南风<3级\n14日（周二）:阴,12/2,东北风<3级\n15日（周三）:晴转多云,8/0,北风<3级\n"},
    {"city": "徐州", "weather": "9日（今天）:多云转晴,3/-4,北风<3级\n10日（明天）:晴转多云,4/-4,西风<3级\n11日（后天）:晴,7/-2,西南风<3级\n12日（周日）:晴,6/-1,西北风<3级\n13日（周一）:晴转阴,10/0,西南风<3级\n14日（周二）:阴转晴,8/-2,东北风<3级\n15日（周三）:晴,5/-2,北风<3级\n"},
    {"city": "无锡", "weather": "9日（今天）:阴转晴,4/-3,西北风<3级\n10日（明天）:晴转阴,4/-2,西北风<3级\n11日（后天）:多云,6/0,西南风<3级\n12日（周日）:晴,9/0,西北风<3级\n13日（周一）:晴,11/1,东南风<3级\n14日（周二）:阴,13/3,东北风<3级\n15日（周三）:晴转多云,9/0,北风<3级\n"},
    {"city": "连云港", "weather": "9日（今天）:晴,2/-7,北风3-4级转<3级\n10日（明天）:晴,4/-5,西风<3级\n11日（后天）:多云,7/-3,西南风<3级\n12日（周日）:晴,7/-2,西北风<3级\n13日（周一）:晴转阴,9/-1,西南风<3级\n14日（周二）:阴转多云,6/-1,东北风<3级\n15日（周三）:晴,5/-4,北风<3级\n"},
    {"city": "宿迁", "weather": "9日（今天）:多云转晴,3/-6,北风<3级\n10日（明天）:晴转多云,4/-4,西风<3级\n11日（后天）:晴,8/-2,西南风<3级\n12日（周日）:晴,7/-2,西北风<3级\n13日（周一）:多云转阴,10/0,南风<3级\n14日（周二）:阴转多云,7/-1,东北风<3级\n15日（周三）:晴,5/-2,北风<3级\n"},
    {"city": "常州", "weather": "9日（今天）:多云,3/-3,北风<3级\n10日（明天）:多云,4/-2,西北风<3级\n11日（后天）:阴转多云,6/-1,西南风<3级\n12日（周日）:多云转晴,9/-1,西北风<3级\n13日（周一）:多云,11/1,南风<3级\n14日（周二）:阴,11/2,东北风<3级\n15日（周三）:阴转多云,9/-1,北风<3级\n"},
    {"city": "淮安", "weather": "9日（今天）:阴转晴,2/-6,北风3-4级\n10日（明天）:晴转多云,4/-5,西北风<3级\n11日（后天）:多云,6/-3,西南风<3级\n12日（周日）:晴,8/-1,西北风<3级\n13日（周一）:多云转阴,10/0,南风<3级\n14日（周二）:阴转多云,8/0,东北风<3级\n15日（周三）:晴,7/-2,北风<3级\n"}
]

import re
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder


# 数据预处理函数
def parse_weather(weather_str):
    days = weather_str.split('\n')
    parsed_data = []
    for day in days:
        parts = day.split(',')
        temp = parts[1].split('/')
        high_temp = int(temp[0])
        low_temp = int(temp[1])
        wind = parts[2].split(' ')[1]
        parsed_data.append((high_temp, low_temp, wind))
    return parsed_data


# 提取特征
def extract_features(data):
    features = []
    labels = []
    for item in data:
        city = item['city']
        weather = parse_weather(item['weather'])
        features.extend(weather[:-3])  # 使用前10天的数据作为特征
        labels.append(weather[-3:])  # 使用最后3天的数据作为标签
    return features, labels


# 编码风向
def encode_wind_direction(wind_directions):
    le = LabelEncoder()
    encoded = le.fit_transform(wind_directions)
    return encoded, le


# 主函数
def main():
    # 解析数据
    features, labels = extract_features(data)

    # 提取风向并编码
    wind_directions = [wind for _, _, wind in features]
    encoded_winds, le = encode_wind_direction(wind_directions)

    # 构建特征矩阵
    X = np.array([(high, low) for high, low, _ in features])
    y = np.array([(high, low, le.transform([wind])[0]) for high, low, wind in labels])

    # 训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # 预测未来三天的天气
    last_10_days = parse_weather(data[0]['weather'])[:-3]  # 使用南京的最后10天数据作为示例
    X_test = np.array([(high, low) for high, low, _ in last_10_days])
    y_pred = knn.predict(X_test[-3:])

    # 解码风向
    predicted_weather = [(high, low, le.inverse_transform([wind])[0]) for high, low, wind in y_pred]

    print("Predicted weather for the next 3 days:")
    for i, (high, low, wind) in enumerate(predicted_weather):
        print(f"Day {i + 1}: High {high}°C, Low {low}°C, Wind {wind}")


if __name__ == "__main__":
    main()