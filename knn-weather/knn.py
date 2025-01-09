from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


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
# 提取特征和标签
X = []
y = []

for entry in data:
    weather_info = entry['weather'].split('\n')
    for day_info in weather_info:
        parts = day_info.split(',')
        if len(parts) < 3:
            continue
        temp_range = parts[1].strip().split('/')
        if len(temp_range) < 2:
            continue
        high_temp = int(temp_range[0])
        low_temp = int(temp_range[1].strip('℃'))
        weather_condition = parts[0].split(':')[1].strip()
        wind_direction = parts[2].split(',')[0].strip().split('风')[0]

        X.append([high_temp, low_temp, weather_condition, wind_direction])
        y.append(entry['city'])

# 将天气情况和风向转换为数值形式
le_weather = LabelEncoder()
le_wind = LabelEncoder()

X_encoded = np.array(X)
X_encoded[:, 2] = le_weather.fit_transform(X_encoded[:, 2])
X_encoded[:, 3] = le_wind.fit_transform(X_encoded[:, 3])

X_encoded = X_encoded.astype(float)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.7, random_state=42)

# 训练KNN模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 进行预测
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")

# 示例预测
sample = np.array([[2, -6, '阴转晴', '北']])  # 示例输入
sample_encoded = sample.copy()
sample_encoded[0, 2] = le_weather.transform([sample[0, 2]])[0]
sample_encoded[0, 3] = le_wind.transform([sample[0, 3]])[0]
sample_encoded = sample_encoded.astype(float)

predicted_city = knn.predict(sample_encoded)
print(f"预测的城市: {predicted_city[0]}")

# 使用交叉验证评估模型
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, X_encoded, y, cv=5)
print(f"交叉验证准确率: {scores.mean() * 100:.2f}%")

# 尝试不同的K值
best_k = 0
best_accuracy = 0
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_encoded, y, cv=5)
    if scores.mean() > best_accuracy:
        best_accuracy = scores.mean()
        best_k = k

print(f"最佳K值: {best_k}, 准确率: {best_accuracy * 100:.2f}%")