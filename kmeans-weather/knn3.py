import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 解析数据
def parse_weather_data(data):
    """
    解析天气数据
    :param data: 原始数据列表
    :return: 解析后的数据列表
    """
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


# 编码特征
def encode_features(df):
    """
    对特征进行编码
    :param df: 数据框
    :return: 编码后的数据框，LabelEncoder 对象
    """
    le_condition = LabelEncoder()
    le_wind = LabelEncoder()
    df['condition_encoded'] = le_condition.fit_transform(df['condition'])
    df['wind_encoded'] = le_wind.fit_transform(df['wind'])
    return df, le_condition, le_wind


# 处理缺失值和无穷大值
def handle_missing_and_infinite_values(df):
    """
    处理缺失值和无穷大值
    :param df: 数据框
    :return: 处理后的数据框
    """
    df.fillna(df.mean(), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


# 标准化数据
def standardize_data(df):
    """
    标准化数据
    :param df: 数据框
    :return: 标准化后的数据框
    """
    scaler = StandardScaler()
    df[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']] = scaler.fit_transform(
        df[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']])
    return df, scaler


# 计算每个城市的平均天气特征
def compute_city_features(df):
    """
    计算每个城市的平均天气特征
    :param df: 数据框
    :return: 包含城市特征的数据框
    """
    city_features = df.groupby('city')[
        ['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']].mean().reset_index()
    return city_features


# 聚类
def perform_clustering(X, n_clusters):
    """
    执行聚类
    :param X: 特征矩阵
    :param n_clusters: 聚类数量
    :return: 聚类模型，标签
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    return kmeans, labels


# 输出聚类结果
def print_clustering_results(clusters, kmeans, cities, le_condition, le_wind):
    """
    输出聚类结果
    :param clusters: 聚类字典
    :param kmeans: 聚类模型
    :param cities: 城市列表
    :param le_condition: 条件编码器
    :param le_wind: 风向编码器
    """
    print("聚类结果:")
    for cluster, cities in clusters.items():
        print(f"簇 {cluster}: {', '.join(cities)}")

        # 获取该簇的中心点
        cluster_center = kmeans.cluster_centers_[cluster]
        print(f"  簇 {cluster} 的中心点特征:")
        print(f"  最高温度: {cluster_center[0]:.2f}")
        print(f"  最低温度: {cluster_center[1]:.2f}")
        print(f"  天气状况: {le_condition.inverse_transform([int(cluster_center[2])])[0]}")
        print(f"  风向: {le_wind.inverse_transform([int(cluster_center[3])])[0]}")
        print()


# 主函数
def main(data):
    # 解析数据
    parsed_data = parse_weather_data(data)
    df = pd.DataFrame(parsed_data)

    # 编码特征
    df, le_condition, le_wind = encode_features(df)

    # 处理缺失值和无穷大值
    df = handle_missing_and_infinite_values(df)

    # 标准化数据
    df, scaler = standardize_data(df)

    # 计算每个城市的平均天气特征
    city_features = compute_city_features(df)

    # 特征和标签
    X = city_features[['temp_high', 'temp_low', 'condition_encoded', 'wind_encoded']]
    cities = city_features['city']

    # 确定最佳聚类数量
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans, labels = perform_clustering(X, n_clusters)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"最佳聚类数量: {best_n_clusters}")

    # 使用最佳聚类数量进行聚类
    kmeans, labels = perform_clustering(X, best_n_clusters)

    # 输出结果
    clusters = {}
    for i, city in enumerate(cities):
        cluster = labels[i]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(city)

    print_clustering_results(clusters, kmeans, cities, le_condition, le_wind)

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X['temp_high'], X['temp_low'], c=labels, cmap='viridis', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100)

    # 添加城市名称
    for i, city in enumerate(cities):
        plt.annotate(city, (X.iloc[i]['temp_high'], X.iloc[i]['temp_low']), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    plt.xlabel('最高温度')
    plt.ylabel('最低温度')
    plt.title('聚类结果')
    plt.show()


# 示例数据
data = [
    {"city": "南京",
     "weather": "4日（今天）:多云,10,东北风3-4级\n5日（明天）:晴,17/8,东北风3-4级转<3级\n6日（后天）:晴,17/7,东北风3-4级转<3级\n7日（周四）:晴转多云,17/8,东风3-4级转<3级\n8日（周五）:阴转多云,16/10,东北风<3级\n9日（周六）:多云转晴,21/11,东风<3级\n10日（周日）:晴,21/10,东北风<3级\n"},
    {"city": "南通",
     "weather": "4日（今天）:多云,11,东北风<3级\n5日（明天）:晴,17/7,北风<3级\n6日（后天）:晴转多云,17/6,北风<3级\n7日（周四）:多云,17/10,东北风<3级\n8日（周五）:阴转多云,18/12,东北风<3级\n9日（周六）:晴,21/12,东风<3级\n10日（周日）:晴,20/10,东北风<3级\n"},
    {"city": "扬州",
     "weather": "4日（今天）:多云,9,北风<3级\n5日（明天）:晴,18/5,北风<3级\n6日（后天）:晴,17/6,东北风<3级\n7日（周四）:晴转多云,18/8,东风<3级\n8日（周五）:阴转多云,17/11,东北风<3级\n9日（周六）:多云转晴,21/12,东风<3级\n10日（周日）:晴,21/9,东风<3级\n"},
    {"city": "无锡",
     "weather": "4日（今天）:多云,11,东北风<3级\n5日（明天）:晴,18/8,北风<3级\n6日（后天）:晴,18/8,东北风<3级\n7日（周四）:多云转晴,17/9,东北风<3级\n8日（周五）:阴,18/13,东北风<3级\n9日（周六）:多云转晴,21/12,东北风<3级\n10日（周日）:晴,21/10,东北风<3级\n"},
    {"city": "镇江",
     "weather": "4日（今天）:多云,10,北风<3级\n5日（明天）:晴,17/6,北风<3级\n6日（后天）:晴,16/7,东北风<3级\n7日（周四）:晴转阴,17/8,东风<3级\n8日（周五）:阴转多云,16/10,东北风<3级\n9日（周六）:晴,20/12,东风<3级\n10日（周日）:晴,20/10,东风<3级\n"},
    {"city": "徐州",
     "weather": "4日（今天）:多云,6,东北风<3级\n5日（明天）:晴,16/4,东风<3级\n6日（后天）:晴,17/5,东南风<3级\n7日（周四）:阴,17/8,东南风<3级\n8日（周五）:多云转晴,14/7,东风<3级\n9日（周六）:多云,18/9,东风<3级\n10日（周日）:多云转晴,20/10,南风<3级\n"},
    {"city": "盐城",
     "weather": "4日（今天）:多云,9,北风<3级\n5日（明天）:晴,18/5,北风<3级\n6日（后天）:晴,17/5,东北风<3级\n7日（周四）:晴转阴,16/8,东风<3级\n8日（周五）:阴,16/10,东风<3级\n9日（周六）:多云转晴,21/10,东风<3级\n10日（周日）:晴转多云,20/10,东北风<3级\n"},
    {"city": "苏州",
     "weather": "4日（今天）:多云,13,北风<3级\n5日（明天）:晴,18/10,北风<3级\n6日（后天）:晴,17/10,东北风<3级\n7日（周四）:多云转晴,17/11,东北风<3级\n8日（周五）:阴转小雨,19/13,东北风<3级\n9日（周六）:阴转晴,22/13,东北风<3级\n10日（周日）:晴,21/12,东北风<3级\n"},
    {"city": "淮安",
     "weather": "4日（今天）:多云,7,北风4-5级\n5日（明天）:晴,14/4,北风<3级\n6日（后天）:晴,17/4,东风<3级\n7日（周四）:晴转阴,16/7,东风<3级\n8日（周五）:阴转多云,16/10,东风<3级\n9日（周六）:多云,22/11,东风<3级\n10日（周日）:晴转多云,21/9,东风<3级\n"},
    {"city": "连云港",
     "weather": "4日（今天）:多云,6,北风<3级\n5日（明天）:晴,15/3,北风<3级\n6日（后天）:晴,17/5,东北风<3级\n7日（周四）:多云,18/7,东南风<3级\n8日（周五）:阴,17/8,东风<3级\n9日（周六）:多云,20/10,东风<3级\n10日（周日）:多云,20/9,东风<3级\n"},
    {"city": "泰州",
     "weather": "4日（今天）:多云,10,北风<3级\n5日（明天）:晴,16/6,北风<3级\n6日（后天）:晴,16/5,东北风<3级\n7日（周四）:晴转阴,17/7,东风<3级\n8日（周五）:阴转多云,16/10,东北风<3级\n9日（周六）:晴,21/11,东风<3级\n10日（周日）:晴,20/9,东风<3级\n"},
    {"city": "常州",
     "weather": "4日（今天）:多云,12,北风<3级\n5日（明天）:晴,17/8,北风<3级\n6日（后天）:晴,17/8,东北风<3级\n7日（周四）:晴转多云,17/9,东风<3级\n8日（周五）:阴转多云,18/12,东北风<3级\n9日（周六）:晴,21/11,东风<3级\n10日（周日）:晴,20/11,东风<3级\n"},
    {"city": "宿迁",
     "weather": "4日（今天）:多云,6,东北风<3级\n5日（明天）:晴,16/5,东北风<3级\n6日（后天）:晴,17/5,东风<3级\n7日（周四）:阴,17/8,东南风<3级\n8日（周五）:多云,15/8,东风<3级\n9日（周六）:晴转多云,20/11,东南风<3级\n10日（周日）:晴转多云,20/10,东南风<3级\n"}
]

main(data)