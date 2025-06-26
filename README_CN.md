# 时间序列预测数据集

[中文版本](README_CN.md) | [English Version](README.md)

本仓库包含了用于时间序列预测研究的多个经典数据集。这些数据集广泛用于学术研究和工业应用中的时间序列预测模型评估。

## 数据集概览

### 📊 包含的数据集

| 数据集 | 描述 | 时间范围 | 特征数量 |
|--------|------|----------|----------|
| **electricity.csv** | 电力负荷数据 | 用电量时间序列数据 | 多变量 |
| **traffic.csv** | 交通流量数据 | 道路交通流量时间序列 | 多变量 |
| **ETTh1.csv** | 电力变压器温度数据集 (小时级) | 每小时采样的变压器数据 | 7个特征 |
| **ETTh2.csv** | 电力变压器温度数据集 (小时级) | 每小时采样的变压器数据 | 7个特征 |
| **ETTm1.csv** | 电力变压器温度数据集 (分钟级) | 每15分钟采样的变压器数据 | 7个特征 |
| **ETTm2.csv** | 电力变压器温度数据集 (分钟级) | 每15分钟采样的变压器数据 | 7个特征 |
| **Exchange.csv** | 汇率数据 | 多国货币汇率时间序列 | 8个货币对 |
| **Weather.csv** | 天气数据 | 气象观测时间序列数据 | 21个气象特征 |

## 🚀 快速开始

### 数据加载示例

```python
import pandas as pd
import numpy as np

# 加载数据集
def load_dataset(file_path):
    """加载时间序列数据集"""
    data = pd.read_csv(file_path)
    return data

# 示例：加载ETT数据集
ett_data = load_dataset('ETTh1.csv')
print(f"数据形状: {ett_data.shape}")
print(f"列名: {ett_data.columns.tolist()}")
```

### 基础数据探索

```python
import matplotlib.pyplot as plt

# 数据基本信息
def explore_dataset(data, dataset_name):
    print(f"\n=== {dataset_name} 数据集信息 ===")
    print(f"数据形状: {data.shape}")
    print(f"时间范围: {data.iloc[0, 0]} 到 {data.iloc[-1, 0]}")
    print(f"缺失值: {data.isnull().sum().sum()}")
    
    # 绘制时间序列图
    plt.figure(figsize=(12, 6))
    plt.plot(data.iloc[:, 1])  # 绘制第一个特征
    plt.title(f'{dataset_name} - 时间序列可视化')
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.show()
```

## 📈 数据集详细说明

### ETT (Electricity Transforming Temperature) 数据集
- **ETTh1/ETTh2**: 小时级别的电力变压器温度数据
- **ETTm1/ETTm2**: 15分钟级别的电力变压器温度数据
- **特征**: 包含目标变量 OT (Oil Temperature) 和其他6个负荷特征

### Electricity 数据集
- 来自UCI的电力负荷数据
- 包含多个客户的用电量时间序列
- 适用于多变量时间序列预测任务

### Traffic 数据集
- 加利福尼亚州高速公路的交通流量数据
- 包含多个传感器的小时级交通流量记录
- 适用于交通预测和城市规划研究

### Exchange 数据集
- 包含8个国家的汇率数据
- 以新加坡元为基准的每日汇率
- 适用于金融时间序列预测

### Weather 数据集
- 德国气象站的天气观测数据
- 包含21个气象特征（温度、湿度、风速等）
- 10分钟级别的高频观测数据

## 🔧 使用建议

### 数据预处理
```python
def preprocess_data(data, target_col, seq_len=96, pred_len=24):
    """
    时间序列数据预处理
    
    Args:
        data: 原始数据
        target_col: 目标列名
        seq_len: 输入序列长度
        pred_len: 预测长度
    """
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # 分离特征和目标变量
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    # 标准化
    features_scaled = scaler.fit_transform(features)
    target_scaled = scaler.fit_transform(target.values.reshape(-1, 1))
    
    return features_scaled, target_scaled, scaler
```

### 推荐的机器学习模型
- **传统方法**: ARIMA, SARIMA, Prophet
- **深度学习**: LSTM, GRU, Transformer
- **最新方法**: Informer, Autoformer, PatchTST

## 📚 相关论文

如果您在研究中使用了这些数据集，请考虑引用相关论文：

- ETT数据集: "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
- 其他数据集的原始来源请参考各自的官方文档

## 🤝 贡献

欢迎提交问题和改进建议！如果您有新的时间序列数据集想要添加到此仓库，请提交 Pull Request。

## 📄 许可证

本仓库中的数据集遵循其原始许可证。请在使用前查看各数据集的具体许可要求。

## 📞 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。

---

**注意**: 部分大型数据集文件使用 Git LFS 存储，首次克隆仓库时请确保已安装 Git LFS。

```bash
# 安装 Git LFS
git lfs install

# 克隆仓库
git clone <repository-url>
```
