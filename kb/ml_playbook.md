# ML Playbook（通用可复用）

## Tabular Regression 快速策略
- 先做 baseline：Linear/Ridge（确认流程 OK）
- 再做 tree boosting：LightGBM / XGBoost / CatBoost（通常很强）
- 指标建议：MAE + RMSE（更直观），也可以补 R2
- 一定要做 train/test split（70/30）并固定 random_state
- 画图建议：Residual histogram、Pred vs True scatter

## 缺失值（Missing Values）
- 数值（numeric）：median 一般稳
- 类别（categorical）：most_frequent 或加一个 "missing" 类别

## 类别编码（Categorical Encoding）
- 低基数：OneHotEncoder
- 高基数：频数编码（frequency encoding）优先；目标编码（target encoding）注意泄露（leakage）

## 防止数据泄露（Leakage）
- 任何“用到 y 信息”的转换必须只在 train 上 fit
- 用 sklearn Pipeline/ColumnTransformer 最稳