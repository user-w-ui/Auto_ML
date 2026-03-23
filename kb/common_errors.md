# Common Errors & Fixes（代码执行常见坑）

## ValueError: could not convert string to float
原因：模型收到字符串特征
修复：
- 使用 OneHotEncoder / OrdinalEncoder
- 或者把非数值列 drop（临时 baseline）

## Found input variables with inconsistent numbers of samples
原因：X 和 y 行数不一致（dropna/drop duplicates 后没同步）
修复：确保对 X/y 采用同样的过滤索引

## train/test split 后忘记对 test 做同样预处理
修复：用 Pipeline/ColumnTransformer 一步到位

## Matplotlib 保存图为空白
修复：先 plt.tight_layout() 再保存；保存后 plt.close()