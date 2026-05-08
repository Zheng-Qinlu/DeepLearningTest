# 导入所需库
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'Heiti TC', 'Songti SC', 'Arial Unicode MS']  # macOS中文字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 强制刷新字体缓存
font_manager._load_fontmanager(try_read_cache=False)

# 设置随机种子以保证可重复性
np.random.seed(42)
tf.random.set_seed(42)

# 1. 加载数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 数据预处理
# 归一化处理（将0-255的像素值压缩到0-1之间）
x_train = x_train / 255.0
x_test = x_test / 255.0

# 为CNN添加通道维度 (28, 28) -> (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 3. 构建优化的CNN神经网络（更简洁高效的结构）
model = keras.Sequential([
    # 第一个卷积块
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # 第二个卷积块
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # 全连接层
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 输出层（10个数字类别）
])

# 4. 编译模型（使用学习率衰减的Adam优化器）
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 打印模型结构
model.build(input_shape=(None, 28, 28, 1))
model.summary()

# 5. 设置回调函数
callbacks = [
    # 学习率衰减
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,  # 更快响应
        min_lr=1e-7,
        verbose=1
    ),
    # 早停法
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,  # 3轮不改善就停止
        restore_best_weights=True,
        verbose=1
    ),
    # 保存最佳模型
    keras.callbacks.ModelCheckpoint(
        'best_mnist_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# 6. 训练模型
print("\n开始训练模型...")
history = model.fit(
    x_train, y_train,
    validation_split=0.1,  # 使用10%数据作为验证集
    epochs=10,  # 限制在10轮以内
    batch_size=256,  # 增加批次大小以加快训练
    callbacks=callbacks,
    verbose=1
)

# 7. 评估模型
print("\n评估模型性能...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"测试集损失：{test_loss:.4f}")
print(f"测试集准确率：{test_acc:.4f} ({test_acc*100:.2f}%)")

# 9. 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 准确率曲线
epochs = range(1, len(history.history['accuracy']) + 1)
line1, = axes[0].plot(epochs, history.history['accuracy'], marker='o', label='训练集准确率')
line2, = axes[0].plot(epochs, history.history['val_accuracy'], marker='s', label='验证集准确率')

# 在每个点上标注纵坐标值
for i, (x, y) in enumerate(zip(epochs, history.history['accuracy'])):
    axes[0].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=8)
for i, (x, y) in enumerate(zip(epochs, history.history['val_accuracy'])):
    axes[0].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                     xytext=(0, -15), ha='center', fontsize=8)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('模型准确率')
axes[0].legend()
axes[0].grid(True)
axes[0].set_xticks(epochs)  # 设置横坐标间隔为1

# 损失曲线
line3, = axes[1].plot(epochs, history.history['loss'], marker='o', label='训练集损失')
line4, = axes[1].plot(epochs, history.history['val_loss'], marker='s', label='验证集损失')

# 在每个点上标注纵坐标值
for i, (x, y) in enumerate(zip(epochs, history.history['loss'])):
    axes[1].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=8)
for i, (x, y) in enumerate(zip(epochs, history.history['val_loss'])):
    axes[1].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                     xytext=(0, -15), ha='center', fontsize=8)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('模型损失')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xticks(epochs)  # 设置横坐标间隔为1

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 预测演示（使用测试集前10张图片）
print("\n详细预测演示（前10张测试图片）：")
predictions = model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].reshape(28, 28), cmap='binary')
    ax.set_title(f'预测: {predicted_labels[i]}\n真实: {y_test[i]}')
    ax.axis('off')
    
    # 如果预测错误，标红
    if predicted_labels[i] != y_test[i]:
        ax.set_title(f'预测: {predicted_labels[i]}\n真实: {y_test[i]}', color='red')

plt.tight_layout()
plt.savefig('predictions_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 混淆矩阵分析
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

all_predictions = model.predict(x_test)
predicted_classes = np.argmax(all_predictions, axis=1)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印分类报告
print("\n分类报告：")
print(classification_report(y_test, predicted_classes, 
                          target_names=[str(i) for i in range(10)]))
