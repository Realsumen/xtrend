import tensorflow as tf
import os
import pickle
from datetime import datetime
from tqdm import tqdm
from change_point_detection import *
from loss_functions import *
from model import *
from dataprocessor import *

macd_timescales = [(8, 24), (16, 28), (32, 96)]
rtn_timescales = [1, 21, 63, 126, 252]
asset_num, context_num = 10, 20
timesteps = 126
folder_path = "data"
# 设置参数
target_std = tf.cast(5e-2, tf.float64)
hidden_dim = 64  # 128
warm_up = 63
features_len = len(macd_timescales) + len(rtn_timescales)

files = [f for f in os.listdir(folder_path) if f.endswith(".csv")][:10]




def run():
    global timesteps
    gaussion_process_list = []

    data_list = process_data_list(files, macd_timescales, rtn_timescales, last_date=20211231)

    pkl_files = [f for f in os.listdir("segments") if f.endswith(".pkl")]
    for file in pkl_files:
        with open("segments/" + file, 'rb') as file:
            data = pickle.load(file)
            gaussion_process_list.extend(data)
            
    print(timesteps)
    # 生成数据： target_set 和 context_set
    target_set, labels, map = generate_tensors(data_list, timesteps, encoder_type = "one-hot", return_map=True)
    with open(f'map.pkl', 'wb') as f:
        pickle.dump(map, f)

    target_set, context_set, labels = gaussian_data_binder(
        data_list,
        target_set,
        labels,
        map=map,
        asset_num=asset_num,
        context_num=context_num,
        gaussion_process_list=gaussion_process_list,
    )

    # declare 数据, 初始化数据集
    x, s = target_set[0], target_set[-1]
    x_c_rtn, x_c, s_c = context_set[0], context_set[0][:, :, :, 1:], context_set[-1]
    dataset = tf.data.Dataset.from_tensor_slices((x_c, x_c_rtn, s_c, x, s, labels))

    timesteps = x.shape[-2]
    features_len = x.shape[-1]
    encoding_size = s.shape[-1]
    x_shape = (None, asset_num, timesteps, features_len)
    s_shape = (None, asset_num, timesteps, encoding_size)

    # 初始化模型
    xtrend_model = ModelWrapper(features_len, hidden_dim, encoding_size, num_heads=4, dropout_rate=0.4)
    xtrend_model.build((x_shape, s_shape))
    train(xtrend_model, dataset, batch_num=64, num_epochs=500, alpha=1e-3)

# 训练模型
def train(model, dataset: tf.data.Dataset, batch_num: int, num_epochs: int, alpha: float, validation_split: float = 0.2, early_stopping_patience: int = 10):
    log_dir = "logs/train_asset_sharpe/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    # 将数据集拆分为训练集和验证集
    dataset = dataset.shuffle(buffer_size=10000)
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).batch(batch_num)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=100.0)
    best_val_loss = np.inf
    global_step = 0
    
    # 开始记录计算图
    
    for epoch in range(num_epochs):
        tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=log_dir)
        # 每个 epoch 开始的时候重新shuffle数据集
        train_dataset = dataset.take(train_size).shuffle(buffer_size=10000).batch(batch_num)
        for batch_data in tqdm(train_dataset, desc=f"训练中...epoch{epoch}"):
            # 对每一个批次进行处理
            x_c, x_c_rtn, s_c, x, s, labels = batch_data
            with tf.GradientTape() as tape:
                result = model(x_c, x_c_rtn, s_c, x, s, training=True, step=global_step, log_dir = log_dir)
                joint_loss, mle, sharpe = joint_loss_function(
                    result, labels, target_std, warm_up, alpha=alpha
                )
            grads = tape.gradient(joint_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            with summary_writer.as_default():
                # for var, grad in zip(model.trainable_variables, grads):
                #     tf.summary.histogram(var.name + '/gradients', grad, step=global_step)
                tf.summary.scalar('Train Joint Loss', joint_loss, step=global_step)
                tf.summary.scalar('Train MLE', mle, step=global_step)
                tf.summary.scalar('Train Sharpe', sharpe, step=global_step)

            global_step += 1

        # 在每个epoch结束后，计算验证集的joint loss
        val_joint_loss = 0.0
        val_mle = 0.0
        val_sharpe = 0.0
        val_steps = 0
        for val_batch_data in val_dataset:
            x_c, x_c_rtn, s_c, x, s, labels = val_batch_data
            result = model(x_c, x_c_rtn, s_c, x, s)
            joint_loss, mle, sharpe = joint_loss_function(
                result, labels, target_std, warm_up, alpha=alpha
            )
            val_joint_loss += joint_loss.numpy()
            val_mle += mle.numpy()
            val_sharpe += sharpe.numpy()
            val_steps += 1

        val_joint_loss /= val_steps
        val_mle /= val_steps
        val_sharpe /= val_steps
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Joint Loss: {val_joint_loss:.2f}, Validation MLE: {val_mle:.2f}, Validation Sharpe: {val_sharpe:.2f}")

        # 写入 TensorBoard 日志
        with summary_writer.as_default():
            tf.summary.scalar('Validation Joint Loss', val_joint_loss, step=epoch)
            tf.summary.scalar('Validation MLE', val_mle, step=epoch)
            tf.summary.scalar('Validation Sharpe', val_sharpe, step=epoch)
        
        tf.summary.trace_export(
            name="model_trace",
            step=epoch,
        )
        
        # 早停检查
        if val_joint_loss < best_val_loss:
            best_val_loss = val_joint_loss
            patience_counter = 0
            model.save(f"model/{epoch}_loss_{val_joint_loss:.2f}.keras")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    # 训练模型
    run()