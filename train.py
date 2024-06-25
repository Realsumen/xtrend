import os
from models import *
from dataprocessor import *
from loss_functions import *
from heartrate import trace
trace(browser=True)

folder_path = 'data'
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
files = [file for file in files if file not in ("CC00.NYB.xlsx", "LB00.CME.xlsx", "ES00.CME.xlsx", "NQ00.CME.xlsx", "YM00.CBT.xlsx")]
data_list = []


batch_size = len(data_list)
macd_timescales = [(8, 24), (16, 28), (32, 96)]
rtn_timescales = [1, 21, 63, 126, 252]
timesteps = 126
hidden_dim = 64 # 128
warm_up = 63
target_std = tf.cast(5e-2, tf.float64)
num_epochs = 100  # Maximum SGD iterations
print_interval = 10

dataset = tf.data.Dataset.load("saved_data")
features_len = len(macd_timescales) + len(rtn_timescales)
encoding_size = len(files) + 1
x_shape = (batch_size, timesteps, features_len)
s_shape = (batch_size, timesteps, encoding_size)

model = ModelWrapper(features_len, hidden_dim, encoding_size, num_heads = 4)
model.build((x_shape, s_shape))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

for epoch in range(num_epochs):
    dataset = dataset.shuffle(buffer_size=10000)
    iter_count = 0
    for batch_data in tqdm(dataset, desc=f"训练中...epoch{epoch}"):
        x_c, x_c_rtn, s_c, x, s, rtn_std, _, _ = batch_data
        with tf.GradientTape() as tape:
            result = model(x_c, x_c_rtn, s_c, x, s)
            loss = joint_loss_function(result, rtn_std, target_std, warm_up, alpha=1)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        iter_count += 1
        if iter_count % print_interval == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Iteration {iter_count}, Loss: {loss.numpy()}")

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.numpy()}")
    
model.save("xtrend.h5")