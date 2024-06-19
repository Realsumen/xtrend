import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, ELU, Add, Softmax, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras as keras


class FFN_j(tf.keras.Model):
    def __init__(self, hidden_dim):
        """初始化 VSN 中针对单个事件点每个特征的FFN

        Args:
            hidden_dim (int): 隐藏层维度
        """
        super(FFN_j, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.linear_1 = Dense(units=hidden_dim)
        self.elu = ELU()
        self.linear_3 = Dense(units=hidden_dim)
    
    def call(self, h_t) -> tf.Tensor:
        """前向传播方法。

        Args:
            h_t (tf.Tensor): 输入的隐藏状态

        Returns:
            tf.Tensor: 输出张量的维度，与隐藏层相等
        """
        add_output = self.linear_1(h_t)
        elu_output = self.elu(add_output)
        output = self.linear_3(elu_output)
        return output


class FFN(tf.keras.Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size, output_dim = None):
        """初始化 Feed Forward network

        Args:
            sequence_length (int): 输入的序列长度
            encoding_size (int): side info encoding的长度
            hidden_dim (int): 隐藏层的维度
        """
        super(FFN, self).__init__()
        if output_dim is None: output_dim = sequence_length
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.encoding_size = encoding_size
        
        self.linear_1 = Dense(units=hidden_dim)
        self.elu = ELU()
        self.linear_2 = Dense(units=hidden_dim)       
        self.linear_3 = Dense(units=output_dim)
    
    def call(self, h_t, s) -> tf.Tensor:
        """前向传播方法。

        Args:
            h_t (tf.Tensor): 输入的隐藏状态
            s (tf.Tensor, optional): side information 的输入张量

        Returns:
            tf.Tensor: 输出张量的维度与输入的序列长度相等
        """
        linear_1_output = self.linear_1(h_t)
        linear_2_output = self.linear_2(s)
        add_output = linear_1_output + linear_2_output
        elu_output = self.elu(add_output)
        output = self.linear_3(elu_output)
        return output


class BaselineNewralForeaster(tf.keras.Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size):
        super(BaselineNewralForeaster, self).__init__()
        self.vsn_model = VSN(sequence_length, hidden_dim, encoding_size=encoding_size)
        self.FFN_3, self.FFN_4 = FFN_j(hidden_dim), FFN_j(hidden_dim)
        self.lstm_model = tf.keras.layers.LSTM(hidden_dim)
        self.layer_norm = LayerNormalization()
        
    def call(self, x, s) -> tf.Tensor:
        x_ = self.vsn_model(x, s)
        return None
class VSN(tf.keras.Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size=None):
        """初始化加权求和网络 (VSN)。

        Args:
            sequence_length (int): 输入的序列长度
            encoding_size (int): side info encoding的长度
            hidden_dim (int): 隐藏层的维度，与输出张量维度相等
        """
        super(VSN, self).__init__()
        self.ffn = FFN(sequence_length, hidden_dim, encoding_size)
        self.softmax = Softmax(axis=2)
        self.sequence_FFN = [FFN_j(hidden_dim = hidden_dim) for _ in range(sequence_length)]
        
    def call(self, x_t, s=None):
        print("VSN call 方法被调用")
        """前向传播方法。

        Args:
            x_t  (tf.Tensor): 输入的样本，形状为 [timestemps, N]
            s (tf.Tensor, optional): side information 的输入张量

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等
        """
        batch_size, time_steps, seq_len = x_t.shape
        ffn_output = self.ffn(x_t, s)
        w_t = self.softmax(ffn_output)
        x_t_reshaped = tf.reshape(x_t, (-1, seq_len))
        outputs = []
        for i in range(self.ffn.sequence_length):
            ff_output = self.sequence_FFN[i](tf.expand_dims(x_t_reshaped[:, i], axis=1))
            outputs.append(ff_output)
        
        print(outputs[0].shape, len(outputs))
        return
        outputs = tf.stack(outputs, axis=1)  # Shape: [batch_size * time_steps, sequence_length, hidden_dim]
        outputs = tf.reshape(outputs, (batch_size, time_steps, seq_len, -1))  # Reshape back

        w_t_expanded = tf.expand_dims(w_t, axis=-1)  # Shape: [batch_size, time_steps, sequence_length, 1]
        weighted_outputs = outputs * w_t_expanded  # Element-wise multiplication
        vsn_output = tf.reduce_sum(weighted_outputs, axis=2)  # Sum over the sequence_length dimension

        return vsn_output
    
batch_size = 5
sequence_length = 4
hidden_dim = 8
encoding_size = 17  # 17 种合约
time_steps = 2  # 每个样本的时间步

x = tf.random.uniform((batch_size, time_steps, sequence_length))  # 示例输入 h_t
s_indices = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, time_steps])
s = tf.one_hot(s_indices + 6, depth=encoding_size) 

vsn_model = VSN(sequence_length, hidden_dim, encoding_size=encoding_size)
x_ = vsn_model(x, s)