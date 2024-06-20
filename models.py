import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, ELU, Add, Softmax, LayerNormalization, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras as keras


class FFN_j(tf.keras.Model):
    def __init__(self, hidden_dim):
        """初始化 FFN_j 类。

        Args:
            hidden_dim (int): 隐藏层维度。
        """
        super(FFN_j, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = Dense(units=hidden_dim)
        self.elu = ELU()
        self.linear_3 = Dense(units=hidden_dim)
    
    def build(self, input_shape):
        """构建模型的子层。

        Args:
            input_shape (tuple): 输入的形状。
        """
        self.linear_1.build(input_shape)
        new_shape = tuple(list(input_shape[:-1]) + [self.hidden_dim])
        self.linear_3.build(new_shape)
        self.elu.build(new_shape)
        super(FFN_j, self).build(input_shape)
    
    def call(self, h_t) -> tf.Tensor:
        """前向传播方法。

        Args:
            h_t (tf.Tensor): 输入的隐藏状态张量

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等
        """
        add_output = self.linear_1(h_t)
        elu_output = self.elu(add_output)
        output = self.linear_3(elu_output)
        return output


class FFN(tf.keras.Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size, output_dim = None):
        """初始化 FFN 类。

        Args:
            sequence_length (int): 序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int): 编码大小。
            output_dim (int, optional): 输出维度。默认为 sequence_length。
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
    
    def build(self, input_shape):
        """初始化 FFN 类。

        Args:
            sequence_length (int): 序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int): 编码大小。
            output_dim (int, optional): 输出维度。默认为 sequence_length。
        """
        h_shape = input_shape[0]
        s_shape = input_shape[1]
        self.linear_1.build(h_shape)
        self.linear_2.build(s_shape)
        new_shape = tuple(list(h_shape[:-1]) + [self.hidden_dim])
        self.elu.build(new_shape)
        self.linear_3.build(new_shape)
        super(FFN, self).build(input_shape)
    
    def call(self, h_t, s) -> tf.Tensor:
        """前向传播方法。

        Args:
            h_t (tf.Tensor): 隐状态张量，形状为 [batch_size, time_steps, feature_dim]
            s (tf.Tensor): 侧信息张量，形状为 [batch_size, time_steps, feature_dim]

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等
        """
        linear_1_output = self.linear_1(h_t)
        linear_2_output = self.linear_2(s)
        add_output = linear_1_output + linear_2_output
        elu_output = self.elu(add_output)
        output = self.linear_3(elu_output)
        return output


class VSN(tf.keras.Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size=None):
        """初始化 VSN 类。

        Args:
            sequence_length (int): 序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int, optional): 编码大小。
        """
        super(VSN, self).__init__()
        self.ffn = FFN(sequence_length, hidden_dim, encoding_size)
        self.softmax = Softmax(axis=2)
        self.sequence_FFN = [FFN_j(hidden_dim = hidden_dim) for _ in range(sequence_length)]
    
    def build(self, input_shape):
        """构建模型的子层。

        Args:
            input_shape (tuple): 输入的形状，包含 x_shape 和 s_shape。
        """
        x_shape = input_shape[0]
        s_shape = input_shape[1] if len(input_shape) > 1 else None
        self.ffn.build((x_shape, s_shape))
        for ffn_j in self.sequence_FFN:
            ffn_j.build((x_shape[0] * x_shape[1], 1))
        self.softmax.build((x_shape[0], x_shape[1], self.ffn.hidden_dim))
        super(VSN, self).build(input_shape)
    
    def call(self, x, s=None):
        """前向传播方法。

        Args:
            x (tf.Tensor): 输入的样本，形状为 [batch_size, time_steps, feature_dim]
            s (tf.Tensor, optional): side information 的输入张量，形状为 [batch_size, time_steps, feature_dim]

        Returns:
            tf.Tensor: 输出张量，形状为 [batch_size, time_steps, hidden_dim]
        """
        batch_size, time_steps, seq_len = x.shape
        ffn_output = self.ffn(x, s)
        w_t = self.softmax(ffn_output)
        x_t_reshaped = tf.reshape(x, (-1, seq_len))
        outputs = []
        for i in range(self.ffn.sequence_length):
            ff_output = self.sequence_FFN[i](tf.expand_dims(x_t_reshaped[:, i], axis=1))
            outputs.append(ff_output)
        
        outputs = tf.stack(outputs, axis=1)  # Shape: [batch_size * time_steps, sequence_length, hidden_dim]
        outputs = tf.reshape(outputs, (batch_size, time_steps, seq_len, -1))  # Reshape back

        w_t_expanded = tf.expand_dims(w_t, axis=-1)  # Shape: [batch_size, time_steps, sequence_length, 1]
        weighted_outputs = outputs * w_t_expanded  # Element-wise multiplication
        vsn_output = tf.reduce_sum(weighted_outputs, axis=2)  # Sum over the sequence_length dimension
        return vsn_output
    
    
class BaselineNeuralForecaster(tf.keras.Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size):
        """初始化 BaselineNeuralForecaster 类。

        Args:
            sequence_length (int): 序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int): 编码大小。
        """
        super(BaselineNeuralForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.vsn_model = VSN(sequence_length, hidden_dim, encoding_size=encoding_size)
        self.FFN_3, self.FFN_4 = FFN_j(hidden_dim), FFN_j(hidden_dim)
        self.layer_norm = LayerNormalization()
        self.lstm_model = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.FFN_2 = FFN(sequence_length, hidden_dim, encoding_size, output_dim=hidden_dim)

    def build(self, input_shape):
        """构建模型的子层。

        Args:
            input_shape (tuple): 输入的形状，包含 x_shape 和 s_shape。
        """
        x_shape = input_shape[0]  
        s_shape = input_shape[1]

        self.vsn_model.build((x_shape, s_shape))
        self.FFN_3.build(s_shape)
        self.FFN_4.build(s_shape)
        self.lstm_model.build((x_shape[0], x_shape[1], self.vsn_model.ffn.hidden_dim))
        self.layer_norm.build((x_shape[0], self.vsn_model.ffn.hidden_dim))
        self.FFN_2.build(((x_shape[0], x_shape[1], self.vsn_model.ffn.hidden_dim), s_shape))

        super(BaselineNeuralForecaster, self).build(input_shape)

    def call(self, x, s) -> tf.Tensor:
        """前向传播方法。

        Args:
            x (tf.Tensor): 输入的样本，形状为 [batch_size, time_steps, feature_dim]
            s (tf.Tensor): side information 的输入张量，形状为 [batch_size, time_steps, feature_dim]

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等。
        """
        x_ = self.vsn_model(x, s)
        h_0, c_0 = self.FFN_3(s), self.FFN_4(s)
        h_0, c_0 = tf.reduce_mean(h_0, axis=1), tf.reduce_mean(c_0, axis=1)
        outputs = self.lstm_model(x_, initial_state=[h_0, c_0])
        a_t = LayerNormalization()(x_ + outputs)
        result = LayerNormalization()(self.FFN_2(a_t, s) + a_t)
        return result
    

class AttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, sequence_length, encoding_size):
        """
        初始化 AttentionWrapper 类。

        Args:
            d_model (int): 模型的维度。
            num_heads (int): 多头注意力的头数。
            ff_dim (int): 前馈网络的维度。
        """
        super(AttentionWrapper, self).__init__()
        self.hidden_dim = d_model
        self.self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.temporal_block = BaselineNeuralForecaster(sequence_length, d_model, encoding_size)
        self.ffn1 = FFN_j(hidden_dim=d_model)
        self.ffn2 = FFN_j(hidden_dim=d_model)
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        
    def build(self, input_shape):
        """
        构建层的权重。

        Args:
            input_shape (tuple): 输入的形状。
        """
        V_shape, K_shape, x_shape, s_shape = input_shape
        self.self_attention.build(value_shape = V_shape, query_shape = V_shape, key_shape = V_shape)
        self.cross_attention.build(value_shape = V_shape, query_shape = K_shape, key_shape = (x_shape[0], x_shape[1], self.hidden_dim))
        self.temporal_block.build((x_shape, s_shape))
        self.ffn1.build(V_shape)
        self.ffn2.build(V_shape)
        self.layer_norm = LayerNormalization()
        super(AttentionWrapper, self).build(input_shape)
        
    def call(self, V, K, s, x) -> tf.Tensor:
        """
        执行前向传播。

        Args:
            V (tf.Tensor): 值向量。
            K (tf.Tensor): 键向量。
            s (tf.Tensor): 上下文信息。
            x (tf.Tensor): 输入数据。

        Returns:
            tf.Tensor: 输出张量。
        """
        q = self.temporal_block(x, s)
        V_ = self.ffn1(self.self_attention(V, V, V))
        attn_output = self.cross_attention(q, K, V_)
        output = self.layer_norm(self.ffn2(attn_output))
        return output
        