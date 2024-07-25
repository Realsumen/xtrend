from typing import assert_type
import tensorflow as tf
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.keras.layers import (
    Dense,
    ELU,
    Dropout,
    Softmax,
    LayerNormalization,
    LSTM,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model


class FFN_j(Model):
    def __init__(self, hidden_dim, dropout_rate):
        """初始化 FFN_j 类。

        Args:
            hidden_dim (int): 隐藏层维度。
        """
        super(FFN_j, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = Dense(units=hidden_dim)
        self.elu = ELU()
        self.dropout = Dropout(dropout_rate)
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

    def call(self, h_t, training=False, step=None, writer=None) -> tf.Tensor:
        """前向传播方法。

        Args:
            h_t (tf.Tensor): 输入的隐藏状态张量

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等
        """
        add_output = self.linear_1(h_t)
        elu_output = self.elu(add_output)
        dropout_output = self.dropout(elu_output)
        output = self.linear_3(dropout_output)
    
        if writer is not None and step is not None:
            with writer.as_default():
                tf.summary.histogram('FFN_j_add_output', add_output, step=step)
                tf.summary.histogram('FFN_j_elu_output', elu_output, step=step)
                tf.summary.histogram('FFN_j_dropout_output', dropout_output, step=step)
                tf.summary.histogram('FFN_j_output', output, step=step)
                
                # 记录每一层的权重
                for layer in self.layers:
                    for weight in layer.weights:
                        tf.summary.histogram(weight.name, weight, step=step)

        return output
    


class FFN(Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size, dropout_rate, output_dim=None):
        """初始化 FFN 类。

        Args:
            sequence_length (int): 序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int): 编码大小。
            output_dim (int, optional): 输出维度。默认为 sequence_length。
        """
        super(FFN, self).__init__()
        if output_dim is None:
            output_dim = sequence_length
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.encoding_size = encoding_size

        self.linear_1 = Dense(units=hidden_dim)
        self.elu = ELU()
        self.dropout = Dropout(dropout_rate)
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
            h_t (tf.Tensor): 隐状态张量，形状为 [batch_size, asset_types, time_steps, feature_dim]
            s (tf.Tensor): 侧信息张量，形状为 [batch_size, asset_types, time_steps, feature_dim]

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等
        """
        linear_1_output = self.linear_1(h_t)
        linear_2_output = self.linear_2(s)
        add_output = linear_1_output + linear_2_output
        elu_output = self.elu(add_output)
        dropout_output = self.dropout(elu_output)
        output = self.linear_3(dropout_output)
        return output


class VSN(Model):
    def __init__(self, sequence_length, hidden_dim, dropout_rate, encoding_size=None):
        """初始化 VSN 类。

        Args:
            sequence_length (int): embedding 长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int, optional): 资产数目长度。
        """
        super(VSN, self).__init__()
        self.ffn = FFN(sequence_length, hidden_dim, encoding_size, dropout_rate)
        self.softmax = Softmax(axis=2)
        self.sequence_FFN = [
            FFN_j(hidden_dim, dropout_rate) for _ in range(sequence_length)
        ]

    def build(self, input_shape):
        """构建模型的子层。

        Args:
            input_shape (tuple): 输入的形状，包含 x_shape 和 s_shape。
        """
        x_shape = input_shape[0]
        s_shape = input_shape[1] if len(input_shape) > 1 else None
        self.ffn.build((x_shape, s_shape))
        for ffn_j in self.sequence_FFN:
            ffn_j.build(
                (None, 1)
            )  # None since the batch_size of the input is undetermined
        self.softmax.build((x_shape[0], x_shape[1], x_shape[2], self.ffn.hidden_dim))
        super(VSN, self).build(input_shape)

    def call(self, x, s=None):
        """前向传播方法。

        Args:
            x (tf.Tensor): 输入的样本，形状为 [batch_size, asset_types, time_steps, feature_dim]
            s (tf.Tensor, optional): side information 的输入张量，形状为 [batch_size, asset_types, time_steps, feature_dim]

        Returns:
            tf.Tensor: 输出张量，形状为 [batch_size, asset_types, time_steps, hidden_dim]
        """
        batch_size, asset_types, time_steps, feature_dim = x.shape
        ffn_output = self.ffn(x, s)
        w_t = self.softmax(ffn_output)
        x_t_reshaped = tf.reshape(x, (-1, feature_dim))
        outputs = []
        for i in range(self.ffn.sequence_length):
            ff_output = self.sequence_FFN[i](tf.expand_dims(x_t_reshaped[:, i], axis=1))
            outputs.append(ff_output)

        outputs = tf.stack(
            outputs, axis=1
        )  # Shape: [batch_size * asset_types * time_steps, feature_dim, hidden_dim]
        outputs = tf.reshape(
            outputs, (batch_size, asset_types, time_steps, feature_dim, -1)
        )  # Reshape back

        w_t_expanded = tf.expand_dims(
            w_t, axis=-1
        )  # Shape: [batch_size, asset_types, time_steps, feature_dim, 1]
        weighted_outputs = (
            outputs * w_t_expanded
        )  # 针对每个特征的Element-wise multiplication
        vsn_output = tf.reduce_sum(
            weighted_outputs, axis=3
        )  # Sum over the feature_dim dimension, 原文中 VSN 的 \sum^|X| 如此
        return vsn_output


class BaselineNeuralForecaster(Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size, dropout_rate):
        """初始化 BaselineNeuralForecaster 类。

        Args:
            sequence_length (int): 序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int): 编码大小。
        """
        super(BaselineNeuralForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.vsn_model = VSN(sequence_length, hidden_dim, dropout_rate, encoding_size=encoding_size)
        self.FFN_3, self.FFN_4 = FFN_j(hidden_dim, dropout_rate), FFN_j(hidden_dim, dropout_rate)
        self.layer_norm = LayerNormalization()
        self.lstm_model = LSTM(hidden_dim, return_sequences=True)
        self.FFN_2 = FFN(
            sequence_length, hidden_dim, encoding_size, dropout_rate, output_dim=hidden_dim
        )

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
        self.lstm_model.build(
            (None, x_shape[2], self.vsn_model.ffn.hidden_dim)
        )  # batch_size * asset_num is not determined
        self.FFN_2.build(
            (
                (x_shape[0], x_shape[1], x_shape[2], self.vsn_model.ffn.hidden_dim),
                s_shape,
            )
        )

        super(BaselineNeuralForecaster, self).build(input_shape)

    def call(self, x, s) -> tf.Tensor:
        """前向传播方法。

        Args:
            x (tf.Tensor): 输入的样本，形状为 [batch_size, asset_types, time_steps, feature_dim]
            s (tf.Tensor): side information 的输入张量，形状为 [batch_size, asset_types, time_steps, feature_dim]

        Returns:
            tf.Tensor: 输出张量的维度与隐藏层相等。
        """
        x_ = self.vsn_model(x, s)
        batch_size, asset_types, time_steps, feature_dim = x_.shape
        x_reshaped = tf.reshape(x_, (batch_size * asset_types, time_steps, feature_dim))
        h_0, c_0 = self.FFN_3(s), self.FFN_4(s)
        h_0, c_0 = tf.reduce_mean(h_0, axis=2), tf.reduce_mean(c_0, axis=2)
        h_0 = tf.reshape(h_0, (batch_size * asset_types, feature_dim))
        c_0 = tf.reshape(c_0, (batch_size * asset_types, feature_dim))
        outputs = self.lstm_model(x_reshaped, initial_state=[h_0, c_0])
        outputs = tf.reshape(
            outputs, (batch_size, asset_types, time_steps, feature_dim)
        )
        a_t = LayerNormalization()(x_ + outputs)
        result = LayerNormalization()(self.FFN_2(a_t, s) + a_t)
        return result


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, sequence_length, encoding_size, dropout_rate):
        """
        初始化 Encoder 类。

        Args:
            d_model (int): 模型的维度。
            num_heads (int): 多头注意力的头数。
            ff_dim (int): 前馈网络的维度。
        """
        super(Encoder, self).__init__()
        self.hidden_dim = d_model
        self.self_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.temporal_block = BaselineNeuralForecaster(
            sequence_length, d_model, encoding_size, dropout_rate=dropout_rate
        )
        self.ffn1 = FFN_j(hidden_dim=d_model, dropout_rate=dropout_rate)
        self.ffn2 = FFN_j(hidden_dim=d_model, dropout_rate=dropout_rate)
        self.layer_norm = LayerNormalization()

    def build(self, input_shape):
        """
        构建层的权重。

        Args:
            input_shape (tuple): 输入的形状。
        """
        V_shape, K_shape, x_shape, s_shape = input_shape
        self.self_attention.build(
            value_shape=V_shape, query_shape=V_shape, key_shape=V_shape
        )
        self.cross_attention.build(
            value_shape=V_shape,
            query_shape=K_shape,
            key_shape=(x_shape[0], x_shape[1], x_shape[2], self.hidden_dim),
        )
        self.temporal_block.build((x_shape, s_shape))
        self.ffn1.build(V_shape)
        self.ffn2.build(V_shape)
        super(Encoder, self).build(input_shape)

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
        if K.shape != q.shape:
            K = tf.transpose(K, perm=[0, 2, 1, 3])  # 把 K, V_ 的时间维度和资产维度交换
            V_ = tf.transpose(V_, perm=[0, 2, 1, 3])
            K = tf.expand_dims(K[:, -1, :, :], axis=1)  # 取最后一个时间维度进行扩展
            V_ = tf.expand_dims(V_[:, -1, :, :], axis=1)
        attn_output = self.cross_attention(q, K, V_)
        output = self.layer_norm(self.ffn2(attn_output))
        return output


class Decoder(Model):
    def __init__(self, sequence_length, hidden_dim, encoding_size, dropout_rate):
        """
        初始化 Decoder 类。

        Args:
            hidden_dim (int): 隐藏层的维度。
        """
        super(Decoder, self).__init__()
        self.vsn_model = VSN(sequence_length, hidden_dim, dropout_rate=dropout_rate, encoding_size=encoding_size)
        self.hidden_dim = hidden_dim
        self.FFN_1 = FFN_j(hidden_dim, dropout_rate)
        self.FFN_3, self.FFN_4 = FFN_j(hidden_dim, dropout_rate), FFN_j(hidden_dim, dropout_rate)
        self.layer_norm = LayerNormalization()
        self.lstm_model = LSTM(hidden_dim, return_sequences=True)
        self.FFN_2 = FFN(hidden_dim, hidden_dim, encoding_size, dropout_rate)
        self.fcn = Dense(units=2)
        self.PTP = FFN_j(1, dropout_rate)

    def build(self, input_shape):
        """
        构建层的权重。

        Args:
            input_shape (tuple): 输入的形状。
        """
        x_shape = input_shape[0]
        s_shape = input_shape[1]

        self.vsn_model.build((x_shape, s_shape))
        self.FFN_1.build((x_shape[0], x_shape[1], x_shape[2], self.hidden_dim * 2))
        self.FFN_3.build(s_shape)
        self.FFN_4.build(s_shape)
        self.lstm_model.build(
            (None, x_shape[2], self.vsn_model.ffn.hidden_dim)
        )  # batch_size * asset_num is not determined
        self.FFN_2.build(
            ((x_shape[0], x_shape[1], x_shape[2], self.hidden_dim), s_shape)
        )
        self.fcn.build((x_shape[0], x_shape[1], x_shape[2], self.hidden_dim))
        self.PTP.build((x_shape[0], x_shape[1], x_shape[2], 2))
        super(Decoder, self).build(input_shape)

    def call(self, x, s, y, training=False, step=None, writer=None) -> tf.Tensor:
        """
        执行前向传播。

        Args:
            x (tf.Tensor): 输入数据。
            s (tf.Tensor): 上下文向量。
            y (tf.Tensor): 辅助输入。

        Returns:
            tf.Tensor: 输出张量。
        """
        tmp = self.vsn_model(x, s)
        x_ = self.layer_norm(self.FFN_1(tf.concat([tmp, y], axis=3)))
        batch_size, asset_nums, time_steps, feature_dim = x_.shape
        x_reshaped = tf.reshape(x_, (batch_size * asset_nums, time_steps, feature_dim))
        h_0, c_0 = self.FFN_3(s), self.FFN_4(s)
        h_0, c_0 = tf.reduce_mean(h_0, axis=2), tf.reduce_mean(c_0, axis=2)
        h_0 = tf.reshape(h_0, (batch_size * asset_nums, feature_dim))
        c_0 = tf.reshape(c_0, (batch_size * asset_nums, feature_dim))
        outputs = self.lstm_model(x_reshaped, initial_state=[h_0, c_0])
        outputs = tf.reshape(outputs, (batch_size, asset_nums, time_steps, feature_dim))
        a = self.layer_norm(outputs + x_)
        result = self.layer_norm(self.FFN_2(a, s) + a)
        properties = self.fcn(result)
        pred_mean = tanh(properties[:, :, :, 0])
        pred_std = sigmoid(properties[:, :, :, 1])
        properties = tf.stack([pred_mean, pred_std], axis=-1)
        PTP_outputs = self.PTP(properties, training=training, step=step)
        positions = tanh(PTP_outputs)
        if writer is not None and step is not None:
            with writer.as_default():
                tf.summary.histogram('lstm_outputs', outputs, step=step)
                tf.summary.histogram('layer_norm_output', a, step=step)
                tf.summary.histogram('FFN_2_output', result, step=step)
                tf.summary.histogram('fcn_output/properties', properties, step=step)
                tf.summary.histogram('pred_mean', properties[:, :, :, 0], step=step)
                tf.summary.histogram('pred_std', properties[:, :, :, -1], step=step)
                tf.summary.histogram('positions', positions, step=step)

        return properties, positions


class ModelWrapper(Model):
    def __init__(self, feature_length, hidden_dim, encoding_size, num_heads, dropout_rate):
        """
        初始化 ModelWrapper 类。

        Args:
            feature_length (int): 特征序列长度。
            hidden_dim (int): 隐藏层维度。
            encoding_size (int): 编码大小。
            num_heads (int): 多头注意力的头数。
        """
        super(ModelWrapper, self).__init__()
        self.hidden_dim = hidden_dim
        self.info_seq_length = feature_length + 1
        self.V_encoder = BaselineNeuralForecaster(
            self.info_seq_length, hidden_dim, encoding_size, dropout_rate=dropout_rate
        )
        self.K_encoder = BaselineNeuralForecaster(
            feature_length, hidden_dim, encoding_size, dropout_rate=dropout_rate
        )
        self.encoder = Encoder(hidden_dim, num_heads, feature_length, encoding_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(feature_length, hidden_dim, encoding_size, dropout_rate=dropout_rate)


    def build(self, input_shape):
        """
        构建层的权重。

        Args:
            input_shape (tuple): 输入的形状。
        """
        x_shape, s_shape = input_shape
        inter_dimensions = (x_shape[0], x_shape[1], x_shape[2], self.hidden_dim)
        self.V_encoder.build(
            ((x_shape[0], x_shape[1], x_shape[2], x_shape[3] + 1), s_shape)
        )
        self.K_encoder.build((x_shape, s_shape))
        self.encoder.build((inter_dimensions, inter_dimensions, x_shape, s_shape))
        self.decoder.build((x_shape, s_shape))
        super(ModelWrapper, self).build(input_shape)


    def call(self, x_c, x_c_r, s_c, x, s, training=False, step=None, writer=None) -> tuple:
        """
        执行前向传播。

        Args:
            x_c (tf.Tensor): 上下文输入数据。
            x_c_r (tf.Tensor): 带未来回报的上下文输入数据。
            s_c (tf.Tensor): 上下文状态向量。
            x_t (tf.Tensor): 目标输入数据。
            s_t (tf.Tensor): 目标状态向量。

        Returns:
            properties: 一个长度为2的tuple, 分别为明日收益率的期望和标准差
            positions: 每一个时间点的下一天持仓
        """
        
        num_classes = s.shape[-1] + 1
        embedding_dim = self.hidden_dim
        embedding_matrix = tf.Variable(tf.random.uniform([num_classes, embedding_dim]), name="embedding_matrix")
        embedded_s = tf.nn.embedding_lookup(embedding_matrix, tf.argmax(s, axis=-1))
        embedded_s_c= tf.nn.embedding_lookup(embedding_matrix, tf.argmax(s_c, axis=-1))
        
        V = self.V_encoder(x_c_r, embedded_s_c)
        K = self.K_encoder(x_c, embedded_s_c)
        y = self.encoder(V, K, embedded_s, x)
        properties, positions = self.decoder(x, embedded_s, y, training=training, step=step, writer=writer)
        properties = tf.cast(properties, tf.float64)
        positions = tf.cast(tf.squeeze(positions, axis=-1), tf.float64)

        # if writer is not None and step is not None:
        #     summary_writer = tf.summary.create_file_writer(log_dir)
        #     with summary_writer.as_default():
        #         tf.summary.histogram('V_encoder_output', V, step=step)
        #         tf.summary.histogram('K_encoder_output', K, step=step)
        #         tf.summary.histogram('encoder_output', y, step=step)
                
        #         # 记录每一层的权重
        #         for layer in self.layers:
        #             for weight in layer.weights:
        #                 tf.summary.histogram(weight.name, weight, step=step)

        return properties, positions
