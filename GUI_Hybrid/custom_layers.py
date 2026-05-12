"""
Custom Keras layers and model builders for the Hybrid CNN-ViT models.
Models were saved with Keras 3.13.1 / TensorFlow backend.
We rebuild architecture from code and load weights only to avoid
deserialization issues across Keras minor versions.
"""
import tensorflow as tf
import keras
from keras import layers, Model


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1,
                 use_qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_qkv_bias = use_qkv_bias
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by "
                f"number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim, use_bias=use_qkv_bias)
        self.key_dense = layers.Dense(embed_dim, use_bias=use_qkv_bias)
        self.value_dense = layers.Dense(embed_dim, use_bias=use_qkv_bias)
        self.combine_heads = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        weights = self.dropout(weights)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout": 0.1,
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1,
                 use_qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.use_qkv_bias = use_qkv_bias
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout,
                                          use_qkv_bias=use_qkv_bias)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


# ---- Model builders (mirror training notebooks exactly) ----

def build_hu_model(input_dim=1025, num_classes=2):
    """CP1: Healthy/Unhealthy — 2-stage CNN compression, 2 Transformer blocks."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)

    # Multi-Scale CNN
    b1 = layers.BatchNormalization()(
        layers.Conv1D(32, 3, padding='same', activation='relu')(x))
    b2 = layers.BatchNormalization()(
        layers.Conv1D(32, 7, padding='same', activation='relu')(x))
    b3 = layers.BatchNormalization()(
        layers.Conv1D(32, 15, padding='same', activation='relu')(x))
    x = layers.Dropout(0.2)(layers.Concatenate()([b1, b2, b3]))

    # CNN compression (2 stages)
    x = layers.Dropout(0.2)(layers.MaxPooling1D(2)(
        layers.BatchNormalization()(
            layers.Conv1D(64, 5, padding='same', activation='relu')(x))))
    x = layers.Dropout(0.3)(layers.MaxPooling1D(2)(
        layers.BatchNormalization()(
            layers.Conv1D(128, 5, padding='same', activation='relu')(x))))

    # Positional encoding
    seq_len = input_dim // 4
    embed_dim = 128
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embed = layers.Embedding(
        input_dim=seq_len, output_dim=embed_dim)(positions)
    x = x + pos_embed

    # 2 Transformer blocks
    x = TransformerBlock(128, 8, 256, 0.1)(x)
    x = TransformerBlock(128, 8, 256, 0.1)(x)

    # Dual pooling
    x_avg = layers.GlobalAveragePooling1D()(x)
    x_max = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([x_avg, x_max])

    # Classification head
    x = layers.Dropout(0.4)(layers.BatchNormalization()(
        layers.Dense(256, activation='relu')(x)))
    x = layers.Dropout(0.3)(layers.BatchNormalization()(
        layers.Dense(128, activation='relu')(x)))
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs, name='Hybrid_CNN_ViT')


def build_disease_model(input_dim=1024, num_classes=5):
    """CP2: 5-class disease — 3-stage CNN compression, 4 Transformer blocks."""
    inputs = layers.Input(shape=(input_dim,), name='eeg_input')
    x = layers.Reshape((input_dim, 1))(inputs)

    # Multi-Scale CNN
    b1 = layers.BatchNormalization()(
        layers.Conv1D(32, 3, padding='same', activation='relu',
                      name='ms_conv_k3')(x))
    b2 = layers.BatchNormalization()(
        layers.Conv1D(32, 7, padding='same', activation='relu',
                      name='ms_conv_k7')(x))
    b3 = layers.BatchNormalization()(
        layers.Conv1D(32, 15, padding='same', activation='relu',
                      name='ms_conv_k15')(x))
    x = layers.Dropout(0.2)(layers.Concatenate(name='ms_concat')([b1, b2, b3]))

    # CNN compression (3 stages: 1024 -> 512 -> 256 -> 128)
    x = layers.Dropout(0.1)(layers.MaxPooling1D(2)(
        layers.BatchNormalization()(
            layers.Conv1D(128, 7, padding='same', activation='relu')(x))))
    x = layers.Dropout(0.1)(layers.MaxPooling1D(2)(
        layers.BatchNormalization()(
            layers.Conv1D(128, 5, padding='same', activation='relu')(x))))
    x = layers.Dropout(0.1)(layers.MaxPooling1D(2)(
        layers.BatchNormalization()(
            layers.Conv1D(128, 3, padding='same', activation='relu',
                          name='patch_projection')(x))))

    # Positional encoding
    seq_len = input_dim // 8
    embed_dim = 128
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embed = layers.Embedding(
        input_dim=seq_len, output_dim=embed_dim,
        name='pos_embedding')(positions)
    x = x + pos_embed

    # 4 Transformer blocks (no QKV bias in disease model)
    for i in range(4):
        x = TransformerBlock(128, 8, 256, 0.1, use_qkv_bias=False,
                             name=f'transformer_block_{i}')(x)

    x = layers.LayerNormalization(epsilon=1e-6, name='final_ln')(x)

    # Dual pooling
    x_avg = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
    x_max = layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    deep = layers.Concatenate(name='deep_features')([x_avg, x_max])

    # Classification head
    x = layers.Dropout(0.4)(layers.BatchNormalization()(
        layers.Dense(256, activation='relu', name='clf_dense1')(deep)))
    x = layers.Dropout(0.3)(layers.BatchNormalization()(
        layers.Dense(128, activation='relu', name='clf_dense2')(x)))
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return Model(inputs=inputs, outputs=outputs,
                 name='Hybrid_CNN_ViT_Disease')


def load_model_weights(builder_fn, weights_path, **kwargs):
    """Build model from code and load weights from .h5 file."""
    model = builder_fn(**kwargs)
    model.load_weights(weights_path)
    return model
