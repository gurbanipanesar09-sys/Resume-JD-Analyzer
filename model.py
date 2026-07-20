import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Lambda,
    Multiply,
)
from tensorflow.keras.optimizers import Adam


def build_siamese_bilstm(
    vocab_size=10000,
    max_length=150,
    embedding_dimension=64,
    lstm_units=64,
):
    """
    Build the Siamese BiLSTM model.

    Both the resume and job description pass through the same
    shared embedding and BiLSTM layers.
    """

    resume_input = Input(
        shape=(max_length,),
        name="resume_input",
    )

    job_description_input = Input(
        shape=(max_length,),
        name="job_description_input",
    )

    shared_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dimension,
        name="shared_embedding",
    )

    shared_bilstm = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,
        ),
        name="shared_bilstm",
    )

    shared_pooling = GlobalMaxPooling1D(
        name="shared_global_max_pooling"
    )

    def encode_text(input_tensor):
        embedded_text = shared_embedding(input_tensor)
        encoded_text = shared_bilstm(embedded_text)
        pooled_text = shared_pooling(encoded_text)

        return pooled_text

    resume_vector = encode_text(resume_input)

    job_description_vector = encode_text(
        job_description_input
    )

    absolute_difference = Lambda(
        lambda tensors: tf.abs(
            tensors[0] - tensors[1]
        ),
        name="absolute_difference",
    )(
        [
            resume_vector,
            job_description_vector,
        ]
    )

    elementwise_product = Multiply(
        name="elementwise_product"
    )(
        [
            resume_vector,
            job_description_vector,
        ]
    )

    combined_features = Concatenate(
        name="combined_features"
    )(
        [
            resume_vector,
            job_description_vector,
            absolute_difference,
            elementwise_product,
        ]
    )

    dense_layer = Dense(
        128,
        activation="relu",
        name="dense_128",
    )(combined_features)

    dense_layer = Dropout(
        0.30,
        name="dropout_1",
    )(dense_layer)

    dense_layer = Dense(
        64,
        activation="relu",
        name="dense_64",
    )(dense_layer)

    dense_layer = Dropout(
        0.20,
        name="dropout_2",
    )(dense_layer)

    output_layer = Dense(
        3,
        activation="softmax",
        name="match_prediction",
    )(dense_layer)

    model = Model(
        inputs=[
            resume_input,
            job_description_input,
        ],
        outputs=output_layer,
        name="Siamese_BiLSTM_Resume_JD_Analyzer",
    )

    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    model = build_siamese_bilstm(
        vocab_size=10000,
        max_length=150,
    )

    model.summary()

    print(
        "\nSiamese BiLSTM model created successfully."
    )


if __name__ == "__main__":
    main()