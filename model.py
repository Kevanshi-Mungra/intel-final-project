import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

st.title("Simple TensorFlow Model")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:", data.head())

    if 'label' in data.columns:
        X = data.drop('label', axis=1).values
        y = data['label'].values

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=5, verbose=0)

        preds = model.predict(X)
        st.write("Predictions:", preds)
    else:
        st.warning("Please ensure your CSV has a 'label' column.")
