import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image


from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import numpy as np
import pandas as pd

model = None  # Digital Recognized Model

with open("style.css") as f:  # css
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Bootstrap
st.markdown(""" 

    """, unsafe_allow_html=True)
with open("nav.html") as f:  # nav
    st.markdown(f'{f.read()}', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('''
    # Walter Alvarado, Data Scientist
    ''', unsafe_allow_html=True)

    image = Image.open('img/ds.jpg')
    st.image(image, use_column_width="auto")

    st.write('### Contact me', unsafe_allow_html=True)
    st.info('''
    - Email: walvdev@gmail.com 
    - Phone: +51 910267566
    - [Linkedin](https://www.linkedin.com/in/walter-alvarado-0179ba1b0/)
    ''')

st.write("## About me")
with st.container():  # About me
    st.markdown("""
    Hello, I'm Walter Alvarado, a talented, ambitious, and hardworking individual, 
    with broad skills and experience in relational databases Management, Data Analysis. 
    Also, build, optimize and Deploy Machine Learning Models.

    """)

st.write("## Skills")
with st.container():  # Skills
    st.markdown("""
    - Python :snake: (numpy, pandas, plotly, matplotlib, streamlit)
    - MySql :dolphin:  PostgreSQL :elephant:  Hive :honeybee: 
    - TensorFlow, scikit-learn
    - Docker :whale:
    - Linux (Ubuntu)z
    - Teamwork - Attention to detail - Creativity - Authenticity - Stress management 
    """)

st.write("## Projects and Experience:")
with st.container():  # Projects
    IMAGE_WIDTH = 180
    st.write("""
    These are only some of my favorite projects.
    Many, many more can be found in my GitHub Repositories.
    """)

    with st.container():  # Digits Recognized
        st.markdown("""
        #### [CNN Digit Recognizer (+98% Acc)](https://www.kaggle.com/code/walteralvarado/cnn-digit-recognizer/notebook)
        """)
        col1, col2 = st.columns([10, 25])
        with col1:
            titanic_img = Image.open('img/digits.jpg')
            st.image(titanic_img, width=IMAGE_WIDTH)
        with col2:
            st.markdown("""
                In this project, I used TensorFlow and Keras to build a handwritten Digit Recognition Convolutional Neural Network,
                 trained with the MNIST Dataset.
            """)
            with st.expander("Try Model, Draw a Digit"):
                if model is None:
                    model = keras.models.load_model('cnn-digit-recognizer98acc.h5')
                # Create a canvas component
                canvas_result = st_canvas(
                    stroke_width=10,
                    stroke_color='#FFFFFF',
                    background_color='#000000',
                    update_streamlit=True,
                    height=300,
                    width=400,
                    drawing_mode='freedraw',
                    display_toolbar=True,
                    key="streamlit_app",
                )

                # Do something interesting with the image data and paths
                if canvas_result.image_data is not None:
                    image = Image.fromarray(canvas_result.image_data)
                    image = image.resize((28, 28))
                    image = image.convert('L')
                    digit = np.asarray(image)
                    # st.write(digit)
                    if (digit == 0).sum() < 760:
                        digit = digit.reshape((1, 28, 28, 1))
                        pred_one_hot = model.predict(digit)
                        y_pred = np.argmax(pred_one_hot, axis=1)
                        st.write(f"## Predict: {y_pred[0]}")
                    else:
                        st.write("## Predict: ?")

    with st.container():  # Titanic Project
        st.markdown("""
        #### [Titanic Survived Prediction (+80% Acc)](https://www.kaggle.com/code/walteralvarado/ml-and-neural-network-80-acc)
        """)
        col1, col2 = st.columns([10, 25])
        with col1:
            titanic_img = Image.open('img/titanic.jpeg')
            st.image(titanic_img, width=IMAGE_WIDTH)
        with col2:
            st.markdown("""
            In this project, I'm going to predict if a Passenger was Survived or not, using ML models and Neural-Network models.
            - For ML used Randon Forest and XGBoost
            - For NN use Three Hidden Layers with Dropout, Batch Normalization and Relu Activation (in the last layer used Sigmoid).
            """)

    with st.container():  # Wine Project
        st.markdown("""
        #### [Wine Quality, Regression vs Classification (keras)](https://www.kaggle.com/code/walteralvarado/wine-quality-regression-vs-classification-keras/notebook)
        """)
        col1, col2 = st.columns([10, 25])
        with col1:
            titanic_img = Image.open('img/wine.jpg')
            st.image(titanic_img, width=IMAGE_WIDTH)
        with col2:
            st.markdown("""
            In this project, I'm going to predict The Quality of a wine, comparing two methods, Classification and Regression.
            - For Classification method, the last layer it's a Dense 10 layer with softmax activation
            - For Regression method, the last layer it's a Dense 1 layer without activation
            """)


