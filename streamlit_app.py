import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

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
    - MySql, PostgresSql 
    - TensorFlow, scikit-learn
       
    """)

st.write("## Projects and Experience:")
with st.container():  # Projects
    IMAGE_WIDTH = 180
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
                In this project, I used TensorFlow and Keras to build a handwritten digit recognition convolutional neural network,
                 trained with the MNIST Dataset.
            """)