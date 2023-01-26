import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title&header
st.title("Animals' Classification Model")
st.header('This model classifies the animals name after uploading their pictures to the app.')
st.write("The supporting animal types are Zebra, Horse, Mule, Deer, Camel, Bull")

#Upload picture
file = st.file_uploader("Upload picture", type=['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)

    #PIL convert image
    img =PILImage.create(file)

    #Model
    model = load_learner('animal1.pkl')

    #Prediction
    pred, pred_id, probs = model.predict(img)

    #Print result
    st.success(f"Prediction: {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.2f}%")

    #Plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
