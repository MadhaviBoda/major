import numpy as np
import streamlit as st
import time
import urllib
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train = '/fruit classification/fruits-data/'

train_datagen = ImageDataGenerator(
  rescale=1./255
)
train_datagen = train_datagen.flow_from_directory(
        train,

        batch_size=32,
        target_size=(256, 256),
        class_mode='sparse')

labels = train_datagen.class_indices
Labels = '\n'.join(sorted(train_datagen.class_indices.keys()))
with open('Labelss.txt', 'w') as file:
  file.write(Labels)
class_names = list(labels.keys())

model_dir = "Models/model_final2.h5"
model = keras.models.load_model(model_dir)


html_temp_2 = '''
    <div style = "padding-bottom: 20px; padding-top: 20px; padding-left: 20px; padding-right: 20px">      
    <center><h2>Fruit Classifier</h2></center>
    <center><h3>Please upload any Fruit Image from the given list</h3></center>
    <center><h3>[apple, avocado, banana, kiwi, orange, pineapple, strawberry] </h3></center>
    </div>
    '''
st.markdown(html_temp_2, unsafe_allow_html=True)
html_temp = """
    <div>
    <h2></h2>
    <center><h3>Please upload any Fruit Image from the given list</h3></center>
    <center><h3>[apple, avocado, banana, kiwi, orange, pineapple, strawberry] </h3></center>
    </div>
    """
st.set_option('deprecation.showfileUploaderEncoding', False)
select = st.selectbox("Please select how you want to upload the image",("Please Select","Upload image via link","Upload image from device"))
if select == "Upload image via link":
    try:
        img = st.text_input('Enter the Image Address')
        img = Image.open(urllib.request.urlopen(img))
    except:
        if st.button('Submit'):
            show = st.error("Please Enter a valid Image Address!")
            time.sleep(4)
            show.empty()

if select == 'Upload image from device':
    file = st.file_uploader('Select', type = ['jpg', 'png', 'jpeg'])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    if file is not None:
        img = Image.open(file)

try:
    if img is not None:
        st.image(img, width = 300, caption = 'Uploaded Image')
        if st.button('Predict'):
            img = np.array(img.resize((256, 256), Image.Resampling.LANCZOS))
            img = np.array(img, dtype='uint8')
            img = np.array(img)/255.0
            
            prediction = model.predict(img[np.newaxis, ...])
            predicted_class = class_names[np.argmax(prediction[0], axis=-1)]
            acc = np.max(prediction[0]) * 100
            
            st.info(f'Classified Class is : "{predicted_class}"   With Accuracy: {acc}')

except:
    pass