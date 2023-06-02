import io
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions
import numpy as np

def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для завантаження зобрадень засобами Streamlit
    uploaded_file = st.file_uploader(
        label='Оберіть зображення для розпізнавання')
    if uploaded_file is not None:
        # Отримання завантаженого зображення
        image_data = uploaded_file.getvalue()
        # Відображення зобнаження на Web-сторінці засобами Streamlit
        st.image(image_data)
        # Повернення зображення у форматі PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None



@st.cache_resource()
def load_model():
    model = EfficientNetB0(weights='imagenet')
    return model

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])

# Завантажуємо попередньо навчену модель
model = load_model()
# Виводимо заголовок сторінки
st.title('Класифікація зображень')
# Виводимо форму зображення
img = load_image()
# Кнопка для запуску розпізнавання
result = st.button('Розпізнати зображення')
# Запускаємо розпізнавання
if result:
    # Попередня обробка
    x = preprocess_image(img)
    # Распізнавання
    preds = model.predict(x)
    # Виводимо заголовок
    st.write('**Результат:**')
    # Виводимо результати розпізнавання
    print_predictions(preds)