import imageio as iio
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam


# Make a function for preprocessing images
def preprocess_image(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to [image_shape, image_shape, color_channel]
    """

    image = tf.image.resize(image, [img_shape, img_shape])
    # image = image/255. # Not required with EfficientNetBX models from tf.keras.applications

    return tf.cast(image, tf.float32), label


class_names = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheesecake",
    "cheese_plate",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "escargots",
    "falafel",
    "filet_mignon",
    "fish_and_chips",
    "foie_gras",
    "french_fries",
    "french_onion_soup",
    "french_toast",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",
    "hot_and_sour_soup",
    "hot_dog",
    "huevos_rancheros",
    "hummus",
    "ice_cream",
    "lasagna",
    "lobster_bisque",
    "lobster_roll_sandwich",
    "macaroni_and_cheese",
    "macarons",
    "miso_soup",
    "mussels",
    "nachos",
    "omelette",
    "onion_rings",
    "oysters",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "peking_duck",
    "pho",
    "pizza",
    "pork_chop",
    "poutine",
    "prime_rib",
    "pulled_pork_sandwich",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "scallops",
    "seaweed_salad",
    "shrimp_and_grits",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "tuna_tartare",
    "waffles",
]

st.title('Hello')

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    # To convert PIL Image to numpy array:
    img = np.array(img)

    # Check the shape of img_array:
    # Should output shape: (height, width, channels)
    st.write(img.shape)

    st.image(img)

    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    st.write(img.shape)

    # Create base model
    input_shape = (*(224, 224), 3)
    base_model = EfficientNetB0(include_top=False)

    # Unfreeze last 5 layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    # Create functional model
    inputs = layers.Input(shape=input_shape, name="input_layer")
    # Note: EfficientNetBX models have rescaling built-in
    x = base_model(
        inputs, training=False
    )  # make sure layers which should be in inference
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(101)(x)
    outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-4),
        metrics=["accuracy"],
    )

    temp = model.load_weights("food_vision_big_fine_tuned.h5", by_name=True)

    pred = model.predict(img)

    st.write(class_names[pred.argmax(axis=1)[0]])