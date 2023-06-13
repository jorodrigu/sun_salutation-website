import tensorflow as models

def predict(data):
    loaded_model = models.load_model("2023_06_08_BNTB_flipped_img")
    return loaded_model.predict(data)
