from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import io
import time
import pandas as pd
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  

MODEL_PATHS = {
    "VGG16": "model/VGG16.h5",
    "MobileNetV2": "model/MobileNetV2.h5"
}
selected_model = "VGG16"
model = load_model(MODEL_PATHS[selected_model])
model.build((None, 224, 224, 3))
_ = model(np.zeros((1, 224, 224, 3)))
print("[INFO] Model siap digunakan.")

def predict_image(file_stream):
    start_time = time.time()

    img = Image.open(file_stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    print(f"\nData gambar mentah:")
    print(f"  Shape: {img_array.shape}")
    print(f"  Contoh pixel [0,0]: {img_array[0,0].astype(int)}")
    print(f"  Nilai min={img_array.min():.2f}, max={img_array.max():.2f}")

    img_array = np.expand_dims(img_array, axis=0) / 255.0
    print(f"\nSetelah normalisasi:")
    print(f"  Contoh pixel [0,0]: {img_array[0,0,0]}")
    print(f"  Nilai min={img_array.min():.4f}, max={img_array.max():.4f}")

    try:
        conv_layer = None

        base_model = model.layers[0] if hasattr(model.layers[0], "layers") else model
        for layer in base_model.layers:
            if "conv" in layer.name.lower():
                conv_layer = layer
                break

        if conv_layer is not None:
            from tensorflow.keras.models import Model
            conv_model = Model(inputs=base_model.input, outputs=conv_layer.output)
            feature_map = conv_model.predict(img_array)

            print(f"\n [STEP 3] Konvolusi pertama:")
            print(f"  Layer: {conv_layer.name}")
            print(f"  Feature map shape: {feature_map.shape}")
            print(f"  Nilai rata-rata feature map: {np.mean(feature_map):.4f}")
            print(f"  Contoh nilai feature_map[0,100,100,0]: {feature_map[0,100,100,0]:.6f}")
        else:
            print("\nidak ditemukan layer konvolusi pada base model.")
    except Exception as e:
        print(f"\n[ERROR] Debug feature map gagal: {e}")


    prediction = model.predict(img_array)[0][0]
    print(f"\nPrediksi Akhir:")
    print(f"  Nilai probabilitas sigmoid: {prediction:.4f}")

    end_time = time.time()
    analysis_time = end_time - start_time
    print(f"\nWaktu analisis total: {analysis_time:.3f} detik")

    label = "Judol" if prediction <= 0.5 else "Bukan Judol"
    print(f" Hasil akhir: {label}")

    return label, float(prediction), img, analysis_time



@app.route('/', methods=['GET', 'POST'])
def index():
    global model, selected_model

    if request.method == 'POST':
        file = request.files.get('image')
        print(f"[DEBUG] Received file: {file.filename if file else 'None'}")
        if not file:
            return render_template('index.html', selected_model=selected_model, result=None, error="No file selected")

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('index.html', selected_model=selected_model, result=None, error="Invalid file type. Only PNG, JPG, or JPEG allowed")

        try:
            result, threshold, img, analysis_time = predict_image(file.stream)

            if result == "Bukan Judol":
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_data = f"data:image/png;base64,{img_base64}"
            else:
                image_data = None

            return render_template(
                'index.html',
                selected_model=selected_model,
                result=result,
                threshold=threshold,
                image_data=image_data,
                analysis_time=analysis_time,
                error=None
            )
        except Exception as e:
            print(f"[ERROR] Error processing image: {str(e)}")
            return render_template('index.html', selected_model=selected_model, result=None, error=f"Error processing image: {str(e)}")

    return render_template('index.html', selected_model=selected_model, result=None, error=None)

@app.route('/set_model', methods=['POST'])
def set_model():
    global model, selected_model
    selected_model = request.form.get('model_name')

    if selected_model in MODEL_PATHS:
        model_path = MODEL_PATHS[selected_model]
        model = load_model(model_path)
        print(f"[INFO] Model aktif diganti ke: {selected_model} ({model_path})")
    else:
        print(f"[WARNING] Model {selected_model} tidak ditemukan!")

    return redirect(url_for('index'))

@app.route('/compare')
def compare():
    csv_path = "hasil_eksperimen.csv"
    if not os.path.exists(csv_path):
        return render_template("compare.html", columns=[], rows=[], best_rows=[])

    df = pd.read_csv(csv_path)

    best_rows = {}
    for model_name in df["Model"].unique():
        subset = df[df["Model"] == model_name]
        best_idx = subset["Akurasi Validasi"].idxmax()
        best_rows[best_idx] = model_name 

    columns = df.columns.tolist()
    rows = df.values.tolist()

    return render_template("compare.html",
                           columns=columns,
                           rows=rows,
                           best_rows=best_rows)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

