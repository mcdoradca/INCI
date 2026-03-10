import os

# --- KRYTYCZNA OPTYMALIZACJA PAMIĘCI RAM (OS LEVEL) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import zipfile
import tempfile
import shutil
import gc
import urllib.parse
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageCms
from rembg import remove, new_session

app = Flask(__name__)
app.secret_key = 'super-secret-professional-key-2026'

GLOBAL_AI_SESSION = None

def get_ai_session():
    global GLOBAL_AI_SESSION
    if GLOBAL_AI_SESSION is None:
        print("Inicjalizacja globalnego modelu AI IS-Net (Zabezpieczenie CPU)...")
        GLOBAL_AI_SESSION = new_session("isnet-general-use", providers=['CPUExecutionProvider'])
        print("Model AI załadowany pomyślnie.")
    return GLOBAL_AI_SESSION

class PackshotProcessor:
    def __init__(self):
        self.srgb_profile = ImageCms.createProfile("sRGB")
        self.ai_session = get_ai_session()

    def remove_background(self, input_data):
        try:
            img = Image.open(io.BytesIO(input_data)).convert("RGBA")
            
            max_ai_dim = 960
            if img.width > max_ai_dim or img.height > max_ai_dim:
                img.thumbnail((max_ai_dim, max_ai_dim), Image.Resampling.LANCZOS)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            optimized_input_data = img_byte_arr.getvalue()

            del img
            gc.collect()

            output_data = remove(
                optimized_input_data,
                session=self.ai_session,
                post_process_mask=True,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=5
            )
            
            result_img = Image.open(io.BytesIO(output_data)).convert("RGBA")
            
            del optimized_input_data
            del output_data
            del img_byte_arr
            gc.collect()
            
            return result_img
            
        except Exception as e:
            print(f"Błąd krytyczny AI podczas szparowania: {e}")
            return None

    def crop_to_content(self, img):
        bbox = img.getbbox()
        return img.crop(bbox) if bbox else img

    def create_platform_image(self, cropped_img, target_size, fill_percentage):
        max_dim = int(target_size * fill_percentage)
        ratio = min(max_dim / cropped_img.width, max_dim / cropped_img.height)
        new_w, new_h = int(cropped_img.width * ratio), int(cropped_img.height * ratio)
        
        resized_img = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        
        canvas.paste(resized_img, (paste_x, paste_y), resized_img)
        return canvas

    def save_image_with_size_limit(self, img, output_path, max_size_mb, format="JPEG"):
        max_bytes = max_size_mb * 1024 * 1024
        quality = 100
        step = 5
        srgb_profile_bytes = ImageCms.ImageCmsProfile(self.srgb_profile).tobytes()

        while quality > 10:
            buffer = io.BytesIO()
            if format.upper() in ["JPEG", "JPG"]:
                img.save(buffer, format="JPEG", quality=quality, optimize=True, icc_profile=srgb_profile_bytes)
            else:
                img.save(buffer, format=format, icc_profile=srgb_profile_bytes)
            
            if buffer.tell() <= max_bytes or format.upper() not in ["JPEG", "JPG"]:
                with open(output_path, "wb") as f:
                    f.write(buffer.getvalue())
                return True
            quality -= step
            
        with open(output_path, "wb") as f:
            f.write(buffer.getvalue())
        return False

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    if 'files' not in request.files:
        return jsonify({'error': 'Nie wybrano plików.'}), 400

    files = request.files.getlist('files')
    asin_prefix = request.form.get('asin', '').strip().replace(" ", "-")

    if not files or files[0].filename == '':
        return jsonify({'error': 'Brak plików do przetworzenia.'}), 400

    # DYNAMICZNE POBIERANIE NAZWY FOLDERU
    folder_name = "Packshoty"
    first_file_path = files[0].filename
    if '/' in first_file_path:
        folder_name = first_file_path.split('/')[0]
    elif '\\' in first_file_path:
        folder_name = first_file_path.split('\\')[0]

    # Zabezpieczamy nazwę przed znakami specjalnymi, ale zostawiamy litery, cyfry i spacje
    safe_folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '-', '_')).strip()
    if not safe_folder_name:
        safe_folder_name = "Gotowe_Zdjecia"

    # Tworzymy ostateczną nazwę pliku ZIP
    zip_download_name = f"{safe_folder_name}_packshot.zip"

    processor = PackshotProcessor()
    
    temp_dir = tempfile.mkdtemp()
    output_zip_path = os.path.join(tempfile.gettempdir(), f"Temp_{os.urandom(4).hex()}.zip")

    try:
        dirs = {
            'allegro': os.path.join(temp_dir, "Allegro_Ready"),
            'amazon': os.path.join(temp_dir, "Amazon_Ready"),
            'emag': os.path.join(temp_dir, "eMag_Ready"),
            'kaufland': os.path.join(temp_dir, "Kaufland_Ready")
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        for file in files:
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff')):
                continue

            # secure_filename ucięłoby nam polskie znaki, zostawiamy oryginalną nazwę bazową bezpiecznie zakodowaną
            base_name = os.path.splitext(os.path.basename(file.filename))[0]
            # Podstawowe czyszczenie dla bezpieczeństwa zapisu na dysku
            base_name = "".join(c for c in base_name if c.isalnum() or c in (' ', '-', '_')).strip()
            
            input_data = file.read()

            img_no_bg = processor.remove_background(input_data)
            if not img_no_bg: continue

            img_cropped = processor.crop_to_content(img_no_bg)

            # Algorytmy platform
            img_allegro = processor.create_platform_image(img_cropped, 2560, 0.90)
            processor.save_image_with_size_limit(img_allegro, os.path.join(dirs['allegro'], f"{base_name}_allegro.jpg"), 20)

            img_amazon = processor.create_platform_image(img_cropped, 3000, 0.95)
            amazon_name = f"{asin_prefix}_{base_name}.MAIN.jpg" if asin_prefix else f"{base_name}.MAIN.jpg"
            processor.save_image_with_size_limit(img_amazon, os.path.join(dirs['amazon'], amazon_name), 9.5)

            img_emag = processor.create_platform_image(img_cropped, 3000, 0.85)
            processor.save_image_with_size_limit(img_emag, os.path.join(dirs['emag'], f"{base_name}_emag.jpg"), 7.5)

            img_kaufland = processor.create_platform_image(img_cropped, 2048, 0.95)
            processor.save_image_with_size_limit(img_kaufland, os.path.join(dirs['kaufland'], f"{base_name}_kaufland.jpg"), 9.5)
            
            gc.collect()

        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', temp_dir)

        with open(output_zip_path, 'rb') as f:
            return_data = io.BytesIO(f.read())
        
        # urllib.parse.quote zabezpiecza polskie znaki w nazwie pobieranego pliku ZIP
        encoded_filename = urllib.parse.quote(zip_download_name)
        
        response = send_file(return_data, mimetype='application/zip', as_attachment=True, download_name=zip_download_name)
        # Dodajemy nagłówki, by skrypt z index.html mógł odczytać dokładną nazwę
        response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{encoded_filename}"
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        return response

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)
        gc.collect()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
