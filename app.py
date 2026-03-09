import os
import io
import zipfile
import tempfile
import shutil
from flask import Flask, request, render_template, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageCms
import rembg

app = Flask(__name__)
app.secret_key = 'super-secret-professional-key-2026'

class PackshotProcessor:
    def __init__(self):
        self.srgb_profile = ImageCms.createProfile("sRGB")

    def remove_background(self, input_data):
        try:
            output_data = rembg.remove(input_data)
            return Image.open(io.BytesIO(output_data)).convert("RGBA")
        except Exception as e:
            print(f"Błąd AI: {e}")
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
        flash('Nie wybrano plików.')
        return redirect(url_for('index'))

    files = request.files.getlist('files')
    asin_prefix = request.form.get('asin', '').strip().replace(" ", "-")

    if not files or files[0].filename == '':
        flash('Brak plików do przetworzenia.')
        return redirect(url_for('index'))

    processor = PackshotProcessor()
    
    temp_dir = tempfile.mkdtemp()
    output_zip_path = os.path.join(tempfile.gettempdir(), f"Packshots_2026_{os.urandom(4).hex()}.zip")

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

            base_name = os.path.splitext(secure_filename(file.filename))[0]
            input_data = file.read()

            img_no_bg = processor.remove_background(input_data)
            if not img_no_bg: continue

            img_cropped = processor.crop_to_content(img_no_bg)

            # Allegro
            img_allegro = processor.create_platform_image(img_cropped, 2560, 0.90)
            processor.save_image_with_size_limit(img_allegro, os.path.join(dirs['allegro'], f"{base_name}_allegro.jpg"), 20)

            # Amazon
            img_amazon = processor.create_platform_image(img_cropped, 3000, 0.95)
            amazon_name = f"{asin_prefix}_{base_name}.MAIN.jpg" if asin_prefix else f"{base_name}.MAIN.jpg"
            processor.save_image_with_size_limit(img_amazon, os.path.join(dirs['amazon'], amazon_name), 9.5)

            # eMag
            img_emag = processor.create_platform_image(img_cropped, 3000, 0.85)
            processor.save_image_with_size_limit(img_emag, os.path.join(dirs['emag'], f"{base_name}_emag.jpg"), 7.5)

            # Kaufland
            img_kaufland = processor.create_platform_image(img_cropped, 2048, 0.95)
            processor.save_image_with_size_limit(img_kaufland, os.path.join(dirs['kaufland'], f"{base_name}_kaufland.jpg"), 9.5)

        shutil.make_archive(output_zip_path.replace('.zip', ''), 'zip', temp_dir)

        with open(output_zip_path, 'rb') as f:
            return_data = io.BytesIO(f.read())
        
        return send_file(return_data, mimetype='application/zip', as_attachment=True, download_name='Gotowe_Packshoty_2026.zip')

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
