from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform, io
from skimage.transform import resize
import uuid

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB upload limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aim')
def aim():
    return render_template('aim.html')

@app.route('/objective')
def objective():
    return render_template('objective.html')

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return render_template('simulation.html', error='No file part')
    
    file = request.files['image']
    if file.filename == '':
        return render_template('simulation.html', error='No selected file')
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    
    try:
        # Load and preprocess the image
        image = io.imread(file_path, as_gray=True)
        # Resize for speed (e.g., 256x256)
        image = resize(image, (256, 256), anti_aliasing=True)
        # Normalize to [0,1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Use fewer angles for speed
        theta = np.linspace(0., 180., 60, endpoint=False)  # 60 angles
        
        # Radon transform
        sinogram = transform.radon(image, theta=theta)
        
        # Inverse Radon (reconstruction)
        reconstructed = transform.iradon(sinogram, theta=theta)
        
        # Save images
        result_id = str(uuid.uuid4())
        sinogram_filename = f"sinogram_{result_id}.png"
        reconstructed_filename = f"reconstructed_{result_id}.png"
        
        # Save sinogram
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], sinogram_filename), sinogram, cmap='gray')
        # Save reconstructed image
        plt.imsave(os.path.join(app.config['UPLOAD_FOLDER'], reconstructed_filename), reconstructed, cmap='gray')
        
        return render_template(
            'result.html',
            original=f"uploads/{unique_filename}",
            sinogram=f"uploads/{sinogram_filename}",
            reconstructed=f"uploads/{reconstructed_filename}"
        )
    except Exception as e:
        return render_template('simulation.html', error=f'Error processing image: {str(e)}')

if __name__ == '__main__':
    # This makes the app accessible to others on your local network!
    app.run(host='0.0.0.0', port=5000, debug=True)
