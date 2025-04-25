# from flask import Flask, request, render_template
# import joblib
# import numpy as np
# import pandas as pd
# import os

# # Use relative paths so it works in Docker
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # Load trained models and encoder
# model1 = joblib.load(os.path.join(BASE_DIR, 'model_decision_tree_gini.pkl'))
# model2 = joblib.load(os.path.join(BASE_DIR, 'model_decision_tree_entropy.pkl'))
# le = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

# # Load symptoms list and additional info
# severity_df = pd.read_csv(os.path.join(BASE_DIR, 'Symptom-severity.csv'))
# desc_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_Description.csv'))
# prec_df = pd.read_csv(os.path.join(BASE_DIR, 'symptom_precaution.csv'))

# # Clean and prepare symptoms list
# symptoms_list = sorted(severity_df['Symptom'].str.strip().str.replace(' ', '_').str.lower().unique())

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html', symptoms=symptoms_list)

# @app.route('/predict', methods=['POST'])
# def predict():
#     selected_symptoms = request.form.getlist('symptoms')
#     input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]
    
#     pred1 = model1.predict([input_data])[0]
#     pred2 = model2.predict([input_data])[0]
#     final_pred = le.inverse_transform([np.bincount([pred1, pred2]).argmax()])[0]

#     description = desc_df[desc_df['Disease'].str.lower() == final_pred]['Description'].values
#     precautions = prec_df[prec_df['Disease'].str.lower() == final_pred].values

#     description_text = description[0] if description.size else "No description available."
#     precaution_list = [p for p in precautions[0][1:] if pd.notna(p)] if precautions.size else []

#     return render_template('result.html', disease=final_pred.title(), 
#                            description=description_text, precautions=precaution_list)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')  # allow external access in Docker
    
    
    
    
from flask import Flask, render_template_string

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Classifier</title>
    <style>
        :root {
            --primary-color: #4361ee;
            --secondary-color: #3f37c9;
            --accent-color: #4895ef;
            --light-color: #f8f9fa;
            --dark-color: #212529;
            --success-color: #4cc9f0;
            --warning-color: #f72585;
            --border-radius: 12px;
            --box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            --transition: all 0.3s ease;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #f5f7ff;
            color: var(--dark-color);
            line-height: 1.6;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }

        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
            text-align: center;
            border-bottom-left-radius: 30px;
            border-bottom-right-radius: 30px;
            box-shadow: var(--box-shadow);
            margin-bottom: 3rem;
        }

        header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        header p {
            font-size: 1.1rem;
            opacity: 0.9;
            max-width: 700px;
            margin: 0 auto;
        }

        .main-content {
            display: flex;
            flex-direction: column;
            gap: 2rem;
            margin-bottom: 3rem;
        }

        .upload-section {
            background: white;
            border-radius: var(--border-radius);
            padding: 2rem;
            box-shadow: var(--box-shadow);
            text-align: center;
        }

        .upload-area {
            border: 2px dashed #ccc;
            border-radius: var(--border-radius);
            padding: 3rem 2rem;
            margin: 1.5rem 0;
            cursor: pointer;
            transition: var(--transition);
            position: relative;
            overflow: hidden;
        }

        .upload-area:hover {
            border-color: var(--accent-color);
            background-color: rgba(72, 149, 239, 0.05);
        }

        .upload-area.active {
            border-color: var(--success-color);
            background-color: rgba(76, 201, 240, 0.05);
        }

        .upload-area i {
            font-size: 3rem;
            color: var(--accent-color);
            margin-bottom: 1rem;
            display: block;
        }

        .upload-area h3 {
            margin-bottom: 0.5rem;
            color: var(--dark-color);
        }

        .upload-area p {
            color: #666;
            margin-bottom: 1rem;
        }

        .btn {
            display: inline-block;
            background: var(--primary-color);
            color: white;
            padding: 0.8rem 1.8rem;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: var(--transition);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 10px rgba(67, 97, 238, 0.3);
        }

        .btn:hover {
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(67, 97, 238, 0.4);
        }

        .btn-outline {
            background: transparent;
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
            box-shadow: none;
        }

        .btn-outline:hover {
            background: var(--primary-color);
            color: white;
        }

        .btn-warning {
            background: var(--warning-color);
            box-shadow: 0 4px 10px rgba(247, 37, 133, 0.3);
        }

        .btn-warning:hover {
            background: #e5177b;
            box-shadow: 0 6px 15px rgba(247, 37, 133, 0.4);
        }

        .preview-section {
            display: none;
            background: white;
            border-radius: var(--border-radius);
            padding: 2rem;
            box-shadow: var(--box-shadow);
        }

        .preview-section h2 {
            margin-bottom: 1.5rem;
            color: var(--dark-color);
            text-align: center;
        }

        .preview-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1.5rem;
            justify-content: center;
        }

        .image-preview {
            width: 250px;
            height: 250px;
            border-radius: var(--border-radius);
            overflow: hidden;
            position: relative;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .image-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: var(--transition);
        }

        .image-preview:hover img {
            transform: scale(1.05);
        }

        .results-section {
            display: none;
            background: white;
            border-radius: var(--border-radius);
            padding: 2rem;
            box-shadow: var(--box-shadow);
        }

        .results-section h2 {
            margin-bottom: 1.5rem;
            color: var(--dark-color);
            text-align: center;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }

        .result-card {
            background: var(--light-color);
            border-radius: var(--border-radius);
            padding: 1.5rem;
            transition: var(--transition);
        }

        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }

        .result-card h3 {
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }

        .confidence-meter {
            height: 10px;
            background: #e9ecef;
            border-radius: 5px;
            margin: 1rem 0;
            overflow: hidden;
        }

        .confidence-level {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-color), var(--success-color));
            border-radius: 5px;
            width: 0%;
            transition: width 1s ease;
        }

        .confidence-value {
            font-weight: bold;
            color: var(--dark-color);
        }

        .top-result {
            grid-column: 1 / -1;
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, rgba(67, 97, 238, 0.1), rgba(76, 201, 240, 0.1));
            border-radius: var(--border-radius);
        }

        .top-result h3 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .top-result .confidence-value {
            font-size: 1.8rem;
            color: var(--success-color);
        }

        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(67, 97, 238, 0.2);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        footer {
            text-align: center;
            padding: 2rem 0;
            background: var(--dark-color);
            color: white;
            margin-top: 3rem;
        }

        footer p {
            opacity: 0.8;
        }

        /* Responsive styles */
        @media (max-width: 768px) {
            header h1 {
                font-size: 2rem;
            }

            header p {
                font-size: 1rem;
            }

            .upload-area {
                padding: 2rem 1rem;
            }

            .btn {
                padding: 0.7rem 1.5rem;
                font-size: 0.9rem;
            }

            .results-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 480px) {
            header {
                padding: 1.5rem 0;
            }

            header h1 {
                font-size: 1.8rem;
            }

            .upload-section, .preview-section, .results-section {
                padding: 1.5rem;
            }

            .image-preview {
                width: 100%;
                height: 200px;
            }
        }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
</head>
<body>
    <header>
        <div class="container">
            <h1>AI Image Classifier</h1>
            <p>Upload your images and let our advanced AI model identify and classify them with remarkable accuracy.</p>
        </div>
    </header>

    <div class="container">
        <main class="main-content">
            <section class="upload-section">
                <h2>Upload Your Image</h2>
                <div class="upload-area" id="uploadArea">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <h3>Drag & Drop Your Image Here</h3>
                    <p>or click to browse files</p>
                    <span class="btn btn-outline">Select Image</span>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;">
                </div>
                <div class="actions">
                    <button class="btn" id="classifyBtn" disabled>Classify Image</button>
                    <button class="btn btn-warning" id="resetBtn" disabled>Reset</button>
                </div>
            </section>

            <div class="loading" id="loadingIndicator">
                <div class="spinner"></div>
                <h3>Analyzing your image...</h3>
                <p>Our AI is working hard to identify the contents of your image.</p>
            </div>

            <section class="preview-section" id="previewSection">
                <h2>Image Preview</h2>
                <div class="preview-container" id="previewContainer">
                    <!-- Preview will be inserted here -->
                </div>
            </section>

            <section class="results-section" id="resultsSection">
                <h2>Classification Results</h2>
                <div class="results-grid" id="resultsGrid">
                    <!-- Results will be inserted here -->
                </div>
            </section>
        </main>
    </div>

    <footer>
        <div class="container">
            <p>&copy; 2023 AI Image Classifier. All rights reserved.</p>
        </div>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // DOM Elements
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            const classifyBtn = document.getElementById('classifyBtn');
            const resetBtn = document.getElementById('resetBtn');
            const previewSection = document.getElementById('previewSection');
            const previewContainer = document.getElementById('previewContainer');
            const resultsSection = document.getElementById('resultsSection');
            const resultsGrid = document.getElementById('resultsGrid');
            const loadingIndicator = document.getElementById('loadingIndicator');

            // Variables
            let selectedFile = null;

            // Event Listeners
            uploadArea.addEventListener('click', () => fileInput.click());
            
            fileInput.addEventListener('change', handleFileSelect);
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('active');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('active');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('active');
                if (e.dataTransfer.files.length) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelect({ target: fileInput });
                }
            });
            
            classifyBtn.addEventListener('click', classifyImage);
            resetBtn.addEventListener('click', resetAll);

            // Functions
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                if (!file.type.match('image.*')) {
                    alert('Please select an image file (JPEG, PNG, etc.)');
                    return;
                }
                
                selectedFile = file;
                displayPreview(file);
                classifyBtn.disabled = false;
                resetBtn.disabled = false;
            }

            function displayPreview(file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewContainer.innerHTML = '';
                    
                    const imgContainer = document.createElement('div');
                    imgContainer.className = 'image-preview';
                    
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.alt = 'Selected image preview';
                    
                    imgContainer.appendChild(img);
                    previewContainer.appendChild(imgContainer);
                    previewSection.style.display = 'block';
                    resultsSection.style.display = 'none';
                };
                
                reader.readAsDataURL(file);
            }

            function classifyImage() {
                if (!selectedFile) return;
                
                // Show loading indicator
                loadingIndicator.style.display = 'block';
                classifyBtn.disabled = true;
                
                // Simulate API call with setTimeout
                setTimeout(() => {
                    // Hide loading indicator
                    loadingIndicator.style.display = 'none';
                    
                    // Get more accurate mock results based on filename
                    const fileName = selectedFile.name.toLowerCase();
                    const results = generateAccurateMockResults(fileName);
                    
                    // Display results
                    displayResults(results);
                    
                    // Show results section
                    resultsSection.style.display = 'block';
                }, 2000);
            }

            function generateAccurateMockResults(filename) {
                // Common animal categories
                const animalCategories = [
                    { label: 'Cat', confidence: 85 },
                    { label: 'Dog', confidence: 75 },
                    { label: 'Bird', confidence: 60 },
                    { label: 'Fish', confidence: 40 },
                    { label: 'Rabbit', confidence: 30 }
                ];
                
                // Common object categories
                const objectCategories = [
                    { label: 'Car', confidence: 80 },
                    { label: 'Building', confidence: 70 },
                    { label: 'Flower', confidence: 65 },
                    { label: 'Tree', confidence: 55 },
                    { label: 'Book', confidence: 45 }
                ];
                
                // Common landscape categories
                const landscapeCategories = [
                    { label: 'Mountain', confidence: 85 },
                    { label: 'Beach', confidence: 75 },
                    { label: 'Forest', confidence: 65 },
                    { label: 'Cityscape', confidence: 55 },
                    { label: 'Sunset', confidence: 45 }
                ];
                
                // Determine which category set to use based on filename
                let selectedCategories;
                
                if (filename.includes('cat') || filename.includes('kitten') || filename.includes('feline')) {
                    selectedCategories = [
                        { label: 'Cat', confidence: 92 },
                        { label: 'Kitten', confidence: 85 },
                        { label: 'Domestic Animal', confidence: 80 },
                        { label: 'Pet', confidence: 75 },
                        { label: 'Mammal', confidence: 70 }
                    ];
                } 
                else if (filename.includes('dog') || filename.includes('puppy') || filename.includes('canine')) {
                    selectedCategories = [
                        { label: 'Dog', confidence: 90 },
                        { label: 'Puppy', confidence: 82 },
                        { label: 'Canine', confidence: 78 },
                        { label: 'Pet', confidence: 75 },
                        { label: 'Mammal', confidence: 70 }
                    ];
                }
                else if (filename.includes('bird') || filename.includes('eagle') || filename.includes('parrot')) {
                    selectedCategories = [
                        { label: 'Bird', confidence: 88 },
                        { label: 'Avian', confidence: 80 },
                        { label: 'Flying Animal', confidence: 75 },
                        { label: 'Wildlife', confidence: 70 },
                        { label: 'Animal', confidence: 65 }
                    ];
                }
                else if (filename.includes('car') || filename.includes('vehicle') || filename.includes('auto')) {
                    selectedCategories = [
                        { label: 'Car', confidence: 95 },
                        { label: 'Vehicle', confidence: 85 },
                        { label: 'Automobile', confidence: 80 },
                        { label: 'Transportation', confidence: 70 },
                        { label: 'Machine', confidence: 65 }
                    ];
                }
                else if (filename.includes('flower') || filename.includes('rose') || filename.includes('tulip')) {
                    selectedCategories = [
                        { label: 'Flower', confidence: 90 },
                        { label: 'Plant', confidence: 85 },
                        { label: 'Rose', confidence: 75 },
                        { label: 'Tulip', confidence: 65 },
                        { label: 'Botany', confidence: 60 }
                    ];
                }
                else if (filename.includes('mountain') || filename.includes('peak') || filename.includes('alps')) {
                    selectedCategories = [
                        { label: 'Mountain', confidence: 95 },
                        { label: 'Landscape', confidence: 85 },
                        { label: 'Nature', confidence: 80 },
                        { label: 'Peak', confidence: 75 },
                        { label: 'Outdoors', confidence: 70 }
                    ];
                }
                else {
                    // Default to animals with some randomization
                    selectedCategories = animalCategories.map(item => ({
                        label: item.label,
                        confidence: Math.min(100, Math.max(5, item.confidence + Math.floor(Math.random() * 20) - 10))
                    })).sort((a, b) => b.confidence - a.confidence);
                }
                
                return selectedCategories;
            }

            function displayResults(results) {
                resultsGrid.innerHTML = '';
                
                // Add top result
                const topResult = document.createElement('div');
                topResult.className = 'top-result result-card';
                topResult.innerHTML = `
                    <h3>Top Prediction</h3>
                    <p>${results[0].label}</p>
                    <div class="confidence-meter">
                        <div class="confidence-level" style="width: ${results[0].confidence}%"></div>
                    </div>
                    <p class="confidence-value">${results[0].confidence}% confidence</p>
                `;
                resultsGrid.appendChild(topResult);
                
                // Add other results
                results.slice(1).forEach(result => {
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.innerHTML = `
                        <h3>${result.label}</h3>
                        <div class="confidence-meter">
                            <div class="confidence-level" style="width: ${result.confidence}%"></div>
                        </div>
                        <p class="confidence-value">${result.confidence}% confidence</p>
                    `;
                    resultsGrid.appendChild(card);
                });
            }

            function resetAll() {
                selectedFile = null;
                fileInput.value = '';
                previewContainer.innerHTML = '';
                previewSection.style.display = 'none';
                resultsSection.style.display = 'none';
                classifyBtn.disabled = true;
                resetBtn.disabled = true;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0')
