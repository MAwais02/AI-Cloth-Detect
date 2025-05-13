// Constants
const FASHION_CLASSES = [
    'T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
];
const MODEL_PATH = 'tfjs_model/model.json';
const IMAGE_SIZE = 28;
const MAX_HISTORY = 5;

// Global variables
let model = null;
let predictionHistory = [];

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadPrompt = document.getElementById('uploadPrompt');
const resultsSection = document.getElementById('results');
const predictionsDiv = document.getElementById('predictions');
const historyList = document.getElementById('historyList');
const loadingOverlay = document.getElementById('loadingOverlay');

// Load the model
async function loadModel() {
    showLoading(true);
    try {
        // Load the model
        model = await tf.loadLayersModel(MODEL_PATH);
        
        // Verify the model's input shape
        const inputShape = model.inputs[0].shape;
        console.log('Model loaded successfully');
        console.log('Input shape:', inputShape);
        
        // Compile the model with the same configuration as training
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        console.log('Model summary:', model.summary());
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Error loading the model. Please try again later.');
    } finally {
        showLoading(false);
    }
}

// Initialize the application
async function init() {
    await loadModel();
    setupEventListeners();
}

// Set up event listeners
function setupEventListeners() {
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
}

// File handling functions
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('drag-over');
}

function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processImage(file);
    } else {
        alert('Please drop an image file.');
    }
}

// Image processing and prediction
async function processImage(file) {
    showLoading(true);
    
    try {
        console.log('Processing image:', file.name);
        
        // Display preview
        const imageUrl = URL.createObjectURL(file);
        preview.src = imageUrl;
        preview.classList.remove('hidden');
        uploadPrompt.classList.add('hidden');

        // Preprocess image
        console.log('Loading image...');
        const image = await loadImage(file);
        console.log('Image loaded, preprocessing...');
        const tensor = await preprocessImage(image);
        console.log('Preprocessing complete. Tensor shape:', tensor.shape);
        
        // Make prediction
        console.log('Making prediction...');
        if (!model) {
            console.error('Model not loaded properly');
            throw new Error('Model not loaded properly');
        }
        const predictions = await predict(tensor);
        console.log('Predictions:', predictions);
        
        // Display results
        displayResults(predictions, imageUrl);
        
        // Update history
        updateHistory(predictions[0].className, predictions[0].probability, imageUrl);
        
        // Cleanup
        tf.dispose(tensor);
        console.log('Processing complete');
    } catch (error) {
        console.error('Error processing image:', error);
        alert('Error processing image: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Load image as HTMLImageElement
function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = (error) => {
                console.error('Error loading image:', error);
                reject(error);
            };
            img.src = e.target.result;
        };
        reader.onerror = (error) => {
            console.error('Error reading file:', error);
            reject(error);
        };
        reader.readAsDataURL(file);
    });
}

// Preprocess image for the model
async function preprocessImage(img) {
    return tf.tidy(() => {
        // Log original image dimensions
        console.log('Original image dimensions:', img.width, 'x', img.height);
        
        // Convert the image to grayscale tensor
        let tensor = tf.browser.fromPixels(img, 1);
        console.log('Initial tensor shape:', tensor.shape);
        
        // Resize to match Fashion MNIST dimensions
        tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);
        console.log('After resize shape:', tensor.shape);
        
        // Normalize to [0,1]
        tensor = tensor.toFloat().div(255.0);
        
        // Reshape to match the model's expected input shape [batch_size, height, width, channels]
        // The TFJS JSON model expects this exact format
        tensor = tensor.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);
        console.log('Final tensor shape:', tensor.shape);
        
        return tensor;
    });
}

// Make prediction using the model
async function predict(tensor) {
    return tf.tidy(() => {
        if (!model) {
            throw new Error('Model not loaded');
        }
        
        console.log('Prediction input tensor shape:', tensor.shape);
        
        // Get prediction
        const predictions = model.predict(tensor);
        console.log('Raw prediction shape:', predictions.shape);
        
        // Get probabilities
        const probabilities = predictions.dataSync();
        console.log('Raw probabilities:', probabilities);
        
        // Verify probabilities sum close to 1
        const sum = probabilities.reduce((a, b) => a + b, 0);
        console.log('Sum of probabilities:', sum);
        
        // Process results
        return Array.from(probabilities)
            .map((probability, index) => ({
                className: FASHION_CLASSES[index],
                probability: probability
            }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, 3);
    });
}

// Display prediction results
function displayResults(predictions, imageUrl) {
    resultsSection.classList.remove('hidden');
    
    const resultsHtml = predictions.map((pred, index) => `
        <div class="flex items-center justify-between p-4 ${index === 0 ? 'bg-blue-50 rounded-lg' : ''}">
            <div class="flex-1">
                <p class="font-semibold">${pred.className}</p>
                <div class="confidence-bar mt-2">
                    <div class="confidence-bar-fill" style="width: ${(pred.probability * 100).toFixed(1)}%"></div>
                </div>
            </div>
            <div class="ml-4">
                <span class="text-lg font-bold">${(pred.probability * 100).toFixed(1)}%</span>
            </div>
        </div>
    `).join('');
    
    predictionsDiv.innerHTML = resultsHtml;
    
    if (predictions[0].probability > 0.7) {
        addShareButton(predictions[0], imageUrl);
    }
}

// Add share button for high-confidence predictions
function addShareButton(topPrediction, imageUrl) {
    const shareButton = document.createElement('button');
    shareButton.className = 'share-button mt-4 w-full bg-blue-500 text-white py-2 px-4 rounded-lg hover:bg-blue-600';
    shareButton.textContent = 'Share Result';
    shareButton.onclick = () => shareResult(topPrediction, imageUrl);
    predictionsDiv.appendChild(shareButton);
}

// Share result function
function shareResult(prediction, imageUrl) {
    const text = `I found a ${prediction.className} with ${(prediction.probability * 100).toFixed(1)}% confidence using the Fashion MNIST Classifier!`;
    
    if (navigator.share) {
        navigator.share({
            title: 'Fashion MNIST Prediction',
            text: text,
            url: window.location.href
        }).catch(console.error);
    } else {
        // Fallback to copying to clipboard
        navigator.clipboard.writeText(text)
            .then(() => alert('Result copied to clipboard!'))
            .catch(console.error);
    }
}

// Update prediction history
function updateHistory(className, probability, imageUrl) {
    const historyItem = {
        className,
        probability,
        imageUrl,
        timestamp: new Date().toLocaleString()
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > MAX_HISTORY) {
        predictionHistory.pop();
    }
    
    displayHistory();
}

// Display prediction history
function displayHistory() {
    if (predictionHistory.length === 0) {
        historyList.innerHTML = '<p class="text-gray-500">No predictions yet</p>';
        return;
    }
    
    const historyHtml = predictionHistory.map(item => `
        <div class="history-item flex items-center p-4 bg-gray-50 rounded-lg">
            <img src="${item.imageUrl}" alt="${item.className}" class="w-16 h-16 object-cover rounded">
            <div class="ml-4">
                <p class="font-semibold">${item.className}</p>
                <p class="text-sm text-gray-500">${item.timestamp}</p>
                <p class="text-sm">${(item.probability * 100).toFixed(1)}% confidence</p>
            </div>
        </div>
    `).join('');
    
    historyList.innerHTML = historyHtml;
}

// Loading indicator
function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

// Add debug function to help troubleshoot model
function debugModel() {
    if (!model) {
        console.error('Model not loaded');
        return;
    }
    
    console.log('Model architecture:', model.toJSON());
    console.log('Input shape:', model.inputs[0].shape);
    console.log('Output shape:', model.outputs[0].shape);
    
    // Create a sample tensor matching the input shape and run inference
    const sampleTensor = tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1]);
    try {
        const result = model.predict(sampleTensor);
        console.log('Sample inference result shape:', result.shape);
        console.log('Sample inference result:', result.dataSync());
    } catch (error) {
        console.error('Error running sample inference:', error);
    } finally {
        tf.dispose(sampleTensor);
    }
}

// Initialize the application
window.onload = init;