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
    try {
        model = await tf.loadLayersModel(MODEL_PATH);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Error loading the model. Please try again later.');
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
        // Display preview
        const imageUrl = URL.createObjectURL(file);
        preview.src = imageUrl;
        preview.classList.remove('hidden');
        uploadPrompt.classList.add('hidden');

        // Preprocess image
        const image = await loadImage(file);
        const tensor = preprocessImage(image);
        
        // Make prediction
        const predictions = await predict(tensor);
        
        // Display results
        displayResults(predictions, imageUrl);
        
        // Update history
        updateHistory(predictions[0].className, predictions[0].probability, imageUrl);
        
        // Cleanup
        tensor.dispose();
    } catch (error) {
        console.error('Error processing image:', error);
        alert('Error processing image. Please try again.');
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
            img.onerror = reject;
            img.src = e.target.result;
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Preprocess image for the model
function preprocessImage(img) {
    // Create a canvas to resize and convert to grayscale
    const canvas = document.createElement('canvas');
    canvas.width = IMAGE_SIZE;
    canvas.height = IMAGE_SIZE;
    const ctx = canvas.getContext('2d');
    
    // Resize and convert to grayscale
    ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const data = imageData.data;
    
    // Convert to grayscale and normalize
    const grayscale = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        grayscale[i / 4] = avg / 255.0;
    }
    
    // Reshape to match model input shape [1, 28, 28, 1]
    return tf.tensor(grayscale).reshape([1, IMAGE_SIZE, IMAGE_SIZE, 1]);
}

// Make prediction using the model
async function predict(tensor) {
    const predictions = await model.predict(tensor).data();
    return Array.from(predictions)
        .map((prob, i) => ({
            className: FASHION_CLASSES[i],
            probability: prob
        }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);
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

// Initialize the application
init(); 