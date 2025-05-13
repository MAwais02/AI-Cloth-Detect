# Fashion MNIST Classifier

A web-based Fashion MNIST image classifier using TensorFlow.js. This application allows users to upload images of clothing items and accessories, and classifies them into one of 10 fashion categories using a pre-trained deep learning model.

## Features

- Pure frontend implementation (no backend required)
- Real-time image classification using TensorFlow.js
- Image upload via drag-and-drop or file selection
- Displays top 3 predictions with confidence scores
- Prediction history tracking
- Share results functionality
- Responsive design using Tailwind CSS
- Loading indicators and error handling

## Fashion MNIST Categories

1. T-shirt/Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

## Technical Details

- Built with vanilla JavaScript and TensorFlow.js
- Uses TensorFlow.js v4.22.0 for model loading and inference
- Converts uploaded images to 28x28 grayscale format
- Implements proper tensor memory management
- Responsive UI using Tailwind CSS

## Project Structure

```
.
├── index.html          # Main HTML file
├── styles.css          # Custom CSS styles
├── app.js             # Application logic
├── tfjs_model/        # TensorFlow.js model files
│   ├── model.json
│   └── weights.bin
├── package.json       # Project dependencies
└── vercel.json        # Vercel deployment configuration
```

## Setup and Development

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open http://localhost:3000 in your browser

## Deployment

This project is configured for deployment on Vercel. Simply push to your GitHub repository and connect it to Vercel for automatic deployments.

## Model Information

The model used in this project is a convolutional neural network trained on the Fashion MNIST dataset. It consists of multiple convolutional layers with batch normalization, max pooling, and dropout layers for regularization. The model achieves high accuracy on the Fashion MNIST test set.

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## License

MIT 