
# Real-time Emotion Detection using FER and OpenCV

This project detects facial emotions in real-time using a webcam feed, and displays an interactive live bar chart of the detected emotions. Additionally, it saves an emotion video with bounding boxes and emotion labels around faces, a GIF of the real-time chart, and cumulative statistics in a plot.

## Overview

The code uses:
- `FER`: A library for facial expression recognition
- `OpenCV`: For video processing and face bounding box drawing
- `Matplotlib`: For real-time chart plotting of detected emotions
- `Pandas`: For storing emotion statistics
- `Imageio`: For creating a GIF of the emotion bar chart

The code is set to detect seven primary emotions: angry, disgust, fear, happy, sad, surprise, and neutral. Each detected emotion is displayed in real-time, and cumulative statistics are saved as images.

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install Required Libraries**
   This code requires the following libraries:
   ```bash
   pip install fer opencv-python-headless matplotlib numpy pandas imageio
   ```

3. **Ensure Webcam Access**
   Check that your webcam is connected and accessible for the real-time emotion detection to work correctly.

## Running the Code

To start emotion detection:
```bash
python emotion_detection.py
```

Upon running, the webcam will turn on, and you’ll see a window with real-time video where each detected face is annotated with the most confident emotion label. A live bar chart of emotion confidences will also update dynamically.

### Ending the Session
Press `ESC` to exit the emotion detection loop and save the output files.

## Outputs

The code generates the following outputs:

1. **Emotion Video (emotion_video.mp4):**
   - This video captures the webcam feed with bounding boxes around detected faces and emotion labels.

2. **Real-time Emotion GIF (emotion_chart.gif):**
   - A GIF showing the dynamic bar chart of emotions updated during detection.

3. **Cumulative Emotion Statistics (cumulative_emotions.jpg):**
   - A line plot that shows cumulative emotion confidence over time for each emotion type.

All outputs are saved in the root directory.

## Example Results

1. **Emotion Video:**
   ![Emotion Video Screenshot](images/emotion_video_screenshot.jpg)

2. **Real-time Emotion GIF:**
   ![Real-time Emotion GIF](images/emotion_chart_screenshot.gif)

3. **Cumulative Emotion Statistics:**
   ![Cumulative Emotion Statistics](images/cumulative_emotions_screenshot.jpg)

> Note: These images are examples of expected outputs. The exact content will vary based on detected emotions.
