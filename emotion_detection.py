from fer import FER
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import time

# Function to update the real-time emotion chart
def update_chart(detected_emotions, ax, fig):
    ax.clear()
    ax.bar(emotion_labels, [detected_emotions.get(emotion, 0) for emotion in emotion_labels], color='lightblue')
    plt.ylim(0, 1)
    plt.ylabel('Confidence')
    plt.title('Real-time Emotion Detection')
    ax.set_xticks(range(len(emotion_labels)))
    ax.set_xticklabels(emotion_labels, rotation=45)
    fig.canvas.draw()
    fig.canvas.flush_events()

# Set up the emotion detector and video capture
detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

# Video writer settings
frame_rate = 4.3
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('emotion_video.mp4', fourcc, frame_rate, (640, 480))

# Set up real-time plot
plt.ion()
fig, ax = plt.subplots()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
bars = ax.bar(emotion_labels, [0]*7, color='lightblue')
plt.ylim(0, 1)
plt.ylabel('Confidence')
plt.title('Real-time Emotion Detection')
ax.set_xticks(range(len(emotion_labels)))
ax.set_xticklabels(emotion_labels, rotation=45)

# GIF writer for emotion chart
gif_writer = imageio.get_writer('emotion_chart.gif', mode='I', duration=0.1)
emotion_statistics = []

# Main loop for real-time emotion detection
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        result = detector.detect_emotions(small_frame)
        largest_face = max(result, key=lambda face: face["box"][2] * face["box"][3], default=None)

        if largest_face:
            box = largest_face["box"]
            current_emotions = largest_face["emotions"]
            emotion_statistics.append(current_emotions)

            x, y, w, h = [int(coord * 2) for coord in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            emotion_type = max(current_emotions, key=current_emotions.get)
            emotion_score = current_emotions[emotion_type]
            emotion_text = f"{emotion_type}: {emotion_score:.2f}"
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            update_chart(current_emotions, ax, fig)
            out.write(frame)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            width, height = fig.canvas.get_width_height()
            try:
                image = image.reshape((height, width, 3))
                gif_writer.append_data(image)
            except ValueError as e:
                print(f"Error reshaping array: {e}")
                continue

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) == 27: 
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)
    out.release()
    gif_writer.close()

    emotion_df = pd.DataFrame(emotion_statistics)

    plt.figure(figsize=(10, 10))
    for emotion in emotion_labels:
        plt.plot(emotion_df[emotion].cumsum(), label=emotion)
    plt.title('Cumulative Emotion Statistics Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Cumulative Confidence')
    plt.legend()
    plt.savefig('cumulative_emotions.jpg')
    plt.close()
