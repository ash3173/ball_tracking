import cv2
import numpy as np
import pandas as pd

# path to the video
video_path = 'AI Assignment video.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

#save the processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (video_width, video_height))

# Define color ranges for different colored balls (in HSV)
color_ranges = {
    'white': ((0, 0, 168), (172, 111, 255)),  # White ball
    'orange': ((5, 150, 150), (15, 255, 255)),  # Orange ball
    'green': ((36, 100, 100), (86, 255, 255)),  # Green ball
    'yellow': ((20, 100, 100), (30, 255, 255))  # Yellow ball
}

# Define quadrants 
quadrants = {
    1: (video_width // 2, video_width, video_height // 2, video_height),  # Bottom right
    2: (0, video_width // 2, video_height // 2, video_height),            # Bottom left
    3: (0, video_width // 2, 0, video_height // 2),                       # Top left
    4: (video_width // 2, video_width, 0, video_height // 2)              # Top right
}

#list to store event logs
events = []

#dictionary to store the last known quadrant of each color
last_known_quadrant = {color: None for color in color_ranges}

# Function to check which quadrant the ball is in
def get_quadrant(x, y):
    for q, (x1, x2, y1, y2) in quadrants.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return q
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect and track balls
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # Filter small blobs
                x, y, w, h = cv2.boundingRect(cnt)
                center_x, center_y = x + w // 2, y + h // 2
                quadrant = get_quadrant(center_x, center_y)
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                # Check if the ball has changed quadrant
                if quadrant != last_known_quadrant[color]:
                    if last_known_quadrant[color] is not None:
                        events.append([timestamp, last_known_quadrant[color], color, 'Exit'])
                    events.append([timestamp, quadrant, color, 'Entry'])
                    last_known_quadrant[color] = quadrant
                
                # Draw rectangle around the ball and label it
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'{color} Entry {quadrant}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save events to text file
df = pd.DataFrame(events, columns=['Time', 'Quadrant', 'Ball Colour', 'Type'])
df.to_csv('events.txt', index=False)

