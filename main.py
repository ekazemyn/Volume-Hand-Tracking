import cv2
import mediapipe as mp
import time
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class HandDetector:
    """
    Hand detection and tracking using MediaPipe
    """
    
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            mode: Static image mode or video mode
            max_hands: Maximum number of hands to detect
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Landmark IDs for fingertips
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.landmark_list = []
    
    def find_hands(self, image, draw=True):
        """
        Find hands in the image
        
        Args:
            image: Input image
            draw: Whether to draw hand landmarks
            
        Returns:
            Image with hand landmarks drawn (if draw=True)
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_image)
        
        # Draw hand landmarks
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return image
    
    def find_position(self, image, hand_no=0, draw=True):
        """
        Find hand landmark positions
        
        Args:
            image: Input image
            hand_no: Hand number (0 for first hand)
            draw: Whether to draw landmarks
            
        Returns:
            List of landmark positions and bounding box
        """
        x_list = []
        y_list = []
        bbox = []
        self.landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                for id, landmark in enumerate(hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    self.landmark_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Calculate bounding box
                if x_list and y_list:
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = [xmin, ymin, xmax, ymax]
                    
                    if draw:
                        cv2.rectangle(image, (bbox[0] - 20, bbox[1] - 20),
                                    (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        
        return self.landmark_list, bbox
    
    def fingers_up(self):
        """
        Check which fingers are up
        
        Returns:
            List of 1s and 0s indicating which fingers are up
        """
        fingers = []
        
        if len(self.landmark_list) != 0:
            # Thumb 
            if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Four fingers 
            for id in range(1, 5):
                if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers
    
    def find_distance(self, p1, p2, image, draw=True):
        """
        Find distance between two landmarks
        
        Args:
            p1, p2: Landmark IDs
            image: Input image
            draw: Whether to draw distance visualization
            
        Returns:
            Distance, image, and coordinate info
        """
        if len(self.landmark_list) == 0:
            return 0, image, []
        
        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, image, [x1, y1, x2, y2, cx, cy]


class VolumeController:
    """
    System volume controller using hand gestures
    """
    
    def __init__(self):
        """Initialize volume controller"""
        try:
            # Get audio device
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get volume range
            self.vol_range = self.volume.GetVolumeRange()
            self.min_vol = self.vol_range[0]
            self.max_vol = self.vol_range[1]
            
            # Volume control parameters
            self.min_distance = 50
            self.max_distance = 300
            
        except Exception as e:
            print(f"Error initializing volume controller: {e}")
            self.volume = None
    
    def set_volume_by_distance(self, distance):
        """
        Set system volume based on hand distance
        
        Args:
            distance: Distance between thumb and index finger
            
        Returns:
            Volume percentage (0-100)
        """
        if self.volume is None:
            return 0
        
        # Map distance to volume
        vol = np.interp(distance, [self.min_distance, self.max_distance], [self.min_vol, self.max_vol])
        vol_percentage = np.interp(distance, [self.min_distance, self.max_distance], [0, 100])
        
        # Set system volume
        try:
            self.volume.SetMasterVolumeLevel(vol, None)
        except Exception as e:
            print(f"Error setting volume: {e}")
        
        return int(vol_percentage)


class HandVolumeControl:
    """
    Main application class combining hand detection and volume control
    """
    
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize hand volume control application
        
        Args:
            camera_id: Camera device ID
            width: Camera width
            height: Camera height
        """
        self.width = width
        self.height = height
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(3, width)
        self.cap.set(4, height)
        
        # Initialize detectors
        self.hand_detector = HandDetector(detection_confidence=0.7)
        self.volume_controller = VolumeController()
        
        # FPS calculation
        self.prev_time = 0
        
        # Volume bar parameters
        self.vol_bar_x = 50
        self.vol_bar_y_top = 150
        self.vol_bar_y_bottom = 400
        self.vol_bar_width = 35
    
    def draw_volume_bar(self, image, volume_percentage):
        """
        Draw volume bar on image
        
        Args:
            image: Input image
            volume_percentage: Volume percentage (0-100)
        """
        # Calculate bar height
        bar_height = np.interp(volume_percentage, [0, 100], 
                              [self.vol_bar_y_bottom, self.vol_bar_y_top])
        
        # Draw volume bar background
        cv2.rectangle(image, (self.vol_bar_x, self.vol_bar_y_top),
                     (self.vol_bar_x + self.vol_bar_width, self.vol_bar_y_bottom),
                     (255, 0, 0), 3)
        
        # Draw volume bar fill
        cv2.rectangle(image, (self.vol_bar_x, int(bar_height)),
                     (self.vol_bar_x + self.vol_bar_width, self.vol_bar_y_bottom),
                     (255, 0, 0), cv2.FILLED)
        
        # Draw volume percentage text
        cv2.putText(image, f'{int(volume_percentage)}%', 
                   (self.vol_bar_x - 10, self.vol_bar_y_bottom + 50),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    def draw_fps(self, image):
        """
        Draw FPS on image
        
        Args:
            image: Input image
        """
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        cv2.putText(image, f'FPS: {int(fps)}', (self.width - 150, 50),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    def run(self):
        """
        Main application loop
        """
        print("Starting Hand Volume Control...")
        print("Use thumb and index finger distance to control volume")
        print("Press 'q' to quit")
        
        while True:
            success, image = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Find hands
            image = self.hand_detector.find_hands(image)
            landmark_list, bbox = self.hand_detector.find_position(image, draw=False)
            
            volume_percentage = 0
            
            if len(landmark_list) != 0:
                # Get distance between thumb and index finger
                distance, image, coords = self.hand_detector.find_distance(4, 8, image)
                
                # Control volume based on distance
                volume_percentage = self.volume_controller.set_volume_by_distance(distance)
                
                # Visual feedback for minimum distance
                if distance < self.volume_controller.min_distance:
                    cv2.circle(image, (coords[4], coords[5]), 15, (0, 255, 0), cv2.FILLED)
            
            # Draw UI elements
            self.draw_volume_bar(image, volume_percentage)
            self.draw_fps(image)
            
            # Display image
            cv2.imshow("Hand Volume Control", image)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")


def main():
    """
    Main function
    """
    try:
        app = HandVolumeControl(camera_id=0)
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error running application: {e}")


if __name__ == "__main__":
    main()