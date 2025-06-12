import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from ultralytics import YOLO
from norfair import Detection, Tracker
from norfair.distances import frobenius
import math
from sklearn.cluster import DBSCAN
from collections import defaultdict

class TennisCourtDetector:
    """Advanced tennis court detection using computer vision techniques"""
    
    def __init__(self):
        # Standard tennis court dimensions (in meters)
        self.court_width = 23.77  # meters
        self.court_height = 10.97  # meters
        
        # Court line configuration - key points and line relationships
        self.reference_court = {
            'outer_lines': ['baseline_top', 'baseline_bottom', 'sideline_left', 'sideline_right'],
            'service_lines': ['service_line_top', 'service_line_bottom'],
            'center_lines': ['center_service_line', 'net_line'],
            'key_intersections': [
                'top_left', 'top_right', 'bottom_left', 'bottom_right',
                'service_top_left', 'service_top_right', 
                'service_bottom_left', 'service_bottom_right'
            ]
        }
        
        # Homography matrix for court mapping
        self.homography_matrix = None
        self.court_keypoints = None
        
    def extract_white_pixels(self, frame):
        """Extract white pixels from the frame using adaptive thresholding"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting conditions
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Additional white pixel extraction using HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for white color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine both methods
        combined_mask = cv2.bitwise_or(adaptive_thresh, white_mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        return cleaned_mask
    
    def detect_lines_hough(self, white_mask):
        """Detect lines using Hough Transform and classify them"""
        # Apply edge detection
        edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is None:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle of the line
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            
            # Classify lines based on angle
            if abs(angle) < 15 or abs(angle) > 165:  # Horizontal lines
                horizontal_lines.append((x1, y1, x2, y2))
            elif 75 < abs(angle) < 105:  # Vertical lines
                vertical_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines, vertical_lines
    
    def cluster_lines(self, lines, is_horizontal=True):
        """Cluster similar lines to reduce duplicates"""
        if not lines:
            return []
        
        # Convert lines to feature vectors for clustering
        features = []
        for x1, y1, x2, y2 in lines:
            if is_horizontal:
                # For horizontal lines, use y-coordinate as primary feature
                features.append([min(y1, y2), max(y1, y2), abs(x2 - x1)])
            else:
                # For vertical lines, use x-coordinate as primary feature
                features.append([min(x1, x2), max(x1, x2), abs(y2 - y1)])
        
        features = np.array(features)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=20, min_samples=1).fit(features)
        
        # Group lines by cluster
        clustered_lines = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            clustered_lines[label].append(lines[i])
        
        # Return representative line from each cluster
        result_lines = []
        for cluster_lines in clustered_lines.values():
            # Take the longest line in each cluster
            longest_line = max(cluster_lines, key=lambda l: math.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))
            result_lines.append(longest_line)
        
        return result_lines
    
    def find_line_intersections(self, horizontal_lines, vertical_lines):
        """Find intersections between horizontal and vertical lines"""
        intersections = []
        
        for h_line in horizontal_lines:
            hx1, hy1, hx2, hy2 = h_line
            for v_line in vertical_lines:
                vx1, vy1, vx2, vy2 = v_line
                
                # Calculate intersection point
                intersection = self.line_intersection(
                    (hx1, hy1, hx2, hy2), 
                    (vx1, vy1, vx2, vy2)
                )
                
                if intersection:
                    intersections.append(intersection)
        
        return intersections
    
    def line_intersection(self, line1, line2):
        """Calculate intersection point of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # Lines are parallel
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        # Calculate intersection point
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        
        return (int(ix), int(iy))
    
    def match_court_configuration(self, horizontal_lines, vertical_lines, intersections):
        """Match detected lines with reference court configuration"""
        court_confidence = 0.0
        detected_court_elements = {
            'baselines': [],
            'sidelines': [],
            'service_lines': [],
            'center_line': None,
            'net_line': None
        }
        
        # Sort lines by position
        horizontal_lines.sort(key=lambda l: (l[1] + l[3]) / 2)  # Sort by y-coordinate
        vertical_lines.sort(key=lambda l: (l[0] + l[2]) / 2)    # Sort by x-coordinate
        
        # Identify baselines (top and bottom horizontal lines)
        if len(horizontal_lines) >= 2:
            detected_court_elements['baselines'] = [horizontal_lines[0], horizontal_lines[-1]]
            court_confidence += 0.3
        
        # Identify sidelines (leftmost and rightmost vertical lines)
        if len(vertical_lines) >= 2:
            detected_court_elements['sidelines'] = [vertical_lines[0], vertical_lines[-1]]
            court_confidence += 0.3
        
        # Identify service lines (middle horizontal lines)
        if len(horizontal_lines) >= 4:
            mid_lines = horizontal_lines[1:-1]
            detected_court_elements['service_lines'] = mid_lines[:2]
            court_confidence += 0.2
        
        # Identify center line (middle vertical line)
        if len(vertical_lines) >= 3:
            center_idx = len(vertical_lines) // 2
            detected_court_elements['center_line'] = vertical_lines[center_idx]
            court_confidence += 0.1
        
        # Calculate court keypoints from intersections
        if len(intersections) >= 4:
            self.court_keypoints = self.extract_court_keypoints(intersections)
            court_confidence += 0.1
        
        return detected_court_elements, court_confidence
    
    def extract_court_keypoints(self, intersections):
        """Extract key court points from line intersections"""
        if len(intersections) < 4:
            return None
        
        # Sort intersections to find corners
        intersections.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
        
        # Approximate court corners
        top_points = intersections[:len(intersections)//2]
        bottom_points = intersections[len(intersections)//2:]
        
        top_points.sort(key=lambda p: p[0])  # Sort by x
        bottom_points.sort(key=lambda p: p[0])
        
        keypoints = {
            'top_left': top_points[0] if top_points else None,
            'top_right': top_points[-1] if top_points else None,
            'bottom_left': bottom_points[0] if bottom_points else None,
            'bottom_right': bottom_points[-1] if bottom_points else None
        }
        
        return keypoints
    
    def calculate_homography(self, detected_keypoints):
        """Calculate homography matrix for court transformation"""
        if not detected_keypoints or len([p for p in detected_keypoints.values() if p]) < 4:
            return None
        
        # Real-world court dimensions (normalized)
        court_points_real = np.array([
            [0, 0],           # top_left
            [1, 0],           # top_right
            [0, 1],           # bottom_left
            [1, 1]            # bottom_right
        ], dtype=np.float32)
        
        # Detected court points
        court_points_detected = np.array([
            detected_keypoints['top_left'],
            detected_keypoints['top_right'],
            detected_keypoints['bottom_left'],
            detected_keypoints['bottom_right']
        ], dtype=np.float32)
        
        # Calculate homography
        self.homography_matrix = cv2.getPerspectiveTransform(
            court_points_detected, court_points_real
        )
        
        return self.homography_matrix
    
    def detect_court(self, frame):
        """Main court detection pipeline"""
        # Step 1: Extract white pixels
        white_mask = self.extract_white_pixels(frame)
        
        # Step 2: Detect and classify lines
        horizontal_lines, vertical_lines = self.detect_lines_hough(white_mask)
        
        # Step 3: Cluster similar lines
        horizontal_lines = self.cluster_lines(horizontal_lines, is_horizontal=True)
        vertical_lines = self.cluster_lines(vertical_lines, is_horizontal=False)
        
        # Step 4: Find intersections
        intersections = self.find_line_intersections(horizontal_lines, vertical_lines)
        
        # Step 5: Match with reference court configuration
        court_elements, confidence = self.match_court_configuration(
            horizontal_lines, vertical_lines, intersections
        )
        
        # Step 6: Calculate homography if possible
        if self.court_keypoints:
            self.calculate_homography(self.court_keypoints)
        
        return {
            'white_mask': white_mask,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'intersections': intersections,
            'court_elements': court_elements,
            'confidence': confidence,
            'keypoints': self.court_keypoints
        }

class EnhancedBallDetector:
    """Enhanced ball detection with tracking and trajectory prediction"""
    
    def __init__(self):
        self.ball_history = []
        self.max_history = 30
        self.ball_tracker = Tracker(distance_function=frobenius, distance_threshold=50)
        
    def detect_balls(self, frame, yolo_model):
        """Detect tennis balls with enhanced accuracy"""
        results = yolo_model(frame, verbose=False)[0]
        ball_detections = []
        
        for result in results.boxes:
            cls = int(result.cls[0])
            conf = float(result.conf[0])
            
            if cls == 32 and conf > 0.3:  # Sports ball with higher confidence
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                center = np.array([[int((x1 + x2) / 2), int((y1 + y2) / 2)]])
                
                # Additional validation for ball detection
                if self.validate_ball_detection(frame, (x1, y1, x2, y2)):
                    ball_detections.append(Detection(points=center))
        
        return ball_detections
    
    def validate_ball_detection(self, frame, bbox):
        """Validate ball detection using shape and color analysis"""
        x1, y1, x2, y2 = bbox
        
        # Extract the detected region
        ball_region = frame[y1:y2, x1:x2]
        if ball_region.size == 0:
            return False
        
        # Check if the region is roughly circular
        height, width = ball_region.shape[:2]
        aspect_ratio = width / height if height > 0 else 0
        
        # Tennis balls should be roughly circular (aspect ratio close to 1)
        if not (0.7 < aspect_ratio < 1.3):
            return False
        
        # Check for typical tennis ball colors (yellow/green in tennis)
        hsv_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        
        # Yellow-green color range for tennis balls
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv_region, lower_yellow, upper_yellow)
        
        # Check if a significant portion is yellow/green
        yellow_ratio = np.sum(yellow_mask > 0) / (height * width)
        
        return yellow_ratio > 0.1  # At least 10% should be yellow/green
    
    def predict_trajectory(self, ball_positions):
        """Predict ball trajectory using physics-based modeling"""
        if len(ball_positions) < 3:
            return None
        
        # Extract recent positions
        recent_positions = ball_positions[-5:]
        
        # Simple trajectory prediction using polynomial fitting
        x_coords = [pos[0] for pos in recent_positions]
        y_coords = [pos[1] for pos in recent_positions]
        t_coords = list(range(len(recent_positions)))
        
        if len(set(x_coords)) > 1 and len(set(y_coords)) > 1:
            # Fit polynomial curves
            x_poly = np.polyfit(t_coords, x_coords, min(2, len(t_coords)-1))
            y_poly = np.polyfit(t_coords, y_coords, min(2, len(t_coords)-1))
            
            # Predict next few positions
            future_t = list(range(len(recent_positions), len(recent_positions) + 5))
            future_x = np.polyval(x_poly, future_t)
            future_y = np.polyval(y_poly, future_t)
            
            return list(zip(future_x, future_y))
        
        return None

class TennisAnalysisSystem:
    """Complete tennis analysis system with court detection and ball tracking"""
    
    def __init__(self, video_path, court_image_path, model_path):
        self.video_path = video_path
        self.court_image_path = court_image_path
        self.model_path = model_path
        
        # Initialize components
        self.court_detector = TennisCourtDetector()
        self.ball_detector = EnhancedBallDetector()
        
        # Load resources
        self.load_resources()
        
        # Analysis state
        self.paused = False
        self.frame_idx = 0
        self.court_detected = False
        self.court_info = None
        
    def load_resources(self):
        """Load all required resources"""
        # Load court background
        self.court_background = cv2.imread(self.court_image_path)
        if self.court_background is None:
            raise FileNotFoundError(f"Court background image not found: {self.court_image_path}")
        
        # Load YOLO model
        self.model = YOLO(self.model_path)
        
        # Setup video capture
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.frame_count / self.fps
        
        # Initialize tracker
        self.tracker = Tracker(distance_function=frobenius, distance_threshold=30)
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        self.fig, ((self.ax_original, self.ax_court_detection), 
                  (self.ax_top_view, self.ax_perspective)) = plt.subplots(2, 2, figsize=(16, 12))
        
        plt.subplots_adjust(bottom=0.15)
        
        # Clock display
        clock_ax = self.fig.add_axes([0.4, 0.05, 0.2, 0.05])
        self.clock_display = clock_ax.text(0.5, 0.5, "", fontsize=12, ha='center', va='center')
        clock_ax.axis("off")
        
        # Play/Pause Button
        btn_ax = self.fig.add_axes([0.7, 0.05, 0.1, 0.075])
        self.btn = Button(btn_ax, "Pause")
        self.btn.on_clicked(self.toggle_play)
        
        # Court detection info
        info_ax = self.fig.add_axes([0.1, 0.05, 0.25, 0.05])
        self.info_display = info_ax.text(0.5, 0.5, "", fontsize=10, ha='center', va='center')
        info_ax.axis("off")
    
    def toggle_play(self, event):
        """Toggle play/pause state"""
        self.paused = not self.paused
        self.btn.label.set_text("Play" if self.paused else "Pause")
    
    def draw_court_lines(self, ax, court_info):
        """Draw detected court lines on the visualization"""
        if not court_info:
            return
        
        # Draw horizontal lines
        for line in court_info['horizontal_lines']:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2, alpha=0.7)
        
        # Draw vertical lines
        for line in court_info['vertical_lines']:
            x1, y1, x2, y2 = line
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
        
        # Draw intersections
        for intersection in court_info['intersections']:
            ax.plot(intersection[0], intersection[1], 'go', markersize=8)
        
        # Draw keypoints if available
        if court_info['keypoints']:
            for name, point in court_info['keypoints'].items():
                if point:
                    ax.plot(point[0], point[1], 'ro', markersize=10)
                    ax.annotate(name, point, xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, color='red')
    
    def run_analysis(self):
        """Main analysis loop"""
        self.setup_visualization()
        plt.ion()
        
        ball_positions = []
        
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect court in first few frames or periodically
                if self.frame_idx < 10 or self.frame_idx % 30 == 0:
                    self.court_info = self.court_detector.detect_court(frame)
                    if self.court_info['confidence'] > 0.5:
                        self.court_detected = True
                
                # Detect players and balls
                results = self.model(frame, verbose=False)[0]
                detections = []
                ball_detections = []
                
                for result in results.boxes:
                    cls = int(result.cls[0])
                    conf = float(result.conf[0])
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    center = np.array([[int((x1 + x2) / 2), int((y1 + y2) / 2)]])
                    
                    if cls == 0 and conf > 0.5:  # Person
                        detections.append(Detection(points=center))
                    elif cls == 32 and conf > 0.3:  # Ball
                        if self.ball_detector.validate_ball_detection(frame, (x1, y1, x2, y2)):
                            ball_detections.append(Detection(points=center))
                            ball_positions.append((center[0][0], center[0][1]))
                
                # Track objects
                tracked_objects = self.tracker.update(detections=detections)
                tracked_balls = self.ball_detector.ball_tracker.update(detections=ball_detections)
                
                # Clear and redraw all subplots
                self.ax_original.clear()
                self.ax_court_detection.clear()
                self.ax_top_view.clear()
                self.ax_perspective.clear()
                
                # Original frame with detections
                self.ax_original.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.ax_original.set_title("Original Frame with Detections")
                self.ax_original.axis("off")
                
                # Court detection visualization
                if self.court_info:
                    self.ax_court_detection.imshow(self.court_info['white_mask'], cmap='gray')
                    self.draw_court_lines(self.ax_court_detection, self.court_info)
                    self.ax_court_detection.set_title("Court Detection")
                    self.ax_court_detection.axis("off")
                
                # Top view (court mapping)
                self.ax_top_view.imshow(self.court_background)
                self.ax_top_view.set_title("Top View")
                self.ax_top_view.axis("off")
                
                # Perspective view
                self.ax_perspective.imshow(self.court_background)
                self.ax_perspective.set_title("Perspective View")
                self.ax_perspective.axis("off")
                
                # Draw tracked objects
                for obj in tracked_objects:
                    x, y = obj.estimate[0]
                    # Draw on original frame
                    self.ax_original.add_patch(Circle((x, y), 15, color='blue', fill=True, alpha=0.7))
                    # Map to court views
                    self.ax_top_view.add_patch(Circle((x * 0.8, y * 0.9), 10, color='blue', fill=True))
                    self.ax_perspective.add_patch(Circle((x * 0.9, y * 1.1), 10, color='blue', fill=True))
                
                # Draw tracked balls
                for ball in tracked_balls:
                    x, y = ball.estimate[0]
                    self.ax_original.add_patch(Circle((x, y), 10, color='yellow', fill=True, alpha=0.8))
                    self.ax_top_view.add_patch(Circle((x * 0.8, y * 0.9), 8, color='yellow', fill=True))
                    self.ax_perspective.add_patch(Circle((x * 0.9, y * 1.1), 8, color='yellow', fill=True))
                
                # Draw ball trajectory
                if len(ball_positions) > 1:
                    trajectory_x = [pos[0] for pos in ball_positions[-10:]]  # Last 10 positions
                    trajectory_y = [pos[1] for pos in ball_positions[-10:]]
                    self.ax_original.plot(trajectory_x, trajectory_y, 'r--', alpha=0.6, linewidth=2)
                
                # Update displays
                current_sec = self.frame_idx / self.fps
                self.clock_display.set_text(f"Time: {current_sec:.1f} / {self.duration_sec:.1f} sec")
                
                if self.court_detected and self.court_info:
                    confidence = self.court_info['confidence']
                    self.info_display.set_text(f"Court Detected: {confidence:.2f} confidence")
                else:
                    self.info_display.set_text("Detecting court...")
                
                self.frame_idx += 1
            
            plt.pause(0.01)
        
        self.cap.release()
        plt.ioff()
        plt.show()

# Usage Example
if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "video.mp4"
    COURT_IMAGE_PATH = "court_background.png"
    MODEL_PATH = "yolov8s.pt"
    
    # Create and run the tennis analysis system
    try:
        tennis_system = TennisAnalysisSystem(VIDEO_PATH, COURT_IMAGE_PATH, MODEL_PATH)
        tennis_system.run_analysis()
    except Exception as e:
        print(f"Error running tennis analysis: {e}")
        print("Please ensure all required files are present:")
        print(f"- Video: {VIDEO_PATH}")
        print(f"- Court background: {COURT_IMAGE_PATH}")
        print(f"- YOLO model: {MODEL_PATH}")