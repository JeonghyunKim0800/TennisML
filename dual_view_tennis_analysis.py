import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from norfair.distances import frobenius
import math
from collections import deque
from scipy.spatial.distance import cdist
import time

# === CONFIGURATION ===
VIDEO_PATH = "video.mp4"
COURT_IMAGE_PATH = "court_background.png"
MODEL_PATH = "yolov8s.pt"

class SuperAccurateBallDetector:
    """Ultra-precise tennis ball detection focusing on yellow/green small circular objects"""
    
    def __init__(self):
        self.ball_history = deque(maxlen=20)
        self.last_ball_position = None
        self.frame_count = 0
        
    def extract_tennis_ball_regions(self, frame):
        """Extract regions that could contain tennis balls with very specific criteria"""
        height, width = frame.shape[:2]
        
        # Convert to multiple color spaces for better ball detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Very specific tennis ball color ranges
        # Tennis balls are typically yellow-green with high saturation
        tennis_ball_masks = []
        
        # HSV ranges for tennis ball yellow-green
        hsv_ranges = [
            ([22, 80, 80], [35, 255, 255]),    # Standard tennis ball yellow
            ([18, 60, 100], [40, 255, 255]),  # Broader yellow range
            ([15, 40, 120], [45, 255, 255]),  # Even broader for different lighting
        ]
        
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for lower, upper in hsv_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Add bright white objects (balls can appear white under strong lighting)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Combine masks
        final_mask = cv2.bitwise_or(combined_mask, bright_mask)
        
        # Clean up the mask - remove noise but preserve small circular objects
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_small)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_small)
        
        return final_mask
    
    def find_circular_objects(self, mask, original_frame):
        """Find circular objects that match tennis ball characteristics"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Tennis ball size filtering (much stricter)
            if area < 15 or area > 800:  # Very specific size range
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio must be close to 1 (circular)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.7 < aspect_ratio < 1.3):
                continue
            
            # Size consistency check
            diameter = (w + h) / 2
            if diameter < 8 or diameter > 35:
                continue
            
            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:  # Must be reasonably circular
                continue
            
            # Check if the object is actually round using HoughCircles
            center_x, center_y = x + w//2, y + h//2
            
            # Extract small region around the candidate
            margin = max(w, h) // 2 + 5
            x1 = max(0, center_x - margin)
            y1 = max(0, center_y - margin)
            x2 = min(original_frame.shape[1], center_x + margin)
            y2 = min(original_frame.shape[0], center_y + margin)
            
            region = original_frame[y1:y2, x1:x2]
            if region.size == 0:
                continue
            
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # Use HoughCircles to detect circular shapes
            circles = cv2.HoughCircles(
                gray_region,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=max(w, h),
                param1=30,  # Lower threshold for edge detection
                param2=15,  # Lower threshold for center detection
                minRadius=max(1, min(w, h) // 3),
                maxRadius=max(w, h)
            )
            
            circle_detected = circles is not None and len(circles[0]) > 0
            
            # Calculate confidence score
            confidence = 0.0
            confidence += circularity * 0.4  # Shape similarity
            confidence += (1.0 - abs(aspect_ratio - 1.0)) * 0.3  # Aspect ratio
            confidence += min(1.0, area / 100) * 0.2  # Size appropriateness
            confidence += (0.3 if circle_detected else 0.0)  # Circle detection
            
            ball_candidates.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'area': area,
                'confidence': confidence,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'diameter': diameter
            })
        
        return ball_candidates
    
    def validate_ball_motion(self, candidates):
        """Validate candidates based on motion consistency"""
        if not candidates:
            return []
        
        if self.last_ball_position is None:
            # First detection - accept the most confident candidate
            best_candidate = max(candidates, key=lambda x: x['confidence'])
            self.last_ball_position = best_candidate['center']
            return [best_candidate]
        
        # Find candidates close to last known position
        valid_candidates = []
        max_movement = 60  # Maximum pixels ball can move between frames
        
        for candidate in candidates:
            distance = np.sqrt(
                (candidate['center'][0] - self.last_ball_position[0])**2 + 
                (candidate['center'][1] - self.last_ball_position[1])**2
            )
            
            if distance <= max_movement:
                # Boost confidence for consistent motion
                candidate['confidence'] += 0.2 * (1.0 - distance / max_movement)
                valid_candidates.append(candidate)
        
        # If no candidates near last position, accept new detections with high confidence
        if not valid_candidates:
            high_confidence = [c for c in candidates if c['confidence'] > 0.7]
            if high_confidence:
                valid_candidates = high_confidence
        
        return valid_candidates
    
    def detect_tennis_ball(self, frame):
        """Main ball detection method"""
        self.frame_count += 1
        
        # Extract potential ball regions
        ball_mask = self.extract_tennis_ball_regions(frame)
        
        # Find circular objects
        candidates = self.find_circular_objects(ball_mask, frame)
        
        # Validate based on motion and history
        valid_balls = self.validate_ball_motion(candidates)
        
        # Keep only the best candidate per frame
        if valid_balls:
            best_ball = max(valid_balls, key=lambda x: x['confidence'])
            if best_ball['confidence'] > 0.5:  # Minimum confidence threshold
                self.last_ball_position = best_ball['center']
                self.ball_history.append(best_ball)
                return [best_ball], ball_mask
        
        return [], ball_mask

class PreciseCourtDetector:
    """Improved court detection focusing on actual court lines"""
    
    def __init__(self):
        self.stable_lines = {'horizontal': [], 'vertical': []}
        self.detection_history = deque(maxlen=5)
        
    def preprocess_for_court_lines(self, frame):
        """Enhanced preprocessing specifically for tennis court lines"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use multiple methods to detect white lines
        # Method 1: Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 21, 2
        )
        
        # Method 2: HSV white detection (more robust)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # More precise white line detection
        lower_white = np.array([0, 0, 180])  # Lower brightness threshold
        upper_white = np.array([180, 40, 255])  # Allow slight color variation
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Method 3: High-pass filtering for line detection
        kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
        high_pass = cv2.filter2D(gray, -1, kernel)
        high_pass_thresh = cv2.threshold(high_pass, 50, 255, cv2.THRESH_BINARY)[1]
        
        # Combine all methods
        combined = cv2.bitwise_or(adaptive, white_mask)
        combined = cv2.bitwise_or(combined, high_pass_thresh)
        
        # Clean up with more precise morphological operations
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_small)
        
        # Connect nearby line segments
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_line)
        
        kernel_line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_line_v)
        
        return combined
    
    def detect_court_lines_improved(self, processed_mask):
        """Improved line detection with better filtering"""
        # Edge detection
        edges = cv2.Canny(processed_mask, 30, 100, apertureSize=3)
        
        # HoughLinesP with more restrictive parameters
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=60,    # Higher threshold
            minLineLength=80,  # Longer minimum length
            maxLineGap=20     # Smaller gaps allowed
        )
        
        if lines is None:
            return [], []
        
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle and length
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Only keep longer lines (court lines are typically long)
            if length < 60:
                continue
            
            # Classify lines with stricter angle criteria
            if abs(angle) <= 8 or abs(angle) >= 172:  # Nearly horizontal
                horizontal_lines.append((x1, y1, x2, y2, length, angle))
            elif 82 <= abs(angle) <= 98:  # Nearly vertical
                vertical_lines.append((x1, y1, x2, y2, length, angle))
        
        return horizontal_lines, vertical_lines
    
    def filter_court_lines(self, horizontal_lines, vertical_lines, frame_shape):
        """Filter lines to keep only actual court lines"""
        height, width = frame_shape[:2]
        
        # Filter horizontal lines
        filtered_horizontal = []
        horizontal_lines.sort(key=lambda x: x[4], reverse=True)  # Sort by length
        
        for line in horizontal_lines[:8]:  # Keep top 8 longest horizontal lines
            x1, y1, x2, y2, length, angle = line
            
            # Must span significant portion of the width
            line_width = abs(x2 - x1)
            if line_width < width * 0.15:  # At least 15% of frame width
                continue
                
            # Filter out lines too close to frame edges (likely not court lines)
            avg_y = (y1 + y2) / 2
            if avg_y < height * 0.1 or avg_y > height * 0.9:
                continue
                
            filtered_horizontal.append((x1, y1, x2, y2))
        
        # Filter vertical lines
        filtered_vertical = []
        vertical_lines.sort(key=lambda x: x[4], reverse=True)  # Sort by length
        
        for line in vertical_lines[:6]:  # Keep top 6 longest vertical lines
            x1, y1, x2, y2, length, angle = line
            
            # Must span significant portion of the height
            line_height = abs(y2 - y1)
            if line_height < height * 0.15:  # At least 15% of frame height
                continue
                
            # Filter out lines too close to frame edges
            avg_x = (x1 + x2) / 2
            if avg_x < width * 0.05 or avg_x > width * 0.95:
                continue
                
            filtered_vertical.append((x1, y1, x2, y2))
        
        return filtered_horizontal, filtered_vertical
    
    def detect_court(self, frame):
        """Main court detection method"""
        # Preprocess frame
        processed_mask = self.preprocess_for_court_lines(frame)
        
        # Detect lines
        h_lines, v_lines = self.detect_court_lines_improved(processed_mask)
        
        # Filter to keep only court lines
        h_lines, v_lines = self.filter_court_lines(h_lines, v_lines, frame.shape)
        
        # Calculate confidence based on detected lines
        confidence = 0.0
        
        # Need at least some lines for a tennis court
        if len(h_lines) >= 2:  # At least 2 horizontal lines (baselines)
            confidence += 0.4
        if len(v_lines) >= 2:  # At least 2 vertical lines (sidelines)
            confidence += 0.4
        if len(h_lines) >= 3:  # Service line
            confidence += 0.1
        if len(v_lines) >= 3:  # Center service line
            confidence += 0.1
        
        court_info = {
            'horizontal_lines': h_lines,
            'vertical_lines': v_lines,
            'white_mask': processed_mask,
            'confidence': min(1.0, confidence)
        }
        
        # Store in history for stability
        self.detection_history.append(court_info)
        
        return court_info

class ImprovedTennisAnalyzer:
    """Main analyzer with improved ball and court detection"""
    
    def __init__(self, video_path, court_image_path, model_path):
        self.video_path = video_path
        self.court_image_path = court_image_path
        self.model_path = model_path
        
        # Initialize improved detectors
        self.ball_detector = SuperAccurateBallDetector()
        self.court_detector = PreciseCourtDetector()
        self.model = YOLO(model_path)
        self.tracker = Tracker(distance_function=frobenius, distance_threshold=50)
        
        # Load court background
        self.court_background = cv2.imread(court_image_path)
        if self.court_background is None:
            print(f"Warning: {court_image_path} not found. Using blank background.")
            self.court_background = np.ones((400, 600, 3), dtype=np.uint8) * 50
        
        # Video setup
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_sec = self.frame_count / self.fps
        
        # Analysis state
        self.frame_idx = 0
        self.paused = False
        self.ball_detections = []
        self.detection_stats = {'total_frames': 0, 'balls_detected': 0, 'false_positives': 0}
        
        print(f"Video loaded: {self.frame_count} frames, {self.fps:.1f} FPS, {self.duration_sec:.1f} seconds")
        print("Improved accuracy mode: Better ball detection and cleaner court detection")
        
    def analyze_frame(self, frame):
        """Analyze single frame with improved methods"""
        analysis_start = time.time()
        
        # 1. Improved ball detection
        detected_balls, ball_mask = self.ball_detector.detect_tennis_ball(frame)
        
        # 2. Improved court detection (every 5 frames)
        court_info = None
        if self.frame_idx % 5 == 0:
            court_info = self.court_detector.detect_court(frame)
        
        # 3. YOLO for player detection only
        yolo_results = self.model(frame, verbose=False)[0]
        
        # 4. Prepare detections for tracking
        detections = []
        
        # Add player detections from YOLO
        for result in yolo_results.boxes:
            cls = int(result.cls[0])
            conf = float(result.conf[0])
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            center = np.array([[int((x1 + x2) / 2), int((y1 + y2) / 2)]])
            
            if cls == 0 and conf > 0.6:  # Person with higher confidence
                detections.append(Detection(points=center))
        
        # Add ball detections
        for ball in detected_balls:
            center = np.array([ball['center']])
            detections.append(Detection(points=center))
            
            # Store ball detection
            self.ball_detections.append({
                'frame': self.frame_idx,
                'position': ball['center'],
                'confidence': ball['confidence'],
                'timestamp': self.frame_idx / self.fps,
                'details': ball
            })
        
        # 5. Update tracking
        tracked_objects = self.tracker.update(detections=detections)
        
        # Update statistics
        self.detection_stats['total_frames'] += 1
        if detected_balls:
            self.detection_stats['balls_detected'] += 1
        
        analysis_time = time.time() - analysis_start
        
        return {
            'frame': frame,
            'court_info': court_info,
            'detected_balls': detected_balls,
            'ball_mask': ball_mask,
            'tracked_objects': tracked_objects,
            'analysis_time': analysis_time,
            'yolo_results': yolo_results
        }
    
    def run_analysis(self):
        """Run the main analysis loop"""
        # Setup matplotlib
        fig, ((ax_original, ax_ball_mask), (ax_court, ax_trajectory)) = plt.subplots(2, 2, figsize=(16, 12))
        plt.subplots_adjust(bottom=0.15)
        
        # Control elements
        def toggle_play(event):
            self.paused = not self.paused
            btn.label.set_text("Play" if self.paused else "Pause")
        
        btn_ax = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        btn = Button(btn_ax, "Pause")
        btn.on_clicked(toggle_play)
        
        # Info display
        info_ax = fig.add_axes([0.1, 0.05, 0.5, 0.05])
        info_text = info_ax.text(0.5, 0.5, "", fontsize=10, ha='center', va='center')
        info_ax.axis("off")
        
        plt.ion()
        
        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video reached")
                        break
                    
                    # Analyze frame
                    results = self.analyze_frame(frame)
                    
                    # Clear plots
                    for ax in [ax_original, ax_ball_mask, ax_court, ax_trajectory]:
                        ax.clear()
                    
                    # 1. Original frame with improved detections
                    ax_original.imshow(cv2.cvtColor(results['frame'], cv2.COLOR_BGR2RGB))
                    ax_original.set_title(f"Frame {self.frame_idx} - Improved Detection")
                    
                    # Draw detected balls with confidence
                    for ball in results['detected_balls']:
                        x, y = ball['center']
                        confidence = ball['confidence']
                        
                        # Green circle for detected ball
                        circle = Circle((x, y), 12, color='lime', fill=False, linewidth=3)
                        ax_original.add_patch(circle)
                        
                        # Confidence text
                        ax_original.text(x, y-20, f'Ball: {confidence:.2f}', 
                                       ha='center', color='lime', fontweight='bold', fontsize=10)
                    
                    # Draw players (blue circles)
                    for result in results['yolo_results'].boxes:
                        cls = int(result.cls[0])
                        conf = float(result.conf[0])
                        
                        if cls == 0 and conf > 0.6:  # Person
                            x1, y1, x2, y2 = map(int, result.xyxy[0])
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            circle = Circle((center_x, center_y), 15, color='blue', fill=False, linewidth=2)
                            ax_original.add_patch(circle)
                    
                    ax_original.axis("off")
                    
                    # 2. Ball detection mask
                    if 'ball_mask' in results:
                        ax_ball_mask.imshow(results['ball_mask'], cmap='hot')
                        ax_ball_mask.set_title("Ball Detection Mask")
                    ax_ball_mask.axis("off")
                    
                    # 3. Improved court detection
                    if results['court_info']:
                        ax_court.imshow(results['court_info']['white_mask'], cmap='gray')
                        
                        # Draw only the filtered court lines
                        for line in results['court_info']['horizontal_lines']:
                            x1, y1, x2, y2 = line
                            ax_court.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=0.8)
                        
                        for line in results['court_info']['vertical_lines']:
                            x1, y1, x2, y2 = line
                            ax_court.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.8)
                        
                        ax_court.set_title(f"Clean Court Lines (Conf: {results['court_info']['confidence']:.2f})")
                    else:
                        ax_court.imshow(np.zeros_like(frame[:,:,0]), cmap='gray')
                        ax_court.set_title("Court Detection - Processing...")
                    
                    ax_court.axis("off")
                    
                    # 4. Ball trajectory
                    if len(self.ball_detections) > 1:
                        recent_detections = self.ball_detections[-30:]  # Last 30 detections
                        x_coords = [det['position'][0] for det in recent_detections]
                        y_coords = [det['position'][1] for det in recent_detections]
                        confidences = [det['confidence'] for det in recent_detections]
                        
                        # Plot trajectory
                        scatter = ax_trajectory.scatter(x_coords, y_coords, c=confidences, 
                                                     cmap='RdYlGn', s=50, alpha=0.8)
                        ax_trajectory.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2)
                        
                        ax_trajectory.set_title("Ball Trajectory (High Confidence)")
                        ax_trajectory.set_xlim(0, frame.shape[1])
                        ax_trajectory.set_ylim(frame.shape[0], 0)
                        
                        # Add colorbar
                        cbar = plt.colorbar(scatter, ax=ax_trajectory)
                        cbar.set_label('Confidence')
                    else:
                        ax_trajectory.set_title("Ball Trajectory - Detecting...")
                        ax_trajectory.set_xlim(0, frame.shape[1])
                        ax_trajectory.set_ylim(frame.shape[0], 0)
                    
                    # Update info
                    detection_rate = (self.detection_stats['balls_detected'] / 
                                    max(1, self.detection_stats['total_frames']))
                    
                    info_text.set_text(
                        f"Time: {self.frame_idx/self.fps:.1f}s | "
                        f"Balls this frame: {len(results['detected_balls'])} | "
                        f"Total detections: {len(self.ball_detections)} | "
                        f"Detection rate: {detection_rate:.1%}"
                    )
                    
                    self.frame_idx += 1
                
                plt.pause(0.001)  # Faster refresh
                
                if plt.get_fignums() == []:
                    break
                    
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cap.release()
            plt.ioff()
            if plt.get_fignums():
                plt.show()
            
            # Print final statistics
            detection_rate = (self.detection_stats['balls_detected'] / 
                            max(1, self.detection_stats['total_frames']))
            
            print(f"\n=== Analysis Complete ===")
            print(f"Total frames processed: {self.detection_stats['total_frames']}")
            print(f"Frames with ball detections: {self.detection_stats['balls_detected']}")
            print(f"Detection rate: {detection_rate:.1%}")
            print(f"Total ball trajectory points: {len(self.ball_detections)}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    analyzer = ImprovedTennisAnalyzer(
        video_path=VIDEO_PATH,
        court_image_path=COURT_IMAGE_PATH,
        model_path=MODEL_PATH
    )
    
    analyzer.run_analysis()