"""
Posture Analyzer module.
Analyzes upper-body posture using pose landmarks.
"""

import numpy as np
from typing import Optional, Dict, Any
from enum import Enum
import config


class PostureType(Enum):
    """Enum for posture types."""
    UPRIGHT = "upright"
    SLOUCHING = "slouching"
    LEANING_FORWARD = "leaning_forward"
    LEANING_BACKWARD = "leaning_backward"
    HEAD_TILT_LEFT = "head_tilt_left"
    HEAD_TILT_RIGHT = "head_tilt_right"
    UNKNOWN = "unknown"


class PostureQuality(Enum):
    """Enum for posture quality levels."""
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNKNOWN = "unknown"


class PostureAnalyzer:
    """
    Analyzes posture using pose landmarks.
    Detects slouching, leaning, and head tilt.
    """
    
    def __init__(
        self,
        upright_threshold: float = config.POSTURE_UPRIGHT_THRESHOLD,
        slouch_threshold: float = config.POSTURE_SLOUCH_THRESHOLD,
        lean_forward_threshold: float = config.POSTURE_LEAN_FORWARD_THRESHOLD,
        lean_back_threshold: float = config.POSTURE_LEAN_BACK_THRESHOLD,
        head_tilt_threshold: float = config.POSTURE_HEAD_TILT_THRESHOLD
    ):
        """
        Initialize the posture analyzer.
        
        Args:
            upright_threshold: Max shoulder angle for upright posture.
            slouch_threshold: Shoulder drop indicating slouch.
            lean_forward_threshold: Nose forward threshold.
            lean_back_threshold: Nose backward threshold.
            head_tilt_threshold: Head tilt angle threshold.
        """
        self.upright_threshold = upright_threshold
        self.slouch_threshold = slouch_threshold
        self.lean_forward_threshold = lean_forward_threshold
        self.lean_back_threshold = lean_back_threshold
        self.head_tilt_threshold = head_tilt_threshold
        
        # Baseline calibration (set during first few frames)
        self.baseline_shoulder_y: Optional[float] = None
        self.baseline_nose_z: Optional[float] = None
        self.calibration_frames = 0
        self.calibration_complete = False
        
        # Metrics tracking
        self.total_frames = 0
        self.good_posture_frames = 0
        self.acceptable_posture_frames = 0
        self.poor_posture_frames = 0
        
        # Smoothing
        self.posture_history = []
        self.history_size = 10
    
    def analyze(self, pose_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze posture from pose detection results.
        
        Args:
            pose_result: Output from PoseDetector.detect()
            
        Returns:
            Dictionary containing posture analysis results.
        """
        analysis_result = {
            "posture_detected": False,
            "posture_type": PostureType.UNKNOWN,
            "posture_quality": PostureQuality.UNKNOWN,
            "issues": [],
            "shoulder_angle": None,
            "posture_score": 0.5,  # 0-1 score
            "calibrated": self.calibration_complete
        }
        
        if not pose_result.get("pose_detected", False):
            return analysis_result
        
        analysis_result["posture_detected"] = True
        landmarks = pose_result.get("landmarks", {})
        
        # Get key landmarks
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        nose = landmarks.get("nose")
        left_ear = landmarks.get("left_ear")
        right_ear = landmarks.get("right_ear")
        
        if not all([left_shoulder, right_shoulder, nose]):
            return analysis_result
        
        # Calculate metrics
        shoulder_angle = self._calculate_shoulder_tilt(left_shoulder, right_shoulder)
        analysis_result["shoulder_angle"] = shoulder_angle
        
        # Calibration during first few frames
        if not self.calibration_complete:
            self._calibrate(left_shoulder, right_shoulder, nose)
            analysis_result["posture_quality"] = PostureQuality.UNKNOWN
            return analysis_result
        
        issues = []
        posture_type = PostureType.UPRIGHT
        
        # Check shoulder tilt (uneven shoulders)
        if abs(shoulder_angle) > self.upright_threshold:
            if shoulder_angle > 0:
                issues.append("right_shoulder_low")
            else:
                issues.append("left_shoulder_low")
        
        # Check for slouching (shoulders dropped from baseline)
        avg_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        if self.baseline_shoulder_y is not None:
            shoulder_drop = avg_shoulder_y - self.baseline_shoulder_y
            if shoulder_drop > self.slouch_threshold / 100:  # Convert to normalized
                posture_type = PostureType.SLOUCHING
                issues.append("slouching")
        
        # Check for head tilt using ears
        if left_ear and right_ear:
            head_tilt = self._calculate_head_tilt(left_ear, right_ear)
            if abs(head_tilt) > self.head_tilt_threshold:
                if head_tilt > 0:
                    posture_type = PostureType.HEAD_TILT_RIGHT
                    issues.append("head_tilted_right")
                else:
                    posture_type = PostureType.HEAD_TILT_LEFT
                    issues.append("head_tilted_left")
        
        # Check for forward/backward lean using nose z-depth
        if self.baseline_nose_z is not None and nose.get("z") is not None:
            nose_z_diff = nose["z"] - self.baseline_nose_z
            if nose_z_diff < -self.lean_forward_threshold / 100:
                posture_type = PostureType.LEANING_FORWARD
                issues.append("leaning_forward")
            elif nose_z_diff > self.lean_back_threshold / 100:
                posture_type = PostureType.LEANING_BACKWARD
                issues.append("leaning_backward")
        
        # Determine quality based on issues
        if len(issues) == 0:
            posture_quality = PostureQuality.GOOD
            self.good_posture_frames += 1
        elif len(issues) == 1:
            posture_quality = PostureQuality.ACCEPTABLE
            self.acceptable_posture_frames += 1
        else:
            posture_quality = PostureQuality.POOR
            self.poor_posture_frames += 1
        
        self.total_frames += 1
        
        # Calculate posture score (0-1)
        posture_score = self._calculate_posture_score(issues, shoulder_angle)
        
        # Smooth the score
        self.posture_history.append(posture_score)
        if len(self.posture_history) > self.history_size:
            self.posture_history.pop(0)
        smoothed_score = np.mean(self.posture_history)
        
        analysis_result["posture_type"] = posture_type
        analysis_result["posture_quality"] = posture_quality
        analysis_result["issues"] = issues
        analysis_result["posture_score"] = smoothed_score
        
        return analysis_result
    
    def _calibrate(
        self, 
        left_shoulder: Dict, 
        right_shoulder: Dict, 
        nose: Dict
    ):
        """Calibrate baseline posture from initial frames."""
        self.calibration_frames += 1
        
        if self.calibration_frames < 30:
            # Accumulate baseline values
            avg_shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            
            if self.baseline_shoulder_y is None:
                self.baseline_shoulder_y = avg_shoulder_y
            else:
                # Running average
                self.baseline_shoulder_y = (
                    self.baseline_shoulder_y * 0.9 + avg_shoulder_y * 0.1
                )
            
            if nose.get("z") is not None:
                if self.baseline_nose_z is None:
                    self.baseline_nose_z = nose["z"]
                else:
                    self.baseline_nose_z = self.baseline_nose_z * 0.9 + nose["z"] * 0.1
        else:
            self.calibration_complete = True
    
    def _calculate_shoulder_tilt(
        self, 
        left_shoulder: Dict, 
        right_shoulder: Dict
    ) -> float:
        """Calculate shoulder tilt angle in degrees."""
        dx = right_shoulder["x"] - left_shoulder["x"]
        dy = right_shoulder["y"] - left_shoulder["y"]
        
        angle_rad = np.arctan2(dy, dx)
        return np.degrees(angle_rad)
    
    def _calculate_head_tilt(self, left_ear: Dict, right_ear: Dict) -> float:
        """Calculate head tilt angle in degrees."""
        dx = right_ear["x"] - left_ear["x"]
        dy = right_ear["y"] - left_ear["y"]
        
        angle_rad = np.arctan2(dy, dx)
        return np.degrees(angle_rad)
    
    def _calculate_posture_score(self, issues: list, shoulder_angle: float) -> float:
        """Calculate a 0-1 posture score based on detected issues."""
        base_score = 1.0
        
        # Deduct for each issue
        issue_penalties = {
            "slouching": 0.25,
            "leaning_forward": 0.2,
            "leaning_backward": 0.15,
            "head_tilted_left": 0.15,
            "head_tilted_right": 0.15,
            "left_shoulder_low": 0.1,
            "right_shoulder_low": 0.1
        }
        
        for issue in issues:
            base_score -= issue_penalties.get(issue, 0.1)
        
        # Additional penalty based on shoulder angle magnitude
        angle_penalty = min(0.2, abs(shoulder_angle) / 90 * 0.2)
        base_score -= angle_penalty
        
        return max(0.0, min(1.0, base_score))
    
    def get_quality_percentage(self) -> Dict[str, float]:
        """Get percentage breakdown of posture quality over time."""
        if self.total_frames == 0:
            return {"good": 0, "acceptable": 0, "poor": 0}
        
        return {
            "good": (self.good_posture_frames / self.total_frames) * 100,
            "acceptable": (self.acceptable_posture_frames / self.total_frames) * 100,
            "poor": (self.poor_posture_frames / self.total_frames) * 100
        }
    
    def reset(self):
        """Reset analyzer state."""
        self.baseline_shoulder_y = None
        self.baseline_nose_z = None
        self.calibration_frames = 0
        self.calibration_complete = False
        self.total_frames = 0
        self.good_posture_frames = 0
        self.acceptable_posture_frames = 0
        self.poor_posture_frames = 0
        self.posture_history = []
