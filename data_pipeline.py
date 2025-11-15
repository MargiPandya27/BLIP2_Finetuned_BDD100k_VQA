import json
import random
import numpy as np
from typing import Dict, Tuple, Optional, List


CLOSE_DISTANCE_THRESHOLD_M = 8.0
MEDIUM_DISTANCE_THRESHOLD_M = 25.0
DEFAULT_IMAGE_HEIGHT = 720 
DEFAULT_IMAGE_WIDTH = 1280 


class VehicleDistanceEstimator:
    """
    Estimates distance to vehicles in front using bounding box information
    without camera calibration parameters, relying on geometric assumptions.
    """

    def __init__(self, image_height: int = DEFAULT_IMAGE_HEIGHT, image_width: int = DEFAULT_IMAGE_WIDTH):
        """
        Initialize the distance estimator with typical BDD100K image dimensions.
        """
        self.image_height = image_height
        self.image_width = image_width

        self.assumed_focal_length = 1000  
        self.assumed_camera_height = 1.2  
        self.avg_vehicle_width = 1.8  
        self.avg_vehicle_height = 1.5  

    def estimate_distance_from_bbox(self, bbox):
        """
        Estimate distance using the Pinhole Camera Model (size-based) and return combined result.

        Args:
            bbox: Dictionary with keys 'x1', 'y1', 'x2', 'y2' (bounding box coordinates)

        Returns:
            Dictionary containing various distance estimates and confidence.
        """

        if isinstance(bbox, dict):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        else:
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_bottom = y2

    
        if bbox_width <= 0 or bbox_height <= 0:
            return {
                'distance_combined': float('inf'),
                'distance_width': float('inf'),
                'distance_height': float('inf'),
                'confidence': 0.0,
                'bbox_info': {'width': 0, 'height': 0, 'aspect_ratio': 0}
            }

        distance_width = (self.avg_vehicle_width * self.assumed_focal_length) / bbox_width

    
        distance_height = (self.avg_vehicle_height * self.assumed_focal_length) / bbox_height

      
        aspect_ratio = bbox_width / bbox_height


        if aspect_ratio > 1.2:  
            width_weight = 0.7
            height_weight = 0.3
        else: 
            width_weight = 0.5
            height_weight = 0.5

        distance_combined = (
            distance_width * width_weight +
            distance_height * height_weight
        )

        normalized_bottom = bbox_bottom / self.image_height

        perspective_factor = 1.0 / max(normalized_bottom, 0.3)
        distance_combined *= (0.7 + 0.3 * perspective_factor)

    
        confidence = self._calculate_confidence(bbox_width, bbox_height, bbox_bottom)

        return {
            'distance_width': distance_width,
            'distance_height': distance_height,
            'distance_combined': distance_combined,
            'confidence': confidence,
            'bbox_info': {
                'width': int(bbox_width),
                'height': int(bbox_height),
                'aspect_ratio': aspect_ratio
            }
        }

    def _calculate_confidence(self, bbox_width, bbox_height, bbox_bottom):

        bbox_area = bbox_width * bbox_height
        image_area = self.image_width * self.image_height
        size_conf = min(bbox_area / (image_area * 0.1), 1.0)

        position_conf = bbox_bottom / self.image_height

        aspect_ratio = bbox_width / bbox_height
        if 0.8 < aspect_ratio < 2.5:
            aspect_conf = 1.0
        else:
            aspect_conf = 0.6

        confidence = (size_conf * 0.4 + position_conf * 0.4 + aspect_conf * 0.2)
        return min(confidence, 1.0)

def find_closest_vehicle(labels):
 
    estimator = VehicleDistanceEstimator()
    image_center = DEFAULT_IMAGE_WIDTH / 2

    
    center_tolerance = DEFAULT_IMAGE_WIDTH * 0.1

    best_vehicle = None
    min_distance = float('inf')

    for label in labels:
        category = label.get('category', '')
        if category not in ['car', 'truck', 'bus', 'train']:
            continue

        bbox = label.get('box2d')
        if not bbox:
            continue


        distance_result = estimator.estimate_distance_from_bbox(bbox)
        distance = distance_result['distance_combined']

        bbox_center_x = (bbox['x1'] + bbox['x2']) / 2
        is_centered = abs(bbox_center_x - image_center) < center_tolerance

  
        if is_centered and distance < min_distance:
            min_distance = distance
            best_vehicle = {
                'label': label,
                'distance': distance,
                'confidence': distance_result['confidence']
            }

    return best_vehicle




def get_traffic_light_color(labels):
    """Extracts the traffic light color from object labels."""
    for obj in labels:
        if obj["category"] == "traffic light":

            return obj["attributes"].get("trafficLightColor", "none")
    return "none"

def get_vehicle_proximity_from_distance(distance_m):
    """Classifies distance (in meters) into 'close', 'medium', or 'far'."""
    if distance_m <= CLOSE_DISTANCE_THRESHOLD_M:
        return "close"
    elif distance_m <= MEDIUM_DISTANCE_THRESHOLD_M:
        return "medium"
    else:
        return "far"


def analyze_front_vehicle_proximity(labels):
    """
    Uses geometric estimation to get the proximity of the vehicle immediately ahead.
    """
    vehicle_data = find_closest_vehicle(labels)

    if vehicle_data is None:
        return {"proximity": "none", "distance_m": float('inf'), "confidence": 1.0}

    distance_m = vehicle_data['distance']
    proximity = get_vehicle_proximity_from_distance(distance_m)
    confidence = vehicle_data['confidence']

    return {"proximity": proximity, "distance_m": distance_m, "confidence": confidence}


def decide_answer(meta, question):
    """Decides the safety answer based on contextual metadata and question type."""
    tl = meta["traffic_light"]
    prox = meta["front_vehicle_proximity"]


    if "traffic condition clear" in question.lower() or "clear ahead" in question.lower():
        if prox in ["far", "none"] and tl in ["none", "green"]:
            return "Yes"
        else:
            return "No"

 
    if tl == "red" or prox == "close":
        return "No"

    if tl == "yellow" or prox == "medium":
        return "slow_down"

    return "Yes"

def generate_rationale(meta, answer, question):
    """Generates a text rationale to explain the decision."""


    if "traffic condition clear" in question.lower() or "clear ahead" in question.lower():
        if answer == "Yes":
            rationale_text = "it is safe to proceed because the vehicle ahead is far away and signals are clear."
            decision_prefix = "Yes"
        else:
            rationale_text = "it is not safe to proceed because a hazard or signal is present."
            decision_prefix = "No"
        return f"{decision_prefix}, {rationale_text}"


    reasons = []
    if meta["traffic_light"] != "none":
        reasons.append(f"the traffic light is {meta['traffic_light']}")

    if meta["front_vehicle_proximity"] != "none" and meta["front_vehicle_distance"] != float('inf'):
        dist_m = meta['front_vehicle_distance']
       
        reasons.append(f"a {meta['front_vehicle_proximity']} vehicle is ahead")

    base = " and ".join(reasons) if reasons else "no immediate hazards were detected"


    if answer == "Yes":
        decision_prefix = "Yes"
        rationale_text = f"it is safe to proceed because {base}."
    elif answer == "slow_down":
      
        if "slow down" in question.lower():
             decision_prefix = "Yes"
             rationale_text = f"the car should slow down because {base}."
        else:
             decision_prefix = "Slow Down"
             rationale_text = f"it is generally safe to proceed, but caution is advised because {base}."
    else: # answer == "No"
        decision_prefix = "No"
        rationale_text = f"it is not safe to proceed because {base}."

    return f"{decision_prefix}, {rationale_text}"

QUESTION_TEMPLATES = [
    "Is it safe to proceed?",
    "Should the car slow down?",
    "Does the car need to stop?",
    "Can the driver continue moving forward safely?",
    "Should the car wait before proceeding?",
    "Is the traffic condition clear ahead?",
]


def build_vqa_dataset(bdd_json_path, output_file):
    """
    Loads BDD data, uses distance estimation, generates VQA entries, and saves.
    """
    try:
        with open(bdd_json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: BDD JSON file not found at {bdd_json_path}. Please set a valid path.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {bdd_json_path}.")
        return

    output = []
 
    for item in data:
        if "labels" not in item or "attributes" not in item:
            continue

        labels = item["labels"]
        attrs = item["attributes"]

        # 1. Use Geometric Distance Logic to determine Proximity
        vehicle_analysis = analyze_front_vehicle_proximity(labels)

        # 2. Compile Metadata for VQA Decision
        meta = {
            "front_vehicle_proximity": vehicle_analysis["proximity"],
            "front_vehicle_distance": vehicle_analysis["distance_m"] if vehicle_analysis["distance_m"] == float('inf') else round(vehicle_analysis["distance_m"], 2),
            "traffic_light": get_traffic_light_color(labels),
            "weather": attrs.get("weather", "unknown"),
            "timeofday": attrs.get("timeofday", "unknown")
        }

        # 3. Generate Question, Answer, and Rationale
        question = random.choice(QUESTION_TEMPLATES)
        answer = decide_answer(meta, question)
        rationale_with_decision = generate_rationale(meta, answer, question)

    
        path = '/kaggle/input/solesensei_bdd100k/bdd100k/bdd100k/images/100k/val/'

        entry = {
            "image": path + item.get("name", "unknown.jpg"),
            "question": question,
            "answer": rationale_with_decision,
            "meta": meta
        }
        output.append(entry)

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✅ Generated {len(output)} samples → {output_file}")




if __name__ == "__main__":
    BDD_JSON_PATH = "bdd100k_labels_images.json"  # Update this path accordingly
    build_vqa_dataset(BDD_JSON_PATH, "bdd100k_risk_vqa.json")

