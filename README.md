# Dog Pose Detection

This project explores real-time pose estimation on a pet dog (Atlas) using Ultralytics YOLOv8 and OpenCV. The goal is to classify Atlas’s pose—**Sitting**, **Laying Down**, or **Unknown**—in a live video stream, and to document the key challenges and how they were overcome.

---

## Overview

- **Subject:** Atlas, a German Shepherd–Golden Retriever mix  
- **Task:** Predict body keypoints on a dog and infer pose (sitting vs. laying)  
- **Applications:** Automated pet behavior monitoring, training reinforcement, simple activity logging  

---

##  Model Training

- **Base Dataset:** Ultralytics Dog Pose Dataset (12 000+ labeled dog images)  
- **Fine-tuning:**  
  - Ran for 100 epochs on Google Colab  
  - Produced `best.pt`, a checkpoint specialized for dog keypoints  

---

##  Inference Pipeline

1. **Capture**  
   - Read frames from an input video (`Atlas_1.mp4`) or webcam  
2. **Rotate & Preprocess**  
   - (Optional) Rotate each frame 90° clockwise  
   - Remove any manual cropping to preserve the full image  
3. **Run YOLOv8**  
   - Load `best.pt` with `model = YOLO("best.pt")`  
   - Call `model(frame, device=device, conf=0.5)` for keypoint predictions  
4. **Pose Logic**  
   - **Sitting**: front-paw keypoints closer than a distance threshold  
   - **Laying Down**: front and back paws at similar heights  
   - **Unknown**: otherwise  
5. **Annotate & Display**  
   - Draw bounding box & keypoints  
   - Place pose label **below** the box so it’s never clipped  
   - Dynamically scale output to fit a resizable OpenCV window  

---

## ⚠Challenges & Solutions

| Challenge                               | Solution                                              |
|-----------------------------------------|-------------------------------------------------------|
| **Jupyter widget lag & cropping**       | Switched to `cv2.imshow` + computed down-scale factor |
| **Hard-coded window size clipping**     | Used `WINDOW_NORMAL` without forced resize; final down-scale to fit |
| **YOLO constructor error**              | Removed `device=` from `YOLO()` and passed it at inference |
| **Mis-placed / commented code blocks**  | Un-commented and re-indented keypoint & pose logic    |
| **Generic COCO labels (“cat,” “cow”)**  | Filtered for dog class (ID=16) and relabeled to “Atlas” |
| **Pose logic edge cases**               | Added front-paw distance check and vertical threshold |

---

# Summary:

This project focuses on computer vision and real-time pose estimation to detect whether a dog is sitting or laying down. We use Ultralytics YOLOv8 for keypoint detection and OpenCV for capture, annotation, and display. Our subject is Atlas, a German Shepherd–Golden Retriever mix, and by monitoring his posture we aim to enable automated pet behavior tracking and training reinforcement.

The model was fine-tuned for 100 epochs on the Ultralytics Dog-Pose Dataset (12 000+ labeled images) in Google Colab, then tested on static images and on video of Atlas (and even a deer to expose COCO mis-labels). We moved from a laggy Jupyter-widget display to a dynamically scaled OpenCV window (~15 FPS), added frame-skipping, filtered for the dog class, and applied simple rules—front-paw distance < 50 px → “Sitting,” vertical paw alignment < 30 px → “Laying Down,” with labels drawn below the box.

Despite all these steps, I was never able to get a fully reliable real-time demo: the video still lagged or clipped, and the pose logic failed in edge cases. Future work will require GPU-accelerated inference, a true dog-specific training set, and a more robust multithreaded pipeline to achieve the intended pet-monitoring application.****
