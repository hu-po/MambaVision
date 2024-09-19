from transformers import AutoModelForImageClassification
import cv2
from PIL import Image
from timm.data.transforms_factory import create_transform
import torch

# Load the pre-trained model
model = AutoModelForImageClassification.from_pretrained("nvidia/MambaVision-T-1K", trust_remote_code=True)
model.cuda().eval()  # set model to eval mode and use GPU

# Camera setup
cap = cv2.VideoCapture(0)  # open video capture on default camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Prepare transform for the model
input_resolution = (3, 224, 224)
transform = create_transform(input_size=input_resolution,
                             is_training=False,
                             mean=model.config.mean,
                             std=model.config.std,
                             crop_mode=model.config.crop_mode,
                             crop_pct=model.config.crop_pct)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the captured frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Transform image for the model input
        inputs = transform(image).unsqueeze(0).cuda()

        # Perform inference
        outputs = model(inputs)
        logits = outputs['logits']
        predicted_class_idx = logits.argmax(-1).item()

        # Display the resulting frame with predicted class
        cv2.imshow('frame', cv2.putText(frame.copy(), model.config.id2label[predicted_class_idx], (50, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA))
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
