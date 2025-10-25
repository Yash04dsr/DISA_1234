import cv2
from ultralytics import YOLO

# --- Load the Pre-trained AI Model ---
# This will download the model automatically on the first run.
# 'yolov8n-seg.pt' is a small and fast segmentation model.
model = YOLO('yolov8n-seg.pt')

# --- Load Your Image ---
image_path = '/Users/yash/Desktop/Disa Images/DISA Code/Pendent Drop Analysis/photogoutte3.jpg'  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # --- Perform Inference ---
    # The model will predict objects, their bounding boxes, and their masks.
    results = model(frame)

    # --- Process and Visualize the Results ---
    if results[0].masks is not None:
        # Plot the segmentation results on the image
        result_plot = results[0].plot()

        # Extract the contour from the mask
        # Note: This part assumes the largest detected mask is the drop.
        masks = results[0].masks.xy
        if len(masks) > 0:
            # Find the largest mask by contour area
            largest_contour = max(masks, key=cv2.contourArea)

            # Create a clean image to draw just the final contour
            contour_image = frame.copy()
            cv2.drawContours(contour_image, [largest_contour.astype(int)], -1, (0, 255, 0), 2)

            print("AI model successfully detected a contour.")

            # Display the images
            cv2.imshow('AI Segmentation Result', result_plot)
            cv2.imshow('Extracted Contour', contour_image)

            # Save the results
            cv2.imwrite('result_ai_segmentation.jpg', result_plot)
            cv2.imwrite('result_ai_contour.jpg', contour_image)

            print("Results saved. Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("AI model ran, but did not find any contours in this image.")

    else:
        print("AI model ran, but did not detect any objects with masks.")