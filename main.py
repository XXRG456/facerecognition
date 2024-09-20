from project import model, Train, Val
from project import PairsDataLoader, DataLoader
from project import anchor_images, plot_anchor_images
from project import detect_face, test_image
from project import create_prediction_pairs, plot_pairs, concat_pair
import numpy as np
import matplotlib.pyplot as plt




(X_train_1, X_train_2, y_train), (X_val_1, X_val_2, y_val) = Train, Val


# results = model.evaluate([X_train_2, X_train_1], y_train)
# print("val loss, val acc:", results)

# plot_anchor_images(anchor_images)
# plt.show()

# plot_pairs(prediction_pairs)
# plt.show()

def make_prediction(image, anchor_images,face = False ,display = False):
    
    classes = ['Drake', 'Messi', 'Eminem', 'Ronaldo']
    if not face: detections = detect_face(image)
    else: detections = [image]
    
    if detections == 'No face detected': return detections
     ## take first detection for now
    detection_results = []
    for detection in detections:
        ## create anchor-image pairs
        prediction_pairs = create_prediction_pairs(anchor_images, detection)
        results = {}
        for clas in classes:

            result = np.mean(model.predict([prediction_pairs[clas][:,0], prediction_pairs[clas][:,1]], verbose = 0))
            if display: print(f"For class {clas}, model output: {result}")
            results[clas] = result
            
        results = dict(sorted(results.items(), key = lambda x: x[1]))
        
        detection_results.append(results)
            
    return detection_results



print(make_prediction(test_image, anchor_images, display=True))
