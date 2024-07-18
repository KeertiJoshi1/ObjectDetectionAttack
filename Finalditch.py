import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
        self.colorlist = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "C:/Users/keert/Desktop/MTech Course Work/Safety Systems/pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print("Model " + self.modelName + " loaded successfully")

    def createBoundingBox(self, image, threshold=0.5):
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]
        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex]
                classColor = self.colorlist[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)
        return image

    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingBox(image, threshold)
        cv2.imwrite(self.modelName + ".png", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fgsm_attack(self, image, epsilon, data_grad):
        sign_data_grad = tf.sign(data_grad)
        perturbed_image = image + epsilon * sign_data_grad
        perturbed_image = tf.clip_by_value(perturbed_image, 0, 1)
        return perturbed_image

    def perturbImage(self, imagePath, epsilon):
        image = cv2.imread(imagePath)
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.float32) / 255.0
        inputTensor = inputTensor[tf.newaxis, ...]

        with tf.GradientTape() as tape:
            tape.watch(inputTensor)
            predictions = self.model(inputTensor)
            loss = tf.reduce_mean(predictions['detection_scores'][0])

        gradient = tape.gradient(loss, inputTensor)
        perturbed_image = self.fgsm_attack(inputTensor, epsilon, gradient)

        perturbed_image = perturbed_image[0].numpy() * 255.0
        perturbed_image = perturbed_image.astype(np.uint8)

        return perturbed_image

    def predictPerturbedImage(self, imagePath, epsilon, threshold=0.5):
        perturbed_image = self.perturbImage(imagePath, epsilon)
        bboxImage = self.createBoundingBox(perturbed_image, threshold)
        cv2.imwrite(self.modelName + "_perturbed.png", bboxImage)
        cv2.imshow("Perturbed Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        original_image = cv2.imread(imagePath)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        perturbed_image_rgb = cv2.cvtColor(perturbed_image, cv2.COLOR_BGR2RGB)

        fig, axarr = plt.subplots(1, 2, figsize=(12, 6))
        axarr[0].imshow(original_image_rgb)
        axarr[0].set_title('Original Image')
        axarr[0].axis('off')

        axarr[1].imshow(perturbed_image_rgb)
        axarr[1].set_title('Perturbed Image')
        axarr[1].axis('off')

        plt.show()

if __name__ == "__main__":
    modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
    classFile = 'coco.names'
    imagePath = "NewTest/new/4.png"
    threshold = 0.5
    epsilon = 0.1

    detector = Detector()
    detector.readClasses(classFile)
    detector.downloadModel(modelURL)
    detector.loadModel()
    detector.predictImage(imagePath, threshold)
    detector.predictPerturbedImage(imagePath, epsilon, threshold)
