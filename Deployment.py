import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
import numpy
import cv2
import joblib

cancers = {1: "Meningioma",
           2: "Glioma",
           3: "Pituitary Tumor"}

# Load pre-trained classification model
mnb = joblib.load('/home/abiggj/Projects/Python/SY IV project/Supervised Learning/MNB.joblib')
#svc = joblib.load('/home/abiggj/Projects/Python/SY IV project/Supervised Learning/SVC.joblib')
knn = joblib.load('/home/abiggj/Projects/Python/SY IV project/Supervised Learning/KNN.joblib')
rfc = joblib.load('/home/abiggj/Projects/Python/SY IV project/Supervised Learning/RFC.joblib')

# Define function to read and preprocess input image
def preprocess_image(image, tumor):

    image = (image - 127.5) / 127.5
    tumor = (tumor - 127.5) / 127.5
    # Preprocess the input image as required by the classification model
    def pca(img):
        pca = PCA(n_components=65)
        img_pca = pca.fit_transform(img)
        img_pca = pca.inverse_transform(img_pca)
        return img_pca

    def bin(img2):
        scaled_img = numpy.interp(img2, (-1, 1), (0, 255))

        # Convert the scaled image to 8-bit unsigned integer format
        uint8_img = numpy.uint8(scaled_img)

        # Threshold the image to get a binary mask
        threshold_value = 75
        max_value = 255
        _, binary_mask = cv2.threshold(uint8_img, threshold_value, max_value, cv2.THRESH_BINARY)

        # Scale the binary mask back to range from -1 to 1
        return numpy.interp(binary_mask, (0, 255), (-1, 1))

    data, mskData = bin(pca(image)), bin(pca(tumor))
    data = data @ mskData.T
    return data.reshape(1,-1)


# Create Streamlit web application
st.title("Cancer Classification")
st.write("Upload an image to predict the type of cancer.")

# Add input file uploader and predict button
uploaded_img = st.file_uploader("Choose an image(Brain Tumor MRI)...", type=["jpg", "jpeg", "png"])
uploaded_msk = st.file_uploader("Choose an image(Tumor Mask)...", type=["jpg", "jpeg", "png"])
if uploaded_img is not None and uploaded_msk is not None:
    image = numpy.asarray(Image.open(uploaded_img).convert('L'))
    mask = numpy.asarray(Image.open(uploaded_msk).convert('L'))
    st.image(image+(mask*2), caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        # Preprocess input image
        preprocessed_image = preprocess_image(image, mask)

        # Predict output using pre-trained model
        knn_pred = knn.predict(preprocessed_image)
        rfc_pred = knn.predict(preprocessed_image)
        mnb_pred = knn.predict(preprocessed_image)

        # Display predicted output
        st.write('Predicted cancer type: KNN-', cancers[knn_pred[0]])
        # st.write('Predicted cancer type: SVC-', svc_pred)
        st.write('Predicted cancer type: RFC-', cancers[rfc_pred[0]])
        st.write('Predicted cancer type: MNB-', cancers[mnb_pred[0]])
