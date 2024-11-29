import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_processing import *

st.title("Görüntü İşleme Uygulaması")

uploaded_file = st.file_uploader("Görüntü Yükleyin", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Yüklenen Görüntü", use_column_width=True)

    option = st.sidebar.selectbox("İşlem Seçin", [
        "Kenar Tespiti",
        "Parlaklık Ayarı",
        "Yeğinlik Dönüşümleri",
        "Histogram",
        "Gürültü Ekleme",
        "Filtreler",
        "Renk Uzayı Dönüşümleri",
        "Morfolojik İşlemler",
    ])

    if option == "Kenar Tespiti":
        low = st.sidebar.slider("Alt Eşik", 0, 255,255 )
        high = st.sidebar.slider("Üst Eşik", 0, 255, 255)
        result = edge_detection(image, low, high)
        st.image(result, caption="Kenar Tespiti Sonucu")
    
    elif option == "Parlaklık Ayarı":
        brightness = st.sidebar.slider("Parlaklık Değeri", -100, 100, 0)
        result = adjust_brightness(image, brightness)
        st.image(result, caption="Parlaklık Ayarı Sonucu")
    
    elif option == "Yeğinlik Dönüşümleri":
        intensity_type = st.sidebar.selectbox("Yeğinlik Türü",["Negatif Dönüşümü","Logaritma Dönüşümü","Gamma Dönüşümü","Kontrast Germe"])
        if intensity_type == "Negatif Dönüşümü":
            result = negative(image)
        elif intensity_type == "Logaritma Dönüşümü":
            result = log_transform(image)
        elif intensity_type == "Gamma Dönüşümü":
            gamma = st.sidebar.slider("Gamma",1,30,1,step=1)
            result = gamma_transform(image,gamma)
        elif intensity_type == "Kontrast Germe":
            result = contrast_stretching(image)

        st.image(result, caption=f"{intensity_type} Sonucu")

    elif option == "Histogram":
        hist_option = st.sidebar.radio("Histogram İşlemi", ["Hesaplama", "Eşitleme","CDF"])
        if hist_option == "Hesaplama":
            bin_option = st.sidebar.slider("Bin Sayısı",1,256,256,step=1)
            hist = calculate_histogram(image,bin_option)
            fig, ax = plt.subplots()
            ax.plot(hist)
            ax.set_title(f"Histogram (Bin: {bin_option})")
            st.pyplot(fig)
        elif hist_option == "Eşitleme":
            result = equalize_histogram(image)
            st.image(result, caption="Histogram Eşitleme Sonucu")
        elif hist_option == "CDF":
            cdf = calculate_cdf(image)
            fig, ax = plt.subplots()
            ax.plot(cdf)
            ax.set_title(f"CDF")
            st.pyplot(fig)


    elif option == "Gürültü Ekleme":
        noise_type = st.sidebar.selectbox("Gürültü Türü", ["Gaussian", "Salt & Pepper", "Uniform"])
        result = add_noise(image, noise_type)
        st.image(result, caption=f"{noise_type} Gürültüsü Eklenmiş Görüntü")

    elif option == "Filtreler":
        filter_type = st.sidebar.selectbox("Filtre Türü",["Box","Blur","Hareketli Ortalama","Medyan","Max","Min","Laplace","Sobel"])
        if filter_type == "Box":
            kernel = st.sidebar.slider("Kernel Boyutu",1,10,3)
            result = box_filter(image,kernel)
        elif filter_type == "Blur":
            kernel = st.sidebar.slider("Kernel Boyutu",1,10,3)
            result = blur_filter(image,kernel)
        elif filter_type == "Hareketli Ortalama":
            kernel = st.sidebar.slider("Kernel Boyutu",1,10,3)
            result = moving_average_filter(image,kernel)
        elif filter_type == "Medyan":
            kernel = st.sidebar.slider("Kernel Boyutu",1,11,3,step=2)
            result = median_filter(image,kernel)
        elif filter_type == "Max":
            kernel = st.sidebar.slider("Kernel Boyutu",1,10,3)
            result = max_filter(image,kernel)
        elif filter_type == "Min":
            kernel = st.sidebar.slider("Kernel Boyutu",1,10,3)
            result = min_filter(image,kernel)
        elif filter_type == "Laplace":
            result = laplacian_filter(image)
        elif filter_type == "Sobel":
            direction = st.sidebar.radio("Direction",["x","y"])
            result = sobel_filter(image,direction)
        
        st.image(result, caption=f"{filter_type} Filtrelenmiş Görüntü")
        
    elif option == "Renk Uzayı Dönüşümleri":
        color_space_type = st.sidebar.selectbox(
            "Dönüştürmek istediğiniz renk uzayını seçin",
            ("BGR", "GRAY", "HSV", "LAB", "YCrCb")
        )
        img_array = np.array(image)
        converted_img = color_space_transform(img_array,color_space_type)
        st.subheader("Dönüştürülmüş Görüntü")
        if color_space_type == "GRAY":
            st.image(converted_img, caption=f"{color_space_type} Görüntü", use_column_width=True, channels="GRAY")
        else:
            st.image(converted_img, caption=f"{color_space_type} Görüntü", use_column_width=True)


    elif option == "Morfolojik İşlemler":
        morph_option = st.sidebar.selectbox("Morfolojik İşlem Türü", ["Erosion", "Dilation", "Opening", "Closing"])
        kernel_size = st.sidebar.slider("Kernel Boyutu", 1, 15, 3, step=1)
        result = morphological_operation(image, morph_option, kernel_size)
        st.image(result, caption=f"{morph_option} İşlemi Sonucu")
