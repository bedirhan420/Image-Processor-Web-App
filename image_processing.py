import cv2
import numpy as np

# Kenar Tespiti
def edge_detection(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)

# Negatif Dönüşümü
def negative(image):
    return 255 - image

# Logaritmik Dönüşüm
def log_transform(image, c=1):
    return (c * np.log(1 + image.astype(np.float32))).clip(0, 255).astype(np.uint8)

# Gamma Dönüşümü
def gamma_transform(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Kontrast Germe
def contrast_stretching(image):
    min_val, max_val = np.min(image), np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Histogram Hesaplama
def calculate_histogram(image, bins=256):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    return hist

# Histogram Eşitleme
def equalize_histogram(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

# CDF Hesaplama
def calculate_cdf(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    return cdf

# 2B Hareketli Ortalama Filtresi
def moving_average_filter(image, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    return cv2.filter2D(image, -1, kernel)

# Medyan Filtre
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Maksimum Filtre
def max_filter(image, kernel_size=3):
    return cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8))

# Minimum Filtre
def min_filter(image, kernel_size=3):
    return cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8))

# Laplas İşleci
def laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Normalize edip uint8'e dönüştürme
    laplacian_normalized = cv2.convertScaleAbs(laplacian)
    return laplacian_normalized

# Birinci Derece Türev
def sobel_filter(image, direction="x"):
    if direction == "x":
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == "y":
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Normalize edip uint8'e dönüştürme
    sobel_normalized = cv2.convertScaleAbs(sobel)
    return sobel_normalized

# Box Filter
def box_filter(image, kernel_size=3):
    return cv2.boxFilter(image, -1, (kernel_size, kernel_size))

# Blur
def blur_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))


# Parlaklık Ayarı
def adjust_brightness(image, brightness):
    return cv2.convertScaleAbs(image, alpha=1, beta=brightness)

# Gürültü Ekleme
def add_noise(image, noise_type):
    h, w, c = image.shape
    if noise_type == "Gaussian":
        mean, sigma = 0, 25
        noise = np.random.normal(mean, sigma, (h, w, c)).astype(np.int16)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    elif noise_type == "Salt & Pepper":
        noisy = image.copy()
        salt = np.ceil(0.05 * image.size * 0.5).astype(int)
        pepper = np.ceil(0.05 * image.size * 0.5).astype(int)
        
        # Salt (Beyaz) ekleme
        salt_coords = [np.random.randint(0, dim, salt) for dim in image.shape[:2]]
        noisy[salt_coords[0], salt_coords[1], :] = 255
        
        # Pepper (Siyah) ekleme
        pepper_coords = [np.random.randint(0, dim, pepper) for dim in image.shape[:2]]
        noisy[pepper_coords[0], pepper_coords[1], :] = 0
    elif noise_type == "Uniform":
        noise = np.random.uniform(-50, 50, (h, w, c)).astype(np.int16)
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"{noise_type} gürültüsü tanınmadı!")
    return noisy

#Renk Uzayı dönüşümleri
def color_space_transform(img_array,color_space_type):
    if color_space_type == "BGR":
        converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif color_space_type == "GRAY":
        converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif color_space_type == "HSV":
        converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    elif color_space_type == "LAB":
        converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    elif color_space_type == "YCrCb":
        converted_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    else:
        raise ValueError(f"Geçersiz Seçim!")
    return converted_img
    
# Morfolojik İşlemler
def morphological_operation(image, operation, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == "Erosion":
        return cv2.erode(image, kernel)
    elif operation == "Dilation":
        return cv2.dilate(image, kernel)
    elif operation == "Opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == "Closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
