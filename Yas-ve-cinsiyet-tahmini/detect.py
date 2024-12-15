import cv2

# Yüz algılama ve çizgileri çizecek fonksiyon
def highlightFace(net, frame, conf_threshold=0.7):
    # Gelen kareyi kopyalayıp boyutlarını alıyoruz
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # DNN için blob oluşturuluyor (yapay zeka modeli için veri hazırlığı)
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Modelin girdisini ayarlıyoruz ve tahmin yapıyoruz
    net.setInput(blob)
    detections = net.forward()

    # Algılanan yüzlerin koordinatları saklanacak
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Algılanan yüzün güveni (oran)
        if confidence > conf_threshold:  # Güven oranı eşikten yüksekse
            # Yüz koordinatlarını hesapla
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # Koordinatları listeye ekle
            faceBoxes.append([x1, y1, x2, y2])

            # Yüzün etrafına dikdörtgen çiz
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, faceBoxes  # Sonuçları döndür

# Yüz, yaş ve cinsiyet modelleri için dosya yolları
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Model için ortalama değerler ve yaş/cinsiyet listeleri
genderList = ['Erkek', 'Kadin']
ageList = ['(0-6)', '(6-14)', '(14-20)', '(21-30)', '(31-40)', '(41-50)', '(51-60)', '(61-100)']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Modelleri yükleme
faceNet = cv2.dnn.readNet(faceModel, faceProto)  # Yüz algılama modeli
ageNet = cv2.dnn.readNet(ageModel, ageProto)  # Yaş tahmin modeli
genderNet = cv2.dnn.readNet(genderModel, genderProto)  # Cinsiyet tahmin modeli

# Video veya resim kaynağı (jpeg,png gibi dosyalar kullanılıyor)
video = cv2.VideoCapture("deniz.jpeg" if "deniz.jpeg" else 0)

padding = 20  # Yüz etrafına ekleme yapılacak pikseller

while cv2.waitKey(1) < 0:  # Sonsuz döngü, 'q' tuşuna basılına kadar devam eder
    hasFrame, frame = video.read()  # Videodan kare oku
    if not hasFrame:  # Kare bulunamazsa döngüyü sonlandır
        cv2.waitKey()
        break

    # Yüz algılama ve dikdörtgen çizme
    resultImg, faceBoxes = highlightFace(faceNet, frame)

    if not faceBoxes:  # Hiç yüz algılanmazsa mesaj yaz
        print("Yüz algılanamadı")

    for faceBox in faceBoxes:  # Algılanan her yüz için
        # Yüz bölgesini belirle
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Cinsiyet tahmini için blob hazırla ve tahmin yap
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]  # En yüksek olasılık
        print(f'Cinsiyet: {gender}')

        # Yaş tahmini için blob hazırla ve tahmin yap
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]  # En yüksek olasılık
        print(f'Yaş: {age[1:-1]} yasinda')

        # Sonuçları ekrana yazı ve görüntüye ekle
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # Sonucu görüntüle
    cv2.imshow("Yas ve Cinsiyet Algilama", resultImg)
