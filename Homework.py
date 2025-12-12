import cv2
import numpy as np
import os

def detect_stop_sign():
    # Kodun çalıştığı yeri bul
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "stop_sign_dataset")
    output_folder = os.path.join(script_dir, "output_results")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.exists(input_folder):
        print(f"Hata: '{input_folder}' bulunamadı.")
        return

    files = os.listdir(input_folder)
    print(f"Klasör bulundu. {len(files)} dosya taranıyor...\n")

    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            
            # --- TÜRKÇE KARAKTER SORUNU ÇÖZÜMÜ (OKUMA) ---
            # cv2.imread yerine bu yöntemi kullanıyoruz:
            try:
                # Dosyayı binary olarak numpy ile oku, sonra decode et
                img_array = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                print(f"Hata: {filename} açılamadı. Sebep: {e}")
                continue

            if img is None:
                print(f"Hata: {filename} bozuk veya okunamıyor.")
                continue

            # --- GÖRÜNTÜ İŞLEME ---
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            full_mask = mask1 | mask2

            kernel = np.ones((3, 3), np.uint8)
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_DILATE, kernel)

            contours, _ = cv2.findContours(full_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            detected = False
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 400:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h
                    
                    if 0.7 < aspect_ratio < 1.4:
                        perimeter = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)
                        corners = len(approx)

                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area == 0: continue
                        solidity = float(area) / hull_area

                        if (6 <= corners <= 12) or (solidity > 0.9):
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                            center_x = x + w // 2
                            center_y = y + h // 2
                            cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)
                            
                            print(f"[{filename}] -> STOP Bulundu!")
                            detected = True
                            break 

            if not detected:
                print(f"[{filename}] -> Bulunamadı.")

            # --- TÜRKÇE KARAKTER SORUNU ÇÖZÜMÜ (KAYDETME) ---
            save_path = os.path.join(output_folder, "islenmis_" + filename)
            try:
                # cv2.imwrite yerine bu yöntemi kullanıyoruz:
                is_success, im_buf_arr = cv2.imencode(".jpg", img)
                im_buf_arr.tofile(save_path)
            except Exception as e:
                print(f"Kaydetme hatası: {e}")

    print("\nİşlem tamamlandı. 'output_results' klasörüne bakabilirsin.")

if __name__ == "__main__":
    detect_stop_sign()