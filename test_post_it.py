import os
import time
import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from collections import deque

# === CONFIGURATION CAMÉRA ENTRÉE ===
SNAPSHOT_URL = "http://192.168.1.103/ISAPI/Streaming/channels/1/picture"
USERNAME = "admin"
PASSWORD = "Hublo75014"

# === CONFIGURATION TRAITEMENT ===
SAVE_DIR = "./cam_entree"    # dossier spécifique pour les images de la caméra d'entrée
INTERVAL = 1.0               # 1 seconde entre snapshots
MAX_IMAGES = 10              # FIFO locale
RAILWAY_API = "https://ai.hublo.eu/upload"  # endpoint d'entrée

# === COULEUR DES GOMMETTES (BLEU FONCÉ SUR FOND CLAIR) ===
LOWER_BLUE = np.array([85, 50, 70])
UPPER_BLUE = np.array([110, 255, 255])
PIXEL_THRESHOLD = 0.02  # % minimal de pixels bleus pour dire "gommette visible"

# === FIFO IMAGES ===
image_queue = deque()

# --- Initialisation du dossier ---
os.makedirs(SAVE_DIR, exist_ok=True)

def cleanup_existing_images():
    """Nettoie les anciennes images au démarrage (max 10 fichiers)"""
    images = sorted(
        [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")],
        key=os.path.getmtime,
        reverse=True
    )
    for old in images[MAX_IMAGES:]:
        try:
            os.remove(old)
            print(f"🧹 Ancienne image supprimée : {old}")
        except:
            pass

cleanup_existing_images()

# === CROPS ===
def crop_zone_util(img):
    """Rogne la zone utile de l'image pour détecter les gommettes"""
    h, w = img.shape[:2]
    return img[int(h * 0.4):int(h * 0.64), int(w * 0.38):int(w * 0.53)]

def crop_upload(img):
    """Rogne la zone à envoyer à Railway"""
    h, w = img.shape[:2]
    return img[int(h * 0.01):int(h * 0.45), int(w * 0.3):int(w * 0.68)]

# === Téléchargement image ===
def download_image():
    """Télécharge une image depuis la caméra Hikvision"""
    try:
        r = requests.get(
            SNAPSHOT_URL,
            timeout=5,
            headers={'User-Agent': 'Mozilla/5.0'},
            auth=HTTPDigestAuth(USERNAME, PASSWORD)
        )
        if r.status_code != 200:
            print(f"❌ Erreur {r.status_code} téléchargement snapshot")
            return None

        img_array = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            print("❌ Erreur : image non décodable")
            return None

        # Crops
        img_crop = crop_zone_util(img)
        img_upload = crop_upload(img)

        timestamp = int(time.time())
        crop_path = os.path.join(SAVE_DIR, f"entree_detect_{timestamp}.jpg")
        upload_path = os.path.join(SAVE_DIR, f"entree_upload_{timestamp}.jpg")

        cv2.imwrite(crop_path, img_crop)
        cv2.imwrite(upload_path, img_upload)

        return crop_path, upload_path

    except Exception as e:
        print(f"⚠️ Exception snapshot : {e}")
        return None

# === Détection de gommettes ===
def detect_postit_zones(img, nb_zones=3):
    """Détecte automatiquement les plus grandes zones bleues"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zones = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 200:
            zones.append((area, (x, y, w, h)))

    zones = sorted(zones, key=lambda z: z[0], reverse=True)[:nb_zones]
    return [z[1] for z in zones]

def postit_visible(img, zone):
    """Vérifie si une zone contient encore du bleu"""
    x, y, w, h = zone
    roi = img[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    ratio = np.sum(mask > 0) / (w * h)
    return ratio > PIXEL_THRESHOLD

def all_postits_visible(img, zones):
    return all(postit_visible(img, z) for z in zones)

# === Upload vers Railway ===
def upload_to_railway(path):
    try:
        with open(path, "rb") as f:
            files = {"image": f}
            r = requests.post(RAILWAY_API, files=files)
        print(f"📤 Envoi Railway: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        print(f"⚠️ Erreur upload Railway : {e}")

# === FIFO local ===
def fifo_cleanup():
    """Supprime les images locales en trop (FIFO)"""
    while len(image_queue) > MAX_IMAGES:
        to_delete = image_queue.popleft()
        try:
            os.remove(to_delete)
            print(f"🗑️ Image supprimée (FIFO): {to_delete}")
        except:
            pass

# === AUTO-DETECTION INITIALE ===
print("🎯 Détection automatique des gommettes bleues foncées en cours (caméra d'entrée)...")
first_image = None
while not first_image:
    result = download_image()
    if result:
        first_image, _ = result
    time.sleep(1)

img0 = cv2.imread(first_image)
POSTIT_ZONES = detect_postit_zones(img0)

if len(POSTIT_ZONES) < 3:
    print("❌ Impossible de détecter 3 gommettes. Vérifie la couleur/éclairage.")
    cv2.imshow("Image pour détection", img0)
    print("Image affichée pour détection automatique. Appuie sur une touche pour quitter.")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        cv2.destroyAllWindows()
        exit(1)
    else:
        cv2.destroyAllWindows()

print(f"✅ Zones détectées: {POSTIT_ZONES}")

# === BOUCLE PRINCIPALE ===
print("📷 Surveillance active (CTRL+C pour arrêter)")

state = "waiting_visible"   # waiting_visible → waiting_hidden → cooldown
hidden_since = None

while True:
    result = download_image()
    if not result:
        time.sleep(INTERVAL)
        continue

    crop_path, upload_path = result
    image_queue.append(crop_path)
    fifo_cleanup()

    img = cv2.imread(crop_path)
    if img is None:
        continue

    visible = all_postits_visible(img, POSTIT_ZONES)

    if state == "waiting_visible":
        if visible:
            print("✅ Gommettes visibles, attente disparition…")
            state = "waiting_hidden"

    elif state == "waiting_hidden":
        if not visible:
            print("⚠️ Gommettes cachées ! Début attente 2s…")
            hidden_since = time.time()
            state = "cooldown"

    elif state == "cooldown":
        if visible:
            print("🔄 Gommettes réapparues, retour état initial")
            state = "waiting_hidden"
            hidden_since = None
        else:
            if time.time() - hidden_since >= 2:
                print("📤 Envoi photo après 2s de couverture")
                upload_to_railway(upload_path)
                try:
                    os.remove(upload_path)
                except:
                    pass
                state = "waiting_visible"
                hidden_since = None

    time.sleep(INTERVAL)