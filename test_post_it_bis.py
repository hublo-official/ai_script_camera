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

# === SEUIL POUR LA DIFFÉRENCE D'IMAGE ===
DIFF_THRESHOLD = 0.02   # proportion minimale de pixels changés pour dire "gommettes cachées"

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
    """Rogne la zone utile de l'image pour la détection (zone des gommettes)"""
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


# === Différence d'image ===
def compute_difference(ref_img, current_img):
    """Calcule la différence entre une image de référence et une image courante"""
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(ref_gray, cur_gray)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    ratio_change = np.sum(thresh > 0) / thresh.size
    return ratio_change


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
print("🎯 Capture image de référence pour détection par différence (caméra d'entrée)...")
first_image = None
while not first_image:
    result = download_image()
    if result:
        first_image, _ = result
    time.sleep(1)

ref_img = cv2.imread(first_image)
if ref_img is None:
    print("❌ Impossible de lire l'image de référence.")
    exit(1)

ref_crop = crop_zone_util(ref_img)
print("✅ Image de référence enregistrée pour la comparaison.")

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

    img_crop = cv2.imread(crop_path)
    if img_crop is None:
        continue

    ratio_change = compute_difference(ref_crop, img_crop)
    visible = ratio_change < DIFF_THRESHOLD

    if state == "waiting_visible":
        if visible:
            print("✅ Gommettes visibles, attente disparition…")
            state = "waiting_hidden"

    elif state == "waiting_hidden":
        if not visible:
            print("⚠️ Changement détecté (gommettes cachées) ! Début attente 2s…")
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