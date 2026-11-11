# ----------- Підрахунок будівель головного кампусу КПІ за двома джерелами знімків -----------------

'''
Функціонал (ЛР №2, рівень III):
підрахунок будівель на двох знімках одного кампусу різними операціями обробки та векторизації:
- google  — Google Maps, еталонне джерело (корекція кольору не потрібна);
- wayback — Esri Wayback, слабше джерело: накладається на кадр Google (SIFT + RANSAC),
            корекція кольору «сірий світ», власні пороги.
Код один, для джерел відрізняються лише CONFIG (параметри, локальні зони, групи зшивання).

Етапи: 1 завантаження -> 2 маска -> 3 контури -> 4 валідація -> 5 об'єднання сегментів -> 6 малювання.

'''

import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

DIR = Path(__file__).resolve().parent
OUT = DIR / 'results'
OUT.mkdir(exist_ok=True)

# ===================== ПАРАМЕТРИ =====================
# Усі координати задані в кадрі Google 1024x571 (знімок Wayback накладається на цей кадр).
PARAMS_GOOGLE = dict(
    smax=46,          # Поріг насиченості сірих дахів
    vmin=52,          # Поріг яскравості
    exg=2.0,          # Поріг індексу зелені ExG
    bmax=None,        # Макс. канал b* (Lab): відсікає теплі: плитку, ґрунт, парковки (None = вимкнено)
    yellow_s=24,      # Мін. насиченість жовтих полів/ґрунту, які виключаються з маски
    open_k=7,         # Видалення дрібних стежок
    close_k=7,        # Базове закриття
    min_area=460,     # Мінімальна площа корпусу
    max_area=28000,   # Максимальна площа корпусу
    min_side=11,      # Мінімальна ширина будівлі
    max_asp=6.5,      # Дозволяє довгі корпуси
    min_rect=0.25,    # Дозволяє гребінчасті та багатокрилі корпуси
    min_sol=0.45,     # Щільність для малих об'єктів
    large_sol=0.34,   # Щільність для великих об'єктів
    min_thick=6.8,    # Мінімальна внутрішня товщина
)

# Wayback: інші кольори й різкість знімка, тому пороги свої (підібрано вручну перебором:
# кількість будівель = еталону і найкращий збіг контурів з Google). yellow_s=255 — фільтр жовтого вимкнено.
PARAMS_WAYBACK = dict(PARAMS_GOOGLE, smax=70, vmin=87, bmax=125, yellow_s=255, open_k=5,
                      min_area=350, min_side=9, min_sol=0.35, min_thick=4.5)

BORDER_MARGIN = 3     # контури, що торкаються краю кадру, відкидаються
EPS = 0.013           # точність апроксимації контуру approxPolyDP (частка периметра)


def rect(x0, y0, x1, y1):
    """Прямокутник як полігон (кути включно)."""
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# --- Межі кадру: дороги, залізниця, площі — прибираються на обох знімках і в обох режимах ---
FRAME_EXCLUDE = [
    rect(0, 548, 1023, 570),       # смуга внизу (Борщагівська вулиця)
    rect(737, 0, 1023, 78),        # вгорі праворуч — вулиця
    rect(0, 0, 80, 570),           # ліва смуга кадру
    rect(0, 239, 178, 570),        # лівий нижній кут (поза кампусом)
    rect(931, 85, 1023, 570),      # права смуга кадру
    [(0, 485), (350, 525), (510, 558), (1024, 560), (1024, 575), (0, 575)],  # залізниця
    rect(555, 90, 725, 198),       # площа й алеї перед головним корпусом
    [(660, 495), (755, 504), (835, 515), (835, 565), (660, 565)],            # майданчик внизу праворуч
    rect(492, 400, 545, 435),      # «хвіст» довгого корпусу на плитку
]

BOTTOM_ROW = (195, 440, 515, 545)    # x0, y0, x1, y1: нижній ряд корпусів, де секції з'єднує закриття 23x9
SPLIT_REGION = (210, 335, 355, 420)  # x0, y0, x1, y1: тут три корпуси злипаються в один сегмент

# --- Локальні зони (ручні налаштування під конкретні знімки) ---
# RELAXED — м'якші пороги всередині полігона (додаються до маски). Ключі як у PARAMS; немає ключа —
#           береться базовий поріг. Додатково: hue=(hmin, hmax, s_low), no_yellow=True.
# EXCLUDE — маска всередині полігона обнуляється (хибні об'єкти).
# CLIP    — у прямокутнику region лишається тільки те, що всередині poly (дах того ж кольору, що площа).
ZONE_L1 = dict(name='будинок L1 (діагональний, ліворуч)', vmin=52, exg=4.0, no_yellow=True,
               poly=[(88, 328), (114, 328), (114, 356), (186, 356), (210, 342), (212, 394), (90, 394)])
ZONE_L2 = dict(name='будинок L2 (над L1)', vmin=52, exg=4.0, no_yellow=True,
               poly=[(148, 280), (213, 280), (213, 308), (150, 304)])
ZONE_A = dict(name='будівля A (праворуч вгорі, світлий дах)', vmin=30, exg=6.0,
              poly=[(898, 54), (950, 62), (952, 100), (900, 92)])
ZONE_C = dict(name='будівля C (світла, ліворуч)', vmin=60, exg=6.0, smax=90, no_yellow=True,
              poly=rect(76, 228, 112, 298))
GAP_WHITE = dict(name='щілина: комплекс | біле здання праворуч', poly=rect(743, 205, 751, 258))

ZONES_GOOGLE = dict(
    RELAXED=[
        # Центральний комплекс: тонкі крила під кронами; відтінок 90-130 = сине-сірі дахи (не зелень)
        dict(name='центральний комплекс', vmin=35, exg=6.0, smax=70, hue=(90, 130, 10),
             poly=[(455, 105), (555, 105), (555, 190), (655, 190), (655, 160),
                   (750, 160), (750, 285), (455, 285)]),
        ZONE_L1, ZONE_L2,
        # Ліве крило головного корпусу: синюваті насичені дахи; b*<=124 відсікає теплий ґрунт
        dict(name='ліве крило головного корпусу', vmin=60, exg=6.0, smax=95, bmax=124, poly=rect(462, 116, 534, 206)),
        ZONE_C, ZONE_A,
        # Будівля B: синюватий дах (S до 59), ґрунт навколо теплий (b*≈130)
        dict(name='будівля B', vmin=60, exg=6.0, smax=70, bmax=125, poly=rect(906, 100, 931, 150)),
    ],
    EXCLUDE=[
        dict(name='мощена площадка праворуч', poly=rect(885, 150, 940, 182)),
        dict(name='ґрунт між будівлею A і ліском', poly=rect(850, 75, 897, 150)),
        # Щілина доведена до зони нижче, щоб не замкнутися (замкнену заповнить крок «внутрішні двори»)
        dict(name='доріжка між корпусом 19 і ангаром', poly=rect(580, 298, 589, 366)),
        dict(name='ґрунт над ангаром', poly=rect(589, 280, 630, 305)),
        dict(name='плитка й ґрунт під корпусом 19', poly=rect(494, 365, 632, 396)),
        dict(name='щілина L2 | плита', poly=rect(213, 276, 224, 314)),
        dict(name='виступ праворуч від головного корпусу', poly=rect(744, 160, 800, 206)),
        GAP_WHITE,
        dict(name='дорожній ріг корпусу 8', poly=rect(790, 400, 858, 455)),
    ],
    CLIP=[
        dict(name='блок 20 (хвіст на площу)', region=(625, 250, 742, 345), poly=rect(660, 250, 722, 318)),
    ],
)

ZONES_WAYBACK = dict(
    RELAXED=[
        # Головний комплекс на Wayback має темно-зелені дахи (V≈20-50); ExG<=12 відсікає яскраву траву.
        # Ліворуч унизу (x<545, y>195) — крони дерев такого ж кольору, тому полігон їх обходить.
        dict(name='головний комплекс (зелені дахи)', vmin=15, exg=12.0, smax=170, bmax=None,
             poly=[(470, 112), (552, 112), (552, 190), (655, 190), (655, 165), (745, 165),
                   (745, 285), (545, 285), (545, 195), (470, 195)]),
        ZONE_L1, ZONE_L2, ZONE_A, ZONE_C,
        # Корпус 20 і малий корпус праворуч від нього: темні дахи (V≈40-100), b*≈125 — без обмеження b*
        dict(name='корпус 20 (темний дах)', vmin=30, exg=6.0, smax=70, bmax=None, poly=rect(515, 298, 583, 360)),
        dict(name='малий корпус праворуч від 20', vmin=30, exg=6.0, smax=70, bmax=None, poly=rect(591, 310, 608, 352)),
        # Сірий будинок праворуч (на Wayback зсунутий на ~10 px через нахил зйомки)
        dict(name='сірий будинок праворуч', vmin=60, exg=6.0, smax=70, poly=rect(873, 222, 902, 262)),
    ],
    EXCLUDE=[
        dict(name='діагональна алея', poly=[(655, 340), (768, 322), (772, 350), (660, 368)]),
        # Виріз праворуч унизу — щоб не зачепити малий корпус праворуч від 20
        dict(name='доріжка під комплексом', poly=[(590, 268), (718, 268), (718, 334), (609, 334), (609, 309), (590, 309)]),
        dict(name='стежка ліворуч від алеї', poly=rect(622, 364, 666, 396)),
        dict(name='стежка внизу', poly=rect(552, 396, 598, 442)),
        dict(name='майданчик (плитка) ліворуч від ротонди', poly=[(362, 205), (398, 205), (396, 262), (362, 262)]),
        GAP_WHITE,
        dict(name='щілина між комплексом і будинком праворуч', poly=rect(732, 240, 744, 305)),
        dict(name='розріз: корпус внизу | будинок угорі праворуч', poly=rect(676, 422, 790, 434)),
        dict(name='тінь ліворуч від будинку угорі праворуч', poly=rect(672, 380, 712, 424)),
        # Хибні об'єкти, яких немає на еталоні: кіоск на дорозі та світла пляма в зелені
        dict(name='кіоск на дорозі вгорі', poly=rect(395, 68, 440, 100)),
        dict(name='світла пляма в зелені ліворуч', poly=rect(180, 400, 215, 425)),
        # Світлі доріжки, які закриття приклеювало до корпусів довгими «шипами»
        dict(name='доріжка над нижнім корпусом ліворуч', poly=rect(215, 428, 520, 443)),
        dict(name='вертикальна доріжка x≈497', poly=rect(493, 330, 503, 400)),
        dict(name='доріжка праворуч від сірого будинку', poly=rect(903, 222, 925, 300)),
        # Щілини, що відділяють сусідні корпуси (ширина 7 px — закриття 7x7 її не заклеює)
        dict(name='щілина: корпус 20 | малий корпус', poly=[(584, 300), (590, 300), (590, 351), (612, 351), (612, 392), (584, 392)]),
        dict(name='щілина: смуга ліворуч | сірий корпус праворуч унизу', poly=rect(886, 380, 893, 470)),
    ],
)

# --- Групи сегментів, що є однією будівлею (точка в кожному сегменті, k — ядро закриття) ---
MERGE_GOOGLE = [
    dict(seeds=[(398, 501), (467, 497)], k=9),                          # корпус з двориком + сусідній блок
    dict(seeds=[(610, 460), (542, 511), (594, 517)], k=21),             # Г-подібний корпус у трьох частинах
    dict(seeds=[(243, 302), (338, 313), (441, 340), (424, 276)], k=15), # довгий корпус з ротондою
    dict(seeds=[(203, 101), (271, 107), (134, 69)], k=33),              # гребінчастий корпус зверху ліворуч
    dict(seeds=[(598, 235), (729, 223), (690, 298), (513, 153)], k=45), # головний (центральний) комплекс
]
# Wayback: комплекс ділиться на 4 частини (+ середня частина, без неї зазор > k і права «рамка» губиться);
# гребінчастий корпус і його прибудова з білим дахом — одна будівля
MERGE_WAYBACK = MERGE_GOOGLE[:4] + [
    dict(seeds=[(513, 153), (598, 235), (640, 230), (729, 223)], k=45),  # головний комплекс
    dict(seeds=[(95, 50), (235, 115)], k=35),                           # гребінчастий корпус + прибудова
]

# ===================== CONFIG ДЖЕРЕЛ =====================
CONFIG = {
    'google':  dict(file='data/google_maps.jpg', gray_world=False,
                    params=PARAMS_GOOGLE, zones=ZONES_GOOGLE, merge=MERGE_GOOGLE),
    'wayback': dict(file='data/wayback.png', gray_world=True,
                    params=PARAMS_WAYBACK, zones=ZONES_WAYBACK, merge=MERGE_WAYBACK),
}


# ===================== 1. ЗАВАНТАЖЕННЯ =====================
def image_read(FileIm):
    image = cv2.imread(str(DIR / FileIm))
    if image is None:
        raise FileNotFoundError(f"Не вдалося відкрити файл: {FileIm}")
    return image


# --- Корекція кольору «сірий світ» (лише Wayback): вирівнює середні значення каналів B, G, R ---
def gray_world(image):
    f = image.astype(np.float32)
    mean = f.reshape(-1, 3).mean(axis=0)
    return np.clip(f * (mean.mean() / mean), 0, 255).astype(np.uint8)


# --- Суміщення: накладає image на кадр reference (SIFT + афінне перетворення, RANSAC) ---
def align_to(reference, image):
    sift = cv2.SIFT_create(5000)
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]          # тест Лоу
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=3)
    print(f"суміщення Wayback -> Google: {len(good)} збігів SIFT, {int(inliers.sum())} inliers")
    return cv2.warpAffine(image, M, (reference.shape[1], reference.shape[0]), flags=cv2.INTER_CUBIC)


# ===================== 2. МАСКА =====================
def poly_mask(shape, poly):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    return m > 0


def image_processing(image, P, zones):
    h, w = image.shape[:2]
    blur = cv2.GaussianBlur(image, (5, 5), 0)                               # Гаусова фільтрація
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lab_b = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)[..., 2]
    yellow = cv2.inRange(hsv, (8, P['yellow_s'], 0), (35, 255, 255))         # жовті поля, ґрунт
    f = cv2.GaussianBlur(image, (9, 9), 0).astype(np.float32)
    exg = f[..., 1] - 0.5 * (f[..., 0] + f[..., 2])                         # індекс зелені ExG = G - (R+B)/2

    def roof_mask(z):
        """Маска дахів: світлий малонасичений піксель, не жовтий, не зелений, не «теплий».
        Пороги беруться із z, а яких немає — з базових P."""
        m = cv2.inRange(hsv, (0, 0, z['vmin']), (180, z.get('smax', P['smax']), 252))
        if 'hue' in z:      # (hmin, hmax, s_low): сине-сірий відтінок АБО майже ахроматичний піксель
            hmin, hmax, s_low = z['hue']
            m &= cv2.inRange(hsv[..., 0], hmin, hmax) | cv2.inRange(hsv[..., 1], 0, s_low)
        if not z.get('no_yellow'):
            m[yellow > 0] = 0
        m[exg > z['exg']] = 0
        bmax = z.get('bmax', P['bmax'])
        if bmax is not None:
            m[lab_b > bmax] = 0
        return m

    # 2.1. Базова маска: сірий дах, без жовтих полів, без зелені
    mask = roof_mask(P)

    # 2.2. Межі кадру: дороги, залізниця, площі
    for poly in FRAME_EXCLUDE:
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 0)

    # 2.3. Локальні зони: м'якші пороги (додаємо), виключення та обрізка (прибираємо)
    for z in zones.get('RELAXED', []):
        mask[poly_mask(image.shape, z['poly']) & (roof_mask(z) > 0)] = 255
    for z in zones.get('EXCLUDE', []):
        mask[poly_mask(image.shape, z['poly'])] = 0
    for z in zones.get('CLIP', []):
        x0, y0, x1, y1 = z['region']
        outside = ~poly_mask(image.shape, z['poly'])
        mask[y0:y1, x0:x1][outside[y0:y1, x0:x1]] = 0

    # 2.4. Морфологія: відкриття прибирає стежки, закриття — розриви в дахах
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((P['open_k'], P['open_k']), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((P['close_k'], P['close_k']), np.uint8))
    x0, y0, x1, y1 = BOTTOM_ROW
    mask[y0:y1, x0:x1] = cv2.morphologyEx(mask[y0:y1, x0:x1], cv2.MORPH_CLOSE, np.ones((9, 23), np.uint8))

    # 2.5. Заповнення внутрішніх дворів (дірки менші за 2500 px)
    outer = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(outer, None, (0, 0), 255)                                 # усе, що досяжне ззовні
    holes = cv2.bitwise_not(outer[1:-1, 1:-1])
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 2500:
            mask[labels == i] = 255
    return mask


# ===================== 3. КОНТУРИ =====================
def image_contours(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    result = []
    for c in cnts:
        area = cv2.contourArea(c)
        x, y = cv2.boundingRect(c)[:2]
        solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1)

        # 3.1. Три злиплі корпуси: розрізаємо відкриттям вузьким вертикальним ядром 5x13
        x0, y0, x1, y1 = SPLIT_REGION
        if 1400 < area < 4500 and solidity < 0.76 and x0 < x < x1 and y0 < y < y1:
            sub = np.zeros_like(mask)
            cv2.drawContours(sub, [c], -1, 255, -1)
            cut = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((13, 5), np.uint8))
            parts = cv2.findContours(cut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(parts) >= 2:
                result.extend(parts)
                continue

        # 3.2. Великий розгалужений кластер: watershed за піками distance transform
        if area > 7500 and solidity < 0.55:
            sub = np.zeros_like(mask)
            cv2.drawContours(sub, [c], -1, 255, -1)
            dist = cv2.GaussianBlur(cv2.distanceTransform(sub, cv2.DIST_L2, 5), (0, 0), 2.5)
            peaks = (dist >= cv2.dilate(dist, np.ones((65, 65), np.uint8)) - 1e-6) & (dist >= 8.0)
            n, markers = cv2.connectedComponents(peaks.astype(np.uint8))
            if n > 2:                                       # фон + щонайменше два піки
                markers = markers + 1                       # фон = 1, піки = 2..n
                markers[(sub > 0) & (markers == 1)] = 0     # невідома область — її ділить watershed
                labels = cv2.watershed(cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR), markers)
                for i in range(2, n + 1):
                    part = (labels == i).astype(np.uint8) * 255
                    parts = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if parts:
                        result.append(max(parts, key=cv2.contourArea))
                continue

        result.append(c)
    return result


# ===================== 4. ВАЛІДАЦІЯ =====================
def image_recognition(image_cont, P, shape):
    """Геометричні фільтри; повертає прийняті контури, спрощені approxPolyDP."""
    h_img, w_img = shape[:2]
    buildings = []
    for c in image_cont:
        area = cv2.contourArea(c)
        if not P['min_area'] <= area <= P['max_area']:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if (x <= BORDER_MARGIN or y <= BORDER_MARGIN or
                x + bw >= w_img - BORDER_MARGIN or y + bh >= h_img - BORDER_MARGIN):
            continue                                                         # торкається краю кадру

        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        short_side = min(rw, rh)
        if short_side < P['min_side']:
            continue                                                         # надто вузький
        if max(rw, rh) / short_side > P['max_asp']:
            continue                                                         # надто витягнутий (дорога)
        if area / (rw * rh) < (P['min_rect'] if area > 1800 else 0.40):
            continue                                                         # погано заповнює свій прямокутник
        if area / cv2.contourArea(cv2.convexHull(c)) < (P['large_sol'] if area > 1800 else P['min_sol']):
            continue                                                         # надто «рвана» форма

        sub = np.zeros((bh + 4, bw + 4), np.uint8)
        cv2.drawContours(sub, [c - (x - 2, y - 2)], -1, 255, -1)
        if cv2.distanceTransform(sub, cv2.DIST_L2, 5).max() < P['min_thick']:
            continue                                                         # тонка смуга (стежка, тінь)

        # Векторизація: апроксимація контуру многокутником
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, EPS * peri, True)
        if 4 <= len(approx) <= 26:
            buildings.append(approx)
    return buildings


# ===================== 5. ОБ'ЄДНАННЯ СЕГМЕНТІВ =====================
def merge_segments(contours, groups, shape):
    """Зшиває секції однієї будівлі: контури, у які потрапили точки seeds, малюються разом
    і закриваються ядром k (трохи більшим за зазор між секціями)."""
    if not contours:
        return contours
    used, merged = set(), []
    for g in groups:
        idx = set()
        for x, y in g['seeds']:
            d = [-cv2.pointPolygonTest(c, (float(x), float(y)), True) for c in contours]  # відстань до контуру (<0 — всередині)
            i = int(np.argmin(d))
            if d[i] <= 8:
                idx.add(i)
        if len(idx) < 2:
            continue
        m = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(m, [contours[i] for i in idx], -1, 255, -1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g['k'], g['k'])))
        # якщо частини не зшилися (зазор > k), кожна лишається окремим контуром — нічого не губиться
        for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            merged.append(cv2.approxPolyDP(c, EPS * cv2.arcLength(c, True), True))
        used |= idx
    return [c for i, c in enumerate(contours) if i not in used] + merged


# ===================== 6. МАЛЮВАННЯ =====================
def draw_result(image, buildings, file_name):
    out = image.copy()
    for i, c in enumerate(buildings, 1):
        cv2.drawContours(out, [c], -1, (0, 255, 0), 3)
        M = cv2.moments(c)
        if M['m00']:
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            cv2.putText(out, str(i), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imwrite(str(file_name), out)
    return out


def run_source(name, image, calibrated=True):
    """Цикл обробки одного джерела (image — уже завантажений і, для Wayback, суміщений знімок).
    calibrated=False вимикає локальні зони й зшивання — автоматичний режим без ручних налаштувань.
    Повертає (зображення після корекції кольору, маска, сегментів, будівлі)."""
    cfg = CONFIG[name]
    if cfg['gray_world']:
        image = gray_world(image)                                    # 1. корекція кольору
    zones = cfg['zones'] if calibrated else {}
    mask = image_processing(image, cfg['params'], zones)             # 2. маска
    image_cont = image_contours(mask)                                # 3. контури
    buildings = image_recognition(image_cont, cfg['params'], image.shape)   # 4. валідація
    segments = len(buildings)
    if calibrated:
        buildings = merge_segments(buildings, cfg['merge'], image.shape)    # 5. об'єднання сегментів
    return image, mask, segments, buildings


# ===================== ЗАПУСК =====================
if __name__ == '__main__':
    show = '--no-show' not in sys.argv

    # --- 1. завантаження; Wayback накладається на кадр Google ---
    google_raw = image_read(CONFIG['google']['file'])
    wayback_raw = align_to(google_raw, image_read(CONFIG['wayback']['file']))
    cv2.imwrite(str(OUT / 'wayback_step0_aligned.jpg'), wayback_raw)           # до корекції кольору

    # --- 2-6. обидва джерела у двох режимах: автоматичному (без ручних зон) і каліброваному ---
    # Файли: <name>_step1_mask.png, <name>_step2_canny.png, <name>_final.jpg (калібрований режим)
    #        <name>_auto_mask.png,  <name>_auto_canny.png,  <name>_auto_final.jpg (автоматичний режим)
    res = {}
    for name, raw in [('google', google_raw), ('wayback', wayback_raw)]:
        for calibrated in (False, True):
            img, mask, seg, build = run_source(name, raw, calibrated)
            m_file, c_file, f_file = ((f'{name}_step1_mask.png', f'{name}_step2_canny.png', f'{name}_final.jpg')
                                      if calibrated else
                                      (f'{name}_auto_mask.png', f'{name}_auto_canny.png', f'{name}_auto_final.jpg'))
            cv2.imwrite(str(OUT / m_file), mask)                                  # бінарна маска
            cv2.imwrite(str(OUT / c_file), cv2.Canny(mask, 50, 150))              # межі Canny (ілюстрація)
            res[name, calibrated] = dict(img=img, seg=seg, build=build,
                                         final=draw_result(img, build, OUT / f_file))
    cv2.imwrite(str(OUT / 'wayback_step0_gray_world.jpg'), res['wayback', True]['img'])  # після «сірого світу»

    g, w = res['google', True], res['wayback', True]
    final = [g['final'], w['final']]
    both = np.hstack(final)                                                  # Google ліворуч, Wayback праворуч
    cv2.imwrite(str(OUT / 'kpi_two_sources.jpg'), both)

    # Візуальна перевірка: суміщення знімків і контури Wayback поверх еталона Google
    cv2.imwrite(str(OUT / 'wayback_registration.jpg'), cv2.addWeighted(google_raw, 0.5, wayback_raw, 0.5, 0))
    overlay = w['img'].copy()
    cv2.drawContours(overlay, g['build'], -1, (0, 255, 0), 2)    # еталон (Google) — зелений
    cv2.drawContours(overlay, w['build'], -1, (0, 0, 255), 2)    # Wayback — червоний
    cv2.imwrite(str(OUT / 'wayback_vs_reference.jpg'), overlay)

    print(f"\n{'джерело':<9}{'сегментів':>10}{'будівель':>10}{'авто':>7}")
    for name, note in [('google', '   (еталон)'), ('wayback', '')]:
        r, a = res[name, True], res[name, False]
        print(f"{name:<9}{r['seg']:>10}{len(r['build']):>10}{len(a['build']):>7}{note}")
    same = 'збігається' if len(w['build']) == len(g['build']) else 'НЕ збігається'
    print(f"Кількість будівель Wayback {same} з еталоном Google.")
    g_build, w_build = g['build'], w['build']

    # --- показ: спершу кожне джерело окремо, потім обидва поруч ---
    if show:
        for img, title in [(final[0], f"Google (еталон): {len(g_build)} будівель"),
                           (final[1], f"Wayback: {len(w_build)} будівель")]:
            plt.figure(figsize=(13, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        for ax, img, title in [(axes[0], final[0], f"Google: {len(g_build)} будівель"),
                               (axes[1], final[1], f"Wayback: {len(w_build)} будівель")]:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
