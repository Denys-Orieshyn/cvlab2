# ----------- Перевірка збігу контурів з еталоном (числа таблиць 3.1–3.2 звіту) -----------------

'''
Функціонал:
порівнює контури будівель Wayback (і варіантів обробки) з еталоном — каліброваним результатом Google.
IoU = площа перетину двох контурів / площа їх об'єднання (1 — повний збіг, 0 — не перетинаються).
Два критерії: «будівлю знайдено» — IoU >= 0.25 (та сама будівля, форма контуру може відрізнятися);
«форма контуру збігається» — IoU >= 0.5. Пари добираються за спаданням IoU (кожен контур — не більше
однієї пари). F1 = 2·TP / (2·TP + FP + FN).
'''

import numpy as np
import cv2

import main as m

m.print = lambda *a, **k: None          # не виводити рядок суміщення з main.align_to


def run(image, params, zones, merge, gray):
    """Повний цикл обробки з довільним набором операцій (для таблиці внеску кроків)."""
    if gray:
        image = m.gray_world(image)
    mask = m.image_processing(image, params, zones)
    buildings = m.image_recognition(m.image_contours(mask), params, image.shape)
    segments = len(buildings)
    if merge:
        buildings = m.merge_segments(buildings, merge, image.shape)
    return segments, buildings


def match(ref, test, shape, thr=0.5):
    """Пари контурів за спаданням IoU (не менше thr); повертає (TP, FP, FN, F1, матриця IoU, пари)."""
    def fill(c):
        z = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(z, [c], -1, 1, -1)
        return z > 0
    R, T = [fill(c) for c in ref], [fill(c) for c in test]
    iou = np.array([[(a & b).sum() / (a | b).sum() for b in T] for a in R]) if R and T else np.zeros((len(R), len(T)))
    used_r, used_t = set(), set()
    for k in np.argsort(-iou, axis=None):
        i, j = divmod(int(k), iou.shape[1])
        if iou[i, j] < thr:
            break
        if i not in used_r and j not in used_t:
            used_r.add(i); used_t.add(j)
    tp = len(used_r); fp = len(T) - tp; fn = len(R) - tp
    return tp, fp, fn, 2 * tp / (2 * tp + fp + fn), iou, used_r, used_t


if __name__ == '__main__':
    google = m.image_read(m.CONFIG['google']['file'])
    wayback = m.align_to(google, m.image_read(m.CONFIG['wayback']['file']))
    G, W = m.CONFIG['google'], m.CONFIG['wayback']
    ref = run(google, G['params'], G['zones'], G['merge'], False)[1]

    variants = [
        ('Google', 'повна обробка', google, G['params'], G['zones'], G['merge'], False),
        ('Google', 'без зшивання секцій', google, G['params'], G['zones'], None, False),
        ('Google', 'без локальних зон', google, G['params'], {}, G['merge'], False),
        ('Google', 'автоматичний режим', google, G['params'], {}, None, False),
        ('Wayback', 'повна обробка', wayback, W['params'], W['zones'], W['merge'], True),
        ('Wayback', 'налаштування Google', wayback, G['params'], G['zones'], G['merge'], False),
        ('Wayback', 'без корекції «сірий світ»', wayback, W['params'], W['zones'], W['merge'], False),
        ('Wayback', 'базові пороги Google', wayback, G['params'], W['zones'], W['merge'], True),
        ('Wayback', 'без локальних зон', wayback, W['params'], {}, W['merge'], True),
        ('Wayback', 'без зшивання секцій', wayback, W['params'], W['zones'], None, True),
        ('Wayback', 'автоматичний режим', wayback, W['params'], {}, None, True),
    ]
    print(f"{'знімок':<8} {'варіант':<28}{'сегм.':>6}{'буд.':>6}{'TP':>4}{'FP':>4}{'FN':>4}{'F1':>6}")
    for src, name, img, p, z, mg, gray in variants:
        seg, b = run(img, p, z, mg, gray)
        tp, fp, fn, f1, *_ = match(ref, b, img.shape)
        print(f"{src:<8} {name:<28}{seg:>6}{len(b):>6}{tp:>4}{fp:>4}{fn:>4}{f1:>6.2f}")

    # Підсумок для Wayback за двома критеріями та розбір контурів без пари
    seg, b = run(wayback, W['params'], W['zones'], W['merge'], True)
    found = match(ref, b, wayback.shape, thr=0.25)
    shape_ok = match(ref, b, wayback.shape, thr=0.5)
    print(f"\nWayback: будівель знайдено (IoU >= 0.25): {found[0]} з {len(ref)}; "
          f"форма контуру збігається (IoU >= 0.5): {shape_ok[0]} з {len(ref)}, F1 = {shape_ok[3]:.2f}")
    iou, used_r, used_t = found[4], found[5], found[6]
    for j in sorted(set(range(len(b))) - used_t):
        i = int(np.argmax(iou[:, j]))
        print(f"  контур Wayback {j + 1:2d} без пари: найближчий еталонний {i + 1:2d}, IoU = {iou[i, j]:.2f}")
