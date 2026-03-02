#!/usr/bin/env python3
"""
stitch_minimaps.py — Собирает глобальную карту из последовательности
перекрывающихся миникарт (скриншотов игровой миникарты).

Использование:
    python3 stitch_minimaps.py [input_dir] [output_file]

    input_dir   — папка с файлами mm_*.png  (по умолчанию: minimaps/minimaps)
    output_file — путь выходного изображения (по умолчанию: full_map.png)

Алгоритм:
  1. Загружает миникарты в натуральном порядке (mm_1, mm_2, …).
  2. Строит маски для синих/белых крестов (маркеры позиции игрока),
     чтобы они не влияли на выравнивание и не попали на итоговую карту.
  3. Вычисляет смещение между каждой парой соседних кадров
     через template matching (поиск шаблона в изображении).
  4. Накладывает все кадры на общий холст, усредняя пиксели
     в зонах перекрытия (маскированные пиксели исключаются).
  5. Убирает мелкие серые точки дорог морфологической фильтрацией.
"""

# --- Импорты ---
# cv2 — OpenCV, библиотека компьютерного зрения (аналог: BufferedImage + ImageIO,
#        но с кучей алгоритмов обработки изображений из коробки)
import cv2
# numpy — библиотека для работы с многомерными массивами (матрицами).
#          Изображение в OpenCV — это numpy-массив shape=(height, width, 3),
#          где 3 канала — Blue, Green, Red.
#          Аналог в Java: двумерный int[][] или BufferedImage.getRaster()
import numpy as np
import os
import re
import sys
# glob — поиск файлов по шаблону (wildcard), аналог Files.newDirectoryStream() в Java
import glob


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def natural_sort_key(path):
    """
    Ключ для натуральной сортировки файлов: mm_1, mm_2, … mm_10, … mm_100
    (а не лексикографической mm_1, mm_10, mm_100, mm_2, …).

    re.split(r'(\\d+)', "mm_123.png") → ["mm_", "123", ".png"]
    Цифровые части превращаем в int, остальное оставляем строками.
    Python сравнивает списки поэлементно — получается правильный порядок.

    Java-аналог: Comparator с ручным парсингом чисел из имени файла.
    """
    # os.path.basename — как Path.getFileName() в Java
    return [int(t) if t.isdigit() else t
            for t in re.split(r'(\d+)', os.path.basename(path))]


def create_cross_mask(img):
    """
    Создаёт бинарную маску (255 = «игнорировать этот пиксель») для
    синих и белых крестов-маркеров на миникарте.

    Идея: переводим изображение из BGR в HSV (Hue-Saturation-Value),
    чтобы удобно фильтровать по цвету, насыщенности и яркости.

    HSV — цветовая модель:
      H (Hue/Оттенок)       — 0..180 в OpenCV (0=красный, 60=зелёный, 120=синий)
      S (Saturation/Насыщ.)  — 0..255 (0=серый, 255=чистый цвет)
      V (Value/Яркость)      — 0..255 (0=чёрный, 255=яркий)

    Пороговые значения подобраны эмпирически (вручную посмотрели пиксели крестов):
      Синий крест — H от 100 до 125, насыщенный (S>120), яркий (V>80)
      Белый крест — любой оттенок, почти без насыщенности (S<35), очень яркий (V>200)
    """
    # cvtColor — конвертация цветового пространства (как ColorConvertOp в Java)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # inRange возвращает бинарную маску: 255 где пиксель попадает в диапазон, 0 где нет.
    # np.array([H_min, S_min, V_min]) — нижняя граница
    # np.array([H_max, S_max, V_max]) — верхняя граница
    blue = cv2.inRange(hsv, np.array([100, 120, 80]),
                             np.array([125, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 200]),
                              np.array([180, 35, 255]))

    # bitwise_or — объединяем обе маски: пиксель маскирован, если он синий ИЛИ белый
    mask = cv2.bitwise_or(blue, white)

    # dilate — «раздуваем» маску на 7×7 пикселей (2 итерации),
    # чтобы захватить полупрозрачные края крестов (антиалиасинг).
    # Аналогия: если крест — это круг, dilate делает его чуть больше.
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return mask


def create_road_mask(img):
    """
    Маска мелких серых точек (дорожная разметка на миникарте).
    Эти точки — мелкий мусор, который портит итоговую карту.

    Алгоритм:
      1. Находим все серые пиксели средней яркости (кандидаты на «дорогу»).
      2. Применяем морфологическое открытие (opening = erosion + dilation) —
         оно убирает мелкие объекты, оставляя крупные.
      3. Вычитаем результат из кандидатов — остаются только мелкие точки.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Конвертируем в оттенки серого (1 канал вместо 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # hsv[:, :, 1] — срез массива: все строки, все столбцы, канал 1 (Saturation).
    # В Java это было бы: for(y) for(x) if(hsv[y][x].saturation < 45 && ...)
    # Но numpy делает это поэлементно для всей матрицы сразу — ОЧЕНЬ быстро.
    candidates = ((hsv[:, :, 1] < 45) &            # низкая насыщенность (серый)
                  (gray > 90) & (gray < 195)        # средняя яркость
                  ).astype(np.uint8) * 255           # bool → 0 или 255

    # Морфологическое открытие ядром 5×5: мелкие точки (< 5px) исчезают
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    # candidates - opened = только то, что исчезло при opening = мелкие точки
    small = cv2.subtract(candidates, opened)
    return small


# ---------------------------------------------------------------------------
# Определение смещения между кадрами
# ---------------------------------------------------------------------------

def find_offset(img1, img2, mask1, mask2):
    """
    Вычисляет смещение (dx, dy) между двумя соседними миникартами.

    Идея: берём центральную часть (50%) первого кадра как «шаблон» (template)
    и ищем, где он лучше всего совпадает со вторым кадром.

    Template matching — это скольжение шаблона по изображению и вычисление
    «похожести» в каждой позиции. Нормализованная кросс-корреляция (NCC)
    даёт значение от -1 до 1, где 1 = идеальное совпадение.

    Аналогия: представь, что кладёшь фрагмент пазла на картинку
    и двигаешь его, пока не найдёшь лучшее совпадение.

    Возвращает (dx, dy, confidence):
      dx, dy — на сколько пикселей сместился кадр 2 относительно кадра 1
      confidence — насколько уверены в совпадении (0..1)
    """
    # Переводим в серый (1 канал), чтобы matchTemplate работал быстрее
    # .astype(np.float32) — приводим к float, как (float) в Java
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Заменяем маскированные пиксели (кресты) средним значением,
    # чтобы они не мешали поиску совпадений.
    # for g, m in ((g1, mask1), (g2, mask2)) — проходим по парам (изобр, маска).
    # В Java: обработать g1+mask1, потом g2+mask2.
    for g, m in ((g1, mask1), (g2, mask2)):
        valid = g[m == 0]                  # пиксели, где маска = 0 (не кресты)
        # np.mean — среднее арифметическое. Заменяем кресты средним цветом фона.
        g[m > 0] = np.mean(valid) if valid.size else 128

    # shape — (height, width). В Java: img.getHeight(), img.getWidth()
    h, w = g1.shape
    # Берём центральные 50%: отступаем по 25% с каждой стороны
    mx, my = w // 4, h // 4                # // — целочисленное деление
    # Срез массива [от:до] — как substring, но для матрицы.
    # g1[my:h-my, mx:w-mx] — вырезаем прямоугольник из центра.
    template = g1[my:h - my, mx:w - mx]

    # Ищем шаблон во втором изображении. Результат — «карта корреляций»:
    # в каждой точке — насколько хорошо шаблон совпал, если его левый верхний
    # угол поставить в эту точку.
    result = cv2.matchTemplate(g2, template, cv2.TM_CCOEFF_NORMED)

    # minMaxLoc находит минимум и максимум в матрице.
    # Нам нужен максимум (лучшее совпадение): conf = значение, loc = координаты.
    # _ — переменные, которые нам не нужны (min_val, min_loc)
    _, conf, _, loc = cv2.minMaxLoc(result)

    # Пересчитываем координаты лучшего совпадения в смещение:
    # Если шаблон был вырезан с отступом (mx, my) от края img1,
    # а найден в позиции loc в img2, то смещение = отступ - найденная позиция.
    dx = mx - loc[0]    # loc[0] = x координата
    dy = my - loc[1]    # loc[1] = y координата
    return dx, dy, conf


# ---------------------------------------------------------------------------
# Постобработка: удаление остаточных точек дорог
# ---------------------------------------------------------------------------

def remove_road_dots(img):
    """
    Заменяет мелкие серые точки дорог на цвет окружающих пикселей.
    Используется inpainting — алгоритм «закрашивания» дыр в изображении
    на основе соседних пикселей (как Content-Aware Fill в Photoshop).
    """
    road = create_road_mask(img)
    # np.count_nonzero — сколько пикселей != 0 в маске.
    # Если 0 — дорожных точек нет, возвращаем изображение как есть.
    if np.count_nonzero(road) == 0:
        return img

    # inpaint — «закрашивает» пиксели, отмеченные в маске road,
    # используя цвет окружающих пикселей в радиусе 3px.
    # INPAINT_TELEA — алгоритм Telea (быстрый, хорошо работает для мелких дыр).
    result = cv2.inpaint(img, road, inpaintRadius=3,
                         flags=cv2.INPAINT_TELEA)
    return result


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def main():
    # sys.argv — аргументы командной строки (как String[] args в Java main).
    # sys.argv[0] = имя скрипта, [1] = первый аргумент, [2] = второй.
    # «X if условие else Y» — тернарный оператор (как «условие ? X : Y» в Java).
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "minimaps/minimaps"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "full_map.png"

    # ==================== ШАГ 1: Поиск файлов ====================
    # glob.glob — находит все файлы по шаблону с wildcard *.
    # sorted(..., key=...) — сортировка с кастомным компаратором.
    # В Java: Files.list(dir).sorted(naturalComparator).collect(toList())
    files = sorted(
        glob.glob(os.path.join(input_dir, "mm_*.png")),
        key=natural_sort_key,
    )
    if not files:
        # f"..." — форматированная строка (как String.format() в Java)
        print(f"ERROR: no mm_*.png files found in '{input_dir}'")
        sys.exit(1)

    n = len(files)    # len() — как .size() или .length в Java
    print(f"[1/5] Found {n} minimaps in '{input_dir}'")

    # ==================== ШАГ 2: Загрузка изображений ====================
    print("[2/5] Loading images & building masks …")
    images = []         # List<Mat> images = new ArrayList<>()
    cross_masks = []    # List<Mat> crossMasks = new ArrayList<>()
    for f in files:     # for (String f : files)
        img = cv2.imread(f)    # ImageIO.read(new File(f))
        if img is None:        # null-check
            print(f"  WARNING: cannot read {f}, skipping")
            continue
        images.append(img)                           # images.add(img)
        cross_masks.append(create_cross_mask(img))   # crossMasks.add(...)

    n = len(images)
    # images[0].shape → (height, width, channels), напр. (251, 272, 3).
    # [:2] — берём только первые 2 элемента (height, width), каналы не нужны.
    # ih, iw = ... — «распаковка кортежа»: ih получает height, iw получает width.
    # В Java: int ih = img.rows(); int iw = img.cols();
    ih, iw = images[0].shape[:2]
    print(f"       {n} images loaded, each {iw}×{ih}")

    # ==================== ШАГ 3: Вычисление смещений ====================
    # Для каждой пары соседних кадров (i, i+1) определяем, на сколько пикселей
    # сместился второй кадр относительно первого. Из попарных смещений
    # собираем абсолютные позиции всех кадров на будущем холсте.
    print("[3/5] Computing offsets …")
    # Список абсолютных позиций. Первый кадр — в точке (0, 0).
    positions = [(0.0, 0.0)]      # List<double[]> positions; positions.add({0,0})
    # range(n-1) — от 0 до n-2 включительно (как for(int i=0; i<n-1; i++))
    for i in range(n - 1):
        # Находим смещение между кадром i и кадром i+1
        dx, dy, conf = find_offset(
            images[i], images[i + 1],
            cross_masks[i], cross_masks[i + 1],
        )
        # positions[-1] — последний элемент списка (как positions.get(size()-1))
        px, py = positions[-1]
        # Абсолютная позиция кадра i+1 = позиция кадра i + смещение
        positions.append((px + dx, py + dy))

        # Предупреждаем, если уверенность в совпадении слишком низкая
        if conf < 0.40:
            print(f"  WARNING: low confidence {conf:.3f} at "
                  f"frame {i}→{i+1} (dx={dx}, dy={dy})")

    # Нормализуем координаты: сдвигаем всё так, чтобы минимум был в (0, 0).
    # Иначе могут быть отрицательные координаты, а у холста индексы от 0.
    # «p[0] for p in positions» — генератор (аналог stream().map(p -> p[0]))
    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    # List comprehension — создание нового списка из старого с преобразованием.
    # В Java: positions.stream().map(p -> new int[]{...}).collect(toList())
    int_positions = [
        (int(round(p[0] - min_x)), int(round(p[1] - min_y)))
        for p in positions
    ]

    # Размер холста = самая дальняя позиция + размер одного кадра
    canvas_w = max(p[0] for p in int_positions) + iw
    canvas_h = max(p[1] for p in int_positions) + ih
    print(f"       Canvas size: {canvas_w}×{canvas_h}")

    # ==================== ШАГ 4: Склейка (композитинг) ====================
    # Создаём пустой холст и «вес» (сколько кадров наложилось на каждый пиксель).
    # В зонах перекрытия несколько кадров покрывают один и тот же участок —
    # мы суммируем их значения и потом делим на количество (среднее).
    print("[4/5] Compositing global map …")
    # np.zeros — массив нулей заданного размера.
    # (canvas_h, canvas_w, 3) — высота × ширина × 3 канала (BGR).
    # dtype=np.float64 — double, чтобы накапливать суммы без потери точности.
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for i in range(n):
        img = images[i]
        # Объединённая маска: кресты + дорожные точки
        cmask = cv2.bitwise_or(cross_masks[i], create_road_mask(img))
        # valid = 1.0 где пиксель валидный (не маскирован), 0.0 где маскирован
        valid = (cmask == 0).astype(np.float64)

        px, py = int_positions[i]
        # Для каждого из 3 цветовых каналов (B, G, R):
        for c in range(3):
            # canvas[py:py+ih, px:px+iw] — прямоугольный регион холста,
            # куда попадает текущий кадр. Это «срез» (slice) — ссылка на
            # часть массива, а не копия. Изменения пишутся прямо в canvas.
            # += — добавляем значения пикселей, умноженные на маску valid.
            # Маскированные пиксели (valid=0) не вносят вклада.
            canvas[py:py + ih, px:px + iw, c] += (
                img[:, :, c].astype(np.float64) * valid
            )
        # Увеличиваем счётчик перекрытий для этого региона
        weight[py:py + ih, px:px + iw] += valid

        # Прогресс каждые 50 кадров
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"       placed {i + 1}/{n}")

    # Делим сумму на количество → получаем среднее значение.
    # np.maximum(weight, 1.0) — чтобы не делить на 0 (где ни один кадр не попал).
    weight = np.maximum(weight, 1.0)
    for c in range(3):
        canvas[:, :, c] /= weight

    # Обрезаем до [0, 255] и приводим к uint8 (обычный формат изображения)
    result = np.clip(canvas, 0, 255).astype(np.uint8)

    # ==================== ШАГ 5: Финальная очистка ====================
    print("[5/5] Cleaning up residual road dots …")
    result = remove_road_dots(result)

    # ==================== Сохранение ====================
    cv2.imwrite(output_file, result)    # ImageIO.write(img, "png", file)
    print(f"Done → '{output_file}'  ({canvas_w}×{canvas_h})")


# Стандартная точка входа Python-скрипта.
# __name__ == "__main__" — True, если файл запущен напрямую (python3 script.py),
# False, если импортирован из другого модуля (import stitch_minimaps).
# В Java аналог — public static void main(String[] args).
if __name__ == "__main__":
    main()
