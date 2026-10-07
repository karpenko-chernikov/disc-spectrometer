#!/usr/bin/env python3
"""
Задача 2 IYPT — спектрометр на CD/DVD.
Один скрипт: диалог, калибровка, обработка фото, графики и метрики.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Без окон GUI — иначе на macOS Python падает при «открыть окна снова»
os.environ["MPLBACKEND"] = "Agg"

import cv2
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

КОРЕНЬ = Path(__file__).resolve().parent
ПАПКА_СЕССИЙ = КОРЕНЬ / "сессии"
ПАПКА_СПРАВОЧНИКОВ = КОРЕНЬ / "справочники"

ПОДПАПКИ_СЕССИИ = (
    "вход",
    "обработанные",
    "результаты",
    "калибровка",
    "калибровка/эталоны",
)

РАСШИРЕНИЯ_ФОТО = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

СПРАВОЧНИКИ = {
    "1": ("неон", "неон.csv"),
    "2": ("водород", "водород.csv"),
    "3": ("гелий", "гелий.csv"),
}

ФОРМУЛЫ = {
    "1": ("линейная", 1),
    "2": ("квадратичная", 2),
    "3": ("кубическая", 3),
}


# ---------------------------------------------------------------------------
# Утилиты диалога
# ---------------------------------------------------------------------------

def печать(текст: str = "") -> None:
    print(текст)


def спросить(приглашение: str, по_умолчанию: str | None = None) -> str:
    if по_умолчанию is not None:
        сырой = input(f"{приглашение} [{по_умолчанию}]: ").strip()
        return сырой if сырой else по_умолчанию
    return input(f"{приглашение}: ").strip()


def спросить_выбор(приглашение: str, варианты: dict[str, str]) -> str:
    while True:
        печать(приглашение)
        for ключ, подпись in варианты.items():
            печать(f"  [{ключ}] {подпись}")
        ответ = input("> ").strip()
        if ответ in варианты:
            return ответ
        печать("Неизвестный пункт, попробуйте ещё раз.\n")


def да_нет(вопрос: str, по_умолчанию: bool = True) -> bool:
    подсказка = "д/н" if по_умолчанию else "д/н"
    дефолт = "д" if по_умолчанию else "н"
    ответ = спросить(f"{вопрос} ({подсказка})", дефолт).lower()
    return ответ.startswith("д") or ответ in {"y", "yes", "д", "да"}


# ---------------------------------------------------------------------------
# Сессии
# ---------------------------------------------------------------------------

def обеспечить_структуру_сессии(сессия: Path) -> None:
    for под in ПОДПАПКИ_СЕССИИ:
        (сессия / под).mkdir(parents=True, exist_ok=True)
    описание = сессия / "описание.txt"
    if not описание.exists():
        описание.write_text(
            "Папка эксперимента спектрометра.\n"
            "Опишите установку: диск (CD/DVD), щель, угол, камера, дата.\n"
            "Новые фото кладите в папку «вход».\n",
            encoding="utf-8",
        )


def список_сессий() -> list[Path]:
    if not ПАПКА_СЕССИЙ.exists():
        return []
    return sorted(
        [p for p in ПАПКА_СЕССИЙ.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )


def создать_сессию() -> Path:
    ПАПКА_СЕССИЙ.mkdir(parents=True, exist_ok=True)
    сегодня = datetime.now().strftime("%Y-%m-%d")
    печать(
        "\nНовая папка эксперимента — это один набор фото\n"
        "с одной и той же настройкой установки (диск, щель, камера).\n"
        "Если подвинули камеру или сменили диск — лучше создать новую папку.\n"
    )
    имя = спросить("Как назвать папку", f"{сегодня}_dvd_неон")
    имя = имя.replace(" ", "_")
    путь = ПАПКА_СЕССИЙ / имя
    if путь.exists():
        печать(f"Такая папка уже есть: {путь}")
    else:
        обеспечить_структуру_сессии(путь)
        печать(f"Создана папка: {путь}")
        печать(f"Положите фото сюда: {путь / 'вход'}")
    return путь


def краткий_статус_папки(сессия: Path) -> str:
    есть_шкала = путь_калибровки(сессия).exists()
    n_in = len(список_фото(сессия / "вход"))
    n_res = 0
    res = сессия / "результаты"
    if res.exists():
        n_res = sum(1 for p in res.iterdir() if p.is_dir())
    шкала = "шкала нм готова" if есть_шкала else "шкала нм ещё не настроена"
    фото = f"новых фото: {n_in}" if n_in else "новых фото нет"
    готово = f"уже обработано: {n_res}" if n_res else "обработанных пока нет"
    return f"{шкала}; {фото}; {готово}"


def выбрать_сессию(арг: str | None = None) -> Path:
    if арг:
        путь = Path(арг)
        if not путь.is_absolute():
            путь = КОРЕНЬ / путь
        обеспечить_структуру_сессии(путь)
        return путь

    печать(
        "\nСначала выберите папку эксперимента.\n"
        "Это просто папка с вашими фото одной настройки установки.\n"
        "Внутри неё программа сама ведёт папки «вход», «результаты» и шкалу длин волн.\n"
    )

    сессии = список_сессий()
    if not сессии:
        печать("Папок экспериментов ещё нет — создадим первую.")
        return создать_сессию()

    варианты: dict[str, str] = {"0": "Создать новую папку эксперимента"}
    for i, s in enumerate(сессии, start=1):
        варианты[str(i)] = f"{s.name}  —  {краткий_статус_папки(s)}"

    выбор = спросить_выбор("Какую папку открыть?", варианты)
    if выбор == "0":
        return создать_сессию()
    return сессии[int(выбор) - 1]


def путь_калибровки(сессия: Path) -> Path:
    return сессия / "калибровка" / "калибровка.json"


def статус_калибровки(сессия: Path) -> str:
    p = путь_калибровки(сессия)
    if not p.exists():
        return "ещё не настроена (пиксели пока без нанометров)"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        формула = data.get("формула", "?")
        n = len(data.get("пары", []))
        return f"готова ({формула}, по {n} эталонным линиям)"
    except Exception:
        return "файл шкалы повреждён — лучше настроить заново"


def подсказка_следующего_шага(сессия: Path) -> str:
    n_in = len(список_фото(сессия / "вход"))
    есть_шкала = путь_калибровки(сессия).exists()
    if n_in == 0 and not есть_шкала:
        return (
            f"Сейчас во «вход» пусто.\n"
            f"1) Положите фото спектров сюда:\n   {сессия / 'вход'}\n"
            f"2) Запустите снова и выберите пункт про настройку шкалы / обработку."
        )
    if n_in > 0 and not есть_шкала:
        return (
            f"Есть {n_in} новых фото, но шкала длин волн ещё не настроена.\n"
            f"Обычно дальше: пункт [3] — настроить шкалу и сразу обработать."
        )
    if n_in > 0 and есть_шкала:
        return (
            f"Шкала уже есть, во «вход» лежит {n_in} фото.\n"
            f"Обычно дальше: пункт [2] — обработать новые фото."
        )
    return (
        "Новых фото во «вход» нет.\n"
        "Положите снимки в папку «вход» или настройте шкалу заново, если меняли установку."
    )

def список_фото(папка: Path) -> list[Path]:
    if not папка.exists():
        return []
    files = [
        p
        for p in sorted(папка.iterdir())
        if p.is_file() and p.suffix.lower() in РАСШИРЕНИЯ_ФОТО
    ]
    return files


# ---------------------------------------------------------------------------
# Загрузка справочников
# ---------------------------------------------------------------------------

@dataclass
class ЛинияСправочника:
    длина_волны_нм: float
    подпись: str
    яркость: str


def загрузить_справочник(имя_файла: str) -> list[ЛинияСправочника]:
    путь = ПАПКА_СПРАВОЧНИКОВ / имя_файла
    линии: list[ЛинияСправочника] = []
    with путь.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            линии.append(
                ЛинияСправочника(
                    длина_волны_нм=float(row["длина_волны_нм"]),
                    подпись=row.get("подпись", ""),
                    яркость=row.get("яркость", ""),
                )
            )
    return линии


# ---------------------------------------------------------------------------
# Обработка изображения
# ---------------------------------------------------------------------------

@dataclass
class Профиль:
    ось: np.ndarray          # координата вдоль спектра (пиксели)
    I: np.ndarray
    Ir: np.ndarray
    Ig: np.ndarray
    Ib: np.ndarray
    roi: tuple[int, int, int, int]  # x0, x1, y0, y1
    ось_дисперсии: str       # "y" или "x"
    исходник: np.ndarray     # BGR


@dataclass
class Пик:
    пиксель: float
    интенсивность: float
    fwhm_px: float
    канал: str = "I"


def загрузить_изображение(путь: Path) -> np.ndarray:
    """BGR uint8 (H, W, 3) через OpenCV — корректно с путями Unicode."""
    data = np.fromfile(str(путь), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось открыть изображение: {путь}")
    return img


def найти_roi(img: np.ndarray) -> tuple[int, int, int, int]:
    """ROI спектра: яркость + морфология OpenCV (чище, чем простой порог)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # лёгкое сглаживание, чтобы шум не рвал маску
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr_val = max(int(blur.max() * 0.12), 8)
    _, mask = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = gray.shape
        return 0, w, 0, h

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    pad = 6
    h, w = gray.shape
    x0 = max(0, x0 - pad)
    x1 = min(w, x1 + pad)
    y0 = max(0, y0 - pad)
    y1 = min(h, y1 + pad)

    # узкая вертикальная колонка — расширим по X для устойчивого усреднения
    ширина = x1 - x0
    высота = y1 - y0
    if высота > ширина * 1.5:
        # центр масс по яркости устойчивее середины bbox
        col_sums = gray[:, x0:x1].sum(axis=0).astype(np.float64)
        if col_sums.sum() > 0:
            weights = col_sums / col_sums.sum()
            cx = x0 + int(np.round(np.sum(np.arange(ширина) * weights)))
        else:
            cx = (x0 + x1) // 2
        half = max(ширина // 2, 10)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half + 1)
    return x0, x1, y0, y1


def определить_ось_дисперсии(roi: tuple[int, int, int, int]) -> str:
    x0, x1, y0, y1 = roi
    return "y" if (y1 - y0) >= (x1 - x0) else "x"


def построить_профиль(img: np.ndarray) -> Профиль:
    x0, x1, y0, y1 = найти_roi(img)
    crop = img[y0:y1, x0:x1]
    # небольшое размытие поперёк спектра снижает шум камеры, линии вдоль оси почти не смазывает
    if определить_ось_дисперсии((x0, x1, y0, y1)) == "y":
        crop_f = cv2.GaussianBlur(crop, (3, 1), 0)
    else:
        crop_f = cv2.GaussianBlur(crop, (1, 3), 0)

    b, g, r = cv2.split(crop_f.astype(np.float64))
    gray = cv2.cvtColor(crop_f, cv2.COLOR_BGR2GRAY).astype(np.float64)
    ось_дисп = определить_ось_дисперсии((x0, x1, y0, y1))

    if ось_дисп == "y":
        Ir = r.mean(axis=1)
        Ig = g.mean(axis=1)
        Ib = b.mean(axis=1)
        I = gray.mean(axis=1)
        ось = np.arange(y0, y1, dtype=np.float64)
    else:
        Ir = r.mean(axis=0)
        Ig = g.mean(axis=0)
        Ib = b.mean(axis=0)
        I = gray.mean(axis=0)
        ось = np.arange(x0, x1, dtype=np.float64)

    return Профиль(
        ось=ось,
        I=I,
        Ir=Ir,
        Ig=Ig,
        Ib=Ib,
        roi=(x0, x1, y0, y1),
        ось_дисперсии=ось_дисп,
        исходник=img,
    )


def найти_пики(профиль: Профиль, канал: str = "I") -> list[Пик]:
    данные = {
        "I": профиль.I,
        "R": профиль.Ir,
        "G": профиль.Ig,
        "B": профиль.Ib,
    }[канал]
    сглаж = gaussian_filter1d(данные, sigma=1.2)
    фон = np.percentile(сглаж, 20)
    высота = max((сглаж.max() - фон) * 0.12, сглаж.max() * 0.05, 1.0)
    расстояние = max(3, len(сглаж) // 80)

    idxs, props = find_peaks(
        сглаж,
        height=фон + высота,
        distance=расстояние,
        prominence=высота * 0.5,
    )
    if len(idxs) == 0:
        return []

    widths, _, _, _ = peak_widths(сглаж, idxs, rel_height=0.5)
    пики: list[Пик] = []
    for i, idx in enumerate(idxs):
        # субпиксель: парабола по 3 точкам
        if 0 < idx < len(сглаж) - 1:
            y0, y1, y2 = сглаж[idx - 1], сглаж[idx], сглаж[idx + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))
        else:
            delta = 0.0
        пикс = float(профиль.ось[idx] + delta)
        пики.append(
            Пик(
                пиксель=пикс,
                интенсивность=float(данные[idx]),
                fwhm_px=float(widths[i]),
                канал=канал,
            )
        )
    пики.sort(key=lambda p: p.пиксель)
    return пики


# ---------------------------------------------------------------------------
# Калибровка
# ---------------------------------------------------------------------------

@dataclass
class ПараКалибровки:
    пиксель: float
    длина_волны_нм: float
    подпись: str
    уверенность: str


def _оценка_совпадения(
    пики_px: np.ndarray,
    линии_нм: np.ndarray,
    scale: float,
    offset: float,
) -> tuple[float, list[tuple[int, int, float]]]:
    """Сопоставляет пики со справочником при λ ≈ scale*p + offset."""
    пары: list[tuple[int, int, float]] = []
    used_line: set[int] = set()
    сум_ош = 0.0
    for i, p in enumerate(пики_px):
        pred = scale * p + offset
        dists = np.abs(линии_нм - pred)
        j = int(np.argmin(dists))
        if j in used_line:
            continue
        err = float(dists[j])
        # допуск: 8 нм или 2% диапазона
        if err > 8.0:
            continue
        used_line.add(j)
        пары.append((i, j, err))
        сум_ош += err * err
    if len(пары) < 2:
        return 1e18, пары
    штраф = сум_ош / len(пары) + 0.5 * (len(пики_px) - len(пары))
    return штраф, пары


def автосопоставление(
    пики: list[Пик],
    справочник: list[ЛинияСправочника],
) -> list[ПараКалибровки]:
    if len(пики) < 2 or len(справочник) < 2:
        return []

    # берём самые яркие пики
    яркие = sorted(пики, key=lambda p: p.интенсивность, reverse=True)[:18]
    яркие = sorted(яркие, key=lambda p: p.пиксель)
    px = np.array([p.пиксель for p in яркие], dtype=np.float64)

    # яркие линии справочника
    вес = {"очень высокая": 3, "высокая": 2, "средняя": 1}
    линии = sorted(справочник, key=lambda L: вес.get(L.яркость, 1), reverse=True)[:20]
    линии = sorted(линии, key=lambda L: L.длина_волны_нм)
    lam = np.array([L.длина_волны_нм for L in линии], dtype=np.float64)

    best_score = 1e18
    best_pairs: list[tuple[int, int, float]] = []
    best_scale = 0.0
    best_offset = 0.0

    # перебор пар якорей пик↔линия для оценки scale/offset
    n_p, n_l = len(px), len(lam)
    for i1 in range(n_p):
        for i2 in range(i1 + 1, n_p):
            dp = px[i2] - px[i1]
            if abs(dp) < 5:
                continue
            for j1 in range(n_l):
                for j2 in range(j1 + 1, n_l):
                    dl = lam[j2] - lam[j1]
                    scale = dl / dp
                    # типичная дисперсия DIY: 0.05 … 2 нм/пикс
                    if not (0.05 <= abs(scale) <= 2.5):
                        continue
                    offset = lam[j1] - scale * px[i1]
                    score, pairs = _оценка_совпадения(px, lam, scale, offset)
                    if score < best_score and len(pairs) >= 3:
                        best_score = score
                        best_pairs = pairs
                        best_scale = scale
                        best_offset = offset

    # запасной план: если мало пар — ослабить требование
    if len(best_pairs) < 3:
        for i1 in range(n_p):
            for i2 in range(i1 + 1, n_p):
                dp = px[i2] - px[i1]
                if abs(dp) < 5:
                    continue
                for j1 in range(n_l):
                    for j2 in range(j1 + 1, n_l):
                        scale = (lam[j2] - lam[j1]) / dp
                        if not (0.05 <= abs(scale) <= 2.5):
                            continue
                        offset = lam[j1] - scale * px[i1]
                        score, pairs = _оценка_совпадения(px, lam, scale, offset)
                        if score < best_score and len(pairs) >= 2:
                            best_score = score
                            best_pairs = pairs
                            best_scale = scale
                            best_offset = offset

    результат: list[ПараКалибровки] = []
    for i, j, err in sorted(best_pairs, key=lambda t: яркие[t[0]].пиксель):
        уверенность = "высокая" if err < 2.0 else ("средняя" if err < 5.0 else "низкая")
        результат.append(
            ПараКалибровки(
                пиксель=float(яркие[i].пиксель),
                длина_волны_нм=float(линии[j].длина_волны_нм),
                подпись=линии[j].подпись,
                уверенность=уверенность,
            )
        )
    # направление: если scale < 0, красное и синее «перевёрнуты» — это ок
    _ = best_scale, best_offset
    return результат


def подобрать_полином(пары: list[ПараКалибровки], степень: int) -> np.ndarray:
    x = np.array([p.пиксель for p in пары], dtype=np.float64)
    y = np.array([p.длина_волны_нм for p in пары], dtype=np.float64)
    степень = min(степень, len(пары) - 1)
    return np.polyfit(x, y, степень)


def применить_калибровку(пиксели: np.ndarray, coeffs: list[float]) -> np.ndarray:
    return np.polyval(np.array(coeffs, dtype=np.float64), пиксели)


def ошибки_калибровки(пары: list[ПараКалибровки], coeffs: list[float]) -> dict:
    pred = применить_калибровку(
        np.array([p.пиксель for p in пары]), coeffs
    )
    true = np.array([p.длина_волны_нм for p in пары])
    err = np.abs(pred - true)
    return {
        "средняя": float(err.mean()),
        "макс": float(err.max()),
        "по_точкам": [float(e) for e in err],
    }


def сохранить_калибровку(
    сессия: Path,
    формула: str,
    coeffs: list[float],
    пары: list[ПараКалибровки],
    эталон: str,
    кадр: str,
    ось_дисперсии: str,
) -> Path:
    ошибки = ошибки_калибровки(пары, coeffs)
    data = {
        "формула": формула,
        "коэффициенты": [float(c) for c in coeffs],
        "эталон": эталон,
        "кадр": кадр,
        "ось_дисперсии": ось_дисперсии,
        "дата": datetime.now().isoformat(timespec="seconds"),
        "пары": [
            {
                "пиксель": p.пиксель,
                "длина_волны_нм": p.длина_волны_нм,
                "подпись": p.подпись,
                "уверенность": p.уверенность,
            }
            for p in пары
        ],
        "ошибка_нм": {
            "средняя": ошибки["средняя"],
            "макс": ошибки["макс"],
        },
    }
    путь = путь_калибровки(сессия)
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return путь


def загрузить_калибровку(сессия: Path) -> Optional[dict]:
    p = путь_калибровки(сессия)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Визуальный стиль графиков (докладный)
# ---------------------------------------------------------------------------

ЦВЕТА = {
    "фон": "#F4F0E8",
    "панель": "#FFFCF7",
    "чернила": "#1B1916",
    "приглушённый": "#6E675C",
    "сетка": "#E4DDD0",
    "акцент": "#B84E2B",
    "линия": "#2A2622",
    "r": "#C0392B",
    "g": "#2E7D4F",
    "b": "#2F5D8C",
}


def настроить_стиль_графиков() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": ЦВЕТА["фон"],
            "axes.facecolor": ЦВЕТА["панель"],
            "axes.edgecolor": ЦВЕТА["чернила"],
            "axes.labelcolor": ЦВЕТА["чернила"],
            "axes.titlecolor": ЦВЕТА["чернила"],
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": ЦВЕТА["приглушённый"],
            "ytick.color": ЦВЕТА["приглушённый"],
            "text.color": ЦВЕТА["чернила"],
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.color": ЦВЕТА["сетка"],
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": ЦВЕТА["фон"],
        }
    )


def длина_волны_в_rgb(нм: float) -> tuple[float, float, float]:
    """Приблизительный цвет видимого спектра (для заливки и линий)."""
    w = float(нм)
    if w < 380 or w > 780:
        return (0.55, 0.55, 0.55)
    if w < 440:
        t = (w - 380) / 60
        r, g, b = 1 - t, 0.0, 1.0
    elif w < 490:
        t = (w - 440) / 50
        r, g, b = 0.0, t, 1.0
    elif w < 510:
        t = (w - 490) / 20
        r, g, b = 0.0, 1.0, 1 - t
    elif w < 580:
        t = (w - 510) / 70
        r, g, b = t, 1.0, 0.0
    elif w < 645:
        t = (w - 580) / 65
        r, g, b = 1.0, 1 - t, 0.0
    else:
        # глубокий красный: чуть темнеет к ИК, чтобы линии не были одним тоном
        t = min(1.0, (w - 645) / 55)
        r, g, b = 1.0, 0.0, 0.0
        r = 1.0 - 0.25 * t
    if w < 420:
        factor = 0.35 + 0.65 * (w - 380) / 40
    elif w > 700:
        factor = 0.35 + 0.65 * (780 - w) / 80
    else:
        factor = 1.0
    return (r * factor, g * factor, b * factor)


def _цветная_полоска(ax, xmin: float, xmax: float, в_нм: bool) -> None:
    if not в_нм or xmax <= xmin:
        ax.axhspan(0, 1, color="#D8CFC0")
        return
    grid = np.linspace(xmin, xmax, 512)
    rgb = np.array([[длина_волны_в_rgb(float(v)) for v in grid]], dtype=np.float64)
    ax.imshow(
        rgb,
        aspect="auto",
        extent=(xmin, xmax, 0, 1),
        origin="lower",
        interpolation="bilinear",
    )


def _оформить_оси(ax, заголовок: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(заголовок, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color=ЦВЕТА["сетка"], lw=0.9)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.1)
        ax.spines[spine].set_color(ЦВЕТА["чернила"])


def _подпись_вне_графика(
    ax,
    текст: str,
    *,
    где: str = "верх_справа",
) -> None:
    """Текст в углу осей — не пересекается с линиями данных."""
    места = {
        "верх_справа": (0.98, 0.96, "right", "top"),
        "верх_слева": (0.02, 0.96, "left", "top"),
        "низ_справа": (0.98, 0.04, "right", "bottom"),
        "низ_слева": (0.02, 0.04, "left", "bottom"),
    }
    x, y, ha, va = места[где]
    ax.text(
        x,
        y,
        текст,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.5,
        color=ЦВЕТА["чернила"],
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#FFFDF8",
            "edgecolor": "#D0C6B6",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        zorder=10,
        clip_on=False,
    )


def _легенда_снаружи(ax, fig=None, **kwargs) -> None:
    """Легенда справа от графика — не наезжает на кривые и линии."""
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="#D0C6B6",
        facecolor="#FFFDF8",
        framealpha=0.96,
        **kwargs,
    )
    if fig is not None:
        fig.subplots_adjust(right=0.82)


def _подписи_пиков(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    в_нм: bool,
    максимум_подписей: int | None = None,
) -> None:
    """Подписывает все пики; соседние поднимает на разные уровни без наложений."""
    if len(xs) == 0:
        return
    индексы = list(range(len(xs)))
    if максимум_подписей is not None and len(индексы) > максимум_подписей:
        индексы = sorted(
            индексы, key=lambda j: ys[j], reverse=True
        )[:максимум_подписей]

    ymax = float(np.max(ys)) if len(ys) else 1.0
    order_x = sorted(индексы, key=lambda j: xs[j])
    # уровни в points; для близких по X берём следующий свободный
    уровни = [14, 30, 46, 62, 22, 38, 54]
    занято: list[tuple[float, int]] = []  # (x, индекс_уровня)
    мин_разрыв_x = (float(np.ptp(xs)) * 0.035) if len(xs) > 1 else 1.0

    for i in order_x:
        x, y = float(xs[i]), float(ys[i])
        текст = f"{x:.1f}" if в_нм else f"{x:.0f}"
        уровень_idx = 0
        for candidate in range(len(уровни)):
            конфликт = any(
                abs(x - x0) < мин_разрыв_x and lvl == candidate
                for x0, lvl in занято
            )
            if not конфликт:
                уровень_idx = candidate
                break
            уровень_idx = candidate
        занято.append((x, уровень_idx))
        вверх = уровни[уровень_idx]
        ax.annotate(
            текст,
            xy=(x, y),
            xytext=(0, вверх),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=ЦВЕТА["чернила"],
            fontweight="normal",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "#FFFDF8",
                "edgecolor": "#D8CFC0",
                "linewidth": 0.55,
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": ЦВЕТА["приглушённый"],
                "lw": 0.5,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            zorder=6,
        )
        маркер = длина_волны_в_rgb(x) if в_нм else ЦВЕТА["акцент"]
        ax.plot(
            [x],
            [y],
            marker="o",
            ms=4.0,
            color=маркер,
            markeredgecolor="#FFFDF8",
            markeredgewidth=0.8,
            zorder=5,
        )
    ax.set_ylim(0, ymax * 1.48)


# ---------------------------------------------------------------------------
# Сохранение результатов и графики
# ---------------------------------------------------------------------------

настроить_стиль_графиков()


def сохранить_csv_профиль(путь: Path, профиль: Профиль, wavelengths: np.ndarray | None) -> None:
    with путь.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["пиксель", "I", "Ir", "Ig", "Ib"]
        if wavelengths is not None:
            header.append("длина_волны_нм")
        w.writerow(header)
        for i in range(len(профиль.ось)):
            row = [
                f"{профиль.ось[i]:.3f}",
                f"{профиль.I[i]:.4f}",
                f"{профиль.Ir[i]:.4f}",
                f"{профиль.Ig[i]:.4f}",
                f"{профиль.Ib[i]:.4f}",
            ]
            if wavelengths is not None:
                row.append(f"{wavelengths[i]:.4f}")
            w.writerow(row)


def сохранить_csv_пики(
    путь: Path,
    пики: list[Пик],
    wavelengths: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
) -> None:
    xs = (
        np.array(wavelengths, dtype=np.float64)
        if wavelengths is not None
        else np.array([p.пиксель for p in пики], dtype=np.float64)
    )
    order = np.argsort(xs)
    gap_после = np.full(len(пики), np.nan)
    for k in range(len(order) - 1):
        gap_после[order[k]] = xs[order[k + 1]] - xs[order[k]]

    with путь.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "номер",
                "пиксель",
                "интенсивность",
                "fwhm_пикс",
                "длина_волны_нм",
                "fwhm_нм",
                "зазор_до_следующего",
                "канал",
            ]
        )
        for i, p in enumerate(пики, start=1):
            lam = f"{wavelengths[i-1]:.4f}" if wavelengths is not None else ""
            fw = f"{fwhm_нм[i-1]:.4f}" if fwhm_нм is not None else ""
            gap = gap_после[i - 1]
            gap_s = f"{gap:.4f}" if np.isfinite(gap) else ""
            w.writerow(
                [
                    i,
                    f"{p.пиксель:.3f}",
                    f"{p.интенсивность:.4f}",
                    f"{p.fwhm_px:.3f}",
                    lam,
                    fw,
                    gap_s,
                    p.канал,
                ]
            )


def метрики(
    пики: list[Пик],
    wavelengths: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
    профиль: Профиль,
) -> dict:
    out: dict = {
        "число_пиков": len(пики),
        "ось_дисперсии": профиль.ось_дисперсии,
        "roi": {
            "x0": профиль.roi[0],
            "x1": профиль.roi[1],
            "y0": профиль.roi[2],
            "y1": профиль.roi[3],
        },
        "макс_интенсивность": float(профиль.I.max()),
        "средняя_интенсивность": float(профиль.I.mean()),
        "доля_почти_засвеченных": float(np.mean(профиль.I > 240)),
        "предупреждение_засветки": bool(профиль.I.max() > 250),
    }
    if wavelengths is not None and len(wavelengths) >= 2:
        order = np.argsort(wavelengths)
        wl = wavelengths[order]
        fw = fwhm_нм[order] if fwhm_нм is not None else None
        gaps = np.diff(wl)
        out["диапазон_нм"] = [float(wl.min()), float(wl.max())]
        out["дисперсия_нм_на_пикс"] = float(
            abs(wavelengths[-1] - wavelengths[0]) / max(len(wavelengths) - 1, 1)
        )
        out["мин_расстояние_между_пиками_нм"] = float(gaps.min()) if len(gaps) else None
        близкие_пары = []
        for k in np.argsort(gaps)[:5]:
            близкие_пары.append(
                {
                    "λ1_нм": float(wl[k]),
                    "λ2_нм": float(wl[k + 1]),
                    "Δλ_нм": float(gaps[k]),
                }
            )
        out["ближайшие_пары"] = близкие_пары
        out["пар_ближе_3_нм"] = int(np.sum(gaps < 3.0))
        out["пар_ближе_5_нм"] = int(np.sum(gaps < 5.0))
        if fw is not None and len(fw):
            median_fwhm = float(np.median(fw))
            out["медианный_fwhm_нм"] = median_fwhm
            out["мин_fwhm_нм"] = float(np.min(fw))
            яркости = np.array([пики[i].интенсивность for i in order])
            top = int(np.argmax(яркости))
            if fw[top] > 1e-9:
                out["R_оценка_по_ярчайшему"] = float(wl[top] / fw[top])
            if median_fwhm > 0:
                out["R_оценка_медианная"] = float(np.median(wl) / median_fwhm)
            # резольв по ближайшей паре: разделены ли относительно FWHM
            if len(gaps):
                k = int(np.argmin(gaps))
                mean_fw = float((fw[k] + fw[k + 1]) / 2)
                out["ближайшая_пара_разрешена"] = bool(gaps[k] > mean_fw)
                out["критерий_разрешения_Δλ_к_FWHM"] = float(
                    gaps[k] / mean_fw if mean_fw > 1e-12 else np.nan
                )
    elif len(пики) >= 2:
        px = np.array([p.пиксель for p in пики])
        gaps = np.diff(np.sort(px))
        out["мин_расстояние_между_пиками_пикс"] = float(gaps.min())
        out["медианный_fwhm_пикс"] = float(np.median([p.fwhm_px for p in пики]))
    return out


def сохранить_превью(
    путь: Path,
    профиль: Профиль,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
) -> None:
    img = профиль.исходник.copy()
    x0, x1, y0, y1 = профиль.roi
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2)
    # подписываем все пики
    for idx, p in enumerate(пики):
        label = f"{peak_wl[idx]:.0f}" if peak_wl is not None else f"{p.пиксель:.0f}"
        if профиль.ось_дисперсии == "y":
            y = int(round(p.пиксель))
            cv2.line(img, (x0, y), (x1, y), (40, 40, 255), 1)
            cv2.putText(
                img,
                label,
                (min(x1 + 4, img.shape[1] - 40), y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        else:
            x = int(round(p.пиксель))
            cv2.line(img, (x, y0), (x, y1), (40, 40, 255), 1)
            cv2.putText(
                img,
                label,
                (x - 10, max(y0 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    if not ok:
        raise RuntimeError(f"Не удалось сохранить превью: {путь}")
    buf.tofile(str(путь))


def сохранить_график_спектра(
    путь: Path,
    профиль: Профиль,
    пики: list[Пик],
    wavelengths: np.ndarray | None,
    peak_wl: np.ndarray | None,
) -> None:
    в_нм = wavelengths is not None
    x = wavelengths if в_нм else профиль.ось
    xlabel = "Длина волны, нм" if в_нм else "Пиксель"

    fig = plt.figure(figsize=(12.5, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.10], hspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0], sharex=ax)

    xmin, xmax = float(np.min(x)), float(np.max(x))

    if в_нм:
        for i in range(len(x) - 1):
            c = длина_волны_в_rgb(float(0.5 * (x[i] + x[i + 1])))
            ax.fill_between(
                x[i : i + 2],
                профиль.I[i : i + 2],
                color=c,
                alpha=0.55,
                linewidth=0,
            )
    else:
        ax.fill_between(x, профиль.I, color="#B84E2B", alpha=0.25, linewidth=0)

    ax.plot(x, профиль.I, color=ЦВЕТА["линия"], lw=1.7, zorder=3)

    px = (
        peak_wl
        if peak_wl is not None
        else np.array([p.пиксель for p in пики], dtype=np.float64)
    )
    py = np.array([p.интенсивность for p in пики], dtype=np.float64)
    for xv in px:
        c = длина_волны_в_rgb(float(xv)) if в_нм else ЦВЕТА["акцент"]
        ax.axvline(xv, color=c, alpha=0.35, lw=1.0, zorder=1)
    _подписи_пиков(ax, px, py, в_нм=в_нм)

    _оформить_оси(ax, "Спектр", "", "Интенсивность (отн.)")
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)

    _цветная_полоска(ax_bar, xmin, xmax, в_нм)
    ax_bar.set_yticks([])
    ax_bar.tick_params(axis="x", pad=2)
    ax_bar.set_xlabel(xlabel, labelpad=10)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.grid(False)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlim(xmin, xmax)
    ax_bar.margins(x=0)

    fig.subplots_adjust(hspace=0.26, bottom=0.12, top=0.90)
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_график_fwhm(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
) -> None:
    if not пики:
        return
    в_нм = peak_wl is not None and fwhm_нм is not None
    x = peak_wl if в_нм else np.array([p.пиксель for p in пики], dtype=np.float64)
    y = fwhm_нм if в_нм else np.array([p.fwhm_px for p in пики], dtype=np.float64)
    colors = [
        длина_волны_в_rgb(float(v)) if в_нм else (0.45, 0.55, 0.70) for v in x
    ]
    width = (np.ptp(x) / max(len(x) * 2.8, 1)) if len(x) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    bars = ax.bar(x, y, width=width, color=colors, edgecolor="#2A2622", linewidth=0.4)
    ymax = float(np.max(y)) if len(y) else 1.0
    for номер, (rect, xv, yv) in enumerate(zip(bars, x, y)):
        # чередуем высоту подписи над столбцом
        dy = 0.04 * ymax + (номер % 2) * 0.05 * ymax
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            yv + dy,
            f"{xv:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=ЦВЕТА["чернила"],
            clip_on=False,
        )
    med = float(np.median(y))
    ax.axhline(med, color=ЦВЕТА["акцент"], ls="--", lw=1.2)
    ax.set_ylim(0, ymax * 1.28)
    _оформить_оси(
        ax,
        "FWHM",
        "Длина волны, нм" if в_нм else "Пиксель",
        "FWHM, нм" if в_нм else "FWHM, пикс",
    )
    ед = "нм" if в_нм else "пикс"
    _подпись_вне_графика(ax, f"медиана = {med:.2f} {ед}", где="верх_справа")
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_карту_линий(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
) -> None:
    if not пики:
        return
    в_нм = peak_wl is not None
    xs = peak_wl if в_нм else np.array([p.пиксель for p in пики], dtype=np.float64)
    intensities = np.array([p.интенсивность for p in пики], dtype=np.float64)
    heights = 0.28 + 0.72 * (intensities / max(intensities.max(), 1e-9))
    xmin, xmax = float(xs.min()), float(xs.max())
    pad = max(0.5, 0.02 * (xmax - xmin + 1e-9))

    fig = plt.figure(figsize=(12.2, 3.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.14], hspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0], sharex=ax)

    for x, h in zip(xs, heights):
        color = длина_волны_в_rgb(float(x)) if в_нм else ЦВЕТА["акцент"]
        ax.vlines(x, 0, h, color=color, lw=2.4)
        ax.plot([x], [h], "o", color=color, ms=5)
    order_x = np.argsort(xs)
    for номер, i in enumerate(order_x):
        y_text = min(heights[i] + 0.06 + (номер % 3) * 0.12, 1.28)
        ax.text(
            xs[i],
            y_text,
            f"{xs[i]:.0f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=ЦВЕТА["чернила"],
        )
    ax.set_ylim(0, 1.38)
    ax.set_yticks([])
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.margins(x=0)
    _оформить_оси(ax, "Карта линий", "", "")
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    ax.grid(False, axis="y")

    _цветная_полоска(ax_bar, xmin - pad, xmax + pad, в_нм)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Длина волны, нм" if в_нм else "Пиксель", labelpad=8)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.grid(False)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlim(xmin - pad, xmax + pad)
    ax_bar.margins(x=0)

    fig.subplots_adjust(hspace=0.22, bottom=0.18, top=0.88)
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_график_близких_пар(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
) -> None:
    if peak_wl is None or len(пики) < 2:
        return
    order = np.argsort(peak_wl)
    wl = peak_wl[order]
    intens = np.array([пики[i].интенсивность for i in order])
    fw = fwhm_нм[order] if fwhm_нм is not None else None
    gaps = np.diff(wl)
    top = np.argsort(gaps)[: min(6, len(gaps))]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    # локальный фрагмент вокруг самой близкой пары
    k0 = int(top[0])
    center = 0.5 * (wl[k0] + wl[k0 + 1])
    span = max(8.0, 4.0 * gaps[k0])
    mask = (wl >= center - span) & (wl <= center + span)
    idxs = list(np.where(mask)[0])
    imax = float(intens[mask].max()) if len(idxs) else 1.0
    for i in idxs:
        c = длина_волны_в_rgb(float(wl[i]))
        ax.vlines(wl[i], 0, intens[i], color=c, lw=2.2)
        ax.plot([wl[i]], [intens[i]], "o", color=c, ms=6)
    for номер, i in enumerate(idxs):
        ax.annotate(
            f"{wl[i]:.1f}",
            (wl[i], intens[i]),
            textcoords="offset points",
            xytext=(0, 10 + (номер % 3) * 10),
            ha="center",
            fontsize=8,
            color=ЦВЕТА["чернила"],
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "#FFFDF8",
                "edgecolor": "#D8CFC0",
                "linewidth": 0.5,
                "alpha": 0.95,
            },
        )
    ax.set_ylim(0, imax * 1.35)
    ax.axvspan(wl[k0], wl[k0 + 1], color=длина_волны_в_rgb(float(center)), alpha=0.15)
    _оформить_оси(
        ax,
        f"Ближайшая пара  Δλ = {gaps[k0]:.2f} нм",
        "Длина волны, нм",
        "Интенсивность",
    )

    ax2 = axes[1]
    labels = [f"{wl[k]:.0f}–{wl[k+1]:.0f}" for k in top]
    vals = [float(gaps[k]) for k in top]
    colors = [длина_волны_в_rgb(float(0.5 * (wl[k] + wl[k + 1]))) for k in top]
    y_pos = np.arange(len(vals))[::-1]
    ax2.barh(y_pos, vals, color=colors, edgecolor="#2A2622", linewidth=0.4, height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    # подписи значений справа от столбцов — не пересекают пунктир
    xmax = max(vals) if vals else 1.0
    if fw is not None:
        med_fw = float(np.median(fw))
        xmax = max(xmax, med_fw)
        ax2.axvline(med_fw, color=ЦВЕТА["акцент"], ls="--", lw=1.2)
        _подпись_вне_графика(ax2, f"FWHM = {med_fw:.2f} нм", где="верх_справа")

    for yp, val in zip(y_pos, vals):
        ax2.text(
            val + 0.04 * xmax,
            yp,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color=ЦВЕТА["чернила"],
            clip_on=False,
        )
    ax2.set_xlim(0, xmax * 1.25)
    _оформить_оси(ax2, "Ближайшие пары", "Δλ, нм", "")
    fig.tight_layout()
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_график_разрешения(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
) -> None:
    """R(λ) = λ / FWHM — ключевая метрика для задачи."""
    if peak_wl is None or fwhm_нм is None or len(пики) == 0:
        return
    mask = fwhm_нм > 1e-9
    if not np.any(mask):
        return
    wl = peak_wl[mask]
    r_vals = wl / fwhm_нм[mask]
    colors = [длина_волны_в_rgb(float(v)) for v in wl]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.scatter(wl, r_vals, c=colors, s=55, edgecolors="#2A2622", linewidths=0.5, zorder=3)
    order = np.argsort(wl)
    ax.plot(wl[order], r_vals[order], color=ЦВЕТА["линия"], lw=1.2, alpha=0.5)
    med = float(np.median(r_vals))
    ax.axhline(med, color=ЦВЕТА["акцент"], ls="--", lw=1.2)
    for x, y in zip(wl, r_vals):
        ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=7, color=ЦВЕТА["приглушённый"])
    ax.set_ylim(0, float(np.max(r_vals)) * 1.25)
    _оформить_оси(ax, "Разрешающая способность", "Длина волны, нм", "R = λ / FWHM")
    _подпись_вне_графика(ax, f"медиана R = {med:.0f}", где="верх_справа")
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_график_критерия_разрешения(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
) -> None:
    """Для каждой соседней пары: Δλ против среднего FWHM (выше диагонали — разрешена)."""
    if peak_wl is None or fwhm_нм is None or len(пики) < 2:
        return
    order = np.argsort(peak_wl)
    wl = peak_wl[order]
    fw = fwhm_нм[order]
    gaps = np.diff(wl)
    mean_fw = 0.5 * (fw[:-1] + fw[1:])
    colors = [
        длина_волны_в_rgb(float(0.5 * (wl[i] + wl[i + 1]))) for i in range(len(gaps))
    ]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lim = float(max(np.max(gaps), np.max(mean_fw)) * 1.15)
    ax.plot([0, lim], [0, lim], ls="--", color=ЦВЕТА["приглушённый"], lw=1.2)
    ax.scatter(mean_fw, gaps, c=colors, s=70, edgecolors="#2A2622", linewidths=0.5, zorder=3)
    for i, (xf, yg) in enumerate(zip(mean_fw, gaps)):
        ax.text(
            xf,
            yg,
            f"{wl[i]:.0f}–{wl[i+1]:.0f}",
            fontsize=7,
            ha="left",
            va="bottom",
            color=ЦВЕТА["чернила"],
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal", adjustable="box")
    _оформить_оси(ax, "Критерий разрешения", "Средний FWHM, нм", "Δλ, нм")
    _подпись_вне_графика(ax, "выше линии: Δλ > FWHM", где="верх_слева")
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_график_дисперсии(
    путь: Path,
    профиль: Профиль,
    wavelengths: np.ndarray | None,
) -> None:
    """Локальная дисперсия нм/пиксель вдоль спектра."""
    if wavelengths is None or len(wavelengths) < 3:
        return
    # dλ/dp вдоль оси профиля
    px = профиль.ось
    dlam = np.gradient(wavelengths, px)
    dlam_abs = np.abs(dlam)

    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12], hspace=0.24)
    ax = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0], sharex=ax)
    xmin, xmax = float(wavelengths.min()), float(wavelengths.max())

    for i in range(len(wavelengths) - 1):
        c = длина_волны_в_rgb(float(0.5 * (wavelengths[i] + wavelengths[i + 1])))
        ax.fill_between(
            wavelengths[i : i + 2],
            dlam_abs[i : i + 2],
            color=c,
            alpha=0.35,
            linewidth=0,
        )
    ax.plot(wavelengths, dlam_abs, color=ЦВЕТА["линия"], lw=1.5)
    med = float(np.median(dlam_abs))
    ax.axhline(med, color=ЦВЕТА["акцент"], ls="--", lw=1.1)
    _оформить_оси(ax, "Дисперсия", "", "нм / пиксель")
    ax.set_xlabel("")
    ax.tick_params(labelbottom=False)
    ax.set_xlim(xmin, xmax)
    ax.margins(x=0)
    _подпись_вне_графика(ax, f"медиана = {med:.3f} нм/пикс", где="верх_справа")

    _цветная_полоска(ax_bar, xmin, xmax, True)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Длина волны, нм", labelpad=8)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.grid(False)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlim(xmin, xmax)
    ax_bar.margins(x=0)
    fig.subplots_adjust(hspace=0.22, bottom=0.14, top=0.90)
    fig.savefig(путь)
    plt.close(fig)
    plt.close("all")


def сохранить_график_каналов(
    путь: Path,
    профиль: Профиль,
    пики: list[Пик],
    wavelengths: np.ndarray | None,
    peak_wl: np.ndarray | None,
) -> None:
    в_нм = wavelengths is not None
    x = wavelengths if в_нм else профиль.ось
    xlabel = "Длина волны, нм" if в_нм else "Пиксель"
    серии = [
        ("R", профиль.Ir, ЦВЕТА["r"]),
        ("G", профиль.Ig, ЦВЕТА["g"]),
        ("B", профиль.Ib, ЦВЕТА["b"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 7.2), sharex=True)
    px = peak_wl if peak_wl is not None else np.array([p.пиксель for p in пики])
    for ax, (name, y, color) in zip(axes, серии):
        ax.fill_between(x, y, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, lw=1.6)
        for xv in px:
            ax.axvline(xv, color=ЦВЕТА["чернила"], alpha=0.08, lw=0.7)
        _оформить_оси(ax, f"Канал {name}", xlabel if ax is axes[-1] else "", "I (отн.)")
        if ax is not axes[-1]:
            ax.set_xlabel("")
    fig.suptitle("Каналы R / G / B", fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.93, hspace=0.28)
    fig.savefig(путь)
    plt.close(fig)


def сохранить_отчёт(
    путь: Path,
    фото_имя: str,
    m: dict,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
) -> None:
    lines = [
        f"# Отчёт по спектру: {фото_имя}",
        "",
        "## Коротко",
        f"- Найдено линий: **{m.get('число_пиков', 0)}**",
    ]
    if "диапазон_нм" in m:
        a, b = m["диапазон_нм"]
        lines.append(f"- Диапазон: **{a:.1f}–{b:.1f} нм**")
    if m.get("медианный_fwhm_нм") is not None:
        lines.append(f"- Медианный FWHM: **{m['медианный_fwhm_нм']:.2f} нм**")
    if m.get("R_оценка_медианная") is not None:
        lines.append(f"- Оценка R (медианная): **{m['R_оценка_медианная']:.0f}**")
    if m.get("мин_расстояние_между_пиками_нм") is not None:
        lines.append(
            f"- Минимальный зазор между линиями: **{m['мин_расстояние_между_пиками_нм']:.2f} нм**"
        )
    if "ближайшая_пара_разрешена" in m:
        ok = "да" if m["ближайшая_пара_разрешена"] else "спорно / нет"
        lines.append(f"- Ближайшая пара выглядит разрешённой (Δλ > FWHM): **{ok}**")
    if m.get("предупреждение_засветки"):
        lines.append("- ⚠ Есть признак засветки камеры — стоит снизить экспозицию.")
    lines += ["", "## Ближайшие пары (важно для неона)", ""]
    for pair in m.get("ближайшие_пары", [])[:5]:
        lines.append(
            f"- {pair['λ1_нм']:.1f} и {pair['λ2_нм']:.1f} нм → Δλ = {pair['Δλ_нм']:.2f} нм"
        )
    if not m.get("ближайшие_пары"):
        lines.append("- Нет данных в нм (шкала не применялась).")
    lines += [
        "",
        "## Файлы",
        "",
        "- `спектр.png`, `карта_линий.png`, `близкие_пары.png`",
        "- `fwhm.png`, `разрешение.png`, `критерий_разрешения.png`, `дисперсия.png`",
        "- `каналы_rgb.png`, `пики.csv`, `метрики.json`",
        "",
    ]
    путь.write_text("\n".join(lines), encoding="utf-8")


def собрать_сводку_сессии(сессия: Path) -> Path | None:
    """Таблица по всем обработанным фото — удобно сравнивать установки."""
    папка = сессия / "результаты"
    if not папка.exists():
        return None
    строки: list[dict] = []
    for d in sorted(p for p in папка.iterdir() if p.is_dir()):
        mj = d / "метрики.json"
        if not mj.exists():
            continue
        m = json.loads(mj.read_text(encoding="utf-8"))
        строки.append(
            {
                "фото": m.get("файл", d.name),
                "пиков": m.get("число_пиков", ""),
                "диапазон_нм": (
                    f"{m['диапазон_нм'][0]:.1f}–{m['диапазон_нм'][1]:.1f}"
                    if m.get("диапазон_нм")
                    else ""
                ),
                "мин_Δλ_нм": m.get("мин_расстояние_между_пиками_нм", ""),
                "мед_FWHM_нм": m.get("медианный_fwhm_нм", ""),
                "R_мед": m.get("R_оценка_медианная", ""),
                "пар_<3нм": m.get("пар_ближе_3_нм", ""),
                "засветка": "да" if m.get("предупреждение_засветки") else "нет",
            }
        )
    if not строки:
        return None

    csv_path = папка / "сводка.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(строки[0].keys()))
        w.writeheader()
        w.writerows(строки)

    md = [
        "# Сводка по эксперименту",
        "",
        f"Папка: `{сессия.name}`",
        "",
        "| фото | пиков | диапазон, нм | мин Δλ | мед. FWHM | R | пар <3 нм | засветка |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for s in строки:
        def fmt(v, digits=2):
            if v == "" or v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{digits}f}"
            return str(v)

        md.append(
            f"| {s['фото']} | {s['пиков']} | {s['диапазон_нм'] or '—'} | "
            f"{fmt(s['мин_Δλ_нм'])} | {fmt(s['мед_FWHM_нм'])} | "
            f"{fmt(s['R_мед'], 0)} | {s['пар_<3нм'] if s['пар_<3нм'] != '' else '—'} | "
            f"{s['засветка']} |"
        )
    md.append("")
    md_path = папка / "сводка.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path


def обработать_одно_фото(
    сессия: Path,
    фото: Path,
    cal: dict | None,
    переносить: bool = True,
) -> Path:
    печать(f"\n→ {фото.name}")
    img = загрузить_изображение(фото)
    профиль = построить_профиль(img)
    пики = найти_пики(профиль, канал="I")

    wavelengths = None
    peak_wl = None
    fwhm_нм = None
    if cal is not None:
        coeffs = cal["коэффициенты"]
        wavelengths = применить_калибровку(профиль.ось, coeffs)
        peak_wl = применить_калибровку(
            np.array([p.пиксель for p in пики], dtype=np.float64), coeffs
        )
        deriv = np.polyder(np.array(coeffs, dtype=np.float64))
        fwhm_нм = np.array(
            [
                abs(float(np.polyval(deriv, p.пиксель))) * p.fwhm_px
                for p in пики
            ],
            dtype=np.float64,
        )

    stem = фото.stem
    out_dir = сессия / "результаты" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    сохранить_превью(out_dir / "превью.jpg", профиль, пики, peak_wl)
    сохранить_csv_профиль(out_dir / "профиль.csv", профиль, wavelengths)
    сохранить_csv_пики(out_dir / "пики.csv", пики, peak_wl, fwhm_нм)
    сохранить_график_спектра(
        out_dir / "спектр.png", профиль, пики, wavelengths, peak_wl
    )
    сохранить_график_fwhm(out_dir / "fwhm.png", пики, peak_wl, fwhm_нм)
    сохранить_карту_линий(out_dir / "карта_линий.png", пики, peak_wl)
    сохранить_график_близких_пар(
        out_dir / "близкие_пары.png", пики, peak_wl, fwhm_нм
    )
    сохранить_график_разрешения(
        out_dir / "разрешение.png", пики, peak_wl, fwhm_нм
    )
    сохранить_график_критерия_разрешения(
        out_dir / "критерий_разрешения.png", пики, peak_wl, fwhm_нм
    )
    сохранить_график_дисперсии(out_dir / "дисперсия.png", профиль, wavelengths)
    сохранить_график_каналов(
        out_dir / "каналы_rgb.png", профиль, пики, wavelengths, peak_wl
    )

    m = метрики(пики, peak_wl, fwhm_нм, профиль)
    m["файл"] = фото.name
    m["калибровка_использована"] = cal is not None
    (out_dir / "метрики.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    сохранить_отчёт(out_dir / "отчёт.md", фото.name, m, пики, peak_wl)

    if cal is not None:
        used = {
            "источник": str(путь_калибровки(сессия).relative_to(сессия)),
            "формула": cal.get("формула"),
            "коэффициенты": cal.get("коэффициенты"),
            "эталон": cal.get("эталон"),
            "дата_калибровки": cal.get("дата"),
            "дата_обработки": datetime.now().isoformat(timespec="seconds"),
        }
        (out_dir / "использованная_калибровка.json").write_text(
            json.dumps(used, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if переносить:
        dest = сессия / "обработанные" / фото.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if фото.resolve() != dest.resolve():
            shutil.move(str(фото), str(dest))

    печать(f"  пиков: {len(пики)} → {out_dir}")
    печать(
        "  файлы: спектр.png, карта_линий.png, разрешение.png, "
        "критерий_разрешения.png, дисперсия.png …"
    )
    return out_dir


# ---------------------------------------------------------------------------
# Диалоги режимов
# ---------------------------------------------------------------------------

def режим_калибровать(сессия: Path) -> None:
    печать("\n=== Настройка шкалы длин волн ===")
    печать(
        "Зачем это нужно: на фото программа видит только пиксели.\n"
        "Шкала переводит «пиксель №312» → «640 нм».\n"
        "Делается один раз для этой папки эксперимента, пока не двигали установку.\n"
    )
    выбор = спросить_выбор(
        "По какому источнику настроить шкалу?",
        {
            "1": "неон (рекомендуем сейчас — яркий, у вас есть)",
            "2": "водород (когда уверенно снимете камерой)",
            "3": "гелий (когда уверенно снимете камерой)",
        },
    )
    имя_эт, файл_спр = СПРАВОЧНИКИ[выбор]
    справочник = загрузить_справочник(файл_спр)

    кандидаты = список_фото(сессия / "калибровка" / "эталоны") + список_фото(
        сессия / "вход"
    )
    # уникальные по пути
    seen = set()
    uniq: list[Path] = []
    for p in кандидаты:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)

    if not uniq:
        печать(
            "Нет фото для калибровки.\n"
            f"Положите кадр в:\n  {сессия / 'калибровка' / 'эталоны'}\n"
            f"или в:\n  {сессия / 'вход'}"
        )
        return

    варианты = {str(i + 1): p.name for i, p in enumerate(uniq)}
    номер = спросить_выбор("Выберите эталонный кадр:", варианты)
    фото = uniq[int(номер) - 1]

    # копируем в эталоны, если ещё не там
    эталон_dir = сессия / "калибровка" / "эталоны"
    эталон_dir.mkdir(parents=True, exist_ok=True)
    эталон_копия = эталон_dir / фото.name
    if фото.resolve() != эталон_копия.resolve():
        shutil.copy2(фото, эталон_копия)

    img = загрузить_изображение(фото)
    профиль = построить_профиль(img)
    пики = найти_пики(профиль)
    печать(f"\nНайдено пиков: {len(пики)}")
    if len(пики) < 2:
        печать("Слишком мало пиков для калибровки. Проверьте фото/экспозицию.")
        return

    пары = автосопоставление(пики, справочник)
    if not пары:
        печать("Автосопоставление не удалось. Попробуйте другой кадр или эталон.")
        return

    печать("\nАвтосопоставление со справочником:")
    печать(f"{'№':>3}  {'пиксель':>10}  {'λ, нм':>10}  {'подпись':<8}  уверенность")
    for i, p in enumerate(пары, start=1):
        печать(
            f"{i:>3}  {p.пиксель:>10.2f}  {p.длина_волны_нм:>10.3f}  "
            f"{p.подпись:<8}  {p.уверенность}"
        )

    if not да_нет("\nПринять это сопоставление?", True):
        печать(
            "Ручное редактирование пар пока упрощённое: "
            "можете удалить сомнительные по номерам."
        )
        сырой = спросить("Номера пар для удаления через пробел (или Enter — ничего)")
        if сырой.strip():
            удалить = {int(x) for x in сырой.split() if x.isdigit()}
            пары = [p for i, p in enumerate(пары, start=1) if i not in удалить]

    if len(пары) < 2:
        печать("Нужно хотя бы 2 пары.")
        return

    # рекомендация формулы
    if len(пары) <= 3:
        рек = "1"
        рек_текст = "линейная"
    elif len(пары) == 4:
        рек = "2"
        рек_текст = "квадратичная"
    else:
        рек = "3"
        рек_текст = "кубическая"

    печать(f"\nПар эталонных линий: {len(пары)}")
    печать(f"Рекомендуем: {рек_текст}")
    ф = спросить_выбор(
        "Формула калибровки:",
        {
            "1": "линейная",
            "2": "квадратичная",
            "3": "кубическая",
        },
    )
    имя_форм, степень = ФОРМУЛЫ[ф]
    coeffs = подобрать_полином(пары, степень)
    ош = ошибки_калибровки(пары, coeffs.tolist())

    путь = сохранить_калибровку(
        сессия=сессия,
        формула=имя_форм,
        coeffs=coeffs.tolist(),
        пары=пары,
        эталон=имя_эт,
        кадр=фото.name,
        ось_дисперсии=профиль.ось_дисперсии,
    )
    печать(f"\nСохранено: {путь}")
    печать(f"Ошибка на эталонах: средняя {ош['средняя']:.3f} нм, макс {ош['макс']:.3f} нм")

    # контрольный график калибровки
    out = сессия / "калибровка" / "график_калибровки.png"
    xs = np.array([p.пиксель for p in пары])
    ys = np.array([p.длина_волны_нм for p in пары])
    grid = np.linspace(профиль.ось.min(), профиль.ось.max(), 400)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(grid, np.polyval(coeffs, grid), color=ЦВЕТА["линия"], lw=2.0, label=имя_форм)
    ax.scatter(
        xs,
        ys,
        s=55,
        color=[длина_волны_в_rgb(float(y)) for y in ys],
        edgecolor=ЦВЕТА["чернила"],
        linewidths=0.6,
        zorder=3,
        label="эталонные линии",
    )
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=ЦВЕТА["приглушённый"],
        )
    _оформить_оси(ax, "Шкала длин волн: пиксель → нм", "Пиксель", "Длина волны, нм")
    _легенда_снаружи(ax, fig)
    fig.savefig(out)
    plt.close(fig)
    печать(f"График: {out}")


def режим_обработать(сессия: Path, только_пиксели: bool = False) -> None:
    печать("\n=== Обработка фото ===")
    cal = None if только_пиксели else загрузить_калибровку(сессия)
    if cal is None and not только_пиксели:
        печать("Шкала длин волн для этой папки ещё не настроена.")
        выбор = спросить_выбор(
            "Что сделать?",
            {
                "1": "Сначала настроить шкалу (нужны нм и разрешение)",
                "2": "Обработать без шкалы (только пиксели, черновик)",
                "3": "Отмена",
            },
        )
        if выбор == "1":
            режим_калибровать(сессия)
            cal = загрузить_калибровку(сессия)
            if cal is None:
                return
        elif выбор == "2":
            только_пиксели = True
        else:
            return

    фотографии = список_фото(сессия / "вход")
    if not фотографии:
        печать("Новых фото пока нет.")
        печать(f"Скопируйте снимки сюда и запустите снова:\n  {сессия / 'вход'}")
        return

    if cal is None:
        печать("Режим: без шкалы (результаты будут в пикселях, не в нм)")
    else:
        печать(f"Шкала длин волн: {статус_калибровки(сессия)}")
    печать(f"Будут обработаны {len(фотографии)} фото:")
    for p in фотографии:
        печать(f"  • {p.name}")
    печать("После обработки исходники переедут в папку «обработанные».")
    if not да_нет("Начать?", True):
        return

    for фото in фотографии:
        обработать_одно_фото(сессия, фото, cal, переносить=True)

    сводка = собрать_сводку_сессии(сессия)
    печать(f"\nГотово. Смотрите результаты здесь:\n  {сессия / 'результаты'}")
    if сводка:
        печать(f"Сводка по всем фото (для сравнения установок):\n  {сводка}")


def режим_калибровать_и_обработать(сессия: Path) -> None:
    режим_калибровать(сессия)
    if путь_калибровки(сессия).exists():
        режим_обработать(сессия)


# ---------------------------------------------------------------------------
# Главное меню
# ---------------------------------------------------------------------------

def главное_меню(сессия: Path) -> None:
    while True:
        n_in = len(список_фото(сессия / "вход"))
        печать("\n" + "=" * 60)
        печать("Спектрометр с диска — обработка фото")
        печать("=" * 60)
        печать(f"Папка эксперимента: {сессия.name}")
        печать(f"Полный путь:        {сессия}")
        печать(f"Шкала длин волн:    {статус_калибровки(сессия)}")
        печать(f"Новых фото во «вход»: {n_in}")
        печать("")
        печать(подсказка_следующего_шага(сессия))
        выбор = спросить_выбор(
            "\nЧто сделать?",
            {
                "1": "Настроить шкалу длин волн (пиксель → нм)",
                "2": "Обработать новые фото из папки «вход»",
                "3": "Настроить шкалу и сразу обработать фото",
                "4": "Собрать сводку по уже обработанным фото",
                "5": "Открыть другую / создать новую папку",
                "6": "Выход",
            },
        )
        if выбор == "1":
            режим_калибровать(сессия)
        elif выбор == "2":
            режим_обработать(сессия)
        elif выбор == "3":
            режим_калибровать_и_обработать(сессия)
        elif выбор == "4":
            сводка = собрать_сводку_сессии(сессия)
            if сводка:
                печать(f"Сводка обновлена:\n  {сводка}")
            else:
                печать("Пока нет обработанных результатов для сводки.")
        elif выбор == "5":
            сессия = выбрать_сессию()
            обеспечить_структуру_сессии(сессия)
        else:
            печать("До встречи.")
            break


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    печать(
        "Задача 2 IYPT — спектрометр с CD/DVD\n"
        "\n"
        "Коротко, как пользоваться:\n"
        "  1. Выбираете папку эксперимента (одна настройка установки).\n"
        "  2. Кладёте фото в подпапку «вход».\n"
        "  3. Настраиваете шкалу длин волн (один раз).\n"
        "  4. Обрабатываете фото — графики и таблицы появятся в «результаты».\n"
    )
    арг_сессии = argv[0] if argv else None
    ПАПКА_СЕССИЙ.mkdir(parents=True, exist_ok=True)
    сессия = выбрать_сессию(арг_сессии)
    обеспечить_структуру_сессии(сессия)
    главное_меню(сессия)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
