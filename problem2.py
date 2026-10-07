#!/usr/bin/env python3
"""
Задача 2 IYPT — спектрометр на CD/DVD.
Один скрипт: диалог, калибровка, обработка фото, графики и метрики.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
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
            "Сессия спектрометра.\n"
            "Опишите установку: диск (CD/DVD), щель, угол, камера, дата.\n",
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
    имя = спросить("Имя сессии", f"{сегодня}_dvd_неон")
    имя = имя.replace(" ", "_")
    путь = ПАПКА_СЕССИЙ / имя
    if путь.exists():
        печать(f"Сессия уже есть: {путь}")
    else:
        обеспечить_структуру_сессии(путь)
        печать(f"Создана сессия: {путь}")
    return путь


def выбрать_сессию(арг: str | None = None) -> Path:
    if арг:
        путь = Path(арг)
        if not путь.is_absolute():
            путь = КОРЕНЬ / путь
        if not путь.exists():
            обеспечить_структуру_сессии(путь)
        else:
            обеспечить_структуру_сессии(путь)
        return путь

    сессии = список_сессий()
    варианты: dict[str, str] = {"0": "Создать новую сессию"}
    for i, s in enumerate(сессии, start=1):
        cal = "есть cal" if (s / "калибровка" / "калибровка.json").exists() else "нет cal"
        n_in = len(список_фото(s / "вход"))
        варианты[str(i)] = f"{s.name}  ({cal}, во входе: {n_in})"

    выбор = спросить_выбор("\nВыберите сессию:", варианты)
    if выбор == "0":
        return создать_сессию()
    return сессии[int(выбор) - 1]


def путь_калибровки(сессия: Path) -> Path:
    return сессия / "калибровка" / "калибровка.json"


def статус_калибровки(сессия: Path) -> str:
    p = путь_калибровки(сессия)
    if not p.exists():
        return "нет"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        формула = data.get("формула", "?")
        n = len(data.get("пары", []))
        return f"есть ({формула}, {n} линий)"
    except Exception:
        return "есть (файл повреждён?)"


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
# Сохранение результатов и графики
# ---------------------------------------------------------------------------

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
                "канал",
            ]
        )
        for i, p in enumerate(пики, start=1):
            lam = f"{wavelengths[i-1]:.4f}" if wavelengths is not None else ""
            fw = f"{fwhm_нм[i-1]:.4f}" if fwhm_нм is not None else ""
            w.writerow(
                [
                    i,
                    f"{p.пиксель:.3f}",
                    f"{p.интенсивность:.4f}",
                    f"{p.fwhm_px:.3f}",
                    lam,
                    fw,
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
        "предупреждение_засветки": bool(профиль.I.max() > 250),
    }
    if wavelengths is not None and len(wavelengths) >= 2:
        order = np.argsort(wavelengths)
        wl = wavelengths[order]
        fw = fwhm_нм[order] if fwhm_нм is not None else None
        gaps = np.diff(wl)
        out["диапазон_нм"] = [float(wl.min()), float(wl.max())]
        out["мин_расстояние_между_пиками_нм"] = float(gaps.min()) if len(gaps) else None
        if fw is not None and len(fw):
            median_fwhm = float(np.median(fw))
            out["медианный_fwhm_нм"] = median_fwhm
            # оценка R по самым ярким пикам в красной области, если есть
            яркости = np.array([пики[i].интенсивность for i in order])
            top = int(np.argmax(яркости))
            if fw[top] > 1e-9:
                out["R_оценка_по_ярчайшему"] = float(wl[top] / fw[top])
            # сколько пар ближе 3 нм (важный результат для неона)
            близкие = int(np.sum(gaps < 3.0))
            out["пар_ближе_3_нм"] = близкие
            if median_fwhm > 0:
                out["R_оценка_медианная"] = float(np.median(wl) / median_fwhm)
    elif len(пики) >= 2:
        px = np.array([p.пиксель for p in пики])
        gaps = np.diff(np.sort(px))
        out["мин_расстояние_между_пиками_пикс"] = float(gaps.min())
        out["медианный_fwhm_пикс"] = float(np.median([p.fwhm_px for p in пики]))
    return out


def сохранить_превью(путь: Path, профиль: Профиль, пики: list[Пик]) -> None:
    img = профиль.исходник.copy()
    x0, x1, y0, y1 = профиль.roi
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    for p in пики:
        if профиль.ось_дисперсии == "y":
            y = int(round(p.пиксель))
            cv2.line(img, (x0, y), (x1, y), (0, 0, 255), 1)
        else:
            x = int(round(p.пиксель))
            cv2.line(img, (x, y0), (x, y1), (0, 0, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
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
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if wavelengths is not None:
        x = wavelengths
        xlabel = "Длина волны, нм"
    else:
        x = профиль.ось
        xlabel = "Пиксель"
    ax.plot(x, профиль.I, color="black", lw=1.2, label="I")
    ax.plot(x, профиль.Ir, color="tab:red", lw=0.8, alpha=0.7, label="R")
    ax.plot(x, профиль.Ig, color="tab:green", lw=0.8, alpha=0.7, label="G")
    ax.plot(x, профиль.Ib, color="tab:blue", lw=0.8, alpha=0.7, label="B")
    for i, p in enumerate(пики):
        xp = float(peak_wl[i]) if peak_wl is not None else p.пиксель
        ax.axvline(xp, color="orange", alpha=0.35, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Интенсивность (отн.)")
    ax.set_title("Профиль спектра")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(путь, dpi=150)
    plt.close(fig)


def сохранить_график_fwhm(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
    fwhm_нм: np.ndarray | None,
) -> None:
    if not пики:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    if peak_wl is not None and fwhm_нм is not None:
        x = peak_wl
        y = fwhm_нм
        ax.set_xlabel("Длина волны, нм")
        ax.set_ylabel("FWHM, нм")
        ax.set_title("Ширина линий (FWHM)")
    else:
        x = np.array([p.пиксель for p in пики])
        y = np.array([p.fwhm_px for p in пики])
        ax.set_xlabel("Пиксель")
        ax.set_ylabel("FWHM, пикс")
        ax.set_title("Ширина линий (FWHM)")
    ax.bar(x, y, width=(np.ptp(x) / max(len(x) * 3, 1)) if len(x) > 1 else 1.0, color="steelblue")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(путь, dpi=150)
    plt.close(fig)


def сохранить_карту_линий(
    путь: Path,
    пики: list[Пик],
    peak_wl: np.ndarray | None,
) -> None:
    """Компактная «линейка» найденных линий — удобно для доклада."""
    if not пики:
        return
    fig, ax = plt.subplots(figsize=(10, 1.8))
    if peak_wl is not None:
        xs = peak_wl
        ax.set_xlabel("Длина волны, нм")
        ax.set_xlim(xs.min() - 10, xs.max() + 10)
    else:
        xs = np.array([p.пиксель for p in пики])
        ax.set_xlabel("Пиксель")
    intensities = np.array([p.интенсивность for p in пики])
    heights = 0.3 + 0.7 * (intensities / max(intensities.max(), 1e-9))
    for x, h in zip(xs, heights):
        ax.vlines(x, 0, h, color="crimson", lw=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_title("Карта найденных линий")
    fig.tight_layout()
    fig.savefig(путь, dpi=150)
    plt.close(fig)


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
        # FWHM в нм: локальная |dλ/dp| * fwhm_px
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

    сохранить_превью(out_dir / "превью.jpg", профиль, пики)
    сохранить_csv_профиль(out_dir / "профиль.csv", профиль, wavelengths)
    сохранить_csv_пики(out_dir / "пики.csv", пики, peak_wl, fwhm_нм)
    сохранить_график_спектра(
        out_dir / "спектр.png", профиль, пики, wavelengths, peak_wl
    )
    сохранить_график_fwhm(out_dir / "fwhm.png", пики, peak_wl, fwhm_нм)
    сохранить_карту_линий(out_dir / "карта_линий.png", пики, peak_wl)

    m = метрики(пики, peak_wl, fwhm_нм, профиль)
    m["файл"] = фото.name
    m["калибровка_использована"] = cal is not None
    (out_dir / "метрики.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8"
    )

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
    return out_dir


# ---------------------------------------------------------------------------
# Диалоги режимов
# ---------------------------------------------------------------------------

def режим_калибровать(сессия: Path) -> None:
    печать("\n=== Калибровка ===")
    выбор = спросить_выбор(
        "Источник эталона (для шкалы длин волн):",
        {
            "1": "неон (есть у вас; также главный образец)",
            "2": "водород (тусклее, когда заведёте камеру)",
            "3": "гелий (тусклее, когда заведёте камеру)",
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
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, color="crimson", zorder=3, label="эталоны")
    ax.plot(grid, np.polyval(coeffs, grid), color="steelblue", label=имя_форм)
    ax.set_xlabel("Пиксель")
    ax.set_ylabel("Длина волны, нм")
    ax.set_title("Калибровка шкалы")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    печать(f"График: {out}")


def режим_обработать(сессия: Path, только_пиксели: bool = False) -> None:
    печать("\n=== Обработка фото ===")
    cal = None if только_пиксели else загрузить_калибровку(сессия)
    if cal is None and not только_пиксели:
        печать(f"Нет файла {путь_калибровки(сессия)}")
        выбор = спросить_выбор(
            "Что сделать?",
            {
                "1": "Сначала откалибровать",
                "2": "Обработать только в пикселях (без нм)",
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
        печать(f"Папка пуста: {сессия / 'вход'}")
        печать("Положите туда необработанные фото и запустите снова.")
        return

    печать(f"Калибровка: {'нет (только пиксели)' if cal is None else статус_калибровки(сессия)}")
    печать(f"Фото во «вход»: {len(фотографии)}")
    for p in фотографии:
        печать(f"  • {p.name}")
    if not да_нет("Продолжить обработку?", True):
        return

    for фото in фотографии:
        обработать_одно_фото(сессия, фото, cal, переносить=True)

    печать(f"\nГотово. Результаты: {сессия / 'результаты'}")


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
        печать("\n" + "=" * 56)
        печать("Задача 2 — спектрометр с диска (problem2.py)")
        печать("=" * 56)
        печать(f"Сессия:     {сессия}")
        печать(f"Калибровка: {статус_калибровки(сессия)}")
        печать(f"Во «вход»:  {n_in} фото")
        выбор = спросить_выбор(
            "\nЧто сделать?",
            {
                "1": "Откалибровать",
                "2": "Обработать фото из «вход»",
                "3": "Калибровать и сразу обработать",
                "4": "Сменить / создать сессию",
                "5": "Выход",
            },
        )
        if выбор == "1":
            режим_калибровать(сессия)
        elif выбор == "2":
            режим_обработать(сессия)
        elif выбор == "3":
            режим_калибровать_и_обработать(сессия)
        elif выбор == "4":
            сессия = выбрать_сессию()
            обеспечить_структуру_сессии(сессия)
        else:
            печать("Выход.")
            break


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    печать("Задача 2 IYPT — обработка спектров с CD/DVD")
    арг_сессии = argv[0] if argv else None
    ПАПКА_СЕССИЙ.mkdir(parents=True, exist_ok=True)
    сессия = выбрать_сессию(арг_сессии)
    обеспечить_структуру_сессии(сессия)
    главное_меню(сессия)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
