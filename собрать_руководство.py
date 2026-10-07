#!/usr/bin/env python3
"""Собирает PDF-руководство рядом со скриптом."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

КОРЕНЬ = Path(__file__).resolve().parent
ВЫХОД = КОРЕНЬ / "руководство.pdf"

pdfmetrics.registerFont(TTFont("DejaVu", "/Library/Fonts/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuBold", "/Library/Fonts/DejaVuSans-Bold.ttf"))

ЧЕРНИЛА = colors.HexColor("#1B1916")
АКЦЕНТ = colors.HexColor("#B84E2B")
ФОН = colors.HexColor("#F4F0E8")
СЕТКА = colors.HexColor("#E4DDD0")
ПРИГЛУШ = colors.HexColor("#6E675C")


def стили():
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle(
            "cover",
            fontName="DejaVuBold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=ЧЕРНИЛА,
            spaceAfter=8,
        ),
        "sub": ParagraphStyle(
            "sub",
            fontName="DejaVu",
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=ПРИГЛУШ,
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="DejaVuBold",
            fontSize=14,
            leading=18,
            textColor=АКЦЕНТ,
            spaceBefore=16,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="DejaVuBold",
            fontSize=11.5,
            leading=15,
            textColor=ЧЕРНИЛА,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="DejaVu",
            fontSize=9.5,
            leading=13.5,
            alignment=TA_JUSTIFY,
            textColor=ЧЕРНИЛА,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="DejaVu",
            fontSize=9.5,
            leading=13,
            textColor=ЧЕРНИЛА,
            leftIndent=4,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="DejaVu",
            fontSize=8.5,
            leading=11.5,
            textColor=ПРИГЛУШ,
        ),
        "cell": ParagraphStyle(
            "cell",
            fontName="DejaVu",
            fontSize=8.2,
            leading=11,
            textColor=ЧЕРНИЛА,
        ),
        "cellb": ParagraphStyle(
            "cellb",
            fontName="DejaVuBold",
            fontSize=8.2,
            leading=11,
            textColor=ЧЕРНИЛА,
        ),
    }


def P(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def таблица(rows, col_widths):
    data = []
    for i, row in enumerate(rows):
        st_key = "cellb" if i == 0 else "cell"
        # rows already Paragraphs or strings
        cells = []
        for j, c in enumerate(row):
            if isinstance(c, Paragraph):
                cells.append(c)
            else:
                cells.append(Paragraph(str(c), стили()[st_key if i == 0 or j == 0 else "cell"]))
        data.append(cells)
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDE6DA")),
                ("TEXTCOLOR", (0, 0), (-1, -1), ЧЕРНИЛА),
                ("FONTNAME", (0, 0), (-1, 0), "DejaVuBold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, СЕТКА),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAF7F2")]),
            ]
        )
    )
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(ПРИГЛУШ)
    canvas.setFont("DejaVu", 8)
    canvas.drawString(18 * mm, 12 * mm, "disc-spectrometer · задача 2 IYPT")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"{doc.page}")
    canvas.setStrokeColor(СЕТКА)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, 16 * mm, A4[0] - 18 * mm, 16 * mm)
    canvas.restoreState()


def main():
    s = стили()
    story = []

    story.append(Spacer(1, 28 * mm))
    story.append(P("Спектрометр с диска", s["cover"]))
    story.append(P("Алгоритм, физика и расшифровка результатов", s["sub"]))
    story.append(
        P(
            "Скрипт <b>problem2.py</b> · обработка фото эмиссионных спектров<br/>"
            "CD/DVD как отражательная дифракционная решётка",
            s["sub"],
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="80%", thickness=1.2, color=АКЦЕНТ, spaceBefore=4, spaceAfter=4))
    story.append(
        P(
            "Документ для команды: как устроена физика измерения, "
            "что считает программа и что означает каждый файл и график.",
            s["body"],
        )
    )
    story.append(PageBreak())

    # --- 1. Физика ---
    story.append(P("1. Алгоритм с точки зрения физики", s["h1"]))
    story.append(
        P(
            "Оптический диск (CD ≈ 1,6 мкм шаг дорожек, DVD ≈ 0,74 мкм) работает как "
            "<b>отражательная дифракционная решётка</b>. Свет от источника через щель "
            "падает на диск и разлагается по длинам волн: разные λ уходят под разными углами. "
            "Камера фиксирует картину линий — эмиссионный спектр (например, неона).",
            s["body"],
        )
    )
    story.append(P("Цепочка измерения", s["h2"]))
    for t in [
        "<b>Источник</b> → атомные/ионные линии с известными или неизвестными λ.",
        "<b>Щель</b> задаёт геометрическую ширину изображения линии и влияет на FWHM.",
        "<b>Решётка (диск)</b> даёт угловую дисперсию; DVD обычно даёт выше разрешение, чем CD.",
        "<b>Камера</b> переводит угол в координату пикселя на матрице.",
        "<b>Программа</b> строит профиль яркости, находит пики, калибрует пиксель → нм и считает метрики разрешения.",
    ]:
        story.append(P("• " + t, s["bullet"]))

    story.append(P("Что делает программа по шагам", s["h2"]))
    for t in [
        "<b>ROI</b> — находит колонку/полосу спектра на чёрном фоне (OpenCV).",
        "<b>Профиль I(x)</b> — усредняет яркость поперёк спектра → одномерный график интенсивности вдоль дисперсии.",
        "<b>Пики</b> — локальные максимумы (линии). Важны высота над фоном и <i>prominence</i> (насколько пик торчит из соседних впадин).",
        "<b>Калибровка</b> — по известным линиям (Ne/H/He) строится λ(p) = полином от пикселя.",
        "<b>FWHM</b> — ширина линии на полувысоте → оценка инструментальной ширины и R ≈ λ / FWHM.",
        "<b>Сверка NIST</b> — найденные λ сравниваются со справочником; смотрим ошибку шкалы Δ.",
        "<b>SNR</b> — отношение (пик − фон) / шум фона.",
        "<b>Вклад щели</b> — оценка, какая часть FWHM связана со щелью (узкие линии ≈ пол щели).",
    ]:
        story.append(P("• " + t, s["bullet"]))

    story.append(P("Зачем это для задачи IYPT", s["h2"]))
    story.append(
        P(
            "Нужно не просто «увидеть радугу», а показать, что конструкция "
            "<b>разрешает близкие линии</b> (для неона — жёлто-красная серия) и оценить "
            "разрешающую способность. Сравнение щелей, CD/DVD, углов — через FWHM, R, Δλ между соседями.",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(P("2. Аббревиатуры и термины", s["h1"]))

    abbr = [
        ["Обозначение", "Расшифровка"],
        ["λ (лямбда)", "Длина волны света, обычно в нанометрах (нм)."],
        ["нм", "Нанометр = 10⁻⁹ м. Видимый свет ≈ 380–780 нм."],
        ["ROI", "Region of Interest — область кадра со спектром."],
        ["I, R, G, B", "Яркость (grayscale) и каналы камеры Red/Green/Blue."],
        ["FWHM", "Full Width at Half Maximum — ширина пика на половине высоты."],
        ["R", "Разрешающая способность ≈ λ / δλ; часто δλ ≈ FWHM."],
        ["Δλ", "Разность длин волн соседних линий."],
        ["SNR", "Signal-to-Noise Ratio — отношение сигнала к шуму."],
        ["NIST", "Справочник атомных линий (здесь — csv в «справочники/»)."],
        ["DVD / CD", "Диски как решётки (~1350 и ~625 штрихов/мм)."],
        ["дисперсия", "Сколько нм приходится на 1 пиксель (нм/пикс)."],
        ["калибровка", "Перевод координаты пикселя в длину волны."],
    ]
    story.append(таблица(abbr, [32 * mm, 140 * mm]))

    story.append(PageBreak())
    story.append(P("3. Папки и файлы", s["h1"]))
    story.append(
        P(
            "Работа идёт в <b>папке эксперимента</b> внутри <font face='DejaVu'>сессии/…</font>. "
            "Одна папка = одна настройка установки.",
            s["body"],
        )
    )

    folders = [
        ["Путь", "Назначение"],
        ["вход/", "Новые необработанные фото."],
        ["обработанные/", "Сюда переносятся фото после обработки."],
        ["калибровка/калибровка.json", "Формула λ(пиксель) и эталонные пары."],
        ["калибровка/эталоны/", "Кадры, по которым строили шкалу."],
        ["параметры.json", "Опционально: проекция_щели_пикс."],
        ["результаты/<имя>/", "Все графики и таблицы по одному фото."],
        ["результаты/сводка.md", "Сводная таблица по всем фото сессии."],
        ["справочники/*.csv", "Известные линии Ne / H / He."],
    ]
    story.append(таблица(folders, [58 * mm, 114 * mm]))

    story.append(Spacer(1, 4 * mm))
    story.append(P("Файлы внутри результаты/&lt;имя&gt;/", s["h2"]))
    files = [
        ["Файл", "Что это"],
        ["превью.jpg", "Исходник с рамкой ROI и метками пиков."],
        ["профиль.csv", "Интенсивность I/R/G/B вдоль спектра (± λ)."],
        ["пики.csv", "Список пиков: λ, FWHM, зазор до следующего."],
        ["метрики.json", "Числовые итоги: R, SNR, NIST-ошибка, вклад щели…"],
        ["отчёт.md", "Короткое текстовое резюме."],
        ["использованная_калибровка.json", "Какая шкала применялась."],
        ["сверка_nist.csv", "Сопоставление пиков со справочником."],
    ]
    story.append(таблица(files, [58 * mm, 114 * mm]))

    story.append(PageBreak())
    story.append(P("4. Графики", s["h1"]))
    plots = [
        ["График", "Смысл"],
        ["спектр.png", "Главный профиль I(λ), заливка цветом видимого спектра, подписи пиков."],
        ["карта_линий.png", "Линейка найденных линий (как «штрихкод» спектра)."],
        ["близкие_пары.png", "Самые близкие Δλ — ключ к демонстрации разрешения неона."],
        ["fwhm.png", "Ширина каждой линии; чем уже, тем лучше."],
        ["разрешение.png", "R = λ / FWHM по пикам."],
        ["критерий_разрешения.png", "Δλ против FWHM: выше диагонали — пара разрешена."],
        ["дисперсия.png", "Локально нм на пиксель вдоль спектра."],
        ["сверка_nist.png", "Спектр + пунктир NIST и график ошибок Δ."],
        ["snr.png", "Отношение сигнал/шум по линиям."],
        ["вклад_щели.png", "Столбики: оценка щели + «прочее» в FWHM."],
        ["каналы_rgb.png", "Ответы матрицы R/G/B (не «цвет волны»)."],
    ]
    story.append(таблица(plots, [48 * mm, 124 * mm]))

    story.append(Spacer(1, 5 * mm))
    story.append(P("5. Типичный вопрос: почему пик «пропал»?", s["h1"]))
    story.append(
        P(
            "Поиск пиков отсекает слабые максимумы по <b>высоте над фоном</b> и по "
            "<b>prominence</b>. Пример: линия неона ~597,5 нм сидит между 594 и 603 нм; "
            "её яркость чуть ниже порога высоты, хотя «торчит» достаточно. "
            "Порог смягчён, чтобы такие линии не терялись. "
            "Если пик всё ещё не ловится — проверьте экспозицию и сглаживание.",
            s["body"],
        )
    )

    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=СЕТКА))
    story.append(
        P(
            "Запуск: <font face='DejaVu'>.venv/bin/python problem2.py</font><br/>"
            "Обновить этот PDF: <font face='DejaVu'>.venv/bin/python собрать_руководство.py</font>",
            s["small"],
        )
    )

    doc = SimpleDocTemplate(
        str(ВЫХОД),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="Спектрометр с диска — руководство",
        author="disc-spectrometer",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"OK → {ВЫХОД}")


if __name__ == "__main__":
    main()
