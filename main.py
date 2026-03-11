"""
Telegram-бот для матчинга резюме и вакансий.

Изменения (OOM-fix + баги):
1. Модель загружается лениво (lazy), а не при импорте — экономит RAM при старте.
2. PyTorch заменён на ONNX Runtime — экономит ~200MB RAM.
3. gc.collect() после тяжёлых операций — возвращает RAM ОС.
4. Убран дублированный блок кода после callback_exp_done (мёртвый код с salary).
5. logger вместо смеси print/logging.
6. Импорты вынесены на верхний уровень (не внутри функций).
7. Lazy pipeline: get_pipeline() вызывается один раз при первом обращении.
"""

import os
import gc
import re
import json
import asyncio
import logging
from datetime import datetime

import httpx
import numpy as np
import pytz
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Reassemble split ONNX model before importing the pipeline
import reassemble_model
reassemble_model.reassemble()

from resume_parser import ResumeParser
from embedding_pipeline import get_pipeline
import database as db

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ── Ленивая загрузка тяжёлых объектов ─────────
# Модель НЕ грузится при импорте — только при первом вызове.
# Это даёт Railway время поднять контейнер без OOM на старте.

resume_parser = ResumeParser()

_pipeline = None


def pipeline():
    """Lazy-загрузка AI-пайплайна. Вызывается один раз."""
    global _pipeline
    if _pipeline is None:
        logger.info("⏳ Загружаю AI модель...")
        _pipeline = get_pipeline()
        gc.collect()
        logger.info("✅ AI модель готова")
    return _pipeline


def embed_resume_safe(resume):
    """Эмбеддинг резюме с защитой от утечки памяти."""
    vec = pipeline().embed_resume(resume)
    gc.collect()
    return vec


def encode_text_safe(text: str):
    """Эмбеддинг произвольного текста с защитой от утечки памяти."""
    vec = pipeline().model.encode(text, normalize_embeddings=True)
    gc.collect()
    return vec


# ── Состояния редактирования ──────────────────
EDIT_POSITION = "edit_position"
EDIT_SKILLS = "edit_skills"
EDIT_EXPERIENCE = "edit_experience"


# ─────────────────────────────────────────────
# /start
# ─────────────────────────────────────────────


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await db.upsert_user(user.id, user.username, user.full_name)

    saved = await db.get_resume(user.id)

    if saved:
        sub = await db.get_subscription(user.id)
        sub_btn_text = "🔔 Подписка: ВКЛ" if sub else "🔕 Включить подписку"
        position = saved.get("position") or "не указана"
        updated = saved["updated_at"].strftime("%d.%m.%Y")
        await update.message.reply_text(
            f"👋 Привет, {user.first_name}!\n\n"
            f"📋 У тебя есть сохранённое резюме:\n"
            f"💼 {position}\n"
            f"🗓 Обновлено: {updated}\n",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("📋 Моё резюме", callback_data="show_myresume"),
                        InlineKeyboardButton("🚀 Найти вакансии", callback_data="search_start"),
                    ],
                    [
                        InlineKeyboardButton(sub_btn_text, callback_data="sub_toggle"),
                        InlineKeyboardButton("⭐️ Избранное", callback_data="show_favorites"),
                    ],
                    [InlineKeyboardButton("📄 Загрузить новое PDF", callback_data="upload_hint")],
                    [
                        InlineKeyboardButton(
                            "👔 Режим HR — найти кандидатов",
                            callback_data="hr_search_start",
                        )
                    ],
                ]
            ),
        )
    else:
        await update.message.reply_text(
            f"👋 Привет, {user.first_name}!\n\n"
            "Я помогу найти подходящие вакансии по твоему резюме.\n\n"
            "С чего начнём:\n"
            "• Отправь резюме PDF → разберу и найду вакансии\n"
            "• Отправь ссылку hh.ru/resume/... → разберу профиль\n"
            "• Напиши специальность → найду вакансии по запросу\n\n"
            "💡 После загрузки резюме можешь подписаться на автоматическую рассылку вакансий — "
            "бот будет присылать новые подборки в удобное время.",
            reply_markup=InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("📄 Загрузить PDF", callback_data="upload_hint")],
                    [InlineKeyboardButton("⭐️ Избранное", callback_data="show_favorites")],
                    [
                        InlineKeyboardButton(
                            "👔 Режим HR — найти кандидатов",
                            callback_data="hr_search_start",
                        )
                    ],
                ]
            ),
        )


# ─────────────────────────────────────────────
# Обработка PDF
# ─────────────────────────────────────────────


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    doc = update.message.document

    if doc.mime_type != "application/pdf":
        await update.message.reply_text("⚠️ Пожалуйста, отправь файл в формате PDF")
        return

    if doc.file_size > 5 * 1024 * 1024:
        await update.message.reply_text("⚠️ Файл слишком большой. Максимум 5MB")
        return

    await update.message.reply_text("⏳ Читаю резюме...")

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        pdf_bytes = await tg_file.download_as_bytearray()
        resume = resume_parser.parse_pdf_bytes(bytes(pdf_bytes))

        if not resume.is_valid():
            await update.message.reply_text(
                "😔 Не удалось извлечь данные из резюме.\n\n"
                "Возможные причины:\n"
                "• PDF создан как изображение (сканированный)\n"
                "• Нестандартное форматирование\n"
                "• Попробуй экспортировать из Word/Google Docs заново"
            )
            return

        await db.upsert_user(user.id, user.username, user.full_name)
        embedding = embed_resume_safe(resume).tolist()
        resume_id = await db.save_resume(user.id, resume, embedding)

        context.user_data["resume"] = resume
        context.user_data["resume_id"] = resume_id

        sub = await db.get_subscription(user.id)
        await update.message.reply_text(
            f"✅ Резюме разобрано и сохранено!\n\n{resume.summary()}\n\n"
            f"Всего навыков: {len(resume.skills)}\n\n"
            "Проверь данные — от этого зависит качество матчинга.",
            reply_markup=_edit_or_search_keyboard(subscribed=bool(sub)),
        )

    except Exception as e:
        logger.error("Ошибка обработки PDF: %s", e, exc_info=True)
        await update.message.reply_text(f"❌ Ошибка: {e}")


def _edit_or_search_keyboard(subscribed: bool = False) -> InlineKeyboardMarkup:
    sub_btn = (
        InlineKeyboardButton("🔔 Подписка: ВКЛ", callback_data="sub_toggle")
        if subscribed
        else InlineKeyboardButton("🔕 Включить подписку", callback_data="sub_toggle")
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✏️ Редактировать", callback_data="edit_start"),
                InlineKeyboardButton("🚀 Начать поиск", callback_data="search_start"),
            ],
            [sub_btn],
            [InlineKeyboardButton("⭐️ Избранное", callback_data="show_favorites")],
        ]
    )


# ─────────────────────────────────────────────
# Кнопка "Начать поиск"
# ─────────────────────────────────────────────


async def callback_search_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user

    resume = context.user_data.get("resume")
    if not resume:
        saved = await db.get_resume(user.id)
        if not saved:
            await query.message.reply_text("⚠️ Резюме не найдено. Загрузи PDF заново.")
            return

    await show_search_filters(query.message, user.id, context)


async def show_search_filters(message, user_id: int, context):
    """Показывает текущие фильтры и кнопки настройки."""
    prefs = context.user_data.get("prefs")
    if not prefs:
        prefs = await db.get_preferences(user_id)
        context.user_data["prefs"] = prefs

    areas_text = ", ".join(prefs.get("area_names") or ["Москва"])
    salary_from = prefs.get("salary_from")
    salary_to = prefs.get("salary_to")
    remote = prefs.get("remote_only", False)
    show_no_salary = prefs.get("show_without_salary", True)
    experience = prefs.get("experience") or [
        "noExperience",
        "between1And3",
        "between3And6",
        "moreThan6",
    ]

    if salary_from and salary_to:
        salary_text = f"{salary_from:,} – {salary_to:,} RUB"
    elif salary_from:
        salary_text = f"от {salary_from:,} RUB"
    elif salary_to:
        salary_text = f"до {salary_to:,} RUB"
    else:
        salary_text = "не указана"

    remote_text = " + удалённо" if remote else ""
    no_salary_text = "✅ показывать" if show_no_salary else "❌ скрывать"

    exp_labels = {
        "noExperience": "без опыта",
        "between1And3": "1-3 года",
        "between3And6": "3-6 лет",
        "moreThan6": "6+ лет",
    }
    exp_text = ", ".join(exp_labels[e] for e in experience if e in exp_labels) or "не выбран"

    await message.reply_text(
        f"⚙️ Параметры поиска:\n\n"
        f"🌍 Локация: {areas_text}{remote_text}\n"
        f"💰 Зарплата: {salary_text}\n"
        f"🙈 Без зарплаты: {no_salary_text}\n"
        f"📅 Опыт: {exp_text}\n\n"
        f"Можешь изменить или сразу искать:",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🌍 Локация", callback_data="filter_location"),
                    InlineKeyboardButton("💰 Зарплата", callback_data="filter_salary"),
                ],
                [
                    InlineKeyboardButton("📅 Опыт", callback_data="filter_experience"),
                    InlineKeyboardButton(
                        f"🙈 Без зарплаты: {'ВКЛ' if show_no_salary else 'ВЫКЛ'}",
                        callback_data="filter_toggle_no_salary",
                    ),
                ],
                [InlineKeyboardButton("🔍 Искать", callback_data="do_search")],
            ]
        ),
    )


# ── Популярные локации (hh.ru area IDs) ───────
POPULAR_AREAS = [
    (113, "Вся Россия"),
    (1, "Москва"),
    (2, "Санкт-Петербург"),
    (4, "Новосибирск"),
    (3, "Екатеринбург"),
    (1438, "Казахстан"),
    (1101, "Беларусь"),
    (5, "Нижний Новгород"),
    (76, "Ростов-на-Дону"),
    (88, "Краснодар"),
    (0, "🌐 Удалённо (весь мир)"),
]


async def callback_filter_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await _show_location_picker(query.message, context)


async def _show_location_picker(message, context):
    selected = context.user_data.get(
        "filter_areas_selected",
        list(context.user_data.get("prefs", {}).get("areas", [1])),
    )
    custom = context.user_data.get("filter_custom_areas", {})

    buttons = []
    row = []
    for area_id, name in POPULAR_AREAS:
        mark = "✅" if area_id in selected else "⬜"
        row.append(InlineKeyboardButton(f"{mark} {name}", callback_data=f"area_{area_id}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    for area_id, name in custom.items():
        if area_id not in dict(POPULAR_AREAS):
            mark = "✅" if area_id in selected else "⬜"
            buttons.append(
                [InlineKeyboardButton(f"{mark} {name} ✕", callback_data=f"area_{area_id}")]
            )

    buttons.append(
        [
            InlineKeyboardButton("🔎 Другой город", callback_data="area_custom"),
            InlineKeyboardButton("🗑 Сбросить всё", callback_data="area_reset"),
            InlineKeyboardButton("✔️ Готово", callback_data="area_done"),
        ]
    )

    await message.reply_text(
        "🌍 Выбери локации (можно несколько):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def callback_area_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    area_id = int(query.data.replace("area_", ""))
    selected = context.user_data.get(
        "filter_areas_selected",
        list(context.user_data.get("prefs", {}).get("areas", [1])),
    )
    custom = context.user_data.get("filter_custom_areas", {})

    if area_id in selected:
        selected.remove(area_id)
    else:
        selected.append(area_id)
    context.user_data["filter_areas_selected"] = selected

    # Перерисовываем
    buttons = []
    row = []
    for aid, name in POPULAR_AREAS:
        mark = "✅" if aid in selected else "⬜"
        row.append(InlineKeyboardButton(f"{mark} {name}", callback_data=f"area_{aid}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    for cid, cname in custom.items():
        if cid not in dict(POPULAR_AREAS):
            mark = "✅" if cid in selected else "⬜"
            buttons.append(
                [InlineKeyboardButton(f"{mark} {cname} ✕", callback_data=f"area_{cid}")]
            )

    buttons.append(
        [
            InlineKeyboardButton("🔎 Другой город", callback_data="area_custom"),
            InlineKeyboardButton("🗑 Сбросить всё", callback_data="area_reset"),
            InlineKeyboardButton("✔️ Готово", callback_data="area_done"),
        ]
    )
    await query.edit_message_reply_markup(InlineKeyboardMarkup(buttons))


async def callback_area_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сбросить все выбранные локации."""
    query = update.callback_query
    await query.answer("Все локации сброшены")
    context.user_data["filter_areas_selected"] = []
    context.user_data["filter_custom_areas"] = {}
    prefs = context.user_data.get("prefs", {})
    prefs["areas"] = []
    prefs["area_names"] = []
    context.user_data["prefs"] = prefs
    await _show_location_picker(query.message, context)


async def callback_area_custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["awaiting_custom_city"] = True
    await query.message.reply_text(
        "🔎 Введи название города (на русском):\n" "например: Алматы, Минск, Тбилиси"
    )


async def callback_area_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    selected = context.user_data.get("filter_areas_selected", [1])
    if not selected:
        await query.answer("Выбери хотя бы одну локацию!", show_alert=True)
        return

    area_map = dict(POPULAR_AREAS)
    custom = context.user_data.get("filter_custom_areas", {})
    area_map.update(custom)

    names = [area_map.get(a, str(a)) for a in selected]

    prefs = context.user_data.get("prefs", {})
    prefs["areas"] = selected
    prefs["area_names"] = names
    context.user_data["prefs"] = prefs
    context.user_data.pop("filter_areas_selected", None)

    await show_search_filters(query.message, update.effective_user.id, context)


async def callback_filter_salary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    prefs = context.user_data.get("prefs", {})
    salary_from = prefs.get("salary_from", "")
    salary_to = prefs.get("salary_to", "")

    context.user_data["awaiting_salary"] = True
    await query.message.reply_text(
        "💰 Введи диапазон зарплаты в рублях:\n\n"
        "Формат: <от>-<до>\n"
        "• 100000-300000 — от 100k до 300k\n"
        "• 150000- — только от 150k (без верхнего лимита)\n"
        "• -250000 — только до 250k (без нижнего лимита)\n"
        "• /skip — без ограничений по зарплате\n\n"
        f"Текущее значение: {salary_from or '∞'} – {salary_to or '∞'} RUB"
    )


async def callback_filter_toggle_no_salary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переключатель — показывать/скрывать вакансии без зарплаты."""
    query = update.callback_query
    await query.answer()
    prefs = context.user_data.get("prefs", {})
    prefs["show_without_salary"] = not prefs.get("show_without_salary", True)
    context.user_data["prefs"] = prefs
    await show_search_filters(query.message, update.effective_user.id, context)


EXP_OPTIONS = [
    ("noExperience", "Без опыта"),
    ("between1And3", "1–3 года"),
    ("between3And6", "3–6 лет"),
    ("moreThan6", "6+ лет"),
]


async def callback_filter_experience(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await _show_experience_picker(query.message, context)


async def _show_experience_picker(message, context):
    selected = context.user_data.get("prefs", {}).get(
        "experience", ["noExperience", "between1And3", "between3And6", "moreThan6"]
    )
    buttons = []
    for key, label in EXP_OPTIONS:
        mark = "✅" if key in selected else "⬜"
        buttons.append([InlineKeyboardButton(f"{mark} {label}", callback_data=f"exp_{key}")])
    buttons.append([InlineKeyboardButton("✔️ Готово", callback_data="exp_done")])

    await message.reply_text(
        "📅 Выбери требуемый опыт (можно несколько):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def callback_exp_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    key = query.data.replace("exp_", "")
    prefs = context.user_data.get("prefs", {})
    selected = list(
        prefs.get("experience")
        or ["noExperience", "between1And3", "between3And6", "moreThan6"]
    )

    if key in selected:
        if len(selected) == 1:
            await query.answer("Выбери хотя бы один вариант!", show_alert=True)
            return
        selected.remove(key)
    else:
        selected.append(key)

    prefs["experience"] = selected
    context.user_data["prefs"] = prefs

    buttons = []
    for k, label in EXP_OPTIONS:
        mark = "✅" if k in selected else "⬜"
        buttons.append([InlineKeyboardButton(f"{mark} {label}", callback_data=f"exp_{k}")])
    buttons.append([InlineKeyboardButton("✔️ Готово", callback_data="exp_done")])
    await query.edit_message_reply_markup(InlineKeyboardMarkup(buttons))


async def callback_exp_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await show_search_filters(query.message, update.effective_user.id, context)


# NOTE: Дублированный блок с salary, который был здесь в оригинале, УДАЛЁН.
# Он содержал неиндентированный код (query = update.callback_query...)
# после callback_exp_done, который никогда не выполнялся корректно.


async def callback_do_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user

    prefs = context.user_data.get("prefs", {})

    await db.save_preferences(
        user_id=user.id,
        areas=prefs.get("areas", [1]),
        area_names=prefs.get("area_names", ["Москва"]),
        salary_from=prefs.get("salary_from"),
        salary_to=prefs.get("salary_to"),
        remote_only=prefs.get("remote_only", False),
        show_without_salary=prefs.get("show_without_salary", True),
        experience=prefs.get("experience")
        or ["noExperience", "between1And3", "between3And6", "moreThan6"],
    )

    resume = context.user_data.get("resume")
    position = resume.desired_position if resume else None

    if not position:
        saved = await db.get_resume(user.id)
        position = saved.get("position") if saved else "аналитик"

    areas = prefs.get("areas", [1])
    area_names = prefs.get("area_names", ["Москва"])
    salary_from = prefs.get("salary_from")
    salary_to = prefs.get("salary_to")

    areas_text = ", ".join(area_names)
    await query.message.reply_text(
        f"🔍 Ищу «{position}»\n" f"🌍 {areas_text}\n" f"⏳ ~15 секунд..."
    )

    all_vacancies = []
    experience = prefs.get("experience") or [
        "noExperience",
        "between1And3",
        "between3And6",
        "moreThan6",
    ]
    show_without_salary = prefs.get("show_without_salary", True)

    hh_areas = [a for a in areas if a != 0]
    use_remotive = 0 in areas
    logger.debug("Поиск: areas=%s, hh_areas=%s, use_remotive=%s", areas, hh_areas, use_remotive)

    async def fetch_one(area_id, exp):
        params = {"per_page": 100, "experience": exp}
        if salary_from:
            params["salary"] = salary_from
        if not show_without_salary or salary_from:
            params["only_with_salary"] = str(not show_without_salary).lower()
        if prefs.get("remote_only"):
            params["schedule"] = "remote"
        return await fetch_vacancies(position, area=area_id, limit=200, **params)

    tasks = [fetch_one(area_id, exp) for area_id in hh_areas for exp in experience]
    if use_remotive:
        tasks.append(fetch_remoteok(position, limit=100))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_vacancies.extend(r)

    # Дедупликация
    seen = set()
    unique = []
    for v in all_vacancies:
        vid = v.get("id")
        if vid not in seen:
            seen.add(vid)
            if salary_to:
                s = v.get("salary")
                if s:
                    v_from = s.get("from") or 0
                    if v_from > salary_to:
                        continue
            unique.append(v)

    if not unique:
        await query.message.reply_text("😔 Вакансий не найдено. Попробуй изменить фильтры.")
        return

    await query.message.reply_text(f"🤖 AI анализирует {len(unique)} вакансий...")

    if resume:
        resume_vector = embed_resume_safe(resume)
    else:
        saved = await db.get_resume(user.id)
        resume_vector = np.array(json.loads(saved.get("embedding", "[]")), dtype=np.float32)

    matches = pipeline().match(resume_vector, unique, top_k=50)
    gc.collect()

    context.user_data["matches"] = matches
    await _send_matches(
        query.message, matches, user_id=user.id, context=context, total=len(matches)
    )


async def handle_filter_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Обрабатывает текстовый ввод для фильтров. Возвращает True если обработал."""

    # Ввод зарплаты
    if context.user_data.get("awaiting_salary"):
        context.user_data.pop("awaiting_salary")
        text = update.message.text.strip()

        if text == "/skip":
            prefs = context.user_data.get("prefs", {})
            prefs["salary_from"] = None
            prefs["salary_to"] = None
            context.user_data["prefs"] = prefs
        else:
            m = re.match(r"^(\d*)-(\d*)$", text.replace(" ", ""))
            if not m:
                await update.message.reply_text(
                    "⚠️ Неверный формат. Примеры:\n"
                    "100000-300000 или 150000- или -250000 или /skip"
                )
                context.user_data["awaiting_salary"] = True
                return True

            from_val = int(m.group(1)) if m.group(1) else None
            to_val = int(m.group(2)) if m.group(2) else None

            prefs = context.user_data.get("prefs", {})
            prefs["salary_from"] = from_val
            prefs["salary_to"] = to_val
            context.user_data["prefs"] = prefs

        await show_search_filters(update.message, update.effective_user.id, context)
        return True

    # Ввод кастомного города
    if context.user_data.get("awaiting_custom_city"):
        context.user_data.pop("awaiting_custom_city")
        city_name = update.message.text.strip()

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.hh.ru/suggests/areas", params={"text": city_name}
                )
                results = resp.json().get("items", [])
        except Exception:
            results = []

        if not results:
            await update.message.reply_text(
                f"😔 Город «{city_name}» не найден. Попробуй другое название."
            )
            return True

        buttons = []
        for item in results[:5]:
            area_id = int(item["id"])
            name = item["text"]
            buttons.append(
                [InlineKeyboardButton(name, callback_data=f"addarea_{area_id}_{name[:20]}")]
            )
        buttons.append([InlineKeyboardButton("❌ Отмена", callback_data="area_done")])

        await update.message.reply_text(
            "Выбери город:", reply_markup=InlineKeyboardMarkup(buttons)
        )
        return True

    return False


async def callback_add_custom_area(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    parts = query.data.replace("addarea_", "").split("_", 1)
    area_id = int(parts[0])
    name = parts[1] if len(parts) > 1 else str(area_id)

    selected = context.user_data.get(
        "filter_areas_selected",
        list(context.user_data.get("prefs", {}).get("areas", [1])),
    )
    if area_id not in selected:
        selected.append(area_id)
    context.user_data["filter_areas_selected"] = selected

    custom = context.user_data.get("filter_custom_areas", {})
    custom[area_id] = name
    context.user_data["filter_custom_areas"] = custom

    await query.message.reply_text(f"✅ «{name}» добавлен в список.")
    await _show_location_picker(query.message, context)


# ─────────────────────────────────────────────
# Кнопка "Редактировать"
# ─────────────────────────────────────────────


async def callback_edit_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    resume = context.user_data.get("resume")
    if not resume:
        await query.message.reply_text("⚠️ Резюме не найдено. Загрузи PDF заново.")
        return

    context.user_data["edit_step"] = EDIT_POSITION
    current = resume.desired_position or "не определена"
    await query.message.reply_text(
        f"✏️ Шаг 1/3 — Должность\n\n"
        f"Сейчас: {current}\n\n"
        f"Введи новую должность или /skip чтобы оставить как есть:"
    )


async def handle_edit_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    step = context.user_data.get("edit_step")

    # Сначала проверяем ввод времени для подписки
    if await handle_sub_time_input(update, context):
        return

    # Проверяем ввод для фильтров поиска
    if await handle_filter_text_input(update, context):
        return

    # Проверяем ввод текста вакансии для HR режима
    if await handle_hr_vacancy_input(update, context):
        return

    if not step:
        await handle_text(update, context)
        return

    text = update.message.text.strip()
    resume = context.user_data.get("resume")
    user = update.effective_user

    if step == EDIT_POSITION:
        if text != "/skip":
            resume.desired_position = text
        context.user_data["edit_step"] = EDIT_SKILLS
        current_skills = ", ".join(resume.skills) if resume.skills else "не определены"
        await update.message.reply_text(
            f"✏️ Шаг 2/3 — Навыки\n\n"
            f"Сейчас: {current_skills}\n\n"
            f"Введи навыки через запятую или /skip:"
        )

    elif step == EDIT_SKILLS:
        if text != "/skip":
            resume.skills = [s.strip().lower() for s in text.split(",") if s.strip()]
        context.user_data["edit_step"] = EDIT_EXPERIENCE
        current_exp = f"{resume.experience_years} лет" if resume.experience_years else "не определён"
        await update.message.reply_text(
            f"✏️ Шаг 3/3 — Опыт\n\n"
            f"Сейчас: {current_exp}\n\n"
            f"Введи количество лет или /skip:"
        )

    elif step == EDIT_EXPERIENCE:
        if text != "/skip":
            try:
                resume.experience_years = float(text.replace(",", "."))
            except ValueError:
                await update.message.reply_text("⚠️ Введи число, например: 3 или 3.5")
                return

        context.user_data["edit_step"] = None
        context.user_data["resume"] = resume

        try:
            embedding = embed_resume_safe(resume).tolist()
            resume_id = await db.save_resume(user.id, resume, embedding)
            context.user_data["resume_id"] = resume_id
        except Exception as e:
            logger.error("Ошибка сохранения резюме: %s", e)

        await update.message.reply_text(
            f"✅ Данные обновлены!\n\n{resume.summary()}",
            reply_markup=_edit_or_search_keyboard(
                subscribed=bool(await db.get_subscription(user.id))
            ),
        )


# ─────────────────────────────────────────────
# Матчинг
# ─────────────────────────────────────────────


async def run_matching(message, resume, user_id: int = None, context=None):
    """Матчинг из объекта ParsedResume."""
    try:
        search_query = resume.desired_position or "аналитик"
        vacancies = await fetch_vacancies(search_query)

        if not vacancies:
            fallback = search_query.split()[0]
            vacancies = await fetch_vacancies(fallback)

        if not vacancies:
            await message.reply_text("😔 Не удалось получить вакансии")
            return

        await message.reply_text(f"🤖 AI анализирует {len(vacancies)} вакансий...")

        resume_vector = embed_resume_safe(resume)
        matches = pipeline().match(resume_vector, vacancies, top_k=50)
        gc.collect()

        if context:
            context.user_data["matches"] = matches
        await _send_matches(message, matches, user_id=user_id, context=context, total=len(matches))

    except Exception as e:
        logger.error("Ошибка матчинга: %s", e, exc_info=True)
        await message.reply_text(f"❌ Ошибка: {e}")


async def run_matching_from_db(message, saved_resume: dict, user_id: int = None, context=None):
    """Матчинг из сохранённого резюме в БД."""
    try:
        search_query = saved_resume.get("position") or "аналитик"
        vacancies = await fetch_vacancies(search_query)

        if not vacancies:
            vacancies = await fetch_vacancies(search_query.split()[0])

        if not vacancies:
            await message.reply_text("😔 Не удалось получить вакансии")
            return

        await message.reply_text(f"🤖 AI анализирует {len(vacancies)} вакансий...")

        embedding_str = saved_resume.get("embedding", "[]")
        resume_vector = np.array(json.loads(embedding_str), dtype=np.float32)

        matches = pipeline().match(resume_vector, vacancies, top_k=50)
        gc.collect()

        if context:
            context.user_data["matches"] = matches
        await _send_matches(message, matches, user_id=user_id, context=context, total=len(matches))

    except Exception as e:
        logger.error("Ошибка матчинга из БД: %s", e, exc_info=True)
        await message.reply_text(f"❌ Ошибка: {e}")


async def _send_matches(
    message, matches, user_id: int = None, context=None, offset: int = 0, total: int = None
):
    if not matches:
        await message.reply_text("😔 Подходящих вакансий не найдено")
        return

    page_size = 5
    page = matches[offset : offset + page_size]
    total = total or len(matches)

    if offset == 0:
        await message.reply_text(f"🎯 Найдено вакансий: {total} — показываю топ по AI-матчингу:")

    for i, match in enumerate(page, offset + 1):
        is_fav = await db.is_favorite(user_id, match.vacancy_id) if user_id else False
        star = "⭐️ В избранном" if is_fav else "☆ В избранное"

        if context and user_id:
            if "vacancies_cache" not in context.user_data:
                context.user_data["vacancies_cache"] = {}
            context.user_data["vacancies_cache"][match.vacancy_id] = {
                "title": match.title,
                "company": match.company,
                "url": match.url,
                "salary_text": match.salary_text,
            }

        await message.reply_text(
            match.format_message(i),
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton(star, callback_data=f"fav_{match.vacancy_id[:50]}")]]
            ),
        )

    next_offset = offset + page_size
    if next_offset < len(matches):
        remaining = len(matches) - next_offset
        await message.reply_text(
            f"Показано {min(next_offset, len(matches))} из {total}",
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            f"➕ Показать ещё 5 (осталось {remaining})",
                            callback_data=f"more_{next_offset}",
                        )
                    ]
                ]
            ),
        )


# ─────────────────────────────────────────────
# Обычный поиск по тексту
# ─────────────────────────────────────────────


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if text == "/skip":
        return

    if ResumeParser.is_hh_url(text):
        await handle_hh_url(update, context, text)
        return

    await update.message.reply_text(f"⏳ Ищу '{text}'...")

    try:
        vacancies = await fetch_vacancies(text, limit=5)

        if not vacancies:
            await update.message.reply_text("😔 Вакансий не найдено")
            return

        for i, job in enumerate(vacancies, 1):
            salary_text = _format_salary(job)
            msg = (
                f"{i}. {job['name']}\n"
                f"🏢 {job['employer']['name']}"
                f"{salary_text}\n"
                f"🔗 {job['alternate_url']}"
            )
            await update.message.reply_text(msg, disable_web_page_preview=True)

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")


async def handle_hh_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    user = update.effective_user
    await update.message.reply_text("⏳ Загружаю профиль с hh.ru...")

    try:
        resume = await resume_parser.parse_hh_url(url)

        if not resume.is_valid():
            await update.message.reply_text(
                "😔 Не удалось получить данные профиля.\n\n"
                "Возможные причины:\n"
                "• Резюме закрыто (видно только работодателям)\n"
                "• Неверная ссылка — нужен формат hh.ru/resume/ID"
            )
            return

        await db.upsert_user(user.id, user.username, user.full_name)
        embedding = embed_resume_safe(resume).tolist()
        resume_id = await db.save_resume(user.id, resume, embedding)

        context.user_data["resume"] = resume
        context.user_data["resume_id"] = resume_id

        await update.message.reply_text(
            f"✅ Профиль разобран!\n\n{resume.summary()}",
            reply_markup=_edit_or_search_keyboard(
                subscribed=bool(await db.get_subscription(user.id))
            ),
        )

    except Exception as e:
        logger.error("Ошибка парсинга hh.ru: %s", e)
        await update.message.reply_text(f"❌ Ошибка: {e}")


# ─────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────


async def fetch_vacancies(query: str, area: int = 1, limit: int = 500, **kwargs) -> list[dict]:
    """Получает вакансии с hh.ru с пагинацией."""
    per_page = 100
    max_pages = min((limit + per_page - 1) // per_page, 20)
    all_items = []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            for page in range(max_pages):
                params = {
                    "text": query,
                    "area": area,
                    "per_page": per_page,
                    "page": page,
                    "order_by": "relevance",
                }
                params.update(kwargs)

                response = await client.get("https://api.hh.ru/vacancies", params=params)
                response.raise_for_status()
                data = response.json()
                items = data.get("items", [])
                all_items.extend(items)

                total_pages = data.get("pages", 1)
                if page + 1 >= total_pages:
                    break
                if len(all_items) >= limit:
                    break

    except Exception as e:
        logger.error("Ошибка получения вакансий: %s", e)

    return all_items[:limit]


def _format_salary(job: dict) -> str:
    if not job.get("salary"):
        return ""
    s = job["salary"]
    from_val, to_val = s.get("from"), s.get("to")
    currency = s.get("currency", "RUB")
    if from_val and to_val:
        return f"\n💰 {from_val:,}–{to_val:,} {currency}"
    elif from_val:
        return f"\n💰 от {from_val:,} {currency}"
    elif to_val:
        return f"\n💰 до {to_val:,} {currency}"
    return ""


# ─────────────────────────────────────────────
# Резюме / Избранное / Удаление
# ─────────────────────────────────────────────


async def callback_show_myresume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await my_resume(update, context)


async def callback_upload_hint(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("📄 Просто отправь PDF файл в этот чат — я его разберу.")


async def my_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/myresume — показывает сохранённое резюме с кнопками."""
    user = update.effective_user
    reply = update.message.reply_text if update.message else update.callback_query.message.reply_text

    resume = context.user_data.get("resume")

    if resume:
        await reply(
            f"📋 Твоё резюме:\n\n{resume.summary()}\n\n"
            f"Все навыки ({len(resume.skills)}):\n{', '.join(resume.skills)}",
            reply_markup=_myresume_keyboard(),
        )
        return

    saved = await db.get_resume(user.id)

    if not saved:
        await reply("😔 Резюме не найдено.\n\nОтправь PDF файл — я разберу и сохраню его.")
        return

    skills = saved.get("skills") or []
    position = saved.get("position") or "не указана"
    name = saved.get("name") or "не указано"
    exp = saved.get("experience_years")
    edu = saved.get("education") or ""

    exp_text = f"📅 Опыт: {exp:.0f} лет\n" if exp else ""
    edu_text = f"🎓 {edu}\n" if edu else ""

    sub = await db.get_subscription(user.id)
    sub_text = "🔔 Подписка активна — новые вакансии каждое утро в 9:00\n" if sub else ""

    await reply(
        f"📋 Твоё резюме:\n\n"
        f"👤 {name}\n"
        f"💼 {position}\n"
        f"{exp_text}"
        f"🔧 Навыки: {', '.join(skills[:12])}\n"
        f"{edu_text}\n"
        f"Всего навыков: {len(skills)}\n"
        f"Обновлено: {saved['updated_at'].strftime('%d.%m.%Y %H:%M')}\n\n"
        f"{sub_text}",
        reply_markup=_myresume_keyboard(subscribed=bool(sub)),
    )


def _myresume_keyboard(subscribed: bool = False) -> InlineKeyboardMarkup:
    sub_btn = (
        InlineKeyboardButton("🔔 Подписка: ВКЛ", callback_data="sub_toggle")
        if subscribed
        else InlineKeyboardButton("🔕 Подписка: ВЫКЛ", callback_data="sub_toggle")
    )
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✏️ Редактировать", callback_data="edit_start"),
                InlineKeyboardButton("🚀 Найти вакансии", callback_data="search_start"),
            ],
            [sub_btn],
            [
                InlineKeyboardButton("⭐️ Избранное", callback_data="show_favorites"),
                InlineKeyboardButton("🗑 Удалить резюме", callback_data="delete_resume"),
            ],
        ]
    )


async def callback_delete_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text(
        "⚠️ Удалить резюме из базы?\n\nЭто действие нельзя отменить.",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Да, удалить", callback_data="confirm_delete"),
                    InlineKeyboardButton("❌ Отмена", callback_data="cancel_delete"),
                ]
            ]
        ),
    )


async def callback_confirm_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    try:
        pool = await db.get_pool()
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM resumes WHERE user_id = $1", user.id)
        context.user_data.pop("resume", None)
        context.user_data.pop("resume_id", None)
        await query.message.reply_text("✅ Резюме удалено.")
    except Exception as e:
        logger.error("Ошибка удаления резюме: %s", e)
        await query.message.reply_text("❌ Ошибка при удалении.")


async def callback_cancel_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text("Отменено.")


# ─────────────────────────────────────────────
# Подписка
# ─────────────────────────────────────────────


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/subscribe — начать настройку подписки."""
    user = update.effective_user
    msg = update.message or update.callback_query.message

    saved = await db.get_resume(user.id)
    if not saved:
        await msg.reply_text(
            "😔 Сначала загрузи резюме — подписка привязывается к твоим навыкам и должности.",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton("📄 Загрузить PDF", callback_data="upload_hint")]]
            ),
        )
        return

    sub = await db.get_subscription(user.id)
    if sub:
        freq_text = _frequency_label(sub.get("frequency", "daily"))
        days_text = _days_label(sub.get("days", [1, 2, 3, 4, 5]))
        hour = sub.get("send_hour", 9)
        minute = sub.get("send_minute", 0)
        await msg.reply_text(
            f"🔔 Подписка уже активна!\n\n"
            f"📅 Частота: {freq_text}\n"
            f"📆 Дни: {days_text}\n"
            f"🕘 Время: {hour:02d}:{minute:02d} МСК\n\n"
            f"Чтобы отключить — /unsubscribe",
        )
        return

    await _ask_frequency(msg)


async def _ask_frequency(msg):
    await msg.reply_text(
        "⏰ Как часто присылать вакансии?",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("1 раз в день", callback_data="freq_daily"),
                    InlineKeyboardButton("2 раза в день", callback_data="freq_twice_daily"),
                ],
                [
                    InlineKeyboardButton("По дням недели", callback_data="freq_weekly"),
                    InlineKeyboardButton("Раз в месяц", callback_data="freq_monthly"),
                ],
            ]
        ),
    )


async def callback_freq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Выбор частоты подписки."""
    query = update.callback_query
    await query.answer()
    freq = query.data.replace("freq_", "")
    context.user_data["sub_frequency"] = freq

    if freq == "weekly":
        context.user_data["sub_days"] = []
        await _ask_days(query.message, context)
    else:
        context.user_data["sub_days"] = [1, 2, 3, 4, 5, 6, 7]
        await _ask_time(query.message)


async def _ask_days(msg, context):
    selected = context.user_data.get("sub_days", [])
    days = [("Пн", 1), ("Вт", 2), ("Ср", 3), ("Чт", 4), ("Пт", 5), ("Сб", 6), ("Вс", 7)]
    buttons = []
    row = []
    for label, num in days:
        mark = "✅" if num in selected else "⬜"
        row.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"day_{num}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("✔️ Готово", callback_data="days_done")])

    await msg.reply_text(
        "📅 Выбери дни недели (можно несколько):",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def callback_day_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Переключение дня недели."""
    query = update.callback_query
    await query.answer()

    day = int(query.data.replace("day_", ""))
    selected = context.user_data.get("sub_days", [])

    if day in selected:
        selected.remove(day)
    else:
        selected.append(day)
    context.user_data["sub_days"] = selected

    days = [("Пн", 1), ("Вт", 2), ("Ср", 3), ("Чт", 4), ("Пт", 5), ("Сб", 6), ("Вс", 7)]
    buttons = []
    row = []
    for label, num in days:
        mark = "✅" if num in selected else "⬜"
        row.append(InlineKeyboardButton(f"{mark} {label}", callback_data=f"day_{num}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("✔️ Готово", callback_data="days_done")])

    await query.message.edit_reply_markup(InlineKeyboardMarkup(buttons))


async def callback_days_done(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Подтверждение выбора дней."""
    query = update.callback_query
    await query.answer()

    selected = context.user_data.get("sub_days", [])
    if not selected:
        await query.answer("Выбери хотя бы один день!", show_alert=True)
        return

    await _ask_time(query.message)


async def _ask_time(msg):
    await msg.reply_text(
        "🕘 В какое время присылать? (МСК)\n\n" "Введи в формате ЧЧ:ММ, например: 09:00"
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/unsubscribe — отключить рассылку."""
    user = update.effective_user
    sub = await db.get_subscription(user.id)

    if not sub:
        await update.message.reply_text("🔕 Подписка и так не активна.")
        return

    await db.deactivate_subscription(user.id)
    await update.message.reply_text(
        "🔕 Подписка отключена.\n\nЧтобы включить снова — /subscribe"
    )


async def callback_sub_toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Кнопка переключения подписки на стартовом экране."""
    query = update.callback_query
    await query.answer()
    user = update.effective_user

    sub = await db.get_subscription(user.id)
    if sub:
        await db.deactivate_subscription(user.id)
        await query.message.reply_text("🔕 Подписка отключена.")
    else:
        await subscribe(update, context)
        return

    await start(update, context)


async def handle_sub_time_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Обрабатывает ввод времени для подписки. Возвращает True если обработал."""
    if "sub_frequency" not in context.user_data:
        return False

    text = update.message.text.strip()
    user = update.effective_user

    m = re.match(r"^(\d{1,2}):(\d{2})$", text)
    if not m:
        await update.message.reply_text(
            "⚠️ Неверный формат. Введи время как ЧЧ:ММ, например: 09:00"
        )
        return True

    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        await update.message.reply_text("⚠️ Неверное время. Например: 09:00 или 18:30")
        return True

    frequency = context.user_data.pop("sub_frequency")
    days = context.user_data.pop("sub_days", [1, 2, 3, 4, 5])

    saved = await db.get_resume(user.id)
    if not saved:
        await update.message.reply_text("😔 Резюме не найдено. Загрузи PDF заново.")
        return True

    search_query = saved.get("position") or "аналитик"
    await db.create_subscription(
        user_id=user.id,
        resume_id=saved["id"],
        search_query=search_query,
        frequency=frequency,
        days=days,
        send_hour=hour,
        send_minute=minute,
    )

    freq_text = _frequency_label(frequency)
    days_text = _days_label(days)

    await update.message.reply_text(
        f"✅ Подписка включена!\n\n"
        f"🔍 Запрос: «{search_query}»\n"
        f"📅 Частота: {freq_text}\n"
        f"📆 Дни: {days_text}\n"
        f"🕘 Время: {hour:02d}:{minute:02d} МСК\n\n"
        f"Чтобы отключить — /unsubscribe"
    )
    return True


def _frequency_label(freq: str) -> str:
    return {
        "daily": "1 раз в день",
        "twice_daily": "2 раза в день",
        "weekly": "По дням недели",
        "monthly": "Раз в месяц",
    }.get(freq, freq)


def _days_label(days: list) -> str:
    names = {1: "Пн", 2: "Вт", 3: "Ср", 4: "Чт", 5: "Пт", 6: "Сб", 7: "Вс"}
    return ", ".join(names[d] for d in sorted(days) if d in names)


# ─────────────────────────────────────────────
# Рассылка дайджеста
# ─────────────────────────────────────────────


async def send_digest_tick(context: ContextTypes.DEFAULT_TYPE):
    """Запускается каждые 30 минут. Проверяет подписки и отправляет дайджест."""
    moscow = pytz.timezone("Europe/Moscow")
    now = datetime.now(moscow)
    current_hour = now.hour
    current_minute = now.minute
    current_weekday = now.isoweekday()
    current_day = now.day

    subscriptions = await db.get_active_subscriptions()

    for sub in subscriptions:
        try:
            freq = sub.get("frequency", "daily")
            send_hour = sub.get("send_hour", 9)
            send_minute = sub.get("send_minute", 0)
            days = sub.get("days") or [1, 2, 3, 4, 5]

            if abs(current_hour * 60 + current_minute - send_hour * 60 - send_minute) > 15:
                continue

            if freq == "weekly" and current_weekday not in days:
                continue
            if freq == "monthly" and current_day != 1:
                continue

            user_id = sub["user_id"]
            search_query = sub["search_query"]
            vacancies = await fetch_vacancies(search_query, limit=100)
            if not vacancies:
                continue

            embedding_str = sub.get("embedding", "[]")
            resume_vector = np.array(json.loads(embedding_str), dtype=np.float32)

            matches = pipeline().match(resume_vector, vacancies, top_k=5)
            gc.collect()

            if not matches:
                continue

            await context.bot.send_message(
                chat_id=user_id,
                text=f"🌅 Новые вакансии по запросу «{search_query}»:",
            )
            for i, match in enumerate(matches, 1):
                await context.bot.send_message(
                    chat_id=user_id,
                    text=match.format_message(i),
                    disable_web_page_preview=True,
                )

            await db.update_subscription_sent(sub["id"])
            logger.info("Рассылка отправлена пользователю %s", user_id)

        except Exception as e:
            logger.error("Ошибка рассылки для %s: %s", sub.get("user_id"), e)


# ─────────────────────────────────────────────
# Избранное
# ─────────────────────────────────────────────


async def callback_favorite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Кнопка ⭐️ под вакансией."""
    query = update.callback_query
    user = update.effective_user

    vacancy_id = query.data[4:]  # убираем "fav_"

    cache = context.user_data.get("vacancies_cache", {})
    vac = cache.get(vacancy_id, {})
    title = vac.get("title", "Вакансия")
    company = vac.get("company", "")
    url = vac.get("url", "")
    salary_text = vac.get("salary_text", "")

    is_fav = await db.is_favorite(user.id, vacancy_id)

    if is_fav:
        await db.remove_favorite(user.id, vacancy_id)
        await query.answer("Удалено из избранного")
        await query.edit_message_reply_markup(
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("☆ В избранное", callback_data=query.data)]]
            )
        )
    else:
        await db.add_favorite(user.id, vacancy_id, title, company, url, salary_text)
        await query.answer("⭐️ Добавлено в избранное!")
        await query.edit_message_reply_markup(
            InlineKeyboardMarkup(
                [[InlineKeyboardButton("⭐️ В избранном", callback_data=query.data)]]
            )
        )


async def _send_favorites(message, user_id: int):
    favs = await db.get_favorites(user_id)

    if not favs:
        await message.reply_text(
            "⭐️ Избранное пусто.\n\nНажми ☆ под любой вакансией чтобы сохранить."
        )
        return

    await message.reply_text(f"⭐️ Избранное — {len(favs)} вакансий:")

    for i, fav in enumerate(favs, 1):
        salary = f"\n💰 {fav['salary_text']}" if fav.get("salary_text") else ""
        saved_date = fav["saved_at"].strftime("%d.%m.%Y")
        await message.reply_text(
            f"{i}. {fav['title']}\n"
            f"🏢 {fav['company']}{salary}\n"
            f"📅 Сохранено: {saved_date}\n"
            f"🔗 {fav['url']}",
            disable_web_page_preview=True,
            reply_markup=InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🗑 Удалить", callback_data=f"fav_{fav['vacancy_id'][:50]}"
                        )
                    ]
                ]
            ),
        )


async def favorites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/favorites — показывает избранные вакансии."""
    await _send_favorites(update.message, update.effective_user.id)


async def callback_show_more(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Кнопка 'Показать ещё 5'."""
    query = update.callback_query
    await query.answer()

    offset = int(query.data.replace("more_", ""))
    matches = context.user_data.get("matches", [])

    if not matches:
        await query.message.reply_text("⚠️ Результаты поиска устарели. Запусти поиск заново.")
        return

    await _send_matches(
        query.message,
        matches,
        user_id=update.effective_user.id,
        context=context,
        offset=offset,
        total=len(matches),
    )


async def callback_show_favorites(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await _send_favorites(query.message, update.effective_user.id)


# ─────────────────────────────────────────────
# RemoteOK
# ─────────────────────────────────────────────

REMOTEOK_TAGS_MAP = {
    "аналитик": "analyst",
    "data аналитик": "analyst",
    "data analyst": "analyst",
    "data scientist": "data",
    "data engineer": "data",
    "разработчик": "dev",
    "программист": "dev",
    "python": "python",
    "backend": "backend",
    "frontend": "frontend",
    "fullstack": "fullstack",
    "devops": "devops",
    "девопс": "devops",
    "тестировщик": "qa",
    "qa": "qa",
    "product manager": "product",
    "дизайнер": "design",
    "ml": "machine-learning",
    "machine learning": "machine-learning",
    "маркетинг": "marketing",
}


async def fetch_remoteok(query: str, limit: int = 100) -> list[dict]:
    """Получает удалённые вакансии с RemoteOK API."""
    query_lower = query.lower().strip()
    tag = None
    for key, t in REMOTEOK_TAGS_MAP.items():
        if key in query_lower:
            tag = t
            break
    if not tag:
        tag = query_lower.split()[0]

    logger.info("RemoteOK: query='%s' → tag='%s'", query, tag)

    try:
        async with httpx.AsyncClient(
            timeout=15, headers={"User-Agent": "JobBot/1.0"}
        ) as client:
            response = await client.get(f"https://remoteok.com/api?tag={tag}")
            response.raise_for_status()
            data = response.json()
            jobs = [j for j in data if isinstance(j, dict) and j.get("id")]

        logger.info("RemoteOK вернул %d вакансий (tag=%s)", len(jobs), tag)

        normalized = []
        for job in jobs[:limit]:
            salary_text = job.get("salary", "") or ""
            desc = re.sub(r"<[^>]+>", " ", job.get("description", "") or "")[:300]
            normalized.append(
                {
                    "id": f"remoteok_{job['id']}",
                    "name": job.get("position", ""),
                    "employer": {"name": job.get("company", "")},
                    "alternate_url": job.get(
                        "url", f"https://remoteok.com/remote-jobs/{job['id']}"
                    ),
                    "salary": None,
                    "snippet": {"requirement": desc, "responsibility": ""},
                    "_salary_text": salary_text,
                    "_source": "remoteok",
                    "published_at": job.get("date", ""),
                }
            )
        return normalized

    except Exception as e:
        logger.error("Ошибка RemoteOK: %s", e)
        return []


# ─────────────────────────────────────────────
# HR режим
# ─────────────────────────────────────────────


async def callback_hr_search_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """HR режим — запрашиваем текст вакансии."""
    query = update.callback_query
    await query.answer()
    context.user_data["awaiting_hr_vacancy"] = True
    await query.message.reply_text(
        "👔 *Режим HR — поиск кандидатов*\n\n"
        "Опиши кого ищешь или вставь текст вакансии:\n\n"
        "_Например:_\n"
        "«Ищем Python разработчика, 3+ лет опыта, знание Django, PostgreSQL, REST API»\n\n"
        "Бот найдёт подходящих кандидатов из нашей базы по AI-матчингу.",
        parse_mode="Markdown",
    )


async def fetch_hh_resumes(vacancy_text: str, limit: int = 20) -> list[dict]:
    """Ищет резюме на hh.ru по тексту вакансии."""
    token = os.getenv("HH_EMPLOYER_TOKEN")
    if not token:
        logger.warning("HH_EMPLOYER_TOKEN не задан — пропускаем hh.ru резюме")
        return []

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://api.hh.ru/resumes",
                params={
                    "text": vacancy_text[:200],
                    "per_page": limit,
                    "order_by": "relevance",
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "JobBot/1.0",
                },
            )
            if response.status_code == 403:
                logger.warning("hh.ru: нет доступа к резюме (нужен платный аккаунт работодателя)")
                return []
            response.raise_for_status()
            items = response.json().get("items", [])
            logger.info("hh.ru резюме: найдено %d", len(items))
            return items
    except Exception as e:
        logger.error("Ошибка поиска резюме hh.ru: %s", e)
        return []


async def handle_hr_vacancy_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Обрабатывает текст вакансии от HR. Возвращает True если обработал."""
    if not context.user_data.get("awaiting_hr_vacancy"):
        return False

    context.user_data.pop("awaiting_hr_vacancy", None)
    vacancy_text = update.message.text.strip()

    await update.message.reply_text("🤖 Ищу подходящих кандидатов...")

    vacancy_vector = encode_text_safe(vacancy_text)

    our_resumes, hh_items = await asyncio.gather(
        db.get_all_resumes(),
        fetch_hh_resumes(vacancy_text, limit=20),
    )

    scored = []

    # --- Наша база ---
    for r in our_resumes:
        emb_str = r.get("embedding")
        if not emb_str:
            continue
        try:
            resume_vec = np.array(json.loads(emb_str), dtype=np.float32)
            score = float(
                np.dot(vacancy_vector, resume_vec)
                / (np.linalg.norm(vacancy_vector) * np.linalg.norm(resume_vec) + 1e-10)
            )
            scored.append(
                {
                    "score": score,
                    "source": "our",
                    "position": r.get("position") or "Позиция не указана",
                    "name": r.get("name") or "Имя не указано",
                    "skills": r.get("skills") or [],
                    "experience_years": r.get("experience_years"),
                    "username": r.get("username"),
                    "user_id": r.get("user_id"),
                }
            )
        except Exception:
            continue

    # --- hh.ru резюме ---
    for item in hh_items:
        title = item.get("title", "")
        skills_hh = [s.get("name", "") for s in item.get("skill_set", [])]
        text_for_embed = f"{title} {' '.join(skills_hh)}"
        try:
            hh_vec = encode_text_safe(text_for_embed)
            score = float(
                np.dot(vacancy_vector, hh_vec)
                / (np.linalg.norm(vacancy_vector) * np.linalg.norm(hh_vec) + 1e-10)
            )
            scored.append(
                {
                    "score": score,
                    "source": "hh",
                    "position": title,
                    "name": item.get("first_name", "") + " " + item.get("last_name", ""),
                    "skills": skills_hh,
                    "experience_years": None,
                    "url": item.get("alternate_url", ""),
                }
            )
        except Exception:
            continue

    gc.collect()

    if not scored:
        await update.message.reply_text(
            "😔 Кандидатов не найдено.\nПригласи соискателей загрузить резюме в бот!"
        )
        return True

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:10]

    our_count = sum(1 for x in scored if x["source"] == "our")
    hh_count = sum(1 for x in scored if x["source"] == "hh")
    await update.message.reply_text(
        f"🎯 Найдено кандидатов: {len(scored)}\n"
        f"👤 Наша база: {our_count} | 🔵 hh.ru: {hh_count}\n"
        f"Показываю топ-{len(top)}:"
    )

    for i, r in enumerate(top, 1):
        score_pct = round(r["score"] * 100)
        filled = round(score_pct / 20)
        bar = "🟩" * filled + "⬜" * (5 - filled)
        skills_str = ", ".join(r["skills"][:5]) if r["skills"] else "не указаны"
        exp = r.get("experience_years")
        exp_str = f"{exp} лет" if exp else "не указан"

        if r["source"] == "our":
            username = r.get("username")
            contact = f"@{username}" if username else f"tg://user?id={r['user_id']}"
            source_badge = "👤 Наш пользователь"
        else:
            contact = r.get("url", "")
            source_badge = "🔵 hh.ru"

        await update.message.reply_text(
            f"{i}. {r['position']}\n"
            f"👤 {r['name'].strip() or 'Имя скрыто'}\n"
            f"🎯 Совпадение: {bar} {score_pct}%\n"
            f"💼 Опыт: {exp_str}\n"
            f"🔧 Навыки: {skills_str}\n"
            f"{source_badge} | {contact}",
            disable_web_page_preview=True,
        )

    return True


# ─────────────────────────────────────────────
# Публикация вакансий в канал
# ─────────────────────────────────────────────

# Список поисковых запросов для рандомных вакансий в канале
CHANNEL_SEARCH_QUERIES = [
    "Python разработчик",
    "Data Analyst",
    "Frontend разработчик",
    "Backend разработчик",
    "DevOps инженер",
    "Product Manager",
    "UX/UI дизайнер",
    "Data Scientist",
    "QA инженер",
    "Machine Learning",
    "Системный аналитик",
    "Java разработчик",
    "Golang разработчик",
    "iOS разработчик",
    "Android разработчик",
    "маркетолог",
    "HR менеджер",
    "проджект менеджер",
    "1С программист",
    "Fullstack разработчик",
]


def _format_channel_vacancy(job: dict, query: str) -> str:
    """Форматирует вакансию для публикации в канале."""
    title = job.get("name", "Вакансия")
    company = job.get("employer", {}).get("name", "")
    url = job.get("alternate_url", "")
    salary = _format_salary(job)

    # Требования из сниппета
    snippet = job.get("snippet", {})
    requirement = (snippet.get("requirement") or "").replace("<highlighttext>", "").replace("</highlighttext>", "")
    responsibility = (snippet.get("responsibility") or "").replace("<highlighttext>", "").replace("</highlighttext>", "")

    # Локация
    area = job.get("area", {}).get("name", "")
    schedule = job.get("schedule", {}).get("name", "")
    location_parts = [p for p in [area, schedule] if p]
    location_text = f"📍 {', '.join(location_parts)}\n" if location_parts else ""

    # Опыт
    experience = job.get("experience", {}).get("name", "")
    exp_text = f"📅 Опыт: {experience}\n" if experience else ""

    lines = [
        f"💼 {title}",
        f"🏢 {company}" if company else "",
        salary.strip() if salary else "",
        location_text.strip() if location_text else "",
        exp_text.strip() if exp_text else "",
        "",
        f"📋 {requirement[:200]}" if requirement else "",
        f"🔧 {responsibility[:200]}" if responsibility else "",
        "",
        f"🔗 Откликнуться: {url}" if url else "",
        "",
        f"🔎 #{query.replace(' ', '_')}",
        "📲 @bobajobabot — подбор вакансий по AI",
    ]

    return "\n".join(line for line in lines if line is not None).strip()


async def post_random_vacancies_to_channel(context: ContextTypes.DEFAULT_TYPE):
    """
    Публикует случайные вакансии в Telegram-канал.
    Запускается по расписанию через job_queue.

    Настройки через переменные окружения:
    - CHANNEL_ID: ID или @username канала (например @my_vacancies)
    - CHANNEL_POST_COUNT: сколько вакансий постить за раз (по умолчанию 5)
    """
    import random

    channel_id = os.getenv("CHANNEL_ID")
    if not channel_id:
        logger.debug("CHANNEL_ID не задан — пропускаем публикацию в канал")
        return

    post_count = int(os.getenv("CHANNEL_POST_COUNT", "5"))

    # Выбираем случайный поисковый запрос
    query = random.choice(CHANNEL_SEARCH_QUERIES)

    # Случайный регион (Москва, Питер, вся Россия)
    area = random.choice([113, 1, 2])

    logger.info("Канал: ищу '%s' (area=%s) для публикации", query, area)

    try:
        vacancies = await fetch_vacancies(query, area=area, limit=50)

        if not vacancies:
            logger.warning("Канал: вакансии не найдены для '%s'", query)
            return

        # Фильтруем вакансии с зарплатой (они интереснее для канала)
        with_salary = [v for v in vacancies if v.get("salary")]
        pool = with_salary if len(with_salary) >= post_count else vacancies

        # Берём случайные, не повторяясь
        selected = random.sample(pool, min(post_count, len(pool)))

        posted = 0
        for job in selected:
            try:
                text = _format_channel_vacancy(job, query)
                await context.bot.send_message(
                    chat_id=channel_id,
                    text=text,
                    disable_web_page_preview=True,
                )
                posted += 1

                # Пауза между постами чтобы не флудить
                await asyncio.sleep(3)

            except Exception as e:
                logger.error("Канал: ошибка отправки вакансии: %s", e)

        logger.info("Канал: опубликовано %d/%d вакансий по запросу '%s'", posted, post_count, query)

    except Exception as e:
        logger.error("Канал: ошибка получения вакансий: %s", e)


# ─────────────────────────────────────────────
# Инициализация и запуск
# ─────────────────────────────────────────────


async def post_init(app):
    """Выполняется после инициализации бота — подключаем БД."""
    try:
        await db.get_pool()
        logger.info("✅ БД подключена")
    except Exception as e:
        logger.error("❌ Ошибка подключения к БД: %s", e)


def main():
    token = os.getenv("BOT_TOKEN")
    if not token:
        logger.error("❌ BOT_TOKEN не найден в .env файле!")
        return

    app = Application.builder().token(token).post_init(post_init).build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("myresume", my_resume))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("favorites", favorites))

    # PDF
    app.add_handler(MessageHandler(filters.Document.PDF, handle_document))

    # Callback-кнопки
    app.add_handler(CallbackQueryHandler(callback_show_myresume, pattern="^show_myresume$"))
    app.add_handler(CallbackQueryHandler(callback_show_more, pattern=r"^more_\d+$"))
    app.add_handler(CallbackQueryHandler(callback_hr_search_start, pattern="^hr_search_start$"))
    app.add_handler(CallbackQueryHandler(callback_show_favorites, pattern="^show_favorites$"))
    app.add_handler(CallbackQueryHandler(callback_upload_hint, pattern="^upload_hint$"))
    app.add_handler(CallbackQueryHandler(callback_search_start, pattern="^search_start$"))
    app.add_handler(
        CallbackQueryHandler(callback_filter_location, pattern="^filter_location$")
    )
    app.add_handler(CallbackQueryHandler(callback_filter_salary, pattern="^filter_salary$"))
    app.add_handler(
        CallbackQueryHandler(callback_filter_experience, pattern="^filter_experience$")
    )
    app.add_handler(
        CallbackQueryHandler(
            callback_filter_toggle_no_salary, pattern="^filter_toggle_no_salary$"
        )
    )
    app.add_handler(CallbackQueryHandler(callback_do_search, pattern="^do_search$"))
    app.add_handler(CallbackQueryHandler(callback_area_toggle, pattern=r"^area_\d+$"))
    app.add_handler(CallbackQueryHandler(callback_area_reset, pattern="^area_reset$"))
    app.add_handler(CallbackQueryHandler(callback_area_custom, pattern="^area_custom$"))
    app.add_handler(CallbackQueryHandler(callback_area_done, pattern="^area_done$"))
    app.add_handler(CallbackQueryHandler(callback_add_custom_area, pattern="^addarea_"))
    app.add_handler(CallbackQueryHandler(callback_exp_toggle, pattern="^exp_(?!done)"))
    app.add_handler(CallbackQueryHandler(callback_exp_done, pattern="^exp_done$"))
    app.add_handler(CallbackQueryHandler(callback_edit_start, pattern="^edit_start$"))
    app.add_handler(CallbackQueryHandler(callback_sub_toggle, pattern="^sub_toggle$"))
    app.add_handler(CallbackQueryHandler(callback_freq, pattern="^freq_"))
    app.add_handler(CallbackQueryHandler(callback_day_toggle, pattern=r"^day_\d+$"))
    app.add_handler(CallbackQueryHandler(callback_days_done, pattern="^days_done$"))
    app.add_handler(CallbackQueryHandler(callback_favorite, pattern="^fav_"))
    app.add_handler(CallbackQueryHandler(callback_delete_resume, pattern="^delete_resume$"))
    app.add_handler(CallbackQueryHandler(callback_confirm_delete, pattern="^confirm_delete$"))
    app.add_handler(CallbackQueryHandler(callback_cancel_delete, pattern="^cancel_delete$"))

    # Текстовый ввод (catch-all)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_edit_input))
    app.add_handler(CommandHandler("skip", lambda u, c: handle_edit_input(u, c)))

    # Джоб-очередь для рассылки
    app.job_queue.run_repeating(
        send_digest_tick,
        interval=1800,
        first=60,
        name="digest_tick",
    )

    # Публикация в канал — каждые 4 часа (настраивается через CHANNEL_INTERVAL_HOURS)
    channel_interval = int(os.getenv("CHANNEL_INTERVAL_HOURS", "4")) * 3600
    if os.getenv("CHANNEL_ID"):
        app.job_queue.run_repeating(
            post_random_vacancies_to_channel,
            interval=channel_interval,
            first=120,  # первый пост через 2 минуты после старта
            name="channel_post",
        )
        logger.info(
            "📢 Публикация в канал %s каждые %s ч.",
            os.getenv("CHANNEL_ID"),
            os.getenv("CHANNEL_INTERVAL_HOURS", "4"),
        )

    logger.info("🤖 Бот запущен!")
    app.run_polling()


if __name__ == "__main__":
    main()