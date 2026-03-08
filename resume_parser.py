"""
resume_parser.py — парсер резюме из PDF и hh.ru профиля
Зависимости: pdfplumber, httpx
"""

import re
import io
import logging
from dataclasses import dataclass, field
from typing import Optional
import pdfplumber
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ParsedResume:
    raw_text: str = ""
    name: Optional[str] = None
    desired_position: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    experience_years: Optional[float] = None
    experience_text: str = ""
    education: str = ""
    languages: list[str] = field(default_factory=list)
    source: str = "unknown"

    def is_valid(self) -> bool:
        return bool(self.raw_text) and (
            bool(self.skills) or bool(self.desired_position)
        )

    def summary(self) -> str:
        lines = []
        if self.name:
            lines.append(f"👤 {self.name}")
        if self.desired_position:
            lines.append(f"💼 {self.desired_position}")
        if self.experience_years is not None:
            lines.append(f"📅 Опыт: {self.experience_years:.0f} лет")
        if self.skills:
            # Сначала показываем "важные" навыки (языки, BI, ETL), потом остальные
            priority = ["python", "sql", "tableau", "power bi", "airflow", "dbt",
                        "spark", "clickhouse", "postgresql", "docker", "react",
                        "javascript", "typescript", "java", "go", "rust"]
            top = sorted(self.skills, key=lambda s: (0 if s in priority else 1, s))[:12]
            lines.append(f"🔧 Навыки: {', '.join(top)}")
        if self.education:
            lines.append(f"🎓 {self.education}")
        return "\n".join(lines) if lines else "⚠️ Данные не извлечены"


class PDFResumeParser:

    # Технологии для поиска по всему тексту
    TECH_KEYWORDS = [
        # Языки
        "python", "sql", "javascript", "typescript", "java", "kotlin",
        "swift", "c++", "c#", "go", "golang", "rust", "php", "ruby",
        "scala", "r", "matlab",
        # Базы данных
        "postgresql", "mysql", "clickhouse", "vertica", "greenplum",
        "mongodb", "redis", "elasticsearch", "sqlite", "oracle",
        # Python библиотеки
        "numpy", "pandas", "scikit-learn", "sklearn", "seaborn",
        "matplotlib", "pytorch", "tensorflow", "keras", "scipy",
        # BI / Аналитика
        "tableau", "power bi", "powerbi", "looker", "metabase",
        "superset", "qlik",
        # ETL / Data
        "airflow", "dbt", "spark", "kafka", "hadoop", "luigi",
        "prefect", "dagster",
        # Бэкенд
        "django", "fastapi", "flask", "spring", "laravel",
        "express", "nestjs", "rails",
        # Фронтенд
        "react", "vue", "angular", "next.js", "nuxt",
        # DevOps / Infra
        "docker", "kubernetes", "k8s", "git", "github", "gitlab",
        "jenkins", "aws", "gcp", "azure", "terraform", "ansible",
        "nginx", "linux",
        # Аналитика
        "excel", "google analytics", "amplitude", "mixpanel",
        "1с", "1c",
        # Реклама
        "dv360", "google ads", "facebook ads",
    ]

    def parse_bytes(self, pdf_bytes: bytes) -> ParsedResume:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if text:
                        pages_text.append(text)
                raw_text = "\n".join(pages_text)

            if not raw_text.strip():
                return ParsedResume(source="pdf")

            return self._parse(raw_text)
        except Exception as e:
            logger.error(f"Ошибка парсинга PDF: {e}")
            return ParsedResume(source="pdf")

    def _parse(self, raw_text: str) -> ParsedResume:
        resume = ParsedResume(raw_text=raw_text, source="pdf")
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

        resume.name = self._extract_name(lines)
        resume.desired_position = self._extract_position(lines)
        resume.skills = self._extract_skills(raw_text, lines)
        resume.experience_years = self._extract_experience_years(raw_text)
        resume.education = self._extract_education(lines)
        return resume

    def _extract_name(self, lines: list[str]) -> Optional[str]:
        """
        Ищет имя тремя способами:
        1. ALL CAPS строки в любом месте первых 40 строк (двухколоночный PDF)
        2. Обычный формат "Иван Иванов"
        3. Из строки "Меня зовут Имя"
        """
        # Способ 1: ALL CAPS — ищем ВСЕ такие строки в первых 40 строках
        # Они могут идти НЕ подряд из-за двухколоночного layout
        caps_words = []
        for line in lines[:40]:
            # Одно слово, полностью заглавными кириллицей или латиницей, без цифр
            if re.match(r'^[А-ЯЁA-Z]{2,20}$', line):
                caps_words.append(line.capitalize())
                if len(caps_words) == 2:
                    return " ".join(caps_words)

        # Способ 2: обычный формат
        for line in lines[:15]:
            if re.match(r'^[А-ЯЁA-Z][а-яёa-z]+(\s+[А-ЯЁA-Z][а-яёa-z]+){1,2}$', line):
                return line

        # Способ 3: "Меня зовут Андрей"
        for line in lines[:10]:
            m = re.search(r'[Мм]еня зовут\s+([А-ЯЁA-Zа-яёa-z]+)', line)
            if m:
                return m.group(1)

        return None

    def _extract_position(self, lines: list[str]) -> Optional[str]:
        """
        Ищет должность.
        Обрабатывает случай когда должность слита с текстом в одну строку
        (двухколоночный PDF): "DATA АНАЛИТИК описание работы..."
        """
        position_markers = [
            "аналитик", "разработчик", "developer", "engineer", "analyst",
            "менеджер", "manager", "дизайнер", "designer", "lead", "лид",
            "data", "backend", "frontend", "fullstack", "devops", "qa",
            "тестировщик", "архитектор", "architect", "маркетолог",
        ]
        for line in lines[:50]:
            line_lower = line.lower()
            if not any(m in line_lower for m in position_markers):
                continue

            # Чистая короткая строка — берём целиком
            if len(line) <= 40:
                if not re.search(r'\b(работа[а-я]*|создани[а-я]*|построени[а-я]*|являюсь|зовут)\b', line_lower):
                    return line

            # Длинная строка — пробуем вырезать должность из начала
            # Паттерн: "DATA АНАЛИТИК " затем описание
            m = re.match(r'^([A-ZА-ЯЁ][A-ZА-ЯЁ\s]{2,30}?)\s+[а-яёa-z]', line)
            if m:
                candidate = m.group(1).strip()
                if any(marker in candidate.lower() for marker in position_markers):
                    return candidate

        return None

    def _extract_skills(self, raw_text: str, lines: list[str]) -> list[str]:
        """
        Два прохода:
        1. Парсим секцию Навыки по паттерну: Заголовок → значения через запятую
        2. Поиск известных технологий по всему тексту
        """
        found = set()

        # --- Проход 1: секция "Навыки" ---
        # Паттерн: строка-категория (SQL, Python, ...) → следующая строка с перечислением
        in_skills = False
        skills_end_markers = re.compile(
            r'(?i)^(опыт работы|образование|контакты|обо мне|languages|experience|education)$'
        )

        for i, line in enumerate(lines):
            if re.search(r'(?i)^навыки$', line):
                in_skills = True
                continue

            if in_skills:
                if skills_end_markers.match(line):
                    break

                # Строка-заголовок категории (SQL, Python, ETL инструменты)
                if re.match(r'^[A-ZА-ЯЁa-zа-яё][A-ZА-ЯЁa-zа-яё\s]{1,30}$', line) and len(line) < 30:
                    # Следующая строка — значения
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        # Значения через запятую — признак что это перечисление
                        if "," in next_line or len(next_line.split()) <= 4:
                            items = re.split(r'[,;]', next_line)
                            for item in items:
                                item = item.strip().lower()
                                if 1 < len(item) <= 30 and not item[0].isdigit():
                                    found.add(item)

        # --- Проход 2: поиск известных технологий по всему тексту ---
        text_lower = raw_text.lower()
        for tech in self.TECH_KEYWORDS:
            # Ищем как отдельное слово/фразу
            pattern = r'(?<![а-яёa-z])' + re.escape(tech) + r'(?![а-яёa-z])'
            if re.search(pattern, text_lower):
                found.add(tech)

        # --- Очистка ---
        # Паттерны мусора который попадает из соседних колонок
        NOISE_PATTERNS = [
            r'ооо\s*["\']',           # OOO "Компания"
            r'ooo\s*["\']',
            r'^\d{4}[/\-]',           # Даты: 2023/02, 2022-06
            r'информационной',        # Из образования
            r'безопасности',
            r'бакалавриат',
            r'магистратура',
            r'университет',
            r'институт',
            r'^\d{4}\s*[-–]\s*\d{4}', # 2016 - 2020
            r'контролировал',         # Глаголы из опыта
            r'формировал',
            r'управлял',
            r'создание',
            r'построение',
            r'подсчет',
            r'аналитика\s+и',
            r'delivery',              # Подзаголовки из опыта
            r'lifecycle',
            r'ownership',
        ]

        clean = []
        for s in found:
            s = s.strip()
            if len(s) < 2 or len(s) > 35:
                continue
            # Проверяем на мусор
            s_lower = s.lower()
            if any(re.search(p, s_lower) for p in NOISE_PATTERNS):
                continue
            # Убираем строки заканчивающиеся союзом/предлогом
            if re.search(r'\s+(и|в|с|на|для|по|из|от|до|при|или|а|но|что)$', s_lower):
                continue
                continue
            clean.append(s)

        clean.sort()
        return clean

    def _extract_experience_years(self, raw_text: str) -> Optional[float]:
        """
        Считает опыт по реальным датам работы — суммирует все периоды.
        Формат дат: YYYY/MM-YYYY/MM или YYYY/MM-настоящее время
        """
        from datetime import date

        today = date.today()
        total_months = 0
        found_any = False

        # Паттерн: 2021/06-2022/06 или 2023/12-настоящее время
        pattern = re.compile(
            r'(\d{4})/(\d{2})\s*[-–−]\s*(?:(\d{4})/(\d{2})|(настоящее\s*время|present|н\.в\.))',
            re.IGNORECASE
        )

        for m in pattern.finditer(raw_text):
            start_year, start_month = int(m.group(1)), int(m.group(2))

            if m.group(5):  # "настоящее время"
                end_year, end_month = today.year, today.month
            else:
                end_year, end_month = int(m.group(3)), int(m.group(4))

            months = (end_year - start_year) * 12 + (end_month - start_month)
            if 0 < months < 600:  # санитарная проверка
                total_months += months
                found_any = True

        if found_any:
            return round(total_months / 12, 1)

        # Фолбэк: явное упоминание в тексте ("более 3 лет опыта")
        fallback_patterns = [
            r'более\s+(\d+)[- ]?летни',
            r'(\d+(?:[.,]\d+)?)\s*(?:лет|года?)\s*опыт',
            r'(\d+(?:[.,]\d+)?)\s*years?\s*(?:of\s*)?experience',
        ]
        for p in fallback_patterns:
            match = re.search(p, raw_text, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(",", "."))

        return None

    def _extract_education(self, lines: list[str]) -> str:
        """Ищет строки с названиями вузов"""
        edu_markers = ["университет", "институт", "академия", "итмо", "мгу",
                       "university", "college", "бакалавр", "магистр", "вуз"]
        edu_lines = []
        for line in lines:
            if any(m in line.lower() for m in edu_markers) and len(line) < 80:
                edu_lines.append(line)

        # Убираем дубли
        seen = set()
        unique = []
        for l in edu_lines:
            key = l.lower()[:30]
            if key not in seen:
                seen.add(key)
                unique.append(l)

        return "; ".join(unique[:3])


# ─────────────────────────────────────────────
# hh.ru парсер
# ─────────────────────────────────────────────

class HHProfileParser:

    HH_API = "https://api.hh.ru"

    async def parse_url(self, url: str) -> ParsedResume:
        resume_id = self._extract_resume_id(url)
        if not resume_id:
            return ParsedResume(source="hh")
        return await self._fetch_resume(resume_id)

    def _extract_resume_id(self, url: str) -> Optional[str]:
        m = re.search(r"hh\.ru/resume/([a-zA-Z0-9]+)", url)
        return m.group(1) if m else None

    async def _fetch_resume(self, resume_id: str) -> ParsedResume:
        headers = {"User-Agent": "JobMatcherBot/1.0 (support@example.com)"}
        url = f"{self.HH_API}/resumes/{resume_id}"

        async with httpx.AsyncClient(timeout=15) as client:
            try:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return self._parse_hh_response(resp.json())
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    logger.warning("hh.ru резюме закрыто")
                else:
                    logger.error(f"hh.ru API: {e}")
                return ParsedResume(source="hh")
            except Exception as e:
                logger.error(f"Ошибка hh.ru: {e}")
                return ParsedResume(source="hh")

    def _parse_hh_response(self, data: dict) -> ParsedResume:
        resume = ParsedResume(source="hh")
        resume.name = f"{data.get('first_name','')} {data.get('last_name','')}".strip() or None
        resume.desired_position = data.get("title")
        resume.skills = [s.lower() for s in data.get("skill_set", [])]

        total_months = sum(e.get("months", 0) or 0 for e in data.get("experience", []))
        if total_months:
            resume.experience_years = round(total_months / 12, 1)

        edu_list = [
            f"{e.get('name','')} ({e.get('year','')})"
            for e in data.get("education", {}).get("primary", [])
        ]
        resume.education = "; ".join(edu_list)
        resume.languages = [l.get("name", "") for l in data.get("languages", [])]

        parts = []
        if resume.desired_position:
            parts.append(resume.desired_position)
        if resume.skills:
            parts.append("Навыки: " + ", ".join(resume.skills))
        for exp in data.get("experience", []):
            desc = exp.get("description", "") or ""
            parts.append(f"{exp.get('company','')} — {exp.get('position','')}\n{desc[:300]}")
        resume.raw_text = "\n\n".join(parts)
        return resume


# ─────────────────────────────────────────────
# Фасад
# ─────────────────────────────────────────────

class ResumeParser:

    def __init__(self):
        self.pdf_parser = PDFResumeParser()
        self.hh_parser = HHProfileParser()

    def parse_pdf_bytes(self, pdf_bytes: bytes) -> ParsedResume:
        return self.pdf_parser.parse_bytes(pdf_bytes)

    async def parse_hh_url(self, url: str) -> ParsedResume:
        return await self.hh_parser.parse_url(url)

    @staticmethod
    def is_hh_url(text: str) -> bool:
        return bool(re.search(r"hh\.ru/resume/", text))