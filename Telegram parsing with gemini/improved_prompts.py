"""
Improved prompts for monthly news review
Ready to copy-paste into your notebook
"""

# =============================================================================
# STAGE 1: IMPROVED FILTERING PROMPT
# =============================================================================

def get_improved_prompt_individual(text):
    """
    Improved prompt for filtering individual posts.
    More structured, clearer instructions, better role definition.
    """
    return f"""Ты - аналитик российского рынка FMCG и e-commerce. Твоя задача: определить, релевантна ли новость для ежемесячного обзора рынка и извлечь ключевые факты.

ОБЛАСТЬ АНАЛИЗА (FMCG/e-commerce РФ 2024-2025):

**Категории продуктов:**
- Напитки: лимонады, кола, энергетики, холодный чай
- Снеки: чипсы, закуски
- Молочная продукция
- Детское питание

**Бренды и компании:**
- Производители: Nestle (детское питание), Черкизово, Мираторг, Эфко, Балтика
- Новые бренды: Добрый Cola, Вкусно и точка, Stars Coffee, азиатские бренды
- Маркетплейсы: Ozon, Wildberries (WB), Яндекс Маркет (ЯМ), СберМегаМаркет
- Ритейл: X5 Group (Пятёрочка, Перекрёсток), Магнит, Лента, ВкусВилл

**Ключевые темы:**
- Импортозамещение и локализация производства
- Параллельный импорт
- СТМ (собственные торговые марки)
- Омниканальность и онлайн-торговля
- Ценообразование и промо

**ИСКЛЮЧИТЬ:**
- B2B-сервисы (не для конечных потребителей)
- Финансовые услуги (банки, кредиты, инвестиции)
- Автомобили и запчасти
- Электроника и бытовая техника
- HR и вакансии

**ИНСТРУКЦИЯ:**

Если новость релевантна:
- Извлеки ключевую информацию: КТО (компания/бренд), ЧТО (событие), ЦИФРЫ (если есть)
- Напиши краткую суть в 2-3 предложениях
- Используй конкретные названия и числа

Если новость НЕ релевантна:
- Напиши только: "НЕРЕЛЕВАНТНО"

**НОВОСТЬ:**
{text}

**ТВОЙ ОТВЕТ:**"""


# =============================================================================
# STAGE 2: IMPROVED SYNTHESIS PROMPT
# =============================================================================

def get_improved_prompt_final(summaries, start_date, num_channels):
    """
    Improved prompt for generating final report.
    Clearer structure, audience definition, handling of missing data.
    """
    return f"""Ты - старший аналитик российского рынка FMCG и e-commerce. Составь ежемесячный аналитический обзор для руководства компании.

**ЦЕЛЕВАЯ АУДИТОРИЯ:** Топ-менеджмент (CEO, CMO, Head of Sales)
**ТОН:** Деловой, лаконичный, фокус на цифрах и бизнес-импликациях
**ПЕРИОД:** {start_date.strftime('%B %Y')}
**ИСТОЧНИК ДАННЫХ:** {len(summaries)} релевантных новостей из {num_channels} отраслевых каналов

---

**ЗАДАЧА:** Структурированный отчет с фокусом на:
- ✅ Повторяющиеся сигналы (тренды, а не единичные события)
- ✅ Количественные данные (цифры, проценты, суммы)
- ✅ Системные изменения (влияние на рынок)
- ❌ Маркетинговые анонсы без бизнес-контекста
- ❌ Единичные новости без влияния на индустрию

---

**СТРУКТУРА ОТЧЕТА (10 РАЗДЕЛОВ):**

**1. Общее состояние рынка**
Основные макротренды месяца: потребительский спрос, ценообразование, конкурентная среда (2-3 ключевых вывода)

**2. Финансовые результаты**
Компании, опубликовавшие отчетность → **[Компания]:** выручка, прибыль, GMV, доля e-commerce, динамика vs прошлый период

**3. Маркетплейсы**
Ozon, Wildberries, Яндекс Маркет, СберМегаМаркет → **[Компания]:** изменения комиссий, логистика, новые сервисы, ключевые метрики

**4. Омниканальный ритейл**
X5, Магнит, Лента, ВкусВилл → **[Компания]:** цифровая трансформация, доставка, приложения, интеграция онлайн/офлайн

**5. Агрегаторы и доставка**
Яндекс Еда, Деливери, Т-банк (Купер) → **[Компания]:** тарифы, технологические новинки, география, партнерства

**6. Производители FMCG**
Бренды и фабрики → **[Компания]:** новые продукты, ценообразование, маркетинговые кампании, проблемы поставок

**7. Новинки и инновации**
Новые запуски → **[Бренд/продукт]:** описание, целевая аудитория, потенциальное влияние на рынок

**8. Цепочки поставок**
Проблемы и решения → **[Проблема/компания]:** суть, масштаб, способы решения, влияние на цены

**9. Регулирование**
Законы, ФАС, инициативы госорганов → **[Событие]:** суть изменений, кого затрагивает, сроки внедрения, бизнес-эффект

**10. Прогноз на следующий месяц**
Ожидаемые тренды и риски (2-3 предложения)

---

**ПРАВИЛА ОФОРМЛЕНИЯ:**
- Используй **жирный шрифт** для названий компаний и брендов
- Обязательно указывай цифры и даты
- Максимум 3 предложения на каждый пункт
- Если по разделу нет значимых новостей, напиши: "*Значимых изменений не зафиксировано*"
- Избегай общих фраз типа "продолжается развитие" - конкретика и факты

---

**ИСХОДНЫЕ ДАННЫЕ (НОВОСТИ):**

{summaries}

---

**АНАЛИТИЧЕСКИЙ ОТЧЕТ:**"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def estimate_tokens(text):
    """
    Rough estimate of token count.
    Russian text: ~1 token per 4 characters
    """
    return len(text) // 4


def truncate_summaries(summaries, max_tokens=100000):
    """
    Keep only most recent summaries within token limit.
    Starts from most recent and works backwards.
    """
    combined = ""
    kept_summaries = []

    for summary in reversed(summaries):
        test_combined = f"- {summary}\n" + combined
        if estimate_tokens(test_combined) > max_tokens:
            break
        combined = test_combined
        kept_summaries.insert(0, summary)

    if len(kept_summaries) < len(summaries):
        print(f"⚠️ Сокращено с {len(summaries)} до {len(kept_summaries)} новостей из-за лимита токенов")

    return kept_summaries


def save_checkpoint(relevant_summaries, total_posts, start_date):
    """Save intermediate results to recover from failures"""
    import json
    from datetime import datetime

    checkpoint_file = f"checkpoint_{start_date.strftime('%Y%m')}.json"

    checkpoint_data = {
        'saved_at': datetime.now().isoformat(),
        'summaries': relevant_summaries,
        'total_posts': total_posts,
        'relevant_count': len(relevant_summaries),
        'start_date': start_date.isoformat()
    }

    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Checkpoint сохранен: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(start_date):
    """Load checkpoint if exists"""
    import json
    import os

    checkpoint_file = f"checkpoint_{start_date.strftime('%Y%m')}.json"

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        print(f"✅ Загружено из checkpoint: {checkpoint['relevant_count']} summaries")
        return checkpoint['summaries']

    return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example text
    test_text = """
    Wildberries увеличила тарифы на логистику с 15 сентября на 5-10% для большинства
    категорий товаров. Исключение составили товары весом до 500г - для них тариф снижен.
    """

    # Test Stage 1 prompt
    print("=" * 60)
    print("STAGE 1 PROMPT:")
    print("=" * 60)
    print(get_improved_prompt_individual(test_text))

    print("\n\n")

    # Test Stage 2 prompt
    print("=" * 60)
    print("STAGE 2 PROMPT:")
    print("=" * 60)
    from datetime import datetime
    test_summaries = ["Summary 1", "Summary 2", "Summary 3"]
    combined = "\n".join(f"- {s}" for s in test_summaries)
    print(get_improved_prompt_final(combined, datetime(2025, 9, 1), 33))

    print("\n\n")

    # Test token estimation
    print("=" * 60)
    print("TOKEN ESTIMATION:")
    print("=" * 60)
    long_text = "Тестовый текст " * 10000
    tokens = estimate_tokens(long_text)
    print(f"Text length: {len(long_text)} chars")
    print(f"Estimated tokens: {tokens:,}")
