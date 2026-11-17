"""
Тестовое задание для компании «Руссмаркет»
Автоматизация утренней сводки support-отдела (CRM + рабочие чаты)
Работает локально (152-ФЗ) с входной/выходной валидацией, ретраями и логгированием.
"""

import json
import logging
import os
from typing import List, Literal
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

# =====================================================================
# 0. НАСТРОЙКА ПРОФЕССИОНАЛЬНОГО ЛОГИРОВАНИЯ
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SupportAutomationService")


# =====================================================================
# 1. КОНФИГУРАЦИЯ ЧЕРЕЗ ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ (Environment Variables)
# =====================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")  # ← изменено под твой Mac

client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # Локальный контур, ключ фиктивный
)


# =====================================================================
# 2. ВХОДНЫЕ СХЕМЫ ДАННЫХ (Валидация входящего JSON из CRM / API)
# =====================================================================
class ChatMessage(BaseModel):
    sender: str
    time: str
    text: str

class CaseInput(BaseModel):
    id: str
    type: str
    crm_status: str
    chat_history: List[ChatMessage]


# =====================================================================
# 3. ВЫХОДНЫЕ СХЕМЫ ДАННЫХ (Structured Outputs через Pydantic)
# =====================================================================
class CaseAnalysis(BaseModel):
    case_id: str = Field(description="ID кейса из CRM (например, C-17)")
    category: str = Field(description="Тип задачи (Оформление, Увольнение, Поставка карт)")
    crm_confirmed: str = Field(description="Что официально подтверждено в CRM")
    chat_info: str = Field(description="Главные факты и выводы из переписки в чате")
    has_conflict_or_risk: bool = Field(description="True, если есть расхождение, задержка или риск")
    risk_explanation: str = Field(description="Суть проблемы/риска или 'Расхождений нет'")
    priority: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Приоритет реакции для support")
    next_step: str = Field(description="Конкретное первое действие для сотрудника support")

class DailySummary(BaseModel):
    current_time_context: str = Field(description="Время формирования сводки")
    total_cases: int
    critical_cases_count: int
    cases: List[CaseAnalysis]


# =====================================================================
# 4. ТЕСТОВЫЕ ДАННЫЕ ИЗ ТЗ «РУССМАРКЕТ» (С ИСТОРИЕЙ ПЕРЕПИСОК)
# =====================================================================
RAW_TEST_CASES = [
    {
        "id": "C-17",
        "type": "Оформление сотрудника",
        "crm_status": "Первый выход завтра. Документы ожидаются.",
        "chat_history": [
            {"sender": "Support Bot", "time": "Вчера, 10:00", "text": "Напоминание: завтра первый рабочий день. Загрузите документы в систему."},
            {"sender": "Сотрудник", "time": "Вчера, 18:40", "text": "Вчера отправил фотографии документов руководителю в личку."},
            {"sender": "Support Bot", "time": "Сегодня, 09:00", "text": "В CRM документы до сих пор не поступили. Требуется подтверждение."}
        ]
    },
    {
        "id": "C-24",
        "type": "Увольнение сотрудника",
        "crm_status": "Последний рабочий день - сегодня. Подтверждение ответственного руководителя отсутствует.",
        "chat_history": [
            {"sender": "Support Bot", "time": "Вчера, 15:00", "text": "Завтра у вас последний день по графику. Сдана ли форма/пропуск?"},
            {"sender": "Сотрудник", "time": "Сегодня, 09:10", "text": "С руководителем всё обсудили, он в курсе."}
        ]
    },
    {
        "id": "П-08",
        "type": "Поставка карт",
        "crm_status": "Доставка на точку запланирована сегодня до 10:00. Статус «В пути».",
        "chat_history": [
            {"sender": "Support Bot", "time": "Сегодня, 09:45", "text": "Курьер, уточните статус по точке, до дедлайна 15 минут."},
            {"sender": "Курьер", "time": "Сегодня, 09:50", "text": "Задерживаюсь, раньше 13:00 не приеду. В пробке на МКАД."}
        ]
    },
    {
        "id": "П-09",
        "type": "Поставка карт",
        "crm_status": "Статус «Принято». Приёмка подтверждена сегодня в 09:20.",
        "chat_history": [
            {"sender": "Support Bot", "time": "Сегодня, 09:15", "text": "Ожидается доставка карт на точку. Просьба отписаться по факту."},
            {"sender": "Получатель на точке", "time": "Сегодня, 09:22", "text": "Карты получили, количество проверили. Всё сходится."}
        ]
    }
]


# =====================================================================
# 5. ФУНКЦИЯ ЗАПРОСА К LLM С МЕХАНИЗМОМ ПОВТОРОВ (Retries)
# =====================================================================
@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.warning(f"Сбой вызова LLM. Попытка ретрая #{retry_state.attempt_number}...")
)
def call_llm_with_retry(system_prompt: str, user_prompt: str) -> DailySummary:
    """Выполняет запрос к локальной LLM с автоповтором при сбоях связи."""
    logger.info(f"Отправка запроса к локальной модели ({MODEL_NAME})...")
    
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=DailySummary,
        temperature=0.0,
    )
    return response.choices[0].message.parsed


def build_daily_summary(validated_cases: List[CaseInput], current_time: str = "Понедельник, 10:00") -> DailySummary:
    system_prompt = (
        "Ты — главный AI-аналитик support-отдела компании «Руссмаркет». "
        "Твоя задача — сопоставлять официальные статусы из CRM с полной историей переписок из рабочих чатов.\n"
        "Правила анализа:\n"
        "1. Четко разделяй, что подтверждено в CRM, а что известно только из диалогов в чате.\n"
        "2. Устные обещания или сообщения в чате (например, 'отправил руководителю') НЕ заменяют официальный статус в CRM.\n"
        "3. Если есть задержка, риск срыва выхода или отсутствие кадровых подтверждений — ставь priority='HIGH'/'MEDIUM' и формулируй четкий первый шаг для support.\n"
        "4. Если всё совпало и успешно подтверждено — ставь priority='LOW' и has_conflict_or_risk=False."
    )

    cases_dict_list = [case.model_dump() for case in validated_cases]

    user_prompt = (
        f"Текущее время: {current_time}.\n"
        f"Проанализируй кейсы с учетом истории чатов и верни JSON строго по схеме:\n"
        f"{json.dumps(cases_dict_list, ensure_ascii=False, indent=2)}"
    )

    return call_llm_with_retry(system_prompt, user_prompt)


# =====================================================================
# 6. ТОЧКА ВХОДА В ПРОГРАММУ
# =====================================================================
if __name__ == "__main__":
    logger.info("Старт сервиса формирования сводки support-отдела...")
    
    try:
        logger.info("Валидация входящих данных...")
        validated_cases = [CaseInput(**raw_case) for raw_case in RAW_TEST_CASES]
        logger.info(f"Успешно обработано входящих кейсов: {len(validated_cases)}")

        summary = build_daily_summary(validated_cases)
        logger.info("Сводка успешно сгенерирована моделью.")

        print("\n" + "=" * 65)
        print(f"📋 УТРЕННЯЯ СВОДКА SUPPORT-ОТДЕЛА | {summary.current_time_context}")
        print(f"Всего кейсов: {summary.total_cases} | Требуют внимания: {summary.critical_cases_count}")
        print("=" * 65)

        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sorted_cases = sorted(summary.cases, key=lambda x: priority_order[x.priority])

        for item in sorted_cases:
            badge = "🔴 [СРОЧНО]" if item.priority == "HIGH" else ("🟡 [ВНИМАНИЕ]" if item.priority == "MEDIUM" else "🟢 [ОК]")
            print(f"\n{badge} Кейс {item.case_id} ({item.category})")
            print(f"  ├─ В CRM подтверждено: {item.crm_confirmed}")
            print(f"  ├─ Выводы из чата:     {item.chat_info}")
            print(f"  ├─ Оценка риска:       {item.risk_explanation}")
            if item.has_conflict_or_risk:
                print(f"  └─ 👉 С чего начать:   {item.next_step}")
            else:
                print(f"  └─ Действия:           Не требуются")
        print("\n" + "=" * 65)

    except Exception as e:
        logger.error(f"Критическая ошибка выполнения сервиса: {e}", exc_info=Tru
