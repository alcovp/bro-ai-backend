import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# --- Конфигурация ---
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Получаем ключ и модель из .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o") # По умолчанию gpt-4o
TELEGRAM_BOT_ID = os.getenv("TELEGRAM_BOT_ID")

if not OPENAI_API_KEY:
    logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
    exit(1) # Завершаем работу, если ключ не найден

# --- Инициализация Flask ---
app = Flask(__name__)

# --- Инициализация LLM ---
# Вы можете добавить параметры, если нужно, например, temperature
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name=OPENAI_MODEL_NAME
)

# --- Определение Агента CrewAI ---
chat_participant_agent = Agent(
    role='Участник группового чата (AI)',
    goal=(
        "Отвечать кратко и по делу на прямые обращения или вопросы. "
        "Вклинивайся редко, только если можешь сказать что-то действительно уместное."
    ),
    backstory=(
        "Ты — AI в чате друзей. Говоришь как обычный чел: уверенно, без пафоса и лишних слов. "
        "Не пытаешься быть слишком крутым или смешным. "
        "Отвечаешь, когда тебя зовут или спрашивают, и редко лезешь сам, только если реально есть что сказать по теме."
    ),
    llm=llm,
    verbose=True, # Включаем логирование работы агента
    allow_delegation=False, # Для MVP агент работает сам
    # max_iter=5 # Ограничение итераций на всякий случай
)

# --- Определение Задачи CrewAI ---
# Магическая строка, которую агент вернет, если решит не отвечать
NO_RESPONSE_MARKER = "NO_RESPONSE"

chat_analysis_task = Task(
    description=(
        "Тебе предоставлена история последних сообщений в чате и самое новое сообщение.\n"
        "Формат истории: '[Имя отправителя]: [Текст сообщения]'\n"
        f"Твой id в истории чата: {TELEGRAM_BOT_ID}.\n"
        "К тебе могут обращаться в чате по имени Бро или Bro с большой буквы.\n"
        "Твоя задача: Проанализируй новое сообщение в контексте истории.\n"
        "Реши, нужно ли тебе ответить на это сообщение или на текущую беседу в целом.\n"
        "Критерии для ответа:\n"
        "- ОБЯЗАТЕЛЬНО отвечай, если к тебе обращаются напрямую (например, 'Бро', 'Bro') или задают вопрос, явно адресованный тебе.\n"
        "- Вклинивайся без обращения ТОЛЬКО если можешь сказать что-то короткое, релевантное и по теме, но делай это редко.\n"
        "- НЕ отвечай, если последнее сообщение было от тебя.\n"
        "- НЕ задавай вопросы в своих ответах.\n"
        "- Пиши кратко, уверенно, без эмодзи и лишней болтовни.\n"
        "- Избегай слэнга, если он звучит неестественно.\n"
        "- Формулируй ответ на РУССКОМ языке.\n"
        f"Если ты решил ответить, напиши текст своего ответа.\n"
        f"Если ты решил НЕ отвечать, ВЕРНИ ТОЛЬКО СТРОКУ: {NO_RESPONSE_MARKER}\n\n"
        "ИСТОРИЯ ЧАТА (последние сообщения):\n"
        "-------------------------------------\n"
        "{chat_history}\n"
        "-------------------------------------\n\n"
        "НОВОЕ СООБЩЕНИЕ:\n"
        "-------------------------------------\n"
        "{new_message}\n"
        "-------------------------------------\n\n"
        f"Твой ответ (или {NO_RESPONSE_MARKER}):"
    ),
    expected_output=(
        "Текст твоего ответа на русском языке, если ты решил ответить. "
        "Или ТОЧНО строка '" + NO_RESPONSE_MARKER + "', если ты решил не отвечать."
    ),
    agent=chat_participant_agent
)

# --- Создание Crew ---
crew = Crew(
    agents=[chat_participant_agent],
    tasks=[chat_analysis_task],
    process=Process.sequential,
    verbose=True
)

# --- Форматирование входных данных для Задачи ---
def format_chat_data(history, new_message):
    """Форматирует историю и новое сообщение в строки для промпта."""
    history_str = "\n".join([f"[{msg['sender']}]: {msg['text']}" for msg in history])
    new_message_str = f"[{new_message['sender']}]: {new_message['text']}"
    return history_str, new_message_str

# --- API Эндпоинт ---
@app.route('/process_message', methods=['POST'])
def handle_process_message():
    """Обрабатывает входящие сообщения от Telegram бота."""
    if not request.is_json:
        logging.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    logging.info(f"Received data: {data}")

    # Валидация входных данных (базовая)
    if not data or 'chat_id' not in data or 'new_message' not in data or 'history' not in data:
        logging.warning("Missing required fields in JSON data")
        return jsonify({"error": "Missing required fields: chat_id, new_message, history"}), 400

    chat_id = data['chat_id']
    new_message = data['new_message'] # Ожидаем {'text': '...', 'sender': '...', ...}
    history = data['history']       # Ожидаем список [ {'text': '...', 'sender': '...'}, ... ]

    # Форматируем данные для CrewAI
    try:
        history_str, new_message_str = format_chat_data(history, new_message)
        logging.info(f"Formatted History for Crew: \n{history_str}")
        logging.info(f"Formatted New Message for Crew: \n{new_message_str}")
    except Exception as e:
        logging.error(f"Error formatting chat data: {e}", exc_info=True)
        return jsonify({"error": "Internal server error formatting data"}), 500

    # Запуск CrewAI
    try:
        logging.info(f"[Chat {chat_id}] Starting Crew kickoff...")
        # Используем словарь для входных данных, как ожидает Task
        inputs = {
            'chat_history': history_str,
            'new_message': new_message_str
        }
        crew_result = crew.kickoff(inputs=inputs)
        logging.info(f"[Chat {chat_id}] Crew kickoff finished. Result: '{crew_result}'")

        # Обработка результата
        response_text = None
        if crew_result and crew_result.raw and crew_result.raw.strip() != NO_RESPONSE_MARKER:
            response_text = crew_result.raw.strip()
            logging.info(f"[Chat {chat_id}] Sending response: '{response_text}'")
        else:
            logging.info(f"[Chat {chat_id}] No response generated by AI.")

        return jsonify({"response_text": response_text})

    except Exception as e:
        logging.error(f"[Chat {chat_id}] Error during Crew kickoff or processing: {e}", exc_info=True)
        return jsonify({"error": "Internal server error processing message with AI"}), 500

# --- Главная точка входа (для запуска Flask) ---
if __name__ == '__main__':
    # Запускаем Flask сервер
    # host='0.0.0.0' делает его доступным извне контейнера/сети
    # debug=True НЕ используйте в продакшене!
    app.run(host='0.0.0.0', port=5000, debug=False)
    # Для продакшена лучше использовать Gunicorn или uWSGI
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app