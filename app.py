import logging.handlers
import uuid
import psycopg
from dotenv import load_dotenv
import asyncio
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
)
from langchain_postgres import PostgresChatMessageHistory
import logging
from telegram import Update
from telegram.ext import filters, ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler
import traceback
import datetime
from utils.custom_formatter import CustomFormatter


model = OllamaLLM(model="gemma3:4B", host="http://ollama:11434")

human_template = f"{{question}}"

system_template = """
Sua persona: Você é Bentinho, um auxiliar da organização da despedida de um rapaz chamado Pedro, vulgo Peu.

Alguns guias: 
1. Responda sempre em português.
2. Não cometa erros.
3. Suas mensagens serão enviadas pelo telegram, então para formatar texto siga as seguintes convenções:
    a. Negrito:  *texto*
    b. Itálico: _texto_
    c. Sublinhado: __texto__
    d. Tachado: ~texto~
    e. Spoiler: ||texto||
4. Use um tom de animado e empolgado, você deve se portar como um amigo próximo que está feliz pela nova etapa da vida do amigo Pedro.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", human_template),
        ("system", system_template)
    ]
)

try:
    os.mkdir('./logs')
except FileExistsError:
    print('"logs/" directory already exists')

fmt = "[%(asctime)s] - [%(levelname)s] - %(message)s"

today = datetime.date.today().strftime("%Y-%m-%d")

file_handler = logging.handlers.TimedRotatingFileHandler('./logs/bentinho.log',
                                                         when="midnight")
file_handler.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(CustomFormatter(fmt))

logging.basicConfig(
        format=fmt,
        style="%",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[file_handler, stdout_handler]
    )

chain = prompt_template | model

table_name = "chat_history"

fixed_session_id = str(uuid.uuid4())


async def get_model_response(user_first_name: str, user_question: str,
                             telegram_chat_id: int) -> str:
    logging.info(f'User: {user_first_name} asked: {user_question}')
    
    try:
        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        session_id = str(uuid.UUID(int=telegram_chat_id))
        
        result = chain_with_history.invoke(
            {"size": "concise", "question": user_question},
            config={"configurable": {"session_id": session_id}}
        )
        
        logging.info(f'Model response: {result.replace("AI:", "")}')
        
        return result.replace("AI:", "")
    except Exception:
        logging.error(f'Error while generating reponse: {traceback.format_exc()}')
        
        return chain_with_history.invoke(
            {"size": "concise", "question": "gere uma mensagem de desculpas informando que você não pode responder a essa questão."},
            config={"configurable": {"session_id": session_id}}
        ).replace("AI:", "")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text='Olá! Vamos mandar uma mensagem?')


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_chat_id = update.effective_chat.id

    user_first_name = update.message.from_user.first_name
    user_last_name = update.message.from_user.last_name

    username = user_first_name + ' ' + user_last_name

    response = await get_model_response(username, update.message.text, telegram_chat_id)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    sync_connection = psycopg.connect(
        dbname=os.getenv('PG_DATABASE'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        host=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT')
    )
    
    return PostgresChatMessageHistory(
        table_name, session_id, sync_connection=sync_connection
    )


if __name__ == '__main__': 
    load_dotenv()
    
    application = ApplicationBuilder().token(os.getenv('TOKEN')).build()
    
    start_handler = CommandHandler('start', start)
    
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    
    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    
    application.run_polling()
