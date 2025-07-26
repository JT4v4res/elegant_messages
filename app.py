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


model = OllamaLLM(model="gemma3:4B", host="http://ollama:11434", temperature=1.2)

human_template = f"{{question}}"

system_template = """
Você é Bentinho, assistente da organização da despedida de Pedro (Peu).

Use sempre português e seja direto ao ponto. Evite repetir frases ou bordões prontos.

Regras:
1. Não use frases genéricas como "que bom te ver!" ou "me conta as novidades".
2. Foque em dar a resposta ou executar o comando o mais rápido possível.
3. Se a mensagem não for clara, peça uma reformulação curta, como: _Não entendi. Pode dizer de outro jeito?_
4. Sempre tente variar a estrutura da resposta. Não use o mesmo começo ou fim toda vez.
5. Use emojis com moderação, apenas se combinarem com o contexto.

Sempre responda com clareza, sem rodeios e com boa disposição. Priorize variedade e objetividade.
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

# Removed the unused `today` variable as it was not used in the code.

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
            {"question": user_question},
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


async def save_elegant_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    telegram_chat_id = update.effective_chat.id
    
    user_first_name = update.message.from_user.first_name
    user_last_name = update.message.from_user.last_name

    username = f"{user_first_name or ''} {user_last_name or ''}".strip() or user_first_name or 'Unknown'
    
    message = " ".join(context.args)
    
    logging.info(f'Inserting new message {message} from user {username} in database')
    
    try:
        insert_elegant_message(username, message)
        
        logging.info(f'Successfully inserted message {message} from user {username} in database')
    except Exception as e:
        logging.error(f'Failed to insert message on database: {traceback.format_exc()}')
    
    model_query = f"Agradeça ao usuário por ter enviado a seguinte mensagem: {message}"
    
    response = await get_model_response(username, model_query, telegram_chat_id)
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


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


def insert_elegant_message(message_author: str, message: str):
    conn = psycopg.connect(
        dbname=os.getenv('PG_DATABASE'),
        user=os.getenv('PG_USER'),
        password=os.getenv('PG_PASSWORD'),
        host=os.getenv('PG_HOST'),
        port=os.getenv('PG_PORT')
    )
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO elegant_messages (message_author, message, created_at)
            VALUES (%s, %s, current_timestamp)
        """, (message_author, message))
        conn.commit()
    conn.close()


if __name__ == '__main__': 
    load_dotenv()
    
    application = ApplicationBuilder().token(os.getenv('TOKEN')).build()
    
    start_handler = CommandHandler('start', start)
    
    elegant_message_handler = CommandHandler('elegant_message', save_elegant_message)
    
    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)
    
    application.add_handler(start_handler)
    application.add_handler(echo_handler)
    application.add_handler(elegant_message_handler)
    
    application.run_polling()
