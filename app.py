import uuid
import psycopg
from dotenv import load_dotenv
import asyncio
import telegram
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_postgres import PostgresChatMessageHistory


model = OllamaLLM(model="gemma3:4B", host="http://ollama:11434")

human_template = f"{{question}}"

prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", human_template),
    ]
)

chain = prompt_template | model

table_name = "chat_history"

fixed_session_id = str(uuid.uuid4())


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


async def main() -> None:
    load_dotenv()
    
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    
    bot = telegram.Bot(os.getenv('TOKEN'))

    async with bot:
        while True:
            user_question = input(">>>>")
            result = chain_with_history.invoke(
                {"size": "concise", "question": user_question},
                config={"configurable": {"session_id": fixed_session_id}}
            )
            
            print(result)


if __name__ == '__main__':
    asyncio.run(main())
