import os
import requests

from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma 
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# import bs4
# from langchain import hub
# from langchain_community.document_loaders import TextLoader
# from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
# from langgraph.graph import END
# from langgraph.graph import MessagesState, StateGraph

# устанавливаем ключ для доступа к mistral api
os.environ["MISTRAL_API_KEY"] = "yvN5pMaZEOjydnFZOncBKz3k7islDBk8"
headers = {"Authorization": f"Bearer {os.environ.get('MISTRAL_API_KEY')}"}

# проверяем, что модель доступна
response = requests.get("https://api.mistral.ai/v1/models", headers=headers)
print(response.status_code)  # должен быть 200

# инициализируем модель чата mistral
llm = init_chat_model("open-mistral-nemo", model_provider="mistralai")

# задаём путь до файла с историей adrian_story.txt 
folder_path = os.path.expanduser("~/Desktop/codeforge")  # file_path = "/Users/louisa/Desktop/codeforge/adrian_story.txt"
file_path = os.path.join(folder_path, "adrian_story.txt")

# загружаем текст из файла и создаём список документов
docs = []
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        docs.append(Document(page_content=text))
else:
    print(f"Файл {file_path} не найден.")

# разбиваем текст на чанки для эмбеддинга
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=10)
all_splits = text_splitter.split_documents(docs)

# создаём объект эмбеддингов от mistral
embeddings = MistralAIEmbeddings(model="mistral-embed")

# удаляем старую базу перед созданием новой
# shutil.rmtree("./chroma_db3", ignore_errors=True)

# создаём локальное векторное хранилище (chroma)
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="./chroma_db3"
)

# отладочный print первых чанков
print(f"\nПример содержимого базы (первые 3 чанка):")
for i, doc in enumerate(all_splits[:3], 1):
    print(f"[{i}] {doc.page_content[:200]}...\n")

# создаём инструмент для поиска похожих фрагментов
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """находит релевантную информацию по запросу"""
    retrieved_docs = vector_store.similarity_search(query, k=1)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}") for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# функция, выбирающая между вызовом инструмента и генерацией ответа
def query_or_respond(state: MessagesState):
    """генерирует запрос к инструменту или даёт ответ напрямую"""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# создаём узел с инструментом
tools = ToolNode([retrieve])

# генерация финального ответа после получения данных
def generate(state: MessagesState):
    """генерирует окончательный ответ на основе найденных данных"""
    recent_tool_messages = [message for message in reversed(state["messages"]) if message.type == "tool"]
    tool_messages = recent_tool_messages[::-1]  # порядок по времени

    # объединяем все ответы инструментов
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    prompt_message = state["prompt_message"]
    system_message_content = f"{prompt_message}\n\n{docs_content}" 

    # берём только user/system/ai без tool_calls
    conversation_messages = [
        message for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages
    response = llm.invoke(prompt)
    return {"messages": [response]}

# инициализация памяти агента
memory = MemorySaver()

# создаём агента react с инструментом и памятью
agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)

# конфигурация 
config = {"configurable": {"thread_id": "abc678"}}

# получаем промпт от пользователя
prompt_message = input("Prompt: ")

# первый ввод пользователя
input_message = input("User: ")
for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}],
     "configurable": {"prompt_message": prompt_message}},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()

# второй ввод пользователя
input_message = input("User: ")
for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}],
     "configurable": {"prompt_message": prompt_message}},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()



# When did Adrian write best?
# What were his two main problems?
# What tricks did he try to stay awake?

# Answer concisely using the provided context.