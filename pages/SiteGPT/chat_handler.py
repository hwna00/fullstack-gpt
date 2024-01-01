from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory

from pages.SiteGPT.utils import ChatCallbackHandler


llm = ChatOpenAI(
    temperature=0.1,
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=150,
    memory_key="chat_history",
    return_messages=True,
)

answers_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                        
            Then, give a score to the answer between 0 and 5.
            If the answer answers the user question the score should be high, else it should be low.
            Make sure to always include the answer's score even if it's 0.
            Context: {context}
                                                        
            Examples:
                                                        
            Question: How far away is the moon?
            Answer: The moon is 384,400 km away.
            Score: 5
                                                        
            Question: How far away is the sun?
            Answer: I don't know
            Score: 0    
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def get_answers(input):
    docs = input["docs"]
    question = input["question"]
    chat_history = input["chat_history"]

    llm.streaming = False
    llm.callbacks = None
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "chat_history": chat_history,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                        "chat_history": chat_history,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.
            Use the answers that have the highest score (more helpful) and favor the most recent ones.
            Cite sources and return the sources of the answers as they are, do not change them.
            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    chat_history = inputs["chat_history"]

    llm.streaming = True
    llm.callbacks = [ChatCallbackHandler()]

    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "answers": condensed,
            "question": question,
            "chat_history": chat_history,
        }
    )
