from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.chatbot import retriever
from rag.chatbot import llm

prompt = ChatPromptTemplate.from_template("""
You are an experienced Telecom NOC Engineer.

You have access to:

1. Current alarms
2. Historical incidents
3. Telecom SOP documents

Determine

- Most likely root cause
- Confidence (%)
- Impact
- Recommended actions

Current Alarms:
{alarms}

Historical Incidents:
{history}

SOP Context:
{context}
""")

parser = StrOutputParser()


def analyze_root_cause(alarm_df, incident_df):

    alarm_text = alarm_df.to_string(index=False)

    history_text = incident_df.to_string(index=False)

    docs = retriever.invoke(alarm_text)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = prompt | llm | parser

    return chain.invoke(
        {
            "alarms": alarm_text,
            "history": history_text,
            "context": context
        }
    )