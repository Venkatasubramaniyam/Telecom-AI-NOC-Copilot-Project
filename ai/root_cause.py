from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.chatbot import retriever, llm


MAX_ALARMS = 30
MAX_HISTORY = 30
MAX_CONTEXT_DOCS = 5
MAX_CONTEXT_CHARS = 12000
MAX_HISTORY_CHARS = 8000
MAX_ALARM_CHARS = 8000


prompt = ChatPromptTemplate.from_template(
    """
You are an experienced Telecom NOC Engineer.

Analyze the supplied information and provide:

- Most likely root cause
- Confidence (%)
- Impact
- Recommended actions

Be concise and practical.
Use only the supplied alarm, historical incident, and SOP information.
Do not invent facts. If evidence is insufficient, say so.

Current Alarms:
{alarms}

Historical Incidents:
{history}

SOP Context:
{context}
"""
)

parser = StrOutputParser()
chain = prompt | llm | parser


def _limit_text(text, max_chars):
    text = str(text or "")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[content truncated]"
    return text


def analyze_root_cause(alarm_df, incident_df):
    alarm_sample = alarm_df.head(MAX_ALARMS)
    history_sample = incident_df.head(MAX_HISTORY)

    alarm_text = _limit_text(
        alarm_sample.to_string(index=False),
        MAX_ALARM_CHARS,
    )

    history_text = _limit_text(
        history_sample.to_string(index=False),
        MAX_HISTORY_CHARS,
    )

    docs = retriever.invoke(_limit_text(alarm_text, 5000))

    context_parts = []
    total_chars = 0

    for doc in docs[:MAX_CONTEXT_DOCS]:
        page = str(getattr(doc, "page_content", "") or "").strip()

        if not page:
            continue

        remaining = MAX_CONTEXT_CHARS - total_chars
        if remaining <= 0:
            break

        page = page[:remaining]
        context_parts.append(page)
        total_chars += len(page)

    context = "\n\n".join(context_parts)

    if not context:
        context = "No relevant SOP context was retrieved."

    return chain.invoke({
        "alarms": alarm_text,
        "history": history_text,
        "context": context,
    })
