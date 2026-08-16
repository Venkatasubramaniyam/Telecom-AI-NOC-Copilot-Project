from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.chatbot import retriever
from rag.chatbot import llm


# Keep the prompt small enough for Render + Bedrock and avoid sending
# an entire large alarm/history dataframe to the LLM.
MAX_ALARMS = 30
MAX_HISTORY = 30
MAX_CONTEXT_DOCS = 5
MAX_CONTEXT_CHARS = 12000
MAX_HISTORY_CHARS = 10000
MAX_ALARM_CHARS = 10000


prompt = ChatPromptTemplate.from_template("""
You are an experienced Telecom NOC Engineer.

You have access to:

1. Current alarms
2. Historical incidents
3. Telecom SOP documents

Analyze the information and determine:

- Most likely root cause
- Confidence (%)
- Impact
- Recommended actions

Be concise and practical. Base the answer only on the supplied alarm,
historical incident, and SOP information. If the evidence is insufficient,
say so instead of inventing details.

Current Alarms:
{alarms}

Historical Incidents:
{history}

SOP Context:
{context}
""")


parser = StrOutputParser()
chain = prompt | llm | parser


def _limit_text(text, max_chars):
    text = str(text)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[content truncated]"
    return text


def analyze_root_cause(alarm_df, incident_df):
    """
    Analyze current alarms using historical incidents and SOP/RAG context.

    The input is deliberately limited to prevent very large prompts from
    causing slow Bedrock calls or Render 502/timeout responses.
    """

    # Use only the most recent/current rows available to this function.
    alarm_sample = alarm_df.head(MAX_ALARMS)
    history_sample = incident_df.head(MAX_HISTORY)

    alarm_text = _limit_text(
        alarm_sample.to_string(index=False),
        MAX_ALARM_CHARS
    )

    history_text = _limit_text(
        history_sample.to_string(index=False),
        MAX_HISTORY_CHARS
    )

    # Use the alarm information as the RAG search query.
    # Limit the query itself so retrieval remains fast.
    retrieval_query = _limit_text(alarm_text, 6000)

    docs = retriever.invoke(retrieval_query)

    selected_docs = docs[:MAX_CONTEXT_DOCS]

    context_parts = []
    total_chars = 0

    for doc in selected_docs:
        page = str(getattr(doc, "page_content", "") or "").strip()

        if not page:
            continue

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        if len(page) > remaining:
            page = page[:remaining] + "\n...[SOP context truncated]"

        context_parts.append(page)
        total_chars += len(page)

    context = "\n\n".join(context_parts)

    if not context:
        context = "No relevant SOP context was retrieved."

    return chain.invoke(
        {
            "alarms": alarm_text,
            "history": history_text,
            "context": context
        }
    )
