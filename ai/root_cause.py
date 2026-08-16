from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RCA uses the LLM directly.
# RAG/SOP retrieval remains available through the separate AI Telecom
# Assistant (/api/sop), but is intentionally not called here because
# OpenAI embeddings + Chroma retrieval was causing the Render request
# to take too long.
from rag.chatbot import llm


MAX_ALARMS = 10
MAX_HISTORY = 10
MAX_ALARM_CHARS = 4000
MAX_HISTORY_CHARS = 4000


prompt = ChatPromptTemplate.from_template(
    """
You are an experienced Telecom NOC Engineer.

Analyze the current alarms and historical incidents.

Provide a concise response with exactly these sections:

Most likely root cause:
Confidence:
Impact:
Recommended actions:

Use only the supplied information.
Do not invent alarm details or historical facts.
If the evidence is insufficient, clearly say so.

Current Alarms:
{alarms}

Historical Incidents:
{history}
"""
)

parser = StrOutputParser()

# Create the chain once when the application starts.
chain = prompt | llm | parser


def _limit_text(text, max_chars):
    text = str(text or "")
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[content truncated]"
    return text


def analyze_root_cause(alarm_df, incident_df):
    """
    Fast RCA path for Render.

    Only a small number of alarm/history rows are sent to the LLM.
    This avoids the additional OpenAI Embeddings + Chroma network call
    that was causing long waits and Gunicorn worker timeouts.
    """

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

    return chain.invoke(
        {
            "alarms": alarm_text,
            "history": history_text,
        }
    )
