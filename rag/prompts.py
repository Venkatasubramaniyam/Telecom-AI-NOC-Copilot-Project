from langchain_core.prompts import ChatPromptTemplate

telecom_prompt = ChatPromptTemplate.from_template(
"""
You are an expert Telecom NOC Engineer.

Answer only using the supplied SOP.

Context:
{context}

Question:
{question}
"""
)

prompt = ChatPromptTemplate.from_template("""
You are an experienced Telecom NOC Engineer.

You have access to:

1. Current alarms
2. Historical incidents
3. Telecom SOP documents

Based on all three sources,

Determine

- Most likely root cause
- Confidence (%)
- Impact
- Recommended actions

Current Alarms

{alarms}

Historical Incidents

{history}

SOP Context

{context}

Return the answer in professional telecom format.
""")

parser = StrOutputParser()