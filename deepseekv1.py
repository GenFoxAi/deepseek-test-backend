import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Optional, Sequence, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_milvus import Milvus
from langchain_core.tools import tool
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

load_dotenv()


app = FastAPI()
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://0.0.0.0:5000",
        "http://localhost:5173",
        "https://comm-it-engage-prototype.vercel.app",
        "https://deepseek-test-frontend-openairouter.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
DEEPOSEEK_API_KEY = os.getenv("DEEPOSEEK_API_KEY")  
MONGO_URI = os.getenv("MONGO_URI")
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_CLOUD_TOKEN = os.getenv("ZILLIZ_CLOUD_API_KEY")

client_mongo = MongoClient(MONGO_URI)
db = client_mongo["pdf_db"]

from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Milvus(
    embedding_function=embedding,
    collection_name="payroll_collection",
    connection_args={
        "uri": ZILLIZ_CLOUD_URI,
        "token": ZILLIZ_CLOUD_TOKEN,
        "secure": True
    }
)
if not vector_store.client.has_collection("payroll_collection"):
    raise ValueError("Collection 'payroll_collection' not found.")

deepseek_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=DEEPOSEEK_API_KEY,
)

class DeepseekLLM:
    def __init__(self, model: str):
        self.model = model

    def invoke(self, messages: List[Any]) -> AIMessage:
        """
        Expects a list of LangChain message objects (HumanMessage and AIMessage).
        Converts them into dictionaries and calls the Deepseek API.
        Returns an AIMessage with the content from Deepseek.
        """
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            else:
               
                api_messages.append({"role": "user", "content": str(msg)})

        response = deepseek_client.chat.completions.create(
            model=self.model,
            messages=api_messages
        )
        content = response.choices[0].message.content
        return AIMessage(content=content)

llm = DeepseekLLM(model="deepseek/deepseek-r1")

class State(TypedDict):
    messages: List[HumanMessage]
    action: Optional[str]
    pending_updates: Optional[dict]

@tool
def fetch_user_data(employee_id: str) -> Dict:
    """
    Retrieve user-related data from MongoDB.
    """
    user_data = {
        "basic_info": db["basic_info"].find_one({"employeeId": employee_id}),
        "employment_details": db["employment_details"].find_one({"employeeId": employee_id}),
        "payroll": db["payroll"].find_one({"employeeId": employee_id}),
        "leaves": db["leaves"].find_one({"employeeId": employee_id}),
        "reimbursement_claims": db["reimbursement_claims"].find_one({"employeeId": employee_id}),
        "attendance": db["attendance"].find_one({"employeeId": employee_id}),
        "roles": db["roles"].find_one({"employeeId": employee_id}),
        "documents": db["documents"].find_one({"employeeId": employee_id}),
        "gosi": db["gosi"].find_one({"employeeId": employee_id}),
    }
    if not any(user_data.values()):
        raise HTTPException(status_code=404, detail=f"Employee with ID {employee_id} not found.")
    for key in user_data:
        if user_data[key]:
            user_data[key].pop("_id", None)
    return user_data

def analyzer(state: State):
    conversation_history = "\n".join(
        [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}" for msg in state["messages"]]
    )
    print("analyzer")
    user_query = state["messages"][-1].content.strip()
    employee_id = "E001" 
    user_data = fetch_user_data(employee_id)

    if "document" in user_query or "pdf" in user_query:
        document_key = None
        docs = user_data.get("documents", {}).get("documents", [])
        query = user_query.lower()
        for doc in docs:
            if doc.get("documentType", "").lower() in query:
                document_key = doc
                break

        if document_key:
            file_id = document_key.get("fileId")
            file_path = f"static/{file_id}.pdf"
            if os.path.exists(file_path):
                return {
                    "messages": state["messages"] + [
                        AIMessage(
                            content=f"Your requested document is ready. Click below to download.",
                            response_metadata={"documentUrl": f"http://localhost:8000/{file_path}"}
                        )
                    ],
                    "action": "document_retrieval_request"
                }
            else:
                return {
                    "messages": state["messages"] + [
                        AIMessage(content="Sorry, I couldn't find the requested document. Please check and try again.")
                    ],
                    "action": None
                }

    if "pending_updates" in state and user_query in ["yes", "no"]:
        return {
            "messages": state["messages"],
            "action": "wait_for_confirmation"
        }

    prompt = f"""
Analyze the user's query and the conversation history to determine the user's intent. Follow these rules:

**Conversation History:**
{conversation_history}

**User's Latest Message:**
{user_query}

**Guidelines for Intent Recognition:**
1. If the user is talking about applying for something and if the context cannot be determined from the conversation history, return "unclear".
2. If the user is asking about applying for leave specifically, return "leave".
3. If the user is asking about applying for a reimbursement specifically, return "reimbursement".
4. If the user is talking about updating or changing details (like bank details (account number, bank name, IBAN) or basic info (email, phone number, address)), return "user_update_request".
5. If the user is asking for policies (leave or reimbursement), return "policy".
6. If the user says something generic like "I want to apply" and the conversation history indicates they were previously talking about leave, return "leave". If the history indicates they were talking about reimbursement, return "reimbursement".
7. If the user's query is unrelated to applying or does not match leave or reimbursement, return "none".

Answer with one of these responses: 'leave', 'reimbursement', 'policy', 'unclear', or 'none'.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    response_content = response.content.strip().lower()
    if response_content == "leave":
        return {"messages": state["messages"] + [AIMessage(content="Your leave application is in progress.")], "action": "open_leave_modal"}
    elif response_content == "reimbursement":
        return {"messages": state["messages"] + [AIMessage(content="Your reimbursement application is in progress.")], "action": "open_reimbursement_modal"}
    elif response_content == "user_update_request":
        return {"messages": state["messages"], "action": "user_update_request"}
    elif response_content == "policy":
        return state
    elif response_content == "unclear":
        return {
            "messages": state["messages"] + [
                AIMessage(content="Could you please clarify what you want to apply for? We have leave and reimbursement applications available.")
            ]
        }
    else:
        return state

def leave_modal(state: State) -> State:
    return state

def reimbursement_modal(state: State) -> State:
    return state

def document_retrieval(state: State) -> State:
    return state

def rag(state: State) -> State:
    action = state.get("action", None)
    last_user_msg = state["messages"][-1].content
    employee_id = "E001"
    user_data = fetch_user_data(employee_id)
    retrieved_docs = vector_store.similarity_search(last_user_msg)
    if not retrieved_docs:
        return {
            "messages": state["messages"] + [AIMessage(content="No relevant documents found.")],
            "action": action
        }
    policy_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    conversation_history = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["messages"]
    )
    prompt = f"""
You are a smart assistant with access to employee payroll data, Saudi labor policies, and prior conversation history.

Conversation History:
{conversation_history}

Employee Data:
{user_data}

Relevant Policies and Formulas:
{policy_context}

User Query:
"{last_user_msg}"

**Instructions for the Bot:**

1. **Concise Responses**: Deliver clear, to-the-point answers with only key results. Avoid unnecessary details unless required for clarity.
2. **Policy Queries**: For policy-related questions, provide a **detailed explanation** including context, applications, and implications using proper markdown formatting.
3. **Calculations**: Perform any required calculations (e.g., overtime pay, leave balance) and present results in **SAR (Saudi Riyal)**.
4. **Employee Details**: If requested, display employee details only up to **basic salary**.
5. **Formatting**: Use **bold**, _italics_, or other Markdown features for readability.
6. **Scope**: Politely inform the user if the query is unrelated or out of scope.

Answer:
"""
    response_content = llm.invoke([HumanMessage(content=prompt)]).content
    return {
        "messages": state["messages"] + [AIMessage(content=response_content)],
        "action": action
    }

def userUpdate(state: State) -> State:
    action = state.get("action", None)
    last_user_msg = state["messages"][-1].content
    employee_id = "E001"
    user_data = fetch_user_data(employee_id)
    retrieved_docs = vector_store.similarity_search(last_user_msg)
    if not retrieved_docs:
        return {
            "messages": state["messages"] + [AIMessage(content="No relevant documents found.")],
            "action": action
        }
    policy_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    conversation_history = "\n".join(
        f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["messages"]
    )

    prompt = f"""
You are a smart assistant specializing in securely updating **bank details** and **basic info** in the employee database. You have access to prior conversation history, organizational policies, and existing employee data for validation and updates.

**Field Categories**:
- **Bank Details**: Includes "bankName", "accountNumber", and "IBAN".
- **Basic Info**: Includes "email", "phone", and "address".

Conversation History:
{conversation_history}

Employee Data:
{user_data}

Relevant Policies:
{policy_context}

User Query:
"{last_user_msg}"

**Instructions for the Bot:**

1. **Parsing and Validation**:
   - Only validate fields explicitly mentioned in the query.
   - Ignore fields not referenced by the user.
   - If user says "update my phone", only validate/process the phone field.
   - Parse the user's query to identify which fields they want to update:
     - For **Bank Details**:
       - Ensure "bankName" is provided and properly formatted.
       - Ensure "accountNumber" is numeric and at least 10 digits.
       - Ensure "IBAN" is provided and matches the standard IBAN format.
     - For **Basic Info**:
       - Ensure "email" is properly formatted as a valid email address.
       - Ensure "phone" is numeric and at least 10 digits.
       - Ensure "address" is non-empty and formatted correctly.
   - If any field is invalid, provide feedback on the specific issue.

2. **Data Extraction**:
   - Extract **ONLY** fields explicitly mentioned in the user's query.
   - Store valid updates in variables (omit fields not mentioned):
     - `bank_details`: {{
         "bankName": "ExtractedBankName"  # ONLY if mentioned
       }}
     - `basic_info`: {{
         "phone": "ExtractedPhone"  # ONLY if mentioned
       }}

3. **Response Format**:
   - If valid updates are identified, respond in the following JSON format:
     {{
         "updated_fields": {{
             "bank_details": bank_details,
             "basic_info": basic_info
         }},
         "confirmation_prompt": "You have requested to update the following details: [list of updates]. Please confirm by responding with 'yes' or 'no'."
     }}
   - If no updates are identified, respond with:
     {{
         "error": "No valid updates identified. Please specify the details to be updated correctly."
     }}
   - If the request is out of scope, respond with:
     {{
         "error": "The request is out of scope. Only **bank details** (bankName, accountNumber, IBAN) and **basic info** (email, phone, address) can be updated."
     }}

Answer:
"""
    response_content = llm.invoke([HumanMessage(content=prompt)]).content

    try:
        parsed_response = json.loads(response_content)
    except json.JSONDecodeError:
        return {
            "messages": state["messages"] + [AIMessage(content="Unable to process your request. Please try again.")],
            "action": "rag"
        }

    if parsed_response.get("updated_fields"):
        updates = parsed_response["updated_fields"]
        confirmation_prompt = parsed_response.get("confirmation_prompt", "Please confirm your updates.")
        if "pending_updates" not in state or state["pending_updates"] is None:
            state["pending_updates"] = {}
        state["pending_updates"] = updates
        return {
            "messages": state["messages"] + [AIMessage(content=confirmation_prompt)],
            "action": "wait_for_confirmation",
            "pending_updates": state["pending_updates"]
        }
    else:
        response_message = parsed_response.get("error", "No valid updates identified. Redirecting your query for further assistance.")
        return {
            "messages": state["messages"] + [AIMessage(content=response_message)],
            "action": "rag"
        }

def processConfirmation(state: State) -> State:
    user_confirmation = state["messages"][-1].content.strip().lower()
    updates = state.get("pending_updates", None)
    employee_id = "E001"

    if not updates:
        return {
            "messages": state["messages"] + [AIMessage(content="No pending updates to confirm.")],
            "action": "rag"
        }

    if user_confirmation == "yes":
        update_success = update_user_data(employee_id, updates)
        if update_success:
            response_message = "The updates have been successfully applied to your profile, and payroll systems will be synchronized."
        else:
            response_message = "There was an error updating your details. Please try again later."
        state.pop("pending_updates", None)
    elif user_confirmation == "no":
        response_message = "Update operation canceled as per your request."
        state.pop("pending_updates", None)
    else:
        response_message = "Invalid response. Please confirm by responding with 'yes' or 'no'."

    return {
        "messages": state["messages"] + [AIMessage(content=response_message)],
        "action": "finalize"
    }

def update_user_data(employee_id: str, updates: Dict[str, Any]) -> bool:
    try:
        # Build partial updates
        if "bank_details" in updates:
            bank_updates = {}
            if "bankName" in updates["bank_details"]:
                bank_updates["bankAccount.bankName"] = updates["bank_details"]["bankName"]
            if "accountNumber" in updates["bank_details"]:
                bank_updates["bankAccount.accountNumber"] = updates["bank_details"]["accountNumber"]
            if "IBAN" in updates["bank_details"]:
                bank_updates["bankAccount.IBAN"] = updates["bank_details"]["IBAN"]

            if bank_updates:
                db["payroll"].update_one(
                    {"employeeId": employee_id},
                    {"$set": bank_updates}
                )

        if "basic_info" in updates:
            basic_info_updates = {}
            if "email" in updates["basic_info"]:
                basic_info_updates["email"] = updates["basic_info"]["email"]
            if "phone" in updates["basic_info"]:
                basic_info_updates["phone"] = updates["basic_info"]["phone"]
            if "address" in updates["basic_info"]:
                basic_info_updates["address"] = updates["basic_info"]["address"]

            if basic_info_updates:
                db["basic_info"].update_one(
                    {"employeeId": employee_id},
                    {"$set": basic_info_updates}
                )

        return True
    except Exception as e:
        print(f"Error updating user data: {e}")
        return False

def route_action(state: State) -> Sequence[str]:
    action = state.get("action")
    user_confirmation = state["messages"][-1].content.strip().lower()

    if action == "open_leave_modal":
        return ["leave_modal"]
    elif action == "open_reimbursement_modal":
        return ["reimbursement_modal"]
    elif action == "document_retrieval_request":
        return ["document_retrieval"]
    elif action == "user_update_request":
        return ["user_update"]
    elif action == "wait_for_confirmation":
        if user_confirmation in ["yes", "no"]:
            return ["process_confirmation"]
        else:
            return ["rag"]
    return ["rag"]

graph_builder = StateGraph(State)
graph_builder.add_node("analyzer", analyzer)
graph_builder.add_node("leave_modal", leave_modal)
graph_builder.add_node("reimbursement_modal", reimbursement_modal)
graph_builder.add_node("rag", rag)
graph_builder.add_node("document_retrieval", document_retrieval)
graph_builder.add_node("user_update", userUpdate)
graph_builder.add_node("process_confirmation", processConfirmation)
graph_builder.add_edge(START, "analyzer")
graph_builder.add_conditional_edges("analyzer", route_action, ["document_retrieval", "leave_modal", "reimbursement_modal", "rag", "user_update", "process_confirmation"])
graph_builder.add_edge("document_retrieval", END)
graph_builder.add_edge("leave_modal", END)
graph_builder.add_edge("reimbursement_modal", END)
graph_builder.add_edge("rag", END)
graph_builder.add_edge("user_update", END)
graph_builder.add_edge("process_confirmation", END)

compiled_graph: CompiledStateGraph = graph_builder.compile()

class ChatRequest(BaseModel):
    thread_id: str
    messages: List[Dict]

class ChatResponse(BaseModel):
    messages: List[str]
    action: Optional[str] = None

conversation_memory: Dict[str, State] = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        thread_id = request.thread_id
        if thread_id not in conversation_memory:
            conversation_memory[thread_id] = {"messages": [], "action": None, "pending_updates": None}

        for msg in request.messages:
            if msg["role"] == "user":
                conversation_memory[thread_id]["messages"].append(HumanMessage(content=msg["content"]))

        current_state = conversation_memory[thread_id]

        result = await compiled_graph.ainvoke(input=current_state)
        conversation_memory[thread_id] = result
        response_messages = []
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                response_content = m.content
                if hasattr(m, "response_metadata") and m.response_metadata:
                    response_content += f"\n[Metadata: {json.dumps(m.response_metadata)}]"
                response_messages.append(response_content)
            else:
                response_messages.append(m.content)

        return ChatResponse(
            messages=response_messages,
            action=result.get("action")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
