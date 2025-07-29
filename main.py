import os
import json
import pandas as pd
from openai import AsyncClient
from datetime import date, timedelta, datetime
import csv
from dotenv import load_dotenv

# FastAPI and WebSocket imports
from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Query
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware


# JWT imports
from jose import JWTError, jwt

# --- 1. Initial Setup & Configuration ---
load_dotenv()

app = FastAPI(
    title="Dumroo.ai Admin Chatbot API",
    description="A WebSocket-based API for querying student data with role-based access.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Authorization", "Set-Cookie"],
)

try:
    openai_client = AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))
except TypeError:
    raise RuntimeError("OpenAI API key not found. Please create a .env file with your key.")


# --- Helper Function to Create Sample Data (Runs at startup) ---
# (This function is identical to the one in the previous script)
def create_sample_csv(filename="student_data.csv"):
    header = [
        "student_id",
        "student_name",
        "grade",
        "class_section",
        "region",
        "assignment_name",
        "submission_status",
        "submission_date",
        "score_percentage",
    ]
    today = date.today()
    students = [
        ("S001", "Alice", "8", "A", "North"),
        ("S002", "Bob", "8", "A", "North"),
        ("S003", "Charlie", "8", "B", "North"),
        ("S004", "Diana", "9", "A", "South"),
        ("S005", "Evan", "9", "A", "South"),
        ("S006", "Fiona", "10", "C", "East"),
        ("S007", "George", "8", "A", "North"),
        ("S008", "Hannah", "10", "C", "East"),
    ]
    assignments = [
        ("Math Homework 1", "submitted", today - timedelta(days=2), 95, students[0]),
        ("Math Homework 1", "pending", None, None, students[1]),
        ("History Essay", "submitted", today - timedelta(days=3), 88, students[2]),
        ("Science Project", "submitted", today - timedelta(days=5), 92, students[3]),
        ("Science Project", "pending", None, None, students[4]),
        ("Math Homework 1", "submitted", today - timedelta(days=8), 78, students[6]),
        ("Algebra Quiz", "scheduled", today + timedelta(days=4), None, students[0]),
        ("Biology Quiz", "scheduled", today + timedelta(days=6), None, students[3]),
        ("Literature Quiz", "scheduled", today + timedelta(days=9), None, students[5]),
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, status, sub_date, score, student_info in assignments:
            row = list(student_info) + [
                name,
                status,
                sub_date.strftime("%Y-%m-%d") if sub_date else "N/A",
                score if score else "N/A",
            ]
            writer.writerow(row)
    print(f"✅ Sample data created in '{filename}'")


@app.on_event("startup")
def on_startup():
    # Only create sample CSV if it doesn't exist or is very small (to preserve expanded data)
    if not os.path.exists("student_data.csv"):
        create_sample_csv()
        print("📄 Created new student_data.csv file")
    else:
        # Check if file is too small (indicating it needs expanded data)
        try:
            df = pd.read_csv("student_data.csv")
            if len(df) < 20:  # If less than 20 records, assume it needs expansion
                print("📈 Existing CSV has minimal data, preserving current file")
            else:
                print(f"📊 Using existing student_data.csv with {len(df)} records")
        except Exception as e:
            print(f"⚠️ Error reading existing CSV: {e}, creating new one")
            create_sample_csv()


# --- 2. User Store & JWT Config ---
USERS = {
    "amit_sharma": {"scope_type": "grade", "scope_value": "8", "role": "Grade 8 Teacher", "full_name": "Amit Sharma"},
    "priya_singh": {
        "scope_type": "region",
        "scope_value": "South",
        "role": "South Region Administrator",
        "full_name": "Priya Singh",
    },
    "raj_kumar": {
        "scope_type": "class",
        "scope_value": ("10", "C"),
        "role": "Class 10-C Teacher",
        "full_name": "Raj Kumar",
    },
}
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# --- 3. Pydantic Models & Auth Utilities ---
class Token(BaseModel):
    access_token: str
    token_type: str


class UserLogin(BaseModel):
    username: str


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user_from_token(token: str):
    """Decodes JWT from WebSocket query param."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        scope: dict = payload.get("scope")
        if username is None or scope is None:
            return None
        return {"username": username, "scope": scope}
    except JWTError:
        return None


def generate_welcome_message(username: str) -> str:
    """Generate a personalized welcome message based on user role."""
    user_info = USERS.get(username)
    if not user_info:
        return "Welcome! How can I help you today?"

    full_name = user_info["full_name"]
    role = user_info["role"]
    scope_type = user_info["scope_type"]
    scope_value = user_info["scope_value"]

    # Generate access description based on scope
    if scope_type == "grade":
        access_desc = f"all Grade {scope_value} student data across all sections"
    elif scope_type == "region":
        access_desc = f"student data from all schools in the {scope_value} region"
    elif scope_type == "class":
        grade, section = scope_value
        access_desc = f"your Class {grade}-{section} student performance data"
    else:
        access_desc = "student data"

    welcome_msg = f"Welcome, {full_name.split()[0]}! 👋\n\n"
    welcome_msg += f"**Role:** {role}\n"
    welcome_msg += f"**Access:** You can view {access_desc}.\n\n"
    welcome_msg += "I'm here to help you analyze student performance, track assignments, and generate insights. "
    welcome_msg += "You can ask me about:\n"
    welcome_msg += "• Student performance and grades\n"
    welcome_msg += "• Assignment submission status\n"
    welcome_msg += "• Upcoming scheduled assignments\n"
    welcome_msg += "• Performance trends and analytics\n\n"
    welcome_msg += "What would you like to know today? 📊"

    return welcome_msg


# --- 4. The Data Querying "Tool" Class (Simplified) ---
class DumrooDataTool:
    def __init__(self, user_context, data_path="student_data.csv"):
        self.user_context = user_context
        full_df = pd.read_csv(data_path)
        full_df["submission_date"] = pd.to_datetime(full_df["submission_date"], errors="coerce")
        self.scoped_df = self._apply_rbac(full_df)

    def _apply_rbac(self, df):
        scope_type = self.user_context.get("scope_type")
        scope_value = self.user_context.get("scope_value")
        print(f"🔐 Applying access controls for WebSocket connection. Scope: {scope_type}='{scope_value}'")
        if scope_type == "grade":
            return df[df["grade"].astype(str) == str(scope_value)]
        if scope_type == "region":
            return df[df["region"] == scope_value]
        if scope_type == "class":
            grade, section = scope_value
            return df[(df["grade"].astype(str) == str(grade)) & (df["class_section"] == section)]
        return pd.DataFrame()

    def query_student_data(self, submission_status=None, date_range=None):
        df = self.scoped_df.copy()
        if submission_status:
            df = df[df["submission_status"] == submission_status]
        if date_range:
            today = pd.Timestamp.today().normalize()
            if date_range == "last_week":
                start_date = today - pd.DateOffset(days=today.weekday() + 7)
                end_date = start_date + pd.DateOffset(days=6)
                df = df[(df["submission_date"] >= start_date) & (df["submission_date"] <= end_date)]
            elif date_range == "next_week":
                start_date = today - pd.DateOffset(days=today.weekday()) + pd.DateOffset(weeks=1)
                end_date = start_date + pd.DateOffset(days=6)
                df = df[(df["submission_date"] >= start_date) & (df["submission_date"] <= end_date)]

        if df.empty:
            return json.dumps({"data": [], "message": "No records found matching your criteria."})

        records = df[
            ["student_name", "grade", "assignment_name", "submission_status", "submission_date", "score_percentage"]
        ].to_dict("records")

        cleaned_records = []
        for row in records:
            cleaned_row = {}
            for key, value in row.items():
                # Check for any pandas null type (NaT, NaN, etc.)
                if pd.isna(value):
                    cleaned_row[key] = None
                # If it's a valid date, format it
                elif isinstance(value, pd.Timestamp):
                    cleaned_row[key] = value.strftime("%Y-%m-%d")
                else:
                    cleaned_row[key] = value
            cleaned_records.append(cleaned_row)

        # Return the cleaned data in a consistent object structure
        return json.dumps({"data": cleaned_records, "message": f"{len(cleaned_records)} record(s) found."})


# --- 5. Tool and System Prompt for OpenAI ---
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "query_student_data",
            "description": "Get student data based on filters like submission status or schedules. Use this to answer any questions about students, homework, quizzes, or performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "submission_status": {
                        "type": "string",
                        "description": "Filter by assignment status.",
                        "enum": ["pending", "submitted", "scheduled"],
                    },
                    "date_range": {
                        "type": "string",
                        "description": "Time period to search, e.g., 'last_week', 'next_week'.",
                        "enum": ["last_week", "next_week"],
                    },
                },
                "required": [],
            },
        },
    }
]
SYSTEM_PROMPT = f"""You are a helpful admin assistant for Dumroo.ai. Answer user questions by calling the available `query_student_data` tool.

**Important formatting guidelines:**
- Present data in clean, readable markdown tables when showing multiple records
- Format dates in readable format (e.g., "July 27, 2025" instead of "2025-07-27")
- Show percentages with % symbol
- Use proper headings and structure for better readability
- Provide brief summaries or insights when presenting data
- Be concise but informative

Today's date is {date.today().strftime('%Y-%m-%d')}."""


# --- 6. Static Files & Routes ---
# Serve the main HTML file at root
@app.get("/")
async def read_index():
    """Serve the main HTML interface."""
    return FileResponse("index.html")


# --- 7. API Endpoints ---
@app.post("/login", response_model=Token)
async def login_for_access_token(form_data: UserLogin):
    """Login endpoint to get a JWT."""
    user = USERS.get(form_data.username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username")
    user_scope = {"scope_type": user["scope_type"], "scope_value": user["scope_value"]}
    access_token = create_access_token(data={"sub": form_data.username, "scope": user_scope})
    return {"access_token": access_token, "token_type": "bearer"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: str = Query(...)):
    """The main chat WebSocket endpoint."""
    user_data = await get_current_user_from_token(token)
    if not user_data:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or expired token")
        return

    await websocket.accept()
    welcome_message = generate_welcome_message(user_data["username"])
    await send_chat_message(welcome_message, websocket)

    # Initialize the data tool and conversation history for this specific connection
    data_tool = DumrooDataTool(user_context=user_data["scope"])
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            user_message = await websocket.receive_text()
            conversation_history.append({"role": "user", "content": user_message})

            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation_history,
                tools=TOOLS_DEFINITION,
                tool_choice="auto",
            )
            response_message = response.choices[0].message
            conversation_history.append(response_message)

            if response_message.tool_calls:
                await handle_tool_calls(response_message.tool_calls, conversation_history, data_tool, websocket)
            else:
                await send_chat_message(response_message.content, websocket)

    except WebSocketDisconnect:
        print(f"Client {user_data['username']} disconnected.")
    except Exception as e:
        print(f"An error occurred in websocket for {user_data['username']}: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=f"An internal error occurred: {e}")


# --- 7. WebSocket Helper Functions ---
async def handle_tool_calls(tool_calls, history: list, data_tool: DumrooDataTool, websocket: WebSocket):
    """Processes tool calls, executes functions, and sends results back to the LLM and client."""
    # First, process ALL tool calls and add their responses to history
    for tool_call in tool_calls:
        if tool_call.function.name == "query_student_data":
            print(f"🤖 LLM requested to call 'query_student_data' tool (ID: {tool_call.id}).")
            args = json.loads(tool_call.function.arguments)
            tool_response_json = data_tool.query_student_data(**args)

            history.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": "query_student_data",
                    "content": tool_response_json,
                }
            )

    # After ALL tool calls are processed, get the summary
    print("🗣️ Getting final summary from LLM...")
    summary_response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
    )
    summary_message = summary_response.choices[0].message.content

    # Add the summary response to conversation history to maintain consistency
    history.append({"role": "assistant", "content": summary_message})

    # Send only the formatted summary message
    await send_chat_message(summary_message, websocket)


async def send_json_response(type: str, data: dict, websocket: WebSocket):
    """Sends a structured JSON message to the client."""
    await websocket.send_text(json.dumps({"type": type, "data": data}))


async def send_chat_message(message: str, websocket: WebSocket):
    """Sends a regular chat message to the client."""
    await send_json_response("chat", {"content": message}, websocket)
