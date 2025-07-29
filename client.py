import asyncio
import websockets
import json
import requests

API_URL = "http://127.0.0.1:8000"
WEBSOCKET_URL = "ws://127.0.0.1:8000/ws/chat"


async def run_chat_client():
    print("--- Dumroo.ai Chat Client ---")

    # 1. Login to get a token
    while True:
        username = input("Enter username (amit_sharma, priya_singh, raj_kumar): ")
        try:
            response = requests.post(f"{API_URL}/login", json={"username": username})
            if response.status_code == 200:
                token = response.json()["access_token"]
                print("✅ Login successful!\n")
                break
            else:
                print(f"❌ Login failed: {response.text}")
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Is the FastAPI server running?")
            return

    # 2. Connect to WebSocket with the token
    uri = f"{WEBSOCKET_URL}?token={token}"
    try:
        async with websockets.connect(uri) as websocket:
            # Receive welcome message
            welcome_message = await websocket.recv()
            print(f"Server: {json.loads(welcome_message)['data']['content']}")

            print("\nType your questions below. Type 'exit' to quit.")

            while True:
                user_input = input(f"[{username}] > ")
                if user_input.lower() == "exit":
                    break

                await websocket.send(user_input)

                # The server might send multiple messages (chat + data)
                while True:
                    try:
                        # Set a timeout to break if no more messages are coming for this query
                        response_str = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        response = json.loads(response_str)

                        if response["type"] == "chat":
                            print(f"\n💡 Dumroo AI: {response['data']['content']}\n")
                        elif response["type"] == "data_result":
                            print("📦 Received structured data:")
                            print(json.dumps(response["data"], indent=2))
                            print("-" * 20)

                    except asyncio.TimeoutError:
                        # No more messages from the server for this query, break to wait for next user input
                        break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ Connection closed by server: {e.reason} (Code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(run_chat_client())
