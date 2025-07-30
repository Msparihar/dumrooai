# 🤖 Dumroo.ai Admin Chatbot

**AI-powered student data analytics with role-based access control**

## 🎬 Demo Video

![Demo](https://www.loom.com/share/52498750f0fc4a89bbe5dcf6669429c7?sid=76d7c7ed-7f13-4836-8d78-b7cf9ff5e0fc)

*👆 Watch the full demo of the Dumroo.ai Admin Chatbot*

---

## 🚀 Quick Start

Get up and running in 3 simple steps:

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd dumrooai

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Run with Docker
docker compose up --build
```

**🌐 Open your browser and go to:** `http://localhost:8000`

---

## ✨ Features

- **🔐 Role-Based Access Control**: Different users see different data based on their role
- **💬 Real-time Chat Interface**: WebSocket-powered AI conversations
- **📊 Beautiful Data Visualization**: Markdown tables with proper formatting
- **🎨 Modern UI**: Clean, responsive interface with loading animations
- **🐳 Docker Ready**: One-command deployment
- **🤖 OpenAI Integration**: Powered by GPT-4o-mini for intelligent responses

---

## 👥 User Roles & Access

### 🧑‍🏫 Grade 8 Teacher (amit_sharma)

- **Access**: All Grade 8 students across sections A & B
- **Data**: 24+ student records from North, South, East, West regions
- **Use Cases**: Grade-level performance analysis, cross-section comparisons

### 🏢 South Region Administrator (priya_singh)

- **Access**: All students from South region schools
- **Data**: 18+ student records across grades 8, 9, 10
- **Use Cases**: Regional performance tracking, multi-grade insights

### 👨‍🏫 Class 10-C Teacher (raj_kumar)

- **Access**: Only Class 10-C students
- **Data**: 6+ specific class records
- **Use Cases**: Class-specific assignments, individual student tracking

---

## 🛠️ Setup Instructions

### Prerequisites

- Docker & Docker Compose
- OpenAI API Key ([Get one here](https://platform.openai.com/api-keys))

### Environment Configuration

1. **Copy the environment template:**

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your API key:**

   ```env
   OPENAI_API_KEY=sk-your-actual-openai-api-key-here
   SECRET_KEY=your_super_secret_jwt_key_here
   ALGORITHM=HS256
   ```

3. **Launch the application:**

   ```bash
   docker compose up --build
   ```

### Accessing the Application

- **Web Interface**: <http://localhost:8000>
- **API Documentation**: <http://localhost:8000/docs> (FastAPI auto-generated)

---

## 🎯 Try These Sample Queries

Once logged in, try asking:

- *"Show me all submitted assignments from last week"*
- *"Which students have pending assignments?"*
- *"What are the upcoming scheduled assignments?"*
- *"Show me student performance in Math subjects"*
- *"Who scored above 90% this week?"*

---

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    FastAPI       │    │   OpenAI API    │
│   (HTML/JS)     │◄──►│   Backend        │◄──►│   (GPT-4o)      │
│                 │    │                  │    │                 │
│ • Login UI      │    │ • JWT Auth       │    │ • Tool Calling  │
│ • Chat Interface│    │ • WebSocket      │    │ • Data Analysis │
│ • Markdown      │    │ • RBAC           │    │ • Formatting    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Student Data   │
                       │     (CSV)        │
                       │                  │
                       │ • 72 Records     │
                       │ • 60 Students    │
                       │ • 20+ Subjects   │
                       └──────────────────┘
```

## 🧬 Technology Stack

- **Backend**: FastAPI, Python 3.12
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **AI**: OpenAI GPT-4o-mini with function calling
- **Authentication**: JWT tokens
- **WebSocket**: Real-time bidirectional communication
- **Containerization**: Docker & Docker Compose
- **Data**: CSV with 72 realistic student records

---

## 📊 Dataset Overview

The included dataset contains:

- **60 unique students** with realistic names
- **72 assignment records** across multiple subjects
- **Subjects**: Math, Science, History, English, Physics, Chemistry, Biology, Art, Music, Languages
- **Score Range**: 76-96% (realistic distribution)
- **Statuses**: Submitted (50), Scheduled (10), Pending (12)
- **Regions**: North, South, East, West (balanced)
- **Grades**: 8, 9, 10 with proper distribution

---

## 🔒 Security Features

- **JWT Authentication**: Secure token-based auth
- **Role-Based Access Control**: Data isolation by user role
- **Input Sanitization**: XSS protection
- **Environment Variables**: Secure API key management
- **CORS Configuration**: Controlled cross-origin requests

---

## 🐳 Docker Configuration

The project includes optimized Docker setup:

- **Multi-stage builds** with UV package manager
- **Environment variable** injection
- **Volume mounting** for data persistence
- **Health checks** and proper logging
- **Production-ready** configuration

---

## 🚀 Development

### Local Development (without Docker)

```bash
# Install dependencies
uv sync

# Set up environment
cp .env.example .env
# Edit .env with your API key

# Run development server
uv run fastapi run main.py --reload

# Open browser
open http://localhost:8000
```

### Project Structure

```
dumrooai/
├── main.py              # FastAPI application
├── index.html           # Frontend interface
├── student_data.csv     # Sample dataset
├── docker-compose.yml   # Docker configuration
├── Dockerfile          # Container definition
├── .env.example        # Environment template
├── pyproject.toml      # Python dependencies
└── README.md           # This file
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

If you encounter any issues:

1. Check the [Issues](../../issues) page
2. Review the demo video above
3. Ensure your OpenAI API key is valid
4. Verify Docker is running properly

---

## 🙏 Acknowledgments

- **OpenAI** for the powerful GPT-4o-mini model
- **FastAPI** for the excellent web framework
- **Docker** for containerization
- **Marked.js** for markdown rendering

---

<div align="center">

**Built with ❤️ for educational data analytics**

⭐ **If this project helped you, please give it a star!** ⭐

</div>
