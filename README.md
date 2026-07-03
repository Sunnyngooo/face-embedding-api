## Face Embedding Model

## Feature

## Project Structure

```
face-embedding-model/
├── main.py
├── requirements.txt
├── README.md
└── app/
    ├── api/
    │   └── router.py         # API endpoint definitions
    └── services/
        └── face_service.py   # Face processing logic
```

## Installation

### Prerequisites

- Python 3.8 or higher

### Setup

1. **Create a virtual environment**

```bash
python -m venv venv

# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Dependencies

- FastAPI
- uvicorn
- DeepFace
- OpenCV Python Headless
- TF-Keras

## How to run

### Start the Server

```bash
uvicorn main:app --reload
```
or 
```bash
uvicorn main:app --host 0.0.0.0  --port 8000
```
when running on Server/Box

The API will be available at `http://localhost:8000` by default.

It is also have Interactive API Documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
