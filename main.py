import uvicorn

from app.api.main import app  # noqa: F401 — imported so uvicorn can reference it


def main():
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
