from fastapi import FastAPI
from app.routers import context

app = FastAPI()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload_includes=["*.html", "*.css", "*.js"],
        reload_excludes=["./node_modules"],
        reload_dirs=["./templates", "./templates/static"],
        reload=True,
    )


app.include_router(context.router,prefix="/context",tags=["context"])

@app.get("/")
async def root():
    return {"message": "Hello World from FastAPI with live reload"}