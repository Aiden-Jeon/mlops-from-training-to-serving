from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item_with_path(item_id: int):
    return {"item_id": item_id}


@app.get("/items/")
def read_item_with_query(item_id: int):
    return {"item_id": item_id}
