from fastapi import FastAPI
from pydantic import BaseModel


#
# Data out schema
#
class ItemOut(BaseModel):
    item_id: int
    item_body: str


# Create a FastAPI instance
app = FastAPI()


@app.get("/item/", response_model=ItemOut)
def read_item_with_query_and_pydantic(item_id: int) -> ItemOut:
    item_body = str(item_id + 1)
    return ItemOut(item_id=item_id, item_body=item_body)
