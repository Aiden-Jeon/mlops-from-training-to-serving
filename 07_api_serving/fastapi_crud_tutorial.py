from fastapi import FastAPI
from pydantic import BaseModel

# create a Fastapi instance
app = FastAPI()

#
# database
#
ITEMS = {0: "default data"}


#
# create
#
class ItemCreateIn(BaseModel):
    item_body: str


class ItemCreateOut(BaseModel):
    item_id: int
    item_body: str


@app.post("/item/", response_model=ItemCreateOut)
def create_item(item_create_in: ItemCreateIn) -> ItemCreateOut:
    item_id = len(ITEMS)
    item_body = item_create_in.item_body
    ITEMS[item_id] = item_body
    return ItemCreateOut(item_id=item_id, item_body=item_body)


#
# read
#
class ItemGetOut(BaseModel):
    item_id: int
    item_body: str


@app.get("/item/", response_model=ItemGetOut)
def read_item(item_id: int) -> ItemGetOut:
    item_body = ITEMS.get(item_id, "Not valid id")
    return ItemGetOut(item_id=item_id, item_body=item_body)


#
# update
#
class ItemUpdateIn(BaseModel):
    item_id: int
    item_body: str


class ItemUpdateOut(BaseModel):
    item_id: int
    item_body: str


@app.put("/item/", response_model=ItemUpdateOut)
def update_item(item_update_in: ItemUpdateIn) -> ItemUpdateOut:
    item_id = item_update_in.item_id
    item_body = item_update_in.item_body
    if item_id not in ITEMS:
        item_body = "Not valid id"
    else:
        ITEMS[item_id] = item_body
    return ItemUpdateOut(item_id=item_id, item_body=item_body)


#
# delete
#
class ItemDeleteIn(BaseModel):
    item_id: int


class ItemDeleteOut(BaseModel):
    item_id: int
    item_body: str


@app.delete("/item/", response_model=ItemDeleteOut)
def delete_item(item_delete_in: ItemDeleteIn) -> ItemDeleteOut:
    item_id = item_delete_in.item_id
    if item_id not in ITEMS:
        item_body = "Not valid id"
    else:
        item_body = ITEMS[item_id]
        del ITEMS[item_id]
    return ItemDeleteOut(item_id=item_id, item_body=item_body)
