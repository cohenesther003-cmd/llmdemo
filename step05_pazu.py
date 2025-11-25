import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Pazu Backend Service")


class ItemPayload(BaseModel):
    name: str
    character: str
    category: str
    rarity: Optional[str] = None
    power: Optional[int] = Field(default=None, ge=0)


class Item(ItemPayload):
    id: str


class WearRequest(BaseModel):
    character: str
    item_id: str


ITEMS: Dict[str, Item] = {}


def list_items(character: Optional[str], category: Optional[str], rarity: Optional[str]) -> List[Item]:
    items = list(ITEMS.values())
    if character:
        items = [i for i in items if i.character.lower() == character.lower()]
    if category:
        items = [i for i in items if i.category.lower() == category.lower()]
    if rarity:
        items = [i for i in items if (i.rarity or "").lower() == rarity.lower()]
    return items


@app.get("/items")
def get_items(character: Optional[str] = None, category: Optional[str] = None, rarity: Optional[str] = None):
    return {"items": list_items(character, category, rarity)}


@app.get("/items/{item_id}")
def get_item(item_id: str):
    item = ITEMS.get(item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    return item


@app.post("/items", status_code=201)
def create_item(payload: ItemPayload):
    item_id = str(uuid.uuid4())
    item = Item(id=item_id, **payload.model_dump())
    ITEMS[item_id] = item
    return item


@app.put("/items/{item_id}")
def update_item(item_id: str, payload: ItemPayload):
    if item_id not in ITEMS:
        raise HTTPException(404, "Item not found")
    ITEMS[item_id] = Item(id=item_id, **payload.model_dump())
    return ITEMS[item_id]


@app.post("/actions/wear")
def wear_item(request: WearRequest):
    item = ITEMS.get(request.item_id)
    if not item:
        raise HTTPException(404, "Item not found")
    if item.character.lower() != request.character.lower():
        raise HTTPException(400, f"{item.name} belongs to {item.character}, not {request.character}")
    return {"message": f"{request.character} is now wearing {item.name}."}