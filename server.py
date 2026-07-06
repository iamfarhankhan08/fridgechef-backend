"""Pantry Rescue backend - FastAPI + MongoDB + Gemini via emergentintegrations."""
from fastapi import FastAPI, APIRouter, HTTPException, Header, Depends, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import re
import json
import logging
import base64
import secrets
import httpx
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta

from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent

ROOT_DIR = Path(__file__).parent
if (ROOT_DIR / '.env').exists():
    load_dotenv(ROOT_DIR / '.env')

MONGO_URL = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
DB_NAME = os.environ.get('DB_NAME', 'fridgechef')
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY', '')

try:
    client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    # Check connection
    # client.admin.command('ping') 
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    db = None

app = FastAPI(title="Pantry Rescue API")
api = APIRouter(prefix="/api")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Pantry Rescue API is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pantry")


# ---------- Models ----------
class User(BaseModel):
    id: str
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    is_guest: bool = False
    is_pro: bool = False
    pro_since: Optional[datetime] = None
    country_preference: str = "Auto"
    unit_preference: str = "Metric"
    diet_preference: str = "No preference"
    skill_level: str = "Beginner"
    spice_level: str = "Medium"
    onboarding_complete: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PantryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    category: str = "Other"
    quantity: float = 1
    unit: str = "pieces"
    expiry_date: Optional[str] = None  # ISO date
    storage_location: str = "Pantry"
    is_leftover: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PantryCreate(BaseModel):
    name: str
    category: Optional[str] = "Other"
    quantity: Optional[float] = 1
    unit: Optional[str] = "pieces"
    expiry_date: Optional[str] = None
    storage_location: Optional[str] = "Pantry"
    is_leftover: Optional[bool] = False


class PantryUpdate(BaseModel):
    name: Optional[str] = None
    category: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    expiry_date: Optional[str] = None
    storage_location: Optional[str] = None
    is_leftover: Optional[bool] = None


class SavedRecipe(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    recipe: Dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GroceryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    category: str = "Other"
    quantity: float = 1
    unit: str = "pieces"
    checked: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GroceryCreate(BaseModel):
    name: str
    category: Optional[str] = "Other"
    quantity: Optional[float] = 1
    unit: Optional[str] = "pieces"


class GroceryUpdate(BaseModel):
    checked: Optional[bool] = None
    name: Optional[str] = None
    quantity: Optional[float] = None


class BrainstormReq(BaseModel):
    mood: Optional[str] = None
    time: Optional[str] = None
    cuisine: Optional[str] = None
    avoid: List[str] = []
    use_leftovers_first: bool = False
    notes: Optional[str] = None
    pantry: List[str] = []


class RecipeReq(BaseModel):
    pantry: List[str] = []
    cuisine: Optional[str] = None
    diet: Optional[str] = None
    cook_time: Optional[str] = None
    servings: int = 2
    budget_mode: bool = False
    leftover_mode: bool = False
    variation: Optional[str] = None  # healthier / spicier / indian / american / budget / faster / kid-friendly / high-protein
    idea_title: Optional[str] = None
    idea_desc: Optional[str] = None


class SubstitutionReq(BaseModel):
    missing_ingredient: str
    pantry: List[str] = []


class OnboardingReq(BaseModel):
    country_preference: Optional[str] = None
    unit_preference: Optional[str] = None
    diet_preference: Optional[str] = None
    skill_level: Optional[str] = None
    spice_level: Optional[str] = None
    onboarding_complete: Optional[bool] = None


class ScanReq(BaseModel):
    image_base64: str  # data URI or raw base64


# ---------- Auth helpers ----------
async def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """Bearer <session_token>. Also accept session_token as bare string."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization")
    token = authorization.replace("Bearer ", "").strip()
    session = await db.sessions.find_one({"session_token": token}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    # optional expiry check
    exp = session.get("expires_at")
    if exp and isinstance(exp, datetime):
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Session expired")
    user_doc = await db.users.find_one({"id": session["user_id"]}, {"_id": 0})
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user_doc)


async def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    await db.sessions.insert_one({
        "session_token": token,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
        "expires_at": datetime.now(timezone.utc) + timedelta(days=30),
    })
    return token


# ---------- Auth routes ----------
@api.post("/auth/guest")
async def auth_guest(device_id: Optional[str] = None):
    """Create or reuse anonymous user by device_id."""
    if device_id:
        existing = await db.users.find_one({"anonymous_device_id": device_id}, {"_id": 0})
        if existing:
            token = await create_session(existing["id"])
            return {"user": User(**existing).dict(), "session_token": token}
    uid = str(uuid.uuid4())
    user = User(id=uid, is_guest=True, name="Guest Cook")
    doc = user.dict()
    doc["anonymous_device_id"] = device_id or uid
    await db.users.insert_one(doc)
    token = await create_session(uid)
    return {"user": user.dict(), "session_token": token}


@api.post("/auth/session")
async def auth_session(x_session_id: str = Header(...)):
    """Exchange Emergent session_id for our session_token.
    Calls Emergent auth service to get user data (email, name, picture, session_token).
    """
    try:
        async with httpx.AsyncClient(timeout=15) as hc:
            resp = await hc.get(
                "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
                headers={"X-Session-ID": x_session_id},
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid Emergent session")
        data = resp.json()
        email = data.get("email")
        name = data.get("name")
        picture = data.get("picture")
        session_token = data.get("session_token")
        if not email:
            raise HTTPException(status_code=401, detail="No email in Emergent session")
        # find or create user
        existing = await db.users.find_one({"email": email}, {"_id": 0})
        if existing:
            user = User(**existing)
        else:
            user = User(id=str(uuid.uuid4()), email=email, name=name, picture=picture, is_guest=False)
            await db.users.insert_one(user.dict())
        # store the emergent session_token as our session token (7 days)
        await db.sessions.insert_one({
            "session_token": session_token,
            "user_id": user.id,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
        })
        return {"user": user.dict(), "session_token": session_token}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("auth_session error")
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/auth/me")
async def me(user: User = Depends(get_current_user)):
    return user.dict()


@api.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        await db.sessions.delete_one({"session_token": token})
    return {"ok": True}


@api.post("/auth/onboarding")
async def onboarding(body: OnboardingReq, user: User = Depends(get_current_user)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    if update:
        await db.users.update_one({"id": user.id}, {"$set": update})
    doc = await db.users.find_one({"id": user.id}, {"_id": 0})
    return User(**doc).dict()


# ---------- Pantry ----------
@api.get("/pantry")
async def list_pantry(user: User = Depends(get_current_user)):
    items = await db.pantry.find({"user_id": user.id}, {"_id": 0}).to_list(500)
    return items


@api.post("/pantry")
async def add_pantry(body: PantryCreate, user: User = Depends(get_current_user)):
    item = PantryItem(user_id=user.id, **body.dict(exclude_none=True))
    await db.pantry.insert_one(item.dict())
    return item.dict()


@api.post("/pantry/bulk")
async def bulk_pantry(items: List[PantryCreate], user: User = Depends(get_current_user)):
    docs = []
    for b in items:
        it = PantryItem(user_id=user.id, **b.dict(exclude_none=True))
        docs.append(it.dict())
    if docs:
        await db.pantry.insert_many(docs)
    # Strip mongo _id before returning to avoid ObjectId serialization errors
    for d in docs:
        d.pop("_id", None)
    return {"inserted": len(docs), "items": docs}


@api.put("/pantry/{item_id}")
async def update_pantry(item_id: str, body: PantryUpdate, user: User = Depends(get_current_user)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    if update:
        await db.pantry.update_one({"id": item_id, "user_id": user.id}, {"$set": update})
    doc = await db.pantry.find_one({"id": item_id, "user_id": user.id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Not found")
    return doc


@api.delete("/pantry/{item_id}")
async def delete_pantry(item_id: str, user: User = Depends(get_current_user)):
    await db.pantry.delete_one({"id": item_id, "user_id": user.id})
    return {"ok": True}


@api.get("/pantry/expiring")
async def expiring(user: User = Depends(get_current_user)):
    items = await db.pantry.find({"user_id": user.id}, {"_id": 0}).to_list(500)
    today = datetime.now(timezone.utc).date()
    result = []
    for it in items:
        ex = it.get("expiry_date")
        if not ex:
            continue
        try:
            d = datetime.fromisoformat(ex).date()
            days = (d - today).days
            if days <= 3:
                it["days_left"] = days
                result.append(it)
        except Exception:
            continue
    result.sort(key=lambda x: x.get("days_left", 99))
    return result


# ---------- Saved Recipes ----------
@api.get("/saved-recipes")
async def list_saved(user: User = Depends(get_current_user)):
    items = await db.saved_recipes.find({"user_id": user.id}, {"_id": 0}).to_list(500)
    return items


@api.post("/saved-recipes")
async def save_recipe(recipe: Dict[str, Any], user: User = Depends(get_current_user)):
    doc = SavedRecipe(user_id=user.id, recipe=recipe)
    await db.saved_recipes.insert_one(doc.dict())
    return doc.dict()


@api.delete("/saved-recipes/{recipe_id}")
async def delete_saved(recipe_id: str, user: User = Depends(get_current_user)):
    await db.saved_recipes.delete_one({"id": recipe_id, "user_id": user.id})
    return {"ok": True}


# ---------- Grocery ----------
@api.get("/grocery")
async def list_grocery(user: User = Depends(get_current_user)):
    return await db.grocery.find({"user_id": user.id}, {"_id": 0}).to_list(500)


@api.post("/grocery")
async def add_grocery(body: GroceryCreate, user: User = Depends(get_current_user)):
    it = GroceryItem(user_id=user.id, **body.dict(exclude_none=True))
    await db.grocery.insert_one(it.dict())
    return it.dict()


@api.post("/grocery/bulk")
async def bulk_grocery(items: List[GroceryCreate], user: User = Depends(get_current_user)):
    docs = [GroceryItem(user_id=user.id, **b.dict(exclude_none=True)).dict() for b in items]
    if docs:
        await db.grocery.insert_many(docs)
    for d in docs:
        d.pop("_id", None)
    return {"inserted": len(docs), "items": docs}


@api.put("/grocery/{item_id}")
async def update_grocery(item_id: str, body: GroceryUpdate, user: User = Depends(get_current_user)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    if update:
        await db.grocery.update_one({"id": item_id, "user_id": user.id}, {"$set": update})
    doc = await db.grocery.find_one({"id": item_id, "user_id": user.id}, {"_id": 0})
    if not doc:
        raise HTTPException(404, "Not found")
    return doc


@api.delete("/grocery/{item_id}")
async def delete_grocery(item_id: str, user: User = Depends(get_current_user)):
    await db.grocery.delete_one({"id": item_id, "user_id": user.id})
    return {"ok": True}


@api.post("/grocery/move-to-pantry")
async def move_to_pantry(user: User = Depends(get_current_user)):
    checked = await db.grocery.find({"user_id": user.id, "checked": True}, {"_id": 0}).to_list(500)
    if not checked:
        return {"moved": 0}
    pantry_docs = []
    for g in checked:
        it = PantryItem(user_id=user.id, name=g["name"], category=g.get("category", "Other"),
                        quantity=g.get("quantity", 1), unit=g.get("unit", "pieces"))
        pantry_docs.append(it.dict())
    if pantry_docs:
        await db.pantry.insert_many(pantry_docs)
    await db.grocery.delete_many({"user_id": user.id, "checked": True})
    return {"moved": len(pantry_docs)}


# ---------- AI Helpers ----------
def _extract_json(text: str) -> Any:
    """Try to extract JSON from LLM output."""
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # find first { or [
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("Could not parse JSON from AI response")


async def _llm_json(system: str, user_text: str, session_id: str, image_b64: Optional[str] = None) -> Any:
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system,
    ).with_model("gemini", "gemini-3-flash-preview")
    file_contents = []
    if image_b64:
        # strip data URI prefix
        if image_b64.startswith("data:"):
            image_b64 = image_b64.split(",", 1)[1]
        file_contents.append(ImageContent(image_base64=image_b64))
    msg = UserMessage(text=user_text, file_contents=file_contents if file_contents else None)
    resp = await chat.send_message(msg)
    if isinstance(resp, str):
        text = resp
    else:
        text = getattr(resp, "content", None) or str(resp)
    return _extract_json(text)


# ---------- AI Endpoints ----------
SCAN_SYSTEM = """You are Pantry Rescue's ingredient detection AI. You look at a photo of a fridge, pantry, groceries, vegetables, or leftovers and identify visible edible ingredients.

Return ONLY valid JSON in this exact schema (no prose, no markdown):
{
  "ingredients": [
    {"name": "Tomato", "category": "Vegetables", "confidence": 0.95, "uncertain": false, "quantity_hint": "3 pieces"}
  ],
  "note": "Optional short note if image is unclear"
}

Rules:
- Only include clearly edible ingredients you can actually see.
- If unsure, set uncertain=true and confidence < 0.6.
- Categories: Vegetables, Fruits, Dairy, Meat, Seafood, Grains, Spices, Sauces, Snacks, Frozen, Leftovers, Other.
- Capitalize ingredient names simply (e.g., "Onion", not "onions").
- If nothing is clearly visible, return {"ingredients": [], "note": "Could not clearly detect ingredients"}.
"""


@api.post("/scan-ingredients")
async def scan_ingredients(body: ScanReq, user: User = Depends(get_current_user)):
    try:
        result = await _llm_json(
            SCAN_SYSTEM,
            "Detect ingredients in this photo. Return JSON only.",
            session_id=f"scan-{user.id}-{uuid.uuid4()}",
            image_b64=body.image_base64,
        )
        return result
    except Exception as e:
        logger.exception("scan error")
        # fallback demo
        return {
            "ingredients": [
                {"name": "Tomato", "category": "Vegetables", "confidence": 0.9, "uncertain": False},
                {"name": "Onion", "category": "Vegetables", "confidence": 0.85, "uncertain": False},
            ],
            "note": f"Demo fallback ({str(e)[:80]})",
        }


RECIPE_SYSTEM = """You are Pantry Rescue's recipe AI. You create practical, safe, delicious recipes using ingredients the user already has.

Return ONLY valid JSON (no prose, no markdown) in this schema:
{
  "title": "Masala Egg Rice",
  "description": "...",
  "cuisine": "Indian",
  "prepTime": "5 min",
  "cookTime": "10 min",
  "totalTime": "15 min",
  "servings": 2,
  "difficulty": "Easy",
  "pantryMatchScore": 92,
  "rescueScore": 3,
  "ingredientsUsed": ["Rice","Eggs","Onion","Tomato"],
  "missingIngredients": ["Coriander"],
  "optionalIngredients": ["Lemon"],
  "substitutions": [{"insteadOf":"Coriander","use":"Spring onion or skip"}],
  "steps": ["...","..."],
  "safetyNotes": ["Cook eggs fully."],
  "storageTips": "Fridge, 1 day",
  "nutritionEstimate": "Approx. 420 kcal / serving",
  "tags": ["Budget","Quick","Indian","High Protein"]
}

Rules: keep steps concise, warn about meat/egg/seafood safety, mark nutrition as approximate, prefer using pantry ingredients first."""


@api.post("/generate-recipes")
async def generate_recipes(body: RecipeReq, user: User = Depends(get_current_user)):
    """Generate a single full recipe."""
    prompt = f"""Generate ONE recipe.
Pantry: {', '.join(body.pantry) or 'none'}
Cuisine: {body.cuisine or 'any'}
Diet: {body.diet or user.diet_preference}
Cook time preference: {body.cook_time or 'any'}
Servings: {body.servings}
Budget mode: {body.budget_mode}
Leftover mode: {body.leftover_mode}
Variation: {body.variation or 'none'}
{"Base on idea: " + body.idea_title + " - " + (body.idea_desc or '') if body.idea_title else ''}

Return JSON only."""
    try:
        recipe = await _llm_json(RECIPE_SYSTEM, prompt, session_id=f"recipe-{user.id}-{uuid.uuid4()}")
        return recipe
    except Exception as e:
        logger.exception("recipe error")
        raise HTTPException(500, f"Recipe AI error: {str(e)[:120]}")


BRAINSTORM_SYSTEM = """You are Pantry Rescue's brainstorm AI. Given mood, time, cuisine, pantry, and a list of past titles the user already saw, propose 5 CREATIVE and DIFFERENT recipe IDEAS (not full recipes).

Return ONLY JSON:
{
  "ideas": [
    {"title":"Charred Corn Elote Bowl","why":"Uses your corn and cheese","time":"15 min","difficulty":"Easy","pantryMatch":90,"tags":["Quick","Mexican"]}
  ]
}

STRICT RULES:
- Exactly 5 ideas.
- DO NOT repeat any title from the "avoid_titles" list. Do not just rename them — propose genuinely different dishes with different cooking techniques, cuisines, or main ingredients.
- Mix cuisines and techniques (roasted, one-pot, no-cook, stir-fry, baked, grilled, cold, soup, wrap, bowl, toast, curry, pasta, salad, taco, ramen, sushi-style, sheet-pan, etc.).
- Keep 'why' short (one sentence).
- Prefer using pantry ingredients first.
"""


@api.post("/brainstorm")
async def brainstorm(body: BrainstormReq, user: User = Depends(get_current_user)):
    # fetch last 30 idea/recipe titles the user has seen, plus saved recipe titles
    recent_history = await db.brainstorm_history.find(
        {"user_id": user.id}, {"_id": 0, "title": 1}
    ).sort("created_at", -1).to_list(30)
    saved = await db.saved_recipes.find(
        {"user_id": user.id}, {"_id": 0, "recipe.title": 1}
    ).sort("created_at", -1).to_list(30)
    avoid_titles = list({h["title"] for h in recent_history if h.get("title")} |
                        {s["recipe"]["title"] for s in saved if s.get("recipe", {}).get("title")})

    variety_seed = uuid.uuid4().hex[:8]
    prompt = f"""Mood: {body.mood or 'any'}
Time available: {body.time or 'any'}
Cuisine: {body.cuisine or 'any'}
Avoid: {', '.join(body.avoid) or 'nothing'}
Use leftovers first: {body.use_leftovers_first}
User note: {body.notes or 'none'}
Pantry: {', '.join(body.pantry) or 'none'}

avoid_titles (already seen — pick genuinely different dishes):
{json.dumps(avoid_titles) if avoid_titles else '[]'}

variety_seed: {variety_seed}
Return 5 fresh, different ideas as JSON."""
    try:
        result = await _llm_json(BRAINSTORM_SYSTEM, prompt, session_id=f"brain-{user.id}-{variety_seed}")
        # store returned titles so next call skips them
        ideas = result.get("ideas", []) if isinstance(result, dict) else []
        if ideas:
            await db.brainstorm_history.insert_many([
                {"user_id": user.id, "title": i.get("title", ""), "created_at": datetime.now(timezone.utc)}
                for i in ideas if i.get("title")
            ])
        return result
    except Exception as e:
        logger.exception("brainstorm error")
        raise HTTPException(500, f"Brainstorm error: {str(e)[:120]}")


@api.post("/surprise")
async def surprise(user: User = Depends(get_current_user)):
    """One tap: pick random pantry subset + generate a bold, unexpected recipe."""
    pantry = await db.pantry.find({"user_id": user.id}, {"_id": 0}).to_list(500)
    names = [p["name"] for p in pantry]
    # avoid recent
    recent = await db.brainstorm_history.find({"user_id": user.id}, {"_id": 0, "title": 1}).sort("created_at", -1).to_list(20)
    saved = await db.saved_recipes.find({"user_id": user.id}, {"_id": 0, "recipe.title": 1}).sort("created_at", -1).to_list(20)
    avoid = list({h["title"] for h in recent if h.get("title")} |
                 {s["recipe"]["title"] for s in saved if s.get("recipe", {}).get("title")})
    seed = uuid.uuid4().hex[:8]
    prompt = f"""Surprise the user with ONE bold, creative recipe using what they have.
Pantry: {', '.join(names) or 'basic staples'}
avoid_titles: {json.dumps(avoid) if avoid else '[]'}
variety_seed: {seed}
Pick a technique the user probably wouldn't guess (e.g., sheet-pan, cold noodles, quesadilla, shakshuka, congee, bibimbap-style, dosa-style, galette, frittata, ramen, pilaf, tagine, one-pot).
Return JSON only."""
    try:
        recipe = await _llm_json(RECIPE_SYSTEM, prompt, session_id=f"surprise-{user.id}-{seed}")
        if isinstance(recipe, dict) and recipe.get("title"):
            await db.brainstorm_history.insert_one({
                "user_id": user.id, "title": recipe["title"], "created_at": datetime.now(timezone.utc),
            })
        return recipe
    except Exception as e:
        logger.exception("surprise error")
        raise HTTPException(500, f"Surprise error: {str(e)[:120]}")


@api.post("/history/reset")
async def reset_history(user: User = Depends(get_current_user)):
    """User can ask for a totally fresh batch of ideas."""
    await db.brainstorm_history.delete_many({"user_id": user.id})
    return {"ok": True}


# ---------- Stats / streaks ----------
@api.get("/stats")
async def stats(user: User = Depends(get_current_user)):
    pantry = await db.pantry.find({"user_id": user.id}, {"_id": 0}).to_list(500)
    saved_count = await db.saved_recipes.count_documents({"user_id": user.id})
    cooked = await db.recipe_history.find({"user_id": user.id}, {"_id": 0}).to_list(500)
    # rescued items = pantry items marked leftover + cooked recipes count as proxies
    rescued = sum(1 for p in pantry if p.get("is_leftover"))
    return {
        "pantry_count": len(pantry),
        "saved_count": saved_count,
        "cooked_count": len(cooked),
        "rescued_count": rescued,
        "streak_days": min(len(cooked), 30),  # simple proxy
    }


@api.post("/history/cook")
async def log_cook(recipe: Dict[str, Any], user: User = Depends(get_current_user)):
    await db.recipe_history.insert_one({
        "id": str(uuid.uuid4()),
        "user_id": user.id,
        "recipe_title": recipe.get("title"),
        "cooked_at": datetime.now(timezone.utc),
    })
    return {"ok": True}


SUB_SYSTEM = """You suggest smart substitutions for a missing ingredient using what the user has.
Return ONLY JSON: {"substitutions":[{"use":"milk + butter","note":"Use 3/4 cup milk + 2 tbsp butter instead of cream"}]}
Give 2-3 practical options."""


@api.post("/substitutions")
async def substitutions(body: SubstitutionReq, user: User = Depends(get_current_user)):
    prompt = f"Missing: {body.missing_ingredient}\nPantry: {', '.join(body.pantry)}\nReturn JSON only."
    try:
        return await _llm_json(SUB_SYSTEM, prompt, session_id=f"sub-{user.id}-{uuid.uuid4()}")
    except Exception as e:
        logger.exception("sub error")
        raise HTTPException(500, f"Substitution error: {str(e)[:120]}")


MEALPLAN_SYSTEM = """You create a simple 7-day meal plan (breakfast/lunch/dinner) using pantry ingredients first.
Return ONLY JSON:
{
  "days": [
    {"day":"Monday","breakfast":"Poha","lunch":"Dal Rice","dinner":"Masala Eggs"}
  ],
  "grocery_needed": ["Coriander","Milk"]
}
Exactly 7 days."""


@api.post("/meal-plan")
async def meal_plan(body: RecipeReq, user: User = Depends(get_current_user)):
    prompt = f"Pantry: {', '.join(body.pantry)}\nDiet: {body.diet or user.diet_preference}\nCuisine: {body.cuisine or 'any'}\nBudget mode: {body.budget_mode}\nServings: {body.servings}\nReturn JSON."
    try:
        return await _llm_json(MEALPLAN_SYSTEM, prompt, session_id=f"plan-{user.id}-{uuid.uuid4()}")
    except Exception as e:
        logger.exception("plan error")
        raise HTTPException(500, f"Meal plan error: {str(e)[:120]}")


# ---------- Ad reward tracking ----------
@api.post("/ad-reward")
async def ad_reward(reward_type: str, user: User = Depends(get_current_user)):
    await db.ad_rewards.insert_one({
        "id": str(uuid.uuid4()),
        "user_id": user.id,
        "reward_type": reward_type,
        "created_at": datetime.now(timezone.utc),
    })
    return {"ok": True}


# ---------- Pro / Subscription (mocked purchase) ----------
class UpgradeReq(BaseModel):
    plan: str = "yearly"  # monthly | yearly | lifetime
    receipt: Optional[str] = None  # placeholder for real IAP receipt


@api.post("/pro/upgrade")
async def upgrade(body: UpgradeReq, user: User = Depends(get_current_user)):
    """Mocked upgrade — a real build would validate a Play/App Store receipt.
    Kept intentionally simple so the paywall UX is complete end-to-end."""
    await db.users.update_one(
        {"id": user.id},
        {"$set": {"is_pro": True, "pro_since": datetime.now(timezone.utc), "pro_plan": body.plan}},
    )
    doc = await db.users.find_one({"id": user.id}, {"_id": 0})
    return {"ok": True, "user": User(**doc).dict(), "plan": body.plan}


@api.post("/pro/cancel")
async def cancel_pro(user: User = Depends(get_current_user)):
    await db.users.update_one({"id": user.id}, {"$set": {"is_pro": False, "pro_since": None}})
    return {"ok": True}


# ---------- Chef Chat (premium multi-turn AI) ----------
CHEF_SYSTEM = """You are ChefAI — a warm, encouraging cooking coach for Pantry Rescue Pro members.

Style:
- Concise, practical, friendly.
- Give real numbers (temperatures, minutes, spoon sizes) not vague terms.
- When the user asks for a recipe, format it clearly with numbered steps.
- If they ask a technique or substitution question, answer in 1–3 short paragraphs — no fluff.
- Suggest a follow-up question at the end when relevant.
- Warn briefly about egg/meat/seafood safety when relevant.
- Never claim certainty about nutrition — say "approximately".
- Answer in plain text (not JSON)."""


class ChefMessageReq(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    include_pantry: bool = True


@api.post("/chef/message")
async def chef_message(body: ChefMessageReq, user: User = Depends(get_current_user)):
    if not user.is_pro:
        raise HTTPException(status_code=402, detail="Chef Chat is a Pantry Rescue Pro feature. Upgrade to unlock.")
    conv_id = body.conversation_id or str(uuid.uuid4())
    pantry_note = ""
    if body.include_pantry:
        pantry_items = await db.pantry.find({"user_id": user.id}, {"_id": 0, "name": 1}).to_list(200)
        if pantry_items:
            pantry_note = f"\n\n(User's pantry: {', '.join(p['name'] for p in pantry_items)}. Prefer these ingredients when relevant.)"
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"chef-{user.id}-{conv_id}",
            system_message=CHEF_SYSTEM,
        ).with_model("gemini", "gemini-3-flash-preview")
        resp = await chat.send_message(UserMessage(text=body.message + pantry_note))
        text = resp if isinstance(resp, str) else (getattr(resp, "content", None) or str(resp))
        # store both sides of the exchange
        now = datetime.now(timezone.utc)
        await db.chef_messages.insert_many([
            {"user_id": user.id, "conversation_id": conv_id, "role": "user", "text": body.message, "created_at": now},
            {"user_id": user.id, "conversation_id": conv_id, "role": "assistant", "text": text, "created_at": now},
        ])
        return {"conversation_id": conv_id, "reply": text}
    except Exception as e:
        logger.exception("chef error")
        raise HTTPException(500, f"Chef error: {str(e)[:120]}")


@api.get("/chef/history")
async def chef_history(user: User = Depends(get_current_user), conversation_id: Optional[str] = None):
    q: Dict[str, Any] = {"user_id": user.id}
    if conversation_id:
        q["conversation_id"] = conversation_id
    msgs = await db.chef_messages.find(q, {"_id": 0}).sort("created_at", 1).to_list(500)
    return msgs


@api.post("/chef/clear")
async def chef_clear(user: User = Depends(get_current_user)):
    await db.chef_messages.delete_many({"user_id": user.id})
    return {"ok": True}


# ---------- Recipe Collections (Pro) ----------
class CollectionCreate(BaseModel):
    name: str
    emoji: Optional[str] = "📚"


@api.get("/collections")
async def list_collections(user: User = Depends(get_current_user)):
    return await db.collections.find({"user_id": user.id}, {"_id": 0}).to_list(200)


@api.post("/collections")
async def create_collection(body: CollectionCreate, user: User = Depends(get_current_user)):
    if not user.is_pro:
        # free users get 1 collection; check count
        count = await db.collections.count_documents({"user_id": user.id})
        if count >= 1:
            raise HTTPException(402, "Free plan is limited to 1 collection. Upgrade to Pro for unlimited.")
    col = {
        "id": str(uuid.uuid4()),
        "user_id": user.id,
        "name": body.name,
        "emoji": body.emoji or "📚",
        "recipe_ids": [],
        "created_at": datetime.now(timezone.utc),
    }
    await db.collections.insert_one(col)
    col.pop("_id", None)
    return col


@api.post("/collections/{col_id}/add")
async def add_to_collection(col_id: str, recipe_id: str, user: User = Depends(get_current_user)):
    await db.collections.update_one(
        {"id": col_id, "user_id": user.id},
        {"$addToSet": {"recipe_ids": recipe_id}},
    )
    return {"ok": True}


@api.delete("/collections/{col_id}")
async def delete_collection(col_id: str, user: User = Depends(get_current_user)):
    await db.collections.delete_one({"id": col_id, "user_id": user.id})
    return {"ok": True}


# ---------- Nutrition weekly summary (Pro) ----------
@api.get("/nutrition/weekly")
async def nutrition_weekly(user: User = Depends(get_current_user)):
    if not user.is_pro:
        raise HTTPException(402, "Weekly nutrition tracking is a Pro feature.")
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    cooked = await db.recipe_history.find(
        {"user_id": user.id, "cooked_at": {"$gte": week_ago}}, {"_id": 0}
    ).to_list(200)
    return {
        "meals_cooked": len(cooked),
        "estimated_calories": len(cooked) * 500,  # rough proxy without per-recipe nutrition
        "avg_prep_min": 20,
        "recipes": [c.get("recipe_title") for c in cooked if c.get("recipe_title")],
    }


# ---------- Health ----------
@api.get("/")
async def root():
    return {"app": "Pantry Rescue", "status": "ok"}


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown():
    client.close()
