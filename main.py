from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import random
from sklearn.ensemble import RandomForestRegressor

# =========================================
# 1. LOAD & TRAIN MODEL
# =========================================

print("Loading nutrition dataset...")

df = pd.read_csv("nutrition.csv")

df = df.rename(columns={
    "Ages": "age",
    "Gender": "gender",
    "Height": "height",
    "Weight": "weight",
    "Activity Level": "activity",
    "Dietary Preference": "diet",
    "Disease": "disease",
    "Calories": "calories",
    "Protein": "protein",
    "Carbohydrates": "carbs",
    "Fat": "fat"
})

# Remove text columns
for col in [
    "Breakfast Suggestion",
    "Lunch Suggestion",
    "Dinner Suggestion",
    "Snack Suggestion"
]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

df["gender"] = df["gender"].map({"Male":1,"Female":0})

activity_map = {
    "Sedentary":0,
    "Lightly Active":1,
    "Moderately Active":2,
    "Very Active":3
}
df["activity"] = df["activity"].map(activity_map)

df = pd.get_dummies(df, columns=["disease","diet"])

target_cols = ["calories","protein","carbs","fat"]
feature_cols = [c for c in df.columns if c not in target_cols]

df = df.dropna().reset_index(drop=True)

X = df[feature_cols]
y = df[target_cols]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

print("Model trained successfully!")

# =========================================
# 2. WHO TDEE
# =========================================

def who_tdee(age, gender, height, weight, activity):

    if gender.lower() == "male":
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161

    activity_factor = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extremely Active": 1.9
    }

    tdee = bmr * activity_factor.get(activity, 1.2)

    protein_pct = random.uniform(0.12, 0.15)
    fat_pct = random.uniform(0.20, 0.30)
    carb_pct = 1 - protein_pct - fat_pct

    return {
        "Calories": round(tdee,1),
        "Protein": round((tdee*protein_pct)/4,1),
        "Carbs": round((tdee*carb_pct)/4,1),
        "Fat": round((tdee*fat_pct)/9,1)
    }

# =========================================
# 3. LOAD FOOD DATA
# =========================================

print("Loading food dataset...")

df_food = pd.read_csv("food.csv", sep=";", decimal=",", engine="python")

df_food = df_food.rename(columns={
    "STT":"stt",
    "Name":"name_vi",
    "Calories":"calories",
    "Chất đạm":"protein",
    "Chất béo":"fat",
    "Carbohydrate":"carbs"
})

df_food = df_food[["stt","name_vi","calories","protein","fat","carbs"]]

for col in ["calories","protein","fat","carbs"]:
    df_food[col] = pd.to_numeric(df_food[col], errors="coerce")

df_food = df_food.dropna().reset_index(drop=True)

print("Food items loaded:", len(df_food))

# =========================================
# 4. RECOMMEND LOGIC (NO DUPLICATE + WEIGHTED)
# =========================================

def recommend_meal(
    target,
    is_disease_user=False,
    used_stt=None,
    max_items=5
):
    selected = []
    remain = target.copy()

    # Remove used foods (no duplicate in whole day)
    if used_stt:
        foods = df_food[~df_food["stt"].isin(used_stt)].copy()
    else:
        foods = df_food.copy()

    def score(food):

        s = abs(food.calories - remain["Calories"]) / max(target["Calories"],1)

        if not is_disease_user:
            s += 1.5 * abs(food.carbs - remain["Carbs"]) / max(target["Carbs"],1)
            s += 0.6 * abs(food.fat - remain["Fat"]) / max(target["Fat"],1)
            s += 0.3 * abs(food.protein - remain["Protein"]) / max(target["Protein"],1)
        else:
            s += 1.2 * abs(food.carbs - remain["Carbs"]) / max(target["Carbs"],1)
            s += 1.2 * abs(food.fat - remain["Fat"]) / max(target["Fat"],1)
            s += 2.5 * abs(food.protein - remain["Protein"]) / max(target["Protein"],1)

        return s

    while (
        remain["Calories"] > target["Calories"] * 0.1
        and len(selected) < max_items
        and not foods.empty
    ):

        foods["score"] = foods.apply(score, axis=1)

        top_k = foods.sort_values("score").head(8)

        weights = 1 / (top_k["score"] + 1e-6)
        weights /= weights.sum()

        best = top_k.sample(1, weights=weights).iloc[0]

        selected.append({
            "stt": int(best.stt),
            "name": best.name_vi,
            "calories": float(best.calories)
        })

        if used_stt is not None:
            used_stt.add(int(best.stt))

        remain["Calories"] -= best.calories
        remain["Protein"] -= best.protein
        remain["Carbs"] -= best.carbs
        remain["Fat"] -= best.fat

        foods = foods.drop(best.name)

        for k in remain:
            remain[k] = max(remain[k], 0)

    return selected


def daily_plan(nutrition, is_disease_user=False, eaten_cal=None):

    ratios = {"Breakfast":0.3, "Lunch":0.4, "Dinner":0.3}
    meals = {}
    used_stt = set()

    for meal, r in ratios.items():

        target_cal = nutrition["Calories"] * r
        eaten = eaten_cal.get(meal, 0) if eaten_cal else 0

        # Nếu đã ăn đủ
        if eaten >= target_cal:
            meals[meal] = []
            continue

        # Phần còn thiếu
        remain_ratio = max((target_cal - eaten) / target_cal, 0)

        meals[meal] = recommend_meal(
            target={
                "Calories": nutrition["Calories"] * r * remain_ratio,
                "Protein": nutrition["Protein"] * r * remain_ratio,
                "Carbs": nutrition["Carbs"] * r * remain_ratio,
                "Fat": nutrition["Fat"] * r * remain_ratio
            },
            is_disease_user=is_disease_user,
            used_stt=used_stt
        )

    return meals

# =========================================
# 5. FASTAPI
# =========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserRequest(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activity: str
    disease: str | None = None
    breakfast_cal: float = 0
    lunch_cal: float = 0
    dinner_cal: float = 0

def convert_gender(gender):
    if gender.lower() in ["nam","male"]:
        return "male"
    return "female"

def convert_activity(activity):
    mapping = {
        "ít vận động": "Sedentary",
        "vận động nhẹ": "Lightly Active",
        "vận động vừa": "Moderately Active",
        "vận động nhiều": "Very Active"
    }
    return mapping.get(activity.lower(), "Sedentary")

@app.post("/recommend")
def recommend(user: UserRequest):

    gender_en = convert_gender(user.gender)
    activity_en = convert_activity(user.activity)

    if not user.disease or user.disease == "None":

        nutrition = who_tdee(
            user.age,
            gender_en,
            user.height,
            user.weight,
            activity_en
        )
        is_disease = False

    else:
        X_user = pd.DataFrame([{
            "age": user.age,
            "gender": 1 if gender_en == "male" else 0,
            "height": user.height,
            "weight": user.weight,
            "activity": 1
        }]).reindex(columns=X.columns, fill_value=0)

        pred = model.predict(X_user)[0]

        nutrition = {
            "Calories": round(pred[0],1),
            "Protein": round(pred[1],1),
            "Carbs": round(pred[2],1),
            "Fat": round(pred[3],1)
        }
        is_disease = True

    # 🔥 Calo đã ăn từ app
    eaten_cal = {
        "Breakfast": user.breakfast_cal,
        "Lunch": user.lunch_cal,
        "Dinner": user.dinner_cal
    }

    meals = daily_plan(nutrition, is_disease, eaten_cal)

    return {
        "nutrition": nutrition,
        "menu": meals
    }
@app.post("/tdee")
def calculate_tdee_api(user: UserRequest):

    gender_en = convert_gender(user.gender)
    activity_en = convert_activity(user.activity)

    if not user.disease or user.disease == "None":
        nutrition = who_tdee(
            user.age,
            gender_en,
            user.height,
            user.weight,
            activity_en
        )
    else:
        X_user = pd.DataFrame([{
            "age": user.age,
            "gender": 1 if gender_en == "male" else 0,
            "height": user.height,
            "weight": user.weight,
            "activity": 1
        }]).reindex(columns=X.columns, fill_value=0)

        pred = model.predict(X_user)[0]

        nutrition = {
            "Calories": round(pred[0],1),
            "Protein": round(pred[1],1),
            "Carbs": round(pred[2],1),
            "Fat": round(pred[3],1)
        }

    return nutrition
