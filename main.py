from unittest import result

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

# 1.Load and tiền xử lý nutrition dataset

print("Loading nutrition dataset...")

df = pd.read_csv("nutrition.csv")
# Đổi tên cột 
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
    "Carbohydrates": "carb",
    "Fat": "fat"
})

# Xóa các cột không cần thiết
for col in [
    "Breakfast Suggestion",
    "Lunch Suggestion",
    "Dinner Suggestion",
    "Snack Suggestion"
]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
# Mã hóa dữ liệu (encoding) 
df["gender"] = df["gender"].map({"Male":1,"Female":0})

activity_map = {
    "Sedentary":0,
    "Lightly Active":1,
    "Moderately Active":2,
    "Very Active":3
}
df["activity"] = df["activity"].map(activity_map)

# One-hot encoding cho các cột disease và diet
df = pd.get_dummies(df, columns=["disease","diet"])
# Chia cột target và feature
target_cols = ["calories","protein","carb","fat"]
feature_cols = [c for c in df.columns if c not in target_cols]

df = df.dropna().reset_index(drop=True)

X = df[feature_cols]
y = df[target_cols]

# Huấn luyện mô hình Random Forest Regressor để dự đoán nhu cầu dinh dưỡng cho người có bệnh lý
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Model trained successfully!")

# Tính TDEE theo công thức của WHO nếu không có bệnh lý

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
     # TDEE = BMR * Activity Factor
    tdee = bmr * activity_factor.get(activity, 1.2)

    protein_pct = random.uniform(0.12, 0.15)
    fat_pct = random.uniform(0.15, 0.30)
    carb_pct = 1 - protein_pct - fat_pct

    return {
        "Calories": round(tdee,1),
        "Protein": round((tdee*protein_pct)/4,1),
        "carb": round((tdee*carb_pct)/4,1),
        "Fat": round((tdee*fat_pct)/9,1)
    }

# Load and tiền xử lý food dataset

print("Loading food dataset...")

df_food = pd.read_csv("food.csv", sep=";", decimal=",", engine="python")

df_food = df_food.rename(columns={
    "STT":"stt",
    "Name":"name_vi",
    "Calories":"calories",
    "Chất đạm":"protein",
    "Chất béo":"fat",
    "Carbohydrate":"carb",
    "Category":"category"
})

df_food = df_food[["stt","name_vi","calories","protein","fat","carb","category"]]

food_category_map = dict(zip(df_food["stt"], df_food["category"]))

for col in ["calories","protein","fat","carb"]:
    df_food[col] = pd.to_numeric(df_food[col], errors="coerce")

df_food = df_food.dropna().reset_index(drop=True)

print("Food items loaded:", len(df_food))
food_array = df_food[["calories","protein","carb","fat"]].values
food_stt = df_food["stt"].values
food_names = df_food["name_vi"].values

# =============================
# TẠO MAP MÓN TƯƠNG TỰ
# =============================

food_name_map = dict(zip(food_stt, food_names))

def is_similar_food(stt, recent_foods):

    if not recent_foods:
        return False

    name = food_name_map.get(stt, "").lower()

    for r in recent_foods:

        rname = food_name_map.get(r, "").lower()

        # ví dụ: bún bò ~ bún riêu
        if rname.split()[0] in name:
            return True

    return False

# Hàm gợi ý món ăn cho từng bữa dựa trên phần còn thiếu của nhu cầu dinh dưỡng và các món đã dùng trong ngày (nếu có)
def recommend_meal(target, is_disease_user=False, used_stt=None, recent_foods=None, max_items=5):

    remain = np.array([
        target["Calories"],
        target["Protein"],
        target["carb"],
        target["Fat"]
    ])

    selected = []
    used = set(used_stt or [])

    for _ in range(max_items):

        mask = ~np.isin(food_stt, list(used))

        foods = food_array[mask]
        stts = food_stt[mask]

        if len(foods) == 0:
            break

        diff = np.abs(foods - remain)

        # =================================
        # BONUS THEO SỞ THÍCH USER
        # =================================

        bonus = np.zeros(len(stts))

        # 🎯 BONUS theo nhóm
        if recent_foods:
            recent_categories = [food_category_map.get(r) for r in recent_foods]

            for i, stt in enumerate(stts):
                cat = food_category_map.get(stt)

                if cat in recent_categories:
                    bonus[i] -= 0.25   # 👈 ưu tiên cùng nhóm

        if recent_foods:

            for i, stt in enumerate(stts):

                if stt in recent_foods:
                    bonus[i] -= 0.2   # món đã ăn

                elif is_similar_food(stt, recent_foods):
                    bonus[i] -= 0.15  # món tương tự

        if not is_disease_user:
            score = (
                diff[:,0] / max(target["Calories"],1) +
                0.5 * diff[:,2] / max(target["carb"],1) +
                0.3 * diff[:,3] / max(target["Fat"],1) +
                0.2 * diff[:,1] / max(target["Protein"],1)
            ) + bonus
        else:
            score = (
                diff[:,0] / max(target["Calories"],1) +
                1.2 * diff[:,2] / max(target["carb"],1) +
                1.0 * diff[:,3] / max(target["Fat"],1) +
                1.5 * diff[:,1] / max(target["Protein"],1)
            ) + bonus

        idx = np.argmin(score)

        best = foods[idx]
        stt = int(stts[idx])

        selected.append({
            "stt": stt,
            "calories": float(best[0])
        })

        used.add(stt)

        remain -= best
        remain = np.maximum(remain, 0)

        if remain[0] < target["Calories"] * 0.1:
            break

    return selected

# Lập kế hoạch ăn uống hàng ngày dựa trên nhu cầu dinh dưỡng và các món đã ăn trong ngày (nếu có)
def daily_plan(nutrition, is_disease_user=False, eaten_cal=None, recent_foods=None):

    ratios = {"Breakfast":0.3, "Lunch":0.4, "Dinner":0.3}
    meals = {}
    used_stt = set()

    # =========================
    # 1. CHECK CALO
    # =========================
    total_eaten = 0
    if eaten_cal:
        total_eaten = (
            eaten_cal.get("Breakfast",0) +
            eaten_cal.get("Lunch",0) +
            eaten_cal.get("Dinner",0)
        )

    if total_eaten >= nutrition["Calories"]:
        return {
            "Breakfast": [],
            "Lunch": [],
            "Dinner": []
        }

    # =========================
    # 2. BUILD MENU
    # =========================
    for meal, r in ratios.items():

        target_cal = nutrition["Calories"] * r
        eaten = eaten_cal.get(meal, 0) if eaten_cal else 0

        if eaten >= target_cal:
            meals[meal] = []
            continue

        remain_ratio = max((target_cal - eaten) / target_cal, 0)

        meals[meal] = recommend_meal(
            target={
                "Calories": nutrition["Calories"] * r * remain_ratio,
                "Protein": nutrition["Protein"] * r * remain_ratio,
                "carb": nutrition["carb"] * r * remain_ratio,
                "Fat": nutrition["Fat"] * r * remain_ratio
            },
            is_disease_user=is_disease_user,
            used_stt=used_stt,
            recent_foods=recent_foods
        )

        for item in meals[meal]:
            used_stt.add(item["stt"])

    # =========================
    # 3. FIX BỮA TRƯA
    # =========================
    lunch = meals.get("Lunch", [])

    has_required = False
    for item in lunch:
        cat = food_category_map.get(item["stt"])
        if cat in ["Tinh bột", "Món nước"]:
            has_required = True
            break

    if not has_required:
        for stt in food_stt:
            cat = food_category_map.get(stt)
            if cat in ["Tinh bột", "Món nước"]:
                meals["Lunch"].append({
                    "stt": int(stt),
                    "calories": float(food_array[list(food_stt).index(stt)][0])
                })
                break

    # =========================
    # 4. FIX SÁNG ≠ TỐI
    # =========================
    breakfast = meals.get("Breakfast", [])
    dinner = meals.get("Dinner", [])

    breakfast_cats = [food_category_map.get(i["stt"]) for i in breakfast]

    for d in dinner:
        cat = food_category_map.get(d["stt"])

        if cat in breakfast_cats and cat != "Khác":

            for stt in food_stt:
                new_cat = food_category_map.get(stt)

                if new_cat != cat:
                    d["stt"] = int(stt)
                    d["calories"] = float(food_array[list(food_stt).index(stt)][0])
                    break

    return meals

# Khởi tạo FastAPI và cấu hình CORS

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
    recent_foods: list[int] | None = None

# Hàm chuyển đổi giới tính 
def convert_gender(gender):
    if gender.lower() in ["nam","male"]:
        return "male"
    return "female"

# Hàm chuyển đổi mức độ vận động
def convert_activity(activity):
    mapping = {
        "ít vận động": "Sedentary",
        "vận động nhẹ": "Lightly Active",
        "vận động vừa": "Moderately Active",
        "vận động nhiều": "Very Active"
    }
    return mapping.get(activity.lower(), "Sedentary")

# Hàm tính nhu cầu dinh dưỡng (TDEE) dựa trên thông tin người dùng 
def calculate_nutrition(user: UserRequest):

    gender_en = convert_gender(user.gender)
    activity_en = convert_activity(user.activity)

    # Nếu không có bệnh dùng WHO
    if not user.disease or user.disease == "None":

        nutrition = who_tdee(
            user.age,
            gender_en,
            user.height,
            user.weight,
            activity_en
        )

        is_disease = False

    # Nếu có bệnh dùng mô hình dự đoán
    else:
        activity_encoded = activity_map.get(activity_en,0)

        X_user = pd.DataFrame([{
            "age": user.age,
            "gender": 1 if gender_en == "male" else 0,
            "height": user.height,
            "weight": user.weight,
            "activity": activity_encoded
        }]).reindex(columns=X.columns, fill_value=0)

        pred = model.predict(X_user)[0]

        nutrition = {
            "Calories": round(pred[0],1),
            "Protein": round(pred[1],1),
            "carb": round(pred[2],1),
            "Fat": round(pred[3],1)
        }

        is_disease = True

    return nutrition, is_disease

# Endpoint để nhận thông tin người dùng và trả về nhu cầu dinh dưỡng cùng gợi ý món ăn hàng ngày
@app.post("/recommend")
async def recommend(user: UserRequest):

    nutrition, is_disease = calculate_nutrition(user)

    eaten_cal = {
        "Breakfast": user.breakfast_cal,
        "Lunch": user.lunch_cal,
        "Dinner": user.dinner_cal
    }

    meals = daily_plan(nutrition, is_disease, eaten_cal, user.recent_foods)

    return {
        "nutrition": nutrition,
        "menu": meals
    }
# Endpoint để chỉ trả về nhu cầu dinh dưỡng (TDEE) 
@app.post("/tdee")
def calculate_tdee_api(user: UserRequest):

    nutrition, _ = calculate_nutrition(user)

    return nutrition
