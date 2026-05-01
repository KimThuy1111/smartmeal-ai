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

    if not name:   # 🔥 FIX
        return False

    for r in recent_foods:
        rname = food_name_map.get(r, "").lower()

        if not rname:
            continue

        # Tách từ để so sánh độ tương đồng cơ bản (ví dụ: "Bún bò" và "Bún riêu")
        words = [w for w in rname.split() if len(w) > 2] # Chỉ lấy các từ có nghĩa
        if len(words) == 0:
            continue

        if any(word in name for word in words):
            return True

    return False

# Hàm gợi ý món ăn cho từng bữa dựa trên phần còn thiếu của nhu cầu dinh dưỡng và các món đã dùng trong ngày (nếu có)
# =============================
# RECOMMEND MEAL (FIXED)
# =============================

def recommend_meal(target, is_disease_user=False, used_stt=None, recent_foods=None, excluded_foods=None, max_items=2):

    remain = np.array([
        target["Calories"],
        target["Protein"],
        target["carb"],
        target["Fat"]
    ])

    selected = []
    used = set(used_stt or [])
    
    # Loại bỏ các món đã được yêu cầu loại trừ
    if excluded_foods:
        used.update(excluded_foods)
    total_cal = 0

    for _ in range(max_items):

        mask = ~np.isin(food_stt, list(used))

        foods = food_array[mask]
        stts = food_stt[mask]

        if len(foods) == 0:
            break

        # CHẶN MÓN VƯỢT CALO: Món đơn lẻ không được vượt quá phần calo còn lại của bữa
        valid_mask = foods[:,0] <= remain[0]
        foods = foods[valid_mask]
        stts = stts[valid_mask]

        if len(foods) == 0:
            break

        diff = np.abs(foods - remain)

        # =================================
        # BONUS THEO SỞ THÍCH USER
        # =================================
        # Phần này xử lý gợi ý món ăn cùng loại hoặc tương tự lịch sử ăn uống

        bonus = np.zeros(len(stts))

        if recent_foods:
            recent_categories = [food_category_map.get(r) for r in recent_foods if r in food_category_map]

            for i, stt in enumerate(stts):
                # Ưu tiên món cùng Category với các món người dùng hay ăn
                if food_category_map.get(stt) in recent_categories:
                    bonus[i] -= 0.4 # Giảm score để món này dễ được chọn hơn

            for i, stt in enumerate(stts):
                # Ưu tiên món có tên tương tự (ví dụ: cùng là Bún, Phở...)
                if stt in recent_foods:
                    bonus[i] -= 0.25
                elif is_similar_food(stt, recent_foods):
                    bonus[i] -= 0.2

        # Thêm noise để thực đơn thay đổi mỗi lần nhấn Refresh
        noise = np.random.uniform(0, 0.05, len(stts))

        if not is_disease_user:
            score = (
                diff[:,0] / max(target["Calories"],1) +
                0.5 * diff[:,2] / max(target["carb"],1) +
                0.3 * diff[:,3] / max(target["Fat"],1) +
                0.2 * diff[:,1] / max(target["Protein"],1)
            ) + bonus + noise
        else:
            score = (
                diff[:,0] / max(target["Calories"],1) +
                1.2 * diff[:,2] / max(target["carb"],1) +
                1.0 * diff[:,3] / max(target["Fat"],1) +
                1.5 * diff[:,1] / max(target["Protein"],1)
            ) + bonus + noise

        idx = np.argmin(score)

        best = foods[idx]
        stt = int(stts[idx])

        # KIỂM TRA NGHIÊM NGẶT: Tổng calo của bữa không được vượt quá target
        if total_cal + best[0] > target["Calories"]:
            break

        selected.append({
            "stt": stt,
            "calories": float(best[0])
        })

        used.add(stt)
        total_cal += best[0]

        remain -= best
        remain = np.maximum(remain, 0)

        if total_cal >= target["Calories"] * 0.9:
            break

    return selected


# Lập kế hoạch ăn uống hàng ngày dựa trên nhu cầu dinh dưỡng và các món đã ăn trong ngày (nếu có)
def daily_plan(nutrition, is_disease_user=False, eaten_cal=None, recent_foods=None, excluded_foods=None):

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
            recent_foods=recent_foods,
            excluded_foods=excluded_foods
        )

        for item in meals[meal]:
            used_stt.add(item["stt"])

    # =========================
    # 3. FIX BỮA TRƯA
    # =========================
    # Ràng buộc: Bữa trưa phải có ít nhất 1 món nước hoặc tinh bột
    lunch = meals.get("Lunch", [])
    target_lunch_cal = nutrition["Calories"] * 0.4
    current_lunch_cal = sum(i["calories"] for i in lunch)

    has_main = any(
        food_category_map.get(i.get("stt")) in ["Tinh bột", "Món nước"]
        for i in lunch
    )

    if not has_main:
        # Tìm món thay thế phù hợp nếu chưa có món chính
        for stt in food_stt:
            if stt in used_stt: continue
            cat = food_category_map.get(stt)

            if cat in ["Tinh bột", "Món nước"]:
                cal = float(food_array[list(food_stt).index(stt)][0])

                # Đảm bảo thêm món chính vào không làm vượt quá calo bữa trưa
                if current_lunch_cal + cal <= target_lunch_cal:
                    lunch.append({
                        "stt": int(stt),
                        "calories": cal
                    })
                    current_lunch_cal += cal
                break

    # Nếu sau khi fix vẫn còn thiếu nhiều calo, bổ sung món giải khát nhẹ
    if current_lunch_cal < target_lunch_cal * 0.9:
        remain = target_lunch_cal - current_lunch_cal
        for stt in food_stt:
            if stt in used_stt: continue
            cat = food_category_map.get(stt)

            if cat == "Giải khát":
                cal = float(food_array[list(food_stt).index(stt)][0])
                if cal <= remain:
                    lunch.append({
                        "stt": int(stt),
                        "calories": cal
                    })
                    break
    meals["Lunch"] = lunch
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

# Định nghĩa Model cho request token
class TokenRequest(BaseModel):
    firebase_uid: str
    email: str

# Thêm Route /token để sửa lỗi 404 từ Frontend
@app.post("/token")
async def get_token(req: TokenRequest):
    return {"access_token": "mock_access_token_for_internal_use"}

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
    # 🔥 NEW: món đã gợi ý trong ngày
    excluded_foods: list[int] | None = None

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
    print("==== RECOMMEND API CALLED ====")
    print("Input:", user.dict())


    nutrition, is_disease = calculate_nutrition(user)

    eaten_cal = {
        "Breakfast": user.breakfast_cal,
        "Lunch": user.lunch_cal,
        "Dinner": user.dinner_cal
    }

    meals = daily_plan(nutrition, is_disease, eaten_cal, user.recent_foods, user.excluded_foods)
    print("Output nutrition:", nutrition)
    print("Output meals:", meals)
    print("=============================")

    return {
        "nutrition": nutrition,
        "menu": meals
    }
# Endpoint để chỉ trả về nhu cầu dinh dưỡng (TDEE) 
@app.post("/tdee")
def calculate_tdee_api(user: UserRequest):

    nutrition, _ = calculate_nutrition(user)
    print("==== TDEE API CALLED ====")
    print("Input:", user.dict())
    print("Output:", nutrition)
    print("=========================")

    return nutrition