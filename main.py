from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import time
import hashlib
import json
import os
import base64

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore


# Khởi tạo FastAPI 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối Firebase
# cred = credentials.Certificate("firebase_key.json")
db = None

try:

    # LOCAL
    if os.path.exists("firebase_key.json"):

        cred = credentials.Certificate(
            "firebase_key.json"
        )

        firebase_admin.initialize_app(cred)

        db = firestore.client()

        print("Firebase LOCAL connected")

    # RAILWAY
    else:

        firebase_base64 = os.getenv(
            "FIREBASE_KEY_BASE64"
        )
        print("BASE64 =", firebase_base64)

        if firebase_base64:

            firebase_json = base64.b64decode(
                firebase_base64
            ).decode("utf-8")
            print(firebase_json[:100])

            cred_dict = json.loads(firebase_json)

            cred = credentials.Certificate(
                cred_dict
            )

            firebase_admin.initialize_app(
                cred
            )

            db = firestore.client()

            print("Firebase RAILWAY connected")

        else:
            print("Firebase ENV missing")

except Exception as e:
    print("Firebase init error:", e)


# Cache layer
food_cache = {}
group_score_cache = {}
training_cache = {}

CACHE_TTL = 300

X_food = None
food_stt = None
food_category = None

model_lr = None
scaler = None

knn = None
knn_scores_cache = None

initialized = False

def initialize_models():
    global X_food
    global food_stt
    global food_category
    global model_lr
    global scaler
    global knn
    global knn_scores_cache
    global initialized

    if initialized:
        return

    print("Loading food data...")

    X_food, food_stt, food_category = load_food_data()

    model_lr, scaler = train_model()

    knn, knn_scores_cache = train_knn(X_food)

    initialized = True

# Huấn luyện mô hình KNN để tìm các món ăn tương tự
def train_knn(X_food):

    # Khởi tạo mô hình với 5 hàng xóm gần nhất
    knn = NearestNeighbors(n_neighbors=5)

    # Huấn luyện mô hình trên dữ liệu món ăn
    knn.fit(X_food)

    # Tính khoảng cách giữa các món ăn
    distances, _ = knn.kneighbors(X_food)

    # Tính điểm tương đồng
    similarity_scores = 1 / (1 + distances.mean(axis=1))

    return knn, similarity_scores

# Request model
class UserRequest(BaseModel):
    age: int
    gender: str
    height: float
    weight: float
    activity: str
    goal: str = "maintain"

    # Lượng calories đã tiêu thụ trong ngày (từng bữa)
    breakfast_cal: float = 0
    lunch_cal: float = 0
    dinner_cal: float = 0

    # Danh sách các món cần loại trừ
    recent_foods: list[int] | None = None
    excluded_foods: list[int] | None = None


# Tạo cache key cho nhóm người dùng có đặc điểm tương tự
def make_group_key(user):
    # Nhóm cân nặng (5kg một nhóm)
    weight_group = int(user.weight // 5) * 5
    # Nhóm chiều cao (5cm một nhóm)
    height_group = int(user.height // 5) * 5

    payload = {
        "gender": user.gender,
        "goal": user.goal,
        "activity": user.activity,
        "weight_group": weight_group,
        "height_group": height_group
    }

    # Tạo hash từ payload để làm cache key
    raw = json.dumps(
        payload,
        sort_keys=True
    )

    return hashlib.md5(
        raw.encode()
    ).hexdigest()


# Kiểm tra tính hợp lệ của cache
def is_cache_valid(item):
    return time.time() - item["time"] < CACHE_TTL


# Load dữ liệu món ăn từ Firestore vào memory. Lấy calories, protein, carb, fat từ từng món
def load_food_data():
    if db is None:
        raise Exception("Firestore database not initialized")
    docs = db.collection("food").stream()
    X = []
    stt_list = []
    category_list = []

    for doc in docs:
        data = doc.to_dict()

        # Cache từng món ăn để tránh query Firestore lại
        food_cache[doc.id] = data

        X.append([
            float(data.get("calories", 0)),
            float(data.get("protein", 0)),
            float(data.get("carb", 0)),
            float(data.get("fat", 0))
        ])

        stt_list.append(int(data.get("stt", 0)))
        category_list.append(data.get("categoryId", ""))

    return np.array(X), np.array(stt_list), np.array(category_list)

# X_food, food_stt, food_category = load_food_data()

# Load dữ liệu huấn luyện từ FirestoreFirestore. Lấy dữ liệu từ suggested_menus (thực đơn gợi ý) và food_diary (nhật ký ăn)
def load_training_data():
    # Kiểm tra training cache
    if "dataset" in training_cache:
        cached = training_cache["dataset"]
        if is_cache_valid(cached):
            return cached["X"], cached["y"]

    X = []  # Mảng đặc trưng
    y = []  # Mảng nhãn (1=thích, 0=không thích)

    # Duyệt qua tất cả thực đơn gợi ý
    docs = db.collection("suggested_menus").stream()

    for doc in docs:
        data = doc.to_dict()
        liked = data.get("liked")

        # Bỏ qua nếu chưa được đánh giá
        if liked is None:
            continue

        # Tạo nhãn (1 thích, 0 không thích)
        label = 1 if liked else 0

        # Duyệt qua từng bữa trong thực đơn
        for meal in data.get("menu", {}):
            # Duyệt qua từng món ăn trong bữa
            for food_id in data["menu"][meal]:
                # Dùng food cache thay vì query Firestore
                if food_id in food_cache:
                    f = food_cache[food_id]
                else:
                    food_doc = db.collection("food").document(food_id).get()

                    if not food_doc.exists:
                        continue

                    f = food_doc.to_dict()
                    food_cache[food_id] = f

                X.append([
                    f.get("calories", 0),
                    f.get("protein", 0),
                    f.get("carb", 0),
                    f.get("fat", 0)
                ])
                y.append(label)

    # Duyệt qua food_diary (những món đã ăn)
    diary_docs = db.collection("food_diary").stream()

    for doc in diary_docs:
        data = doc.to_dict()
        food_id = data.get("foodId")

        if food_id in food_cache:
            f = food_cache[food_id]
        else:
            food_doc = db.collection("food").document(food_id).get()

            if not food_doc.exists:
                continue

            f = food_doc.to_dict()
            food_cache[food_id] = f

        X.append([
            f.get("calories", 0),
            f.get("protein", 0),
            f.get("carb", 0),
            f.get("fat", 0)
        ])

        y.append(1)  

    X_np = np.array(X)
    y_np = np.array(y)

    # Save training cache
    training_cache["dataset"] = {
        "X": X_np,
        "y": y_np,
        "time": time.time()
    }

    return X_np, y_np


# Huấn luyện mô hình Logistic Regression để dự đoán món ăn được thích hay không
def train_model():
    # Load dữ liệu huấn luyện
    X_train, y_train = load_training_data()

    # Kiểm tra nếu không đủ dữ liệu huấn luyện
    if len(X_train) == 0 or len(set(y_train)) < 2:
        print("Không đủ data train")
        return None, None

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Huấn luyện mô hình Logistic Regression
    model = LogisticRegression()
    model.fit(X_scaled, y_train)

    print("Train xong:", len(X_train), "samples")

    return model, scaler


# model_lr, scaler = train_model()


# # Huấn luyện mô hình KNN để tìm những món ăn tương tự. Tính toán khoảng cách giữa các món ăn dựa trên thành phần dinh dưỡng
# knn = NearestNeighbors(n_neighbors=5)
# knn.fit(X_food)

# # Tính điểm tương tự dựa trên khoảng cách trung bình đến 5 hàng xóm gần nhất
# distances, _ = knn.kneighbors(X_food)
# knn_scores_cache = 1 / (
#     1 + distances.mean(axis=1)
# )


# Hàm tính điểm ML cho món ăn dựa trên mô hình Logistic Regression. Dự đoán xác suất người dùng sẽ thích món ăn này
def lr_score(food):
    if model_lr is None:
        return 0

    return model_lr.predict_proba(scaler.transform([food]))[0][1]



# Tính điểm cho tất cả các món ăn theo nhóm người dùng. 
def calculate_group_scores(user):
    # Tạo key nhóm dựa trên đặc điểm người dùng
    group_key = make_group_key(user)

    # Kiểm tra nếu đã có cache nhóm thì dùng lại
    if group_key in group_score_cache:
        cached = group_score_cache[group_key]
        if is_cache_valid(cached):
            return cached["scores"]

    scores = []

    # Duyệt qua tất cả các món ăn
    for i, food in enumerate(X_food):
        stt = int(food_stt[i])

        # Tính điểm dinh dưỡng (40% calories, 30% protein, 20% carb, 10% fat)
        nutrition_score = (
            (food[0] / 500) * 0.4 +
            (food[1] / 50) * 0.3 +
            (food[2] / 100) * 0.2 +
            (food[3] / 30) * 0.1
        )

        ml_score = lr_score(food)

        similarity_score = (knn_scores_cache[i])

        # Tổng hợp điểm
        total_score = (nutrition_score + ml_score + similarity_score
        )

        scores.append({
            "stt": stt,
            "food": food,
            "score": total_score
        })

    # Sắp xếp giảm dần theo score

    scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Lưu cache nhóm để tái sử dụng
    group_score_cache[group_key] = {
        "scores": scores,
        "time": time.time()
    }

    return scores


# Hàm tính TDEE và nhu cầu dinh dưỡng hàng ngày dựa trên thông tin người dùng
@app.post("/tdee")
def who_tdee(user: UserRequest):
    # Tính BMR dựa trên giới tính
    if user.gender.lower() == "nam":
        bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age + 5
    else:
        bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age - 161

    # Hệ số hoạt động 
    activity_map = {
        "ít vận động": 1.2,
        "vận động nhẹ": 1.375,
        "vận động vừa": 1.55,
        "vận động nhiều": 1.725,
        "vận động cực nhiều": 1.9
    }

    # Tính TDEE = BMR * hệ số hoạt động
    tdee = bmr * activity_map.get(user.activity.lower(), 1.2)

    # Điều chỉnh theo mục tiêu cân nặng
    goal = user.goal.lower()

    if goal == "giảm cân":
        tdee -= 500
        protein_ratio = 0.15
        carb_ratio = 0.55
        fat_ratio = 0.30

    elif goal == "tăng cân":
        tdee += 500
        protein_ratio = 0.10
        carb_ratio = 0.75
        fat_ratio = 0.15

    else:
        # duy trì cân nặng
        protein_ratio = 0.125
        carb_ratio = 0.65
        fat_ratio = 0.225

    return {
        "Calories": tdee,
        "Protein": tdee * protein_ratio / 4,  # 1g protein = 4 kcal
        "carb": tdee * carb_ratio / 4,  # 1g carb = 4 kcal
        "Fat": tdee * fat_ratio / 9  # 1g fat = 9 kcal
    }

# Hàm kiểm tra xem món ăn có phù hợp để gợi ý hay không dựa trên lượng dinh dưỡng còn thiếu 
def is_valid_food(food, remain):
    return (
        food[0] > 0 and
        food[1] > 0 and
        food[0] <= remain[0] * 1.2 and
        food[1] <= remain[1] * 1.5 and
        food[3] <= remain[3] * 1.5
    )

# Hàm tính điểm dinh dưỡng dựa trên sự khác biệt giữa thành phần món ăn.
def calculate_nutrition_score(food, remain):

    diff = np.abs(remain - food)

    cal_diff = diff[0] / max(remain[0], 1)
    protein_diff = diff[1] / max(remain[1], 1)
    carb_diff = diff[2] / max(remain[2], 1)
    fat_diff = diff[3] / max(remain[3], 1)

    # calories là ưu tiên chính

    return (
        cal_diff * 3.5 +
        protein_diff * 1.5 +
        carb_diff * 1.2 +
        fat_diff * 1.0
    )

# Tính điểm thưởng nếu món ăn thuộc loại yêu thích và điểm phạt nếu đã ăn gần đây
def calculate_bonus_penalty(
        stt,
        recent_foods,
        favorite_categories
):
    category_bonus = 0
    recent_penalty = 0

    idx = np.where(food_stt == stt)[0]

    if len(idx) > 0:
        category = food_category[idx[0]]

        if category in favorite_categories:
            category_bonus = 0.25

    if recent_foods and stt in recent_foods:
        recent_penalty = 0.15

    return category_bonus, recent_penalty

def calculate_final_score(
        nutrition_score,
        recent_penalty,
        category_bonus,
        ml_score
):

    # nutrition là chính
    # ML chỉ ưu tiên nhẹ

    return (
        nutrition_score +
        recent_penalty * 0.2 -
        category_bonus * 0.15 -
        ml_score * 0.05
    )

# Hàm gợi ý món ăn cho từng bữa 
def recommend_meal(target, scored_foods, recent_foods=None, excluded_foods=None, used=None):
    recent_foods = recent_foods or []
    excluded_foods = excluded_foods or []
    remain = np.array([target["Calories"], target["Protein"], target["carb"], target["Fat"]])
    used = set(used or [])
    
    #Lưu các món đã được chọn để tránh gợi ý lại
    if excluded_foods:
        used.update(excluded_foods)
    selected = []
    favorite_categories = []
    
    if recent_foods is None:
        recent_foods = []

    # Xác định các loại món ăn yêu thích dựa trên những món đã ăn gần đây
    if recent_foods:
        for food_id in recent_foods:
            idx = np.where(food_stt == food_id)[0]

            if len(idx) > 0:
                favorite_categories.append(food_category[idx[0]])
    
    # Chọn tối đa 4 món cho 1 bữa
    for _ in range(4):
        best_food = None
        best_score = float("inf")
        for item in scored_foods:
            stt = item["stt"]
            food = item["food"]

            if stt in used:
                continue

            if not is_valid_food(food, remain):
                continue
            # bỏ món lệch calories quá nhiều

            cal_ratio = food[0] / max(remain[0], 1)

            if cal_ratio < 0.25 or cal_ratio > 1.1:
                continue
            nutrition_score = calculate_nutrition_score(food, remain)
            category_bonus, recent_penalty = (calculate_bonus_penalty(stt, recent_foods, favorite_categories))
            score = calculate_final_score(nutrition_score, recent_penalty, category_bonus, item["score"])

            if score < best_score:
                best_score = score
                best_food = item

        if best_food is None:
            break
        stt = best_food["stt"]
        food = best_food["food"]
        selected.append({"stt": stt, "calories": float(food[0])})
        used.add(stt)
        remain -= food
        remain = np.maximum(remain, 0)
        if remain[0] <= 80:
            break
        
    return selected


# Hàm gợi ý thực đơn hàng ngày (sáng, trưa, tối). Tính toán nhu cầu dinh dưỡng cho từng bữa dựa trên TDEE
def daily_plan(user):
    # Tính nhu cầu dinh dưỡng hàng ngày
    nutrition = who_tdee(user)

    # Lượng calories đã tiêu thụ trong ngày theo từng bữa
    eaten = {"Breakfast": user.breakfast_cal, "Lunch": user.lunch_cal, "Dinner": user.dinner_cal}

    # Tỷ lệ calo phân bổ cho từng bữa. Breakfast 30%, Lunch 40%, Dinner 30%
    ratios = {"Breakfast": 0.3, "Lunch": 0.4, "Dinner": 0.3}
    meals = {}
    used = set()

    # Lấy danh sách các món ăn theo nhóm người dùng
    scored_foods = calculate_group_scores(user)

    for meal, r in ratios.items():
        # Tính target calories cho bữa ăn này
        target = nutrition["Calories"] * r

        # Nếu đã ăn đủ rồi thì bỏ qua bữa này
        if eaten[meal] >= target:
            meals[meal] = []
            continue

        # Tính tỷ lệ dinh dưỡng còn lại cần cung cấp cho bữa này
        ratio_remain = (target - eaten[meal]) / target

        # Lấy các món gợi ý cho từng bữa
        meals[meal] = recommend_meal(
            target={
                "Calories": nutrition["Calories"] * r * ratio_remain,
                "Protein": nutrition["Protein"] * r * ratio_remain,
                "carb": nutrition["carb"] * r * ratio_remain,
                "Fat": nutrition["Fat"] * r * ratio_remain
            },
            scored_foods=scored_foods,
            recent_foods=user.recent_foods,
            excluded_foods=user.excluded_foods,
            used=used
        )

        # Cập nhật danh sách những món đã gợi ý để không trùng 
        for f in meals[meal]:
            used.add(f["stt"])

    # Trả về thực đơn gợi ý cho 3 bữa
    return meals


# Nhận thông tin người dùng từ Flutter và trả về thực đơn gợi ý
@app.post("/recommend")
async def recommend(user: UserRequest):

    try:
        if not initialized:
            initialize_models()

        result = {
            "menu": daily_plan(user)
        }

        return result

    except Exception as e:
        return {
            "error": str(e)
        }
@app.get("/test-firebase")
def test_firebase():

    docs = db.collection("food").limit(1).stream()

    result = []

    for doc in docs:
        result.append(doc.to_dict())

    return result