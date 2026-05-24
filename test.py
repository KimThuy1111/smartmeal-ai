import numpy as np
import pandas as pd
import main

from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, r2_score

from main import (
    db,
    UserRequest,
    daily_plan,
    who_tdee,
    initialize_models
)

initialize_models()

X_food = main.X_food
food_stt = main.food_stt


# =========================
# TÍNH TỔNG DINH DƯỠNG MENU
# =========================

def calculate_menu_total(menu):

    total = {
        "Calories": 0,
        "Protein": 0,
        "carb": 0,
        "Fat": 0
    }

    for _, foods in menu.items():

        for item in foods:

            stt = item["stt"]

            idx = np.where(food_stt == stt)[0]

            if len(idx) == 0:
                continue

            food = X_food[idx[0]]

            total["Calories"] += food[0]
            total["Protein"] += food[1]
            total["carb"] += food[2]
            total["Fat"] += food[3]

    return total


# =========================
# LOAD USER FIREBASE
# =========================

def load_firebase_users():

    users = []

    users_ref = db.collection("users").stream()

    for doc in users_ref:

        data = doc.to_dict()

        try:

            if (
                "age" not in data or
                "gender" not in data or
                "height" not in data or
                "weight" not in data or
                "activity" not in data
            ):
                continue

            user = UserRequest(
                age=int(data["age"]),
                gender=data["gender"],
                height=float(data["height"]),
                weight=float(data["weight"]),
                activity=data["activity"],
                goal=data.get("goal", "duy trì"),

                breakfast_cal=0,
                lunch_cal=0,
                dinner_cal=0,

                recent_foods=data.get("recent_foods") or [],
                excluded_foods=data.get("excluded_foods") or []
            )

            users.append(user)

        except Exception as e:

            print("Lỗi user:", doc.id, str(e))

    return users


# =========================
# LOAD NUTRITION CSV
# =========================

def load_nutrition_users():

    df = pd.read_csv("nutrition.csv")

    users = []

    for _, row in df.iterrows():

        try:

            # =====================
            # AGE
            # =====================

            age = int(
                row.get("Age", 25)
            )

            if age < 15 or age > 70:
                continue

            # =====================
            # GENDER
            # =====================

            gender_raw = str(
                row.get("Gender", "Male")
            ).lower()

            gender = (
                "nam"
                if "male" in gender_raw
                else "nữ"
            )

            # =====================
            # HEIGHT
            # =====================

            height = float(
                row.get("Height", 165)
            )

            if height < 140 or height > 210:
                continue

            # =====================
            # WEIGHT
            # =====================

            weight = float(
                row.get("Weight", 60)
            )
            # =====================
            # BMI FILTER
            # =====================

            bmi = weight / ((height / 100) ** 2)

            # bỏ user quá gầy hoặc quá béo

            if bmi < 16 or bmi > 35:
                continue

            if weight < 35 or weight > 180:
                continue
            
            # =====================
            # ACTIVITY
            # =====================

            activity_raw = str(
                row.get("ActivityLevel", "")
            ).lower()

            if (
                "sedentary" in activity_raw
            ):
                activity = "ít vận động"

            elif (
                "light" in activity_raw
            ):
                activity = "vận động nhẹ"

            elif (
                "moderate" in activity_raw
            ):
                activity = "vận động vừa"

            elif (
                "very active" in activity_raw or
                "high" in activity_raw
            ):
                activity = "vận động nhiều"

            else:
                activity = "vận động vừa"

            # =====================
            # GOAL
            # =====================

            disease = str(
                row.get("Disease", "")
            ).lower()

            goal = "duy trì"

            if (
                "obesity" in disease or
                "weight loss" in disease or
                "diabetes" in disease or
                "fat" in disease
            ):
                goal = "giảm cân"

            elif (
                "underweight" in disease or
                "gain" in disease or
                "thin" in disease
            ):
                goal = "tăng cân"

            # =====================
            # USER
            # =====================

            user = UserRequest(

                age=age,
                gender=gender,

                height=height,
                weight=weight,

                activity=activity,
                goal=goal,

                breakfast_cal=0,
                lunch_cal=0,
                dinner_cal=0,

                recent_foods=[],
                excluded_foods=[]
            )

            users.append(user)

        except:
            continue

    return users


# =========================
# EVALUATE
# =========================

def evaluate_users(users, title):

    recommended_calories = []
    recommended_protein = []
    recommended_carb = []
    recommended_fat = []

    meal_calories = []
    meal_protein = []
    meal_carb = []
    meal_fat = []

    tested_users = 0

    for user in users:

        try:

            target = who_tdee(user)
            

            menu = daily_plan(user)

            total = calculate_menu_total(menu)

            recommended_calories.append(target["Calories"])
            recommended_protein.append(target["Protein"])
            recommended_carb.append(target["carb"])
            recommended_fat.append(target["Fat"])

            meal_calories.append(total["Calories"])
            meal_protein.append(total["Protein"])
            meal_carb.append(total["carb"])
            meal_fat.append(total["Fat"])

            tested_users += 1

        except Exception as e:

            print("Lỗi:", str(e))

    if tested_users == 0:
        print("\nKhông có dữ liệu test")
        return

    mae_cal = np.mean(np.abs((np.array(recommended_calories) - np.array(meal_calories)) / np.array(recommended_calories))) * 100
    mae_pro = np.mean(np.abs((np.array(recommended_protein) - np.array(meal_protein)) / np.array(recommended_protein))) * 100
    mae_carb = np.mean(np.abs((np.array(recommended_carb) - np.array(meal_carb)) / np.array(recommended_carb))) * 100
    mae_fat = np.mean(np.abs((np.array(recommended_fat) - np.array(meal_fat)) / np.array(recommended_fat))) * 100

    rmse_cal = np.sqrt(mean_squared_error(recommended_calories, meal_calories)) / np.mean(recommended_calories) * 100
    rmse_pro = np.sqrt(mean_squared_error(recommended_protein, meal_protein)) / np.mean(recommended_protein) * 100
    rmse_carb = np.sqrt(mean_squared_error(recommended_carb, meal_carb)) / np.mean(recommended_carb) * 100
    rmse_fat = np.sqrt(mean_squared_error(recommended_fat, meal_fat)) / np.mean(recommended_fat) * 100

    r2_cal = r2_score(recommended_calories, meal_calories)
    r2_pro = r2_score(recommended_protein, meal_protein)
    r2_carb = r2_score(recommended_carb, meal_carb)
    r2_fat = r2_score(recommended_fat, meal_fat)

    avg_error = (mae_cal + mae_pro + mae_carb + mae_fat) / 4

    accuracy = max(0, 100 - avg_error)

    table = PrettyTable()

    table.field_names = [
        "Metric",
        "Calories",
        "Protein",
        "Carbs",
        "Fat"
    ]

    table.add_row([
        "MAE %",
        round(mae_cal, 2),
        round(mae_pro, 2),
        round(mae_carb, 2),
        round(mae_fat, 2)
    ])

    table.add_row([
        "RMSE %",
        round(rmse_cal, 2),
        round(rmse_pro, 2),
        round(rmse_carb, 2),
        round(rmse_fat, 2)
    ])

    table.add_row([
        "R2",
        round(r2_cal, 3),
        round(r2_pro, 3),
        round(r2_carb, 3),
        round(r2_fat, 3)
    ])

    print(f"\n===== {title} =====\n")

    print(table)

    print("\nAccuracy:", round(accuracy, 2), "%")

    print("Total Tested Users:", tested_users)


# =========================
# TEST FIREBASE
# =========================

def test_firebase_users():

    users = load_firebase_users()

    evaluate_users(
        users,
        "FIREBASE USERS RESULT"
    )


# =========================
# TEST NUTRITION CSV
# =========================

def test_nutrition_users():

    users = load_nutrition_users()

    evaluate_users(
        users,
        "NUTRITION CSV RESULT"
    )


# =========================
# RUN TEST
# =========================

if __name__ == "__main__":

    test_firebase_users()

    test_nutrition_users()