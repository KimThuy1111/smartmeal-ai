import pandas as pd
import numpy as np

from prettytable import PrettyTable

import main

from main import (
    UserRequest,
    initialize_models,
    who_tdee,
    daily_plan
)

# Khởi tạo mô hình

initialize_models()

X_food = main.X_food
food_stt = main.food_stt


# Tính tổng dinh dưỡng của thực đơn được gợi ý

def calculate_menu_total(menu):

    total = {
        "Calories": 0,
        "Protein": 0,
        "carb": 0,
        "Fat": 0
    }

    for meal_name, foods in menu.items():

        for item in foods:

            stt = item["stt"]

            idx = np.where(food_stt == stt)[0]

            if len(idx) == 0:
                continue

            food = X_food[idx[0]]

            total["Calories"] += float(food[0])
            total["Protein"] += float(food[1])
            total["carb"] += float(food[2])
            total["Fat"] += float(food[3])

    return total


# Chuyển mục tiêu cân nặng từ cột Disease

def get_goal(disease):

    if pd.isna(disease):
        return "duy trì"

    disease = str(disease).lower()

    if "weight gain" in disease:
        return "tăng cân"

    if "weight loss" in disease:
        return "giảm cân"

    return "duy trì"


# Chuyển giới tính từ dataset sang hệ thống

def map_gender(gender):

    gender = str(gender).strip().lower()

    if gender == "male":
        return "nam"

    return "nữ"


# Chuyển mức vận động từ dataset sang hệ thống

def map_activity(activity):

    activity = str(activity).strip().lower()

    mapping = {
        "sedentary": "ít vận động",
        "lightly active": "vận động nhẹ",
        "moderately active": "vận động vừa",
        "very active": "vận động nhiều",
        "extremely active": "vận động cực nhiều"
    }

    return mapping.get(
        activity,
        "vận động vừa"
    )


# Đọc dữ liệu người dùng từ file csv

def load_users_from_csv(csv_path):

    df = pd.read_csv(csv_path)

    users = []

    for _, row in df.iterrows():

        try:

            user = UserRequest(
                age=int(row["Ages"]),
                gender=map_gender(row["Gender"]),
                height=float(row["Height"]),
                weight=float(row["Weight"]),
                activity=map_activity(
                    row["Activity Level"]
                ),
                goal=get_goal(
                    row["Disease"]
                ),
                breakfast_cal=0,
                lunch_cal=0,
                dinner_cal=0,
                recent_foods=[],
                excluded_foods=[]
            )

            users.append(user)

        except Exception:
            continue

    return users


# Đánh giá hệ thống trên toàn bộ người dùng

def evaluate_system(users):

    target_calories = []
    target_protein = []
    target_carbs = []
    target_fat = []

    meal_calories = []
    meal_protein = []
    meal_carbs = []
    meal_fat = []

    success_count = 0

    for user in users:

        try:

            target = who_tdee(user)

            menu = daily_plan(user)

            meal_total = calculate_menu_total(menu)

            target_calories.append(
                target["Calories"]
            )

            target_protein.append(
                target["Protein"]
            )

            target_carbs.append(
                target["carb"]
            )

            target_fat.append(
                target["Fat"]
            )

            meal_calories.append(
                meal_total["Calories"]
            )

            meal_protein.append(
                meal_total["Protein"]
            )

            meal_carbs.append(
                meal_total["carb"]
            )

            meal_fat.append(
                meal_total["Fat"]
            )

            success_count += 1

        except Exception:
            continue

    avg_target_cal = np.mean(
        target_calories
    )

    avg_target_pro = np.mean(
        target_protein
    )

    avg_target_carb = np.mean(
        target_carbs
    )

    avg_target_fat = np.mean(
        target_fat
    )

    avg_meal_cal = np.mean(
        meal_calories
    )

    avg_meal_pro = np.mean(
        meal_protein
    )

    avg_meal_carb = np.mean(
        meal_carbs
    )

    avg_meal_fat = np.mean(
        meal_fat
    )

    error_cal = (
        abs(avg_target_cal - avg_meal_cal)
        / avg_target_cal
        * 100
    )

    error_pro = (
        abs(avg_target_pro - avg_meal_pro)
        / avg_target_pro
        * 100
    )

    error_carb = (
        abs(avg_target_carb - avg_meal_carb)
        / avg_target_carb
        * 100
    )

    error_fat = (
        abs(avg_target_fat - avg_meal_fat)
        / avg_target_fat
        * 100
    )

    average_error = (
        error_cal +
        error_pro +
        error_carb +
        error_fat
    ) / 4

    accuracy = 100 - average_error

    table = PrettyTable()

    table.field_names = [
        "Nutrient",
        "Recommended",
        "MealTotal",
        "Error %"
    ]

    table.add_row([
        "Calories",
        round(avg_target_cal, 1),
        round(avg_meal_cal, 1),
        round(error_cal, 2)
    ])

    table.add_row([
        "Protein",
        round(avg_target_pro, 1),
        round(avg_meal_pro, 1),
        round(error_pro, 2)
    ])

    table.add_row([
        "Carbs",
        round(avg_target_carb, 1),
        round(avg_meal_carb, 1),
        round(error_carb, 2)
    ])

    table.add_row([
        "Fat",
        round(avg_target_fat, 1),
        round(avg_meal_fat, 1),
        round(error_fat, 2)
    ])

    print()
    print(table)

    print(
        "\nAccuracy:",
        round(accuracy, 2),
        "%"
    )

    print(
        "Total Users:",
        success_count
    )


# Hàm chạy chương trình

def main_test():

    users = load_users_from_csv(
        "nutrition.csv"
    )

    print(
        "Loaded users:",
        len(users)
    )

    evaluate_system(users)


if __name__ == "__main__":
    main_test()