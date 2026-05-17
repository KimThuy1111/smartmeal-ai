import numpy as np

from prettytable import PrettyTable

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from main import (
    db,
    UserRequest,
    daily_plan,
    who_tdee,
    X_food,
    food_stt
)


# Hàm tính tổng nutrition menu

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

            idx = np.where(
                food_stt == stt
            )[0]

            if len(idx) == 0:
                continue

            food = X_food[idx[0]]

            total["Calories"] += food[0]
            total["Protein"] += food[1]
            total["carb"] += food[2]
            total["Fat"] += food[3]

    return total


# Hàm test toàn bộ user

def test_all_users():

    users_ref = db.collection(
        "users"
    ).stream()

    recommended_calories = []
    recommended_protein = []
    recommended_carb = []
    recommended_fat = []

    meal_calories = []
    meal_protein = []
    meal_carb = []
    meal_fat = []

    tested_users = 0

    for doc in users_ref:

        data = doc.to_dict()

        try:

            # Skip user thiếu data

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

                breakfast_cal=0,
                lunch_cal=0,
                dinner_cal=0,

                recent_foods=[],
                excluded_foods=[]
            )

            # Nutrition target

            target = who_tdee(user)

            # Generate menu

            menu = daily_plan(user)

            # Nutrition menu

            total = calculate_menu_total(
                menu
            )

            # Save target

            recommended_calories.append(
                target["Calories"]
            )

            recommended_protein.append(
                target["Protein"]
            )

            recommended_carb.append(
                target["carb"]
            )

            recommended_fat.append(
                target["Fat"]
            )

            # Save menu

            meal_calories.append(
                total["Calories"]
            )

            meal_protein.append(
                total["Protein"]
            )

            meal_carb.append(
                total["carb"]
            )

            meal_fat.append(
                total["Fat"]
            )

            tested_users += 1

        except Exception as e:

            print(
                "Lỗi user:",
                doc.id,
                str(e)
            )

    # MAE %

    mae_cal = np.mean(
        np.abs(
            (
                np.array(recommended_calories) -
                np.array(meal_calories)
            ) / np.array(recommended_calories)
        )
    ) * 100

    mae_pro = np.mean(
        np.abs(
            (
                np.array(recommended_protein) -
                np.array(meal_protein)
            ) / np.array(recommended_protein)
        )
    ) * 100

    mae_carb = np.mean(
        np.abs(
            (
                np.array(recommended_carb) -
                np.array(meal_carb)
            ) / np.array(recommended_carb)
        )
    ) * 100

    mae_fat = np.mean(
        np.abs(
            (
                np.array(recommended_fat) -
                np.array(meal_fat)
            ) / np.array(recommended_fat)
        )
    ) * 100

    # RMSE %

    rmse_cal = np.sqrt(
        mean_squared_error(
            recommended_calories,
            meal_calories
        )
    ) / np.mean(
        recommended_calories
    ) * 100

    rmse_pro = np.sqrt(
        mean_squared_error(
            recommended_protein,
            meal_protein
        )
    ) / np.mean(
        recommended_protein
    ) * 100

    rmse_carb = np.sqrt(
        mean_squared_error(
            recommended_carb,
            meal_carb
        )
    ) / np.mean(
        recommended_carb
    ) * 100

    rmse_fat = np.sqrt(
        mean_squared_error(
            recommended_fat,
            meal_fat
        )
    ) / np.mean(
        recommended_fat
    ) * 100

    # R2

    r2_cal = r2_score(
        recommended_calories,
        meal_calories
    )

    r2_pro = r2_score(
        recommended_protein,
        meal_protein
    )

    r2_carb = r2_score(
        recommended_carb,
        meal_carb
    )

    r2_fat = r2_score(
        recommended_fat,
        meal_fat
    )

    # Accuracy

    avg_error = (
        mae_cal +
        mae_pro +
        mae_carb +
        mae_fat
    ) / 4

    accuracy = max(
        0,
        100 - avg_error
    )

    # Pretty table

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

    print("\n===== Evaluation Result =====\n")

    print(table)

    print(
        "\nAccuracy:",
        round(accuracy, 2),
        "%"
    )

    print(
        "Total Tested Users:",
        tested_users
    )


# Run test

if __name__ == "__main__":

    test_all_users()