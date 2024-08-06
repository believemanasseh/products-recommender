import pandas as pd
from faker import Faker
from faker.providers import DynamicProvider


def generate_sythentic_data():
    products_df = pd.read_csv("./phones_data.csv")
    brand_names = products_df["brand_name"].unique().tolist()
    brands_provider = DynamicProvider(
        provider_name="brands",
        elements=brand_names,
    )
    faker = Faker()
    faker.add_provider(brands_provider)

    # User data
    user_data = []
    for user_id in range(1, 1225):
        user_data.append(
            {
                "user_id": user_id,
                "age": faker.random_int(min=18, max=65),
                "gender": faker.random_element(elements=("Male", "Female")),
                "location": faker.city(),
                "preferences": ",".join(
                    faker.random_elements(
                        elements=brand_names,
                        unique=True,
                    )[:3]
                ),
            }
        )
    users_df = pd.DataFrame(user_data)

    # Purchase history data
    purchase_history_data = []
    for _ in range(1224 * 5):
        user_id = faker.random_element(elements=users_df["user_id"].unique().tolist())
        product_id = faker.random_element(
            elements=products_df[products_df.columns[0]].unique().tolist()
        )
        purchase_date = faker.date_between(start_date="-5y")
        quantity = faker.random_int(min=1, max=5)
        purchase_history_data.append(
            {
                "user_id": user_id,
                "product_id": product_id,
                "purchase_date": purchase_date,
                "quantity": quantity,
            }
        )
    purchase_history_df = pd.DataFrame(purchase_history_data)

    users_df.to_csv("users_data.csv", index=False)
    purchase_history_df.to_csv("purchase_history_data.csv", index=False)

    return users_df, products_df, purchase_history_df


if __name__ == "__main__":
    generate_sythentic_data()
