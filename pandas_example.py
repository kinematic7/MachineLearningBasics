import pandas as pd

class Item:
    def __init__(self, name, color, price):
        self.name = name
        self.color = color
        self.price = price


class Shirt(Item):
    def __init__(self, name, color, price, size):
        super().__init__(name, color, price)
        self.size = size


class Pants(Item):
    def __init__(self, name, color, price, waist):
        super().__init__(name, color, price)
        self.waist = waist


class ShoppingCart:
    def __init__(self):
        self.items = pd.DataFrame(columns=["Item", "Quantity"])

    def add_item(self, item, quantity):
        new_row = pd.DataFrame([{
            "Item": item,
            "Quantity": quantity
        }])

        self.items = pd.concat([self.items, new_row], ignore_index=True)

    def calculate_total(self):
        total = 0

        for index, row in self.items.iterrows():
            item = row["Item"]
            quantity = row["Quantity"]

            total += item.price * quantity

        return total


if __name__ == "__main__":

    shirt1 = Shirt("T-Shirt", "Blue", 19.99, "M")
    shirt2 = Shirt("Dress Shirt", "White", 39.99, "L")
    shirt3 = Shirt("Polo Shirt", "Red", 29.99, "S")

    pants1 = Pants("Jeans", "Blue", 49.99, 32)
    pants2 = Pants("Chinos", "Khaki", 39.99, 34)
    pants3 = Pants("Shorts", "Black", 29.99, 30)

    cart = ShoppingCart()

    cart.add_item(shirt1, 2)
    cart.add_item(pants1, 1)
    cart.add_item(shirt2, 1)
    cart.add_item(pants2, 3)

    # Display cleaner output
    display_df = cart.items.copy()
    display_df["Item"] = display_df["Item"].apply(lambda x: x.name)

    print(display_df)

    print(f"\nTotal Sales: ${cart.calculate_total():.2f}")

    result_total_quantity = (
        display_df.groupby("Item")["Quantity"]
        .sum()
        .reset_index()
    )

    print("\nTotal Quantity of Each Item:")
    print(result_total_quantity)

    result_sales_greater_than_one = (
        display_df[display_df["Quantity"] > 1]
    )

    print("\nItems with Quantity Greater than 1:")
    print(result_sales_greater_than_one)
