import pandas as pd


class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


class Employee(Person):
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id


class EmployeeManager:
    def __init__(self):
        self.employees = pd.DataFrame(
            columns=["name", "age", "employee_id"]
        )

    def add_employee(self, employee):
        new_row = pd.DataFrame([{
            "name": employee.name,
            "age": employee.age,
            "employee_id": employee.employee_id
        }])

        self.employees = pd.concat(
            [self.employees, new_row],
            ignore_index=True
        )

    def get_employee_by_id(self, employee_id):
        employee_data = self.employees.loc[
            self.employees["employee_id"] == employee_id
        ]

        if not employee_data.empty:
            return employee_data.iloc[0].to_dict()

        return None

    def get_all_employees(self):
        return self.employees.to_dict(orient="records")

    def group_employees_by_age(self):
        age_groups = {
            age: group.to_dict(orient="records")
            for age, group in self.employees.groupby("age")
        }

        return age_groups

    def delete_employee_by_id(self, employee_id):
        self.employees = self.employees.loc[
            self.employees["employee_id"] != employee_id
        ]


if __name__ == "__main__":
    manager = EmployeeManager()

    emp1 = Employee("Alice", 30, "E001")
    emp2 = Employee("Bob", 25, "E002")
    emp3 = Employee("Charlie", 30, "E003")

    manager.add_employee(emp1)
    manager.add_employee(emp2)
    manager.add_employee(emp3)

    print("All Employees:")
    print(manager.get_all_employees())

    print("\nEmployee By ID:")
    print(manager.get_employee_by_id("E002"))

    print("\nEmployees Grouped By Age:")
    print(manager.group_employees_by_age())