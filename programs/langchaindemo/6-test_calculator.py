from langchain_core.tools import tool


@tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    Perform arithmetic calculations.

    operation can be:
    add
    subtract
    multiply
    divide
    """

    if operation == "add":
        return a + b

    elif operation == "subtract":
        return a - b

    elif operation == "multiply":
        return a * b

    elif operation == "divide":

        if b == 0:
            return "Cannot divide by zero"

        return a / b

    else:
        return "Unknown operation"


result = calculator.invoke({
    "a": 25,
    "b": 10,
    "operation": "multiply"
})

print(result)