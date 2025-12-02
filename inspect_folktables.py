import folktables
import inspect

print("Available prediction tasks in folktables:")
for name, obj in inspect.getmembers(folktables):
    if isinstance(obj, folktables.BasicProblem):
        print(f"- {name}")
