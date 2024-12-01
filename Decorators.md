- Modifies the behavior of a function
- Executes each time a function is called. 
- function is wrapped inside the wrapper function
- The flow of operation is:
  ```mermaid
	graph TD
    Start([Start]) --> DefineDecorator[Define Decorator Function]
    DefineDecorator --> TakeFunctionInput[Take Function as Input]
    TakeFunctionInput --> DefineWrapper[Define Wrapper Function]
    DefineWrapper --> AddLogic[Add Additional Logic in Wrapper]
    AddLogic --> CallOriginalFunction[Call Original Function in Wrapper]
    CallOriginalFunction --> ReturnResult[Return the Result of the Function]
    ReturnResult --> ReturnWrapper[Return the Wrapper Function]
    ReturnWrapper --> ApplyDecorator[Apply Decorator to Target Function]
    ApplyDecorator --> CallDecoratedFunction[Call the Decorated Function]
    CallDecoratedFunction --> ExecuteWrapper[Execute Wrapper Logic]
    ExecuteWrapper --> ExecuteOriginal[Execute Original Function Logic]
    ExecuteOriginal --> End([End])

  
	```
# Problem 1: Timing Function ExecutionProblem:
### Write a decorator that measures the time a function takes to execute.
```
import time
def decorator(func):
	def wrapper(*args ,**kwargs):
		start=time.time()
		# we are assuming outpout will be a single parameter 
		result=func(*args, **kwargs)
		end=time.time()
		print(f"{func.__name__} ran in {end-start} time")
		return result
	return wrapper
@decorator
def base_func(n):
	sum=0
	time.sleep(n)
base_func(2)
```
# Program 2: Debugging Function Calls
### Problem: Create a decorator to print the function name and the values of its arguments every time the function is called.
```
def decorator(func):
	def wrapper(*args ,**kwargs):
		start=time.time()
		print(f"The name of the function is {func.__name__}") 
		result=func(*args, **kwargs)
		end=time.time()
		list_args=', ',join(i for i in args)
		key&value=', '.join(f"{key}={value}" for key,value in kwargs.items())
		return result
	return wrapper
@decorator
def greet(name, greeting="Hello")
	print(f"{greeting},{name}")
greet(omi)
```