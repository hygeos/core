import asyncio
from prefect import task
from prefect import get_client


"""
This module is imported by the deptree module.
It is used to isolate the part which requires prefect.
"""

async def set_concurrency_limit(d: dict):
    """
    Asynchronous function to set concurrency limits for tasks
    """
    async with get_client() as client:
        for item in d :
            await client.create_concurrency_limit(
                tag=item, 
                concurrency_limit=d[item]
                )


async def remove_concurrency_limit(d: dict):
    """
    Asynchronous function to remove concurrency limits for tasks
    """
    for item in d:
        async with get_client() as client:
            await client.delete_concurrency_limit_by_tag(tag=item)


def decorator_concurrency_limit(d: dict|None):
    """
    Decorator to apply concurrency limits to a function
    """
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            try:
                asyncio.run(set_concurrency_limit(d))  # Set concurrency limits
                result = func(*args, **kwargs)         # Execute the function
            finally:
                asyncio.run(remove_concurrency_limit(d))  # Remove concurrency limits
            return result
        if (d is None) or len(d) == 0:
            # bypass the decorator if useless
            return func
        else:
            return wrapper
    return actual_decorator


def wraptask(obj):
    """
    Wrap a Product so that its dependencies are passed as futures to the wrapped
    function arguments
    """
    classname = str(obj.__class__.__name__)
    def wrapper(*args):
        # execute the run method
        return obj.run()
    return task(name=classname, tags=[classname])(wrapper)


def gen_prefect_flow(prod):
    
    if prod.exists():
        return None

    # recursively trigger dependencies
    futures = [gen_prefect_flow(x) for x in prod.dependencies() if not x.exists()]

    # submit the current task by passing the futures dependencies as arguments
    future = wraptask(prod).submit(*futures)

    return future

