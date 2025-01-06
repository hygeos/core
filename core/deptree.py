from concurrent.futures import ThreadPoolExecutor
from time import sleep
from typing import Type

from core import log

class BaseProduct:
    """
    Base class for products, which are vertices of the dependency tree.

    Must hold an attribute `executor_id` (see function `gen`).

    Should typically hold an attribute `path` for storing the product path.
    """

    def depends(self) -> list:
        """
        Returns a list of dependencies, each being derived from BaseProduct
        """
        return []

    def run(self):
        """
        Implements the actual execution code
        """
        pass

    def exists(self):
        """
        Whether the product exists
        """
        return False


def gen(obj: Type[BaseProduct], executors: dict, graph_executor=None):
    """
    Recursively generate dependencies of `obj`, then call its `run` method in a
    custom executor.
    `gen` is defined as a function instead of an obj method because it includes
    non-serializable code, and obj can be serialized (for example by the dask
    distributed executor).

    Args:
        obj: product derived from BaseProduct
        executors: dictionary of executors, whose keys must include obj.executor_id
        graph_executor: an instance of ThreadPoolExecutor used for traversing the
            dependency tree with the `gen` function. It is useful to override the
            default ThreadPoolExecutor(512) if the tree is too large.

    Example:
        from concurrent.futures import ThreadPoolExecutor
        from distributed import Client # dask executor

        gen(obj, executors={
            'local' : ThreadPoolExecutor(),
            'dask' : Client(), # dask executor
                               # Note: can also use dask_jobqueue to instantiate
                               # an executor backed by a job queuing system like
                               # HTCondor
        })

    For details on how to implement `obj`, please check function core.deptree.example.
    """
    if graph_executor is None:
        graph_executor = ThreadPoolExecutor(512)

    if not obj.exists():
        # run all dependencies on the graph executor
        deps = obj.depends()
        if len(deps) > graph_executor._max_workers:
            raise RuntimeError(
                f"Object {obj} has {len(deps)} dependencies, but graph_executor only "
                f"has {graph_executor._max_workers} maximum workers. Please use a "
                f"graph_executor with additional workers."
            )
        futures = [
            graph_executor.submit(gen, d, executors, graph_executor) for d in deps
        ]

        # wait for all tasks to finish
        # if any error occurred in the task, it is raised now
        for t in futures:
            t.result()

        # get the executor for current `obj`
        executor = executors[obj.executor_id]

        # execute `obj.run` on this executor and wait for result
        future = executor.submit(obj.run)
        return future.result()


def example():
    """
    Usage example of deptree.gen

    Execute it with:
        python -m core.deptree
    """

    class Level1(BaseProduct):
        def __init__(self, id):
            self.id = id
            self.executor_id = "local"

        def run(self):
            log.info(f"Downloading level1 {self.id} on {self.executor_id}")
            sleep(1)

    class Level2(BaseProduct):
        def __init__(self, id):
            self.id = id
            self.executor_id = "dask"

        def depends(self) -> list:
            # each Level2 depends on a single Level1
            return [Level1(self.id)]

        def run(self):
            log.info(f"Generating level2 {self.id} on {self.executor_id}")
            sleep(1)

    class Composite(BaseProduct):
        def __init__(self, id):
            self.id = id
            self.executor_id = "local"

        def depends(self) -> list:
            # list required OLCI level 2 for current composite
            return [Level2(i) for i in range(5)]

        def run(self):
            log.info(
                f"Generating monthly composite {self.id} on {self.executor_id} from "
                f"{len(self.depends())} dependencies."
            )
            sleep(1)

    gen(
        Composite("202410"),
        executors={
            # the "local" executor has a limit of 2 workers
            # (for example, to limit parallel downloads)
            "local": ThreadPoolExecutor(2),
            "dask": ThreadPoolExecutor(),
        },
    )


if __name__ == "__main__":
    example()
