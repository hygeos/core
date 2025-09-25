import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import argv
from time import sleep
from typing import Literal

from core.interpolate import product_dict
from core import log


class Task:
    """
    Base class for tasks implementing a dependency tree structure for execution workflow.

    This class provides the foundation for creating tasks that can be organized in a
    dependency graph and executed in the correct order. Tasks can be generated and run
    using the `gen` function with different execution modes (sync, executor, or prefect).

    Attributes:
        output: Optional attribute for storing the task's output path or result.
        deps: Optional list of Task instances that this task depends on.

    Methods:
        dependencies(): Returns the list of dependent tasks. The default implementation
            returns the `deps` attribute, if it exists.
        run(): Implements the actual execution code
        done(): Checks if the task is completed. The default implementation checks if
            the `output` Path exists, if defined.

    Usage:
        Subclass Task and implement at minimum:
        - The `run()` method with your task's execution logic
        - Optionally override `dependencies()` if your task has dependencies
        - Set `output` attribute if your task produces output files

    Example:
        See the `sample` function for a complete example of Task definition and usage.
    """

    def dependencies(self) -> list:
        """
        Returns a list of dependencies, each being a `Task`.
        """
        if hasattr(self, "deps"):
            return getattr(self, "deps")
        else:
            return []

    def run(self):
        """
        Implements the actual execution code
        """
        pass

    def done(self) -> bool:
        """
        Whether the task is done
        """
        if hasattr(self, 'output'):
            return Path(self.output).exists() # type: ignore
        else:
            return False


def gen(
    task: Task,
    mode: Literal['sync', 'executor', 'prefect'] = 'sync',
    **kwargs
):
    """
    Run a task and its dependencies

    Arguments:
        mode (str):
            - 'sync': sequential
            - 'executor': using executors
            - 'prefect': using prefect as a worflow manager
    """
    if mode == 'sync':
        return gen_sync(task, **kwargs)
    elif mode == 'executor':
        return gen_executor(task, **kwargs)
    elif mode == 'prefect':
        from core.deptree_prefect import gen_prefect
        return gen_prefect(task, **kwargs)
    else:
        raise ValueError(f"Invalid mode '{mode}'")


def gen_sync(
    task: Task,
    verbose: bool = False,
):
    """Run a Task in sequential mode

    Args:
        task (Task): The task to generate
    """
    if not task.done():
        for d in task.dependencies():
            gen_sync(d, verbose=verbose)
        if verbose:
            log.info(f"Generating {task}...")
        return task.run()


def gen_executor(
    task: Task,
    executors: dict | None = None,
    default_executor=None,
    graph_executor=None,
    verbose: bool = False,
):
    """
    Recursively generate dependencies of `task`, then call its `run` method in a
    custom executor.

    Args:
        task: a Task object
        executors: dictionary of executors
        default_executor: the executor to use for all tasks not included in executors. Defaults
            to ThreadPoolExecutor
        graph_executor: an instance of ThreadPoolExecutor used for traversing the
            dependency tree with the `gen` function. It is useful to override the
            default ThreadPoolExecutor(512) if the tree is too large.

    Example:
        from concurrent.futures import ThreadPoolExecutor
        from distributed import Client # dask executor

        gen(task,
            executors={'Level1' : ThreadPoolExecutor(2)},
            default_executor=Client()  # dask executor
                # Note: can also use dask_jobqueue to instantiate
                # an executor backed by a job queuing system like
                # HTCondor
        })

    For details on how to implement `task`, please check function core.deptree.sample.
    """
    # Note: `gen` is defined as a function instead of an task method because it includes
    # non-serializable code, and task can be serialized (for example by the dask
    # distributed executor).

    if graph_executor is None:
        graph_executor = ThreadPoolExecutor(512)
    
    if default_executor is None:
        default_executor = ThreadPoolExecutor()

    if not task.done():
        # run all dependencies on the graph executor
        deps = task.dependencies()
        if len(deps) > graph_executor._max_workers:
            raise RuntimeError(
                f"Task {task} has {len(deps)} dependencies, but graph_executor only "
                f"has {graph_executor._max_workers} maximum workers. Please use a "
                f"graph_executor with additional workers."
            )
        futures = [
            graph_executor.submit(
                gen_executor,
                d,
                executors=executors,
                default_executor=default_executor,
                graph_executor=graph_executor,
                verbose = verbose,
            )
            for d in deps
        ]

        # wait for all tasks to finish
        # if any error occurred in the task, it is raised now
        for t in futures:
            t.result()

        # get the executor for current `task`
        clsname = task.__class__.__name__
        if (executors is not None) and (clsname in executors):
            executor = executors[clsname]
        else:
            executor = default_executor

        # execute `task.run` on this executor and wait for result
        future = executor.submit(task.run)
        if verbose:
            log.info(f"Generating {task}...")
        return future.result()


class TaskList(Task):
    def __init__(self, deps: list):
        """
        Construct a Task that depends on a list of dependencies (other tasks)

        Ex: TaskList([MyTask1(), MyTask2()])
        depends on [
            MyTask1(),
            MyTask2(),
        ]
        """
        self.deps = deps

    def dependencies(self):
        return self.deps


class TaskProduct(Task):
    def __init__(self, cls, others={}, **kwargs):
        """
        Construct a Task that depends on the cartesian product of
        the given arguments.
        The `other` arguments are passed directly to the class.

        cls: a class that inherits from Task

        Ex: TaskProduct(MyTask, {
            'a': [1, 2],
            'b': [3, 4],
        })
        depends on [
            MyTask(a=1, b=3),
            MyTask(a=1, b=4),
            MyTask(a=2, b=3),
            MyTask(a=2, b=4),
        ]
        """
        self.cls = cls
        for k, v in kwargs.items():
            assert isinstance(v, list)
            assert k not in others
        self.kwargs = kwargs
        self.others = others

    def dependencies(self):
        return [self.cls(**d, **self.others) for d in product_dict(**self.kwargs)]


def sample():
    """
    Generate a sample Product for demonstration purposes
    """
    class Level1(Task):
        def __init__(self, id):
            self.id = id

        def run(self):
            log.info(f"Downloading level1 {self.id}")
            sleep(1)

    class Level2(Task):
        def __init__(self, id, **kwargs):
            self.id = id
            self.kwargs = kwargs

        def dependencies(self) -> list:
            # each Level2 depends on a single Level1
            return [Level1(self.id, **self.kwargs)]

        def run(self):
            log.info(f"Generating level2 {self.id} on {socket.gethostname()}")
            sleep(1)

    class Composite(Task):
        def __init__(self, id, **kwargs):
            self.id = id
            self.kwargs = kwargs

        def dependencies(self) -> list:
            # list required level 2 products for current composite
            return [Level2(i, **self.kwargs) for i in range(5)]

        def run(self):
            log.info(
                f"Generating monthly composite {self.id} from "
                f"{len(self.dependencies())} dependencies."
            )
            sleep(1)

    return Composite("202410")


if __name__ == "__main__":
    """
    python -m core.deptree sync
    python -m core.deptree executor
    python -m core.deptree prefect
    """

    composite = sample()

    if argv[1] == 'sync':
        gen_sync(composite)
    elif argv[1] == 'executor':
        gen_executor(
            composite,
            executors={"Level1": ThreadPoolExecutor(2)},
            # default_executor=get_htcondor_client(),
        )
    else:
        from core.deptree_prefect import gen_prefect
        if argv[1] == 'prefect':
            task_runner = None
        elif argv[1] == 'prefect_dask':
            from prefect_dask import DaskTaskRunner
            task_runner = DaskTaskRunner
        elif argv[1] == 'prefect_condor':
            from core.condor import get_htcondor_runner
            task_runner=get_htcondor_runner()
        else:
            raise ValueError(f"Invalid mode {argv[1]}")

        gen_prefect(
            composite,
            task_runner=task_runner,
            # concurrency_limit={"Level1": 2},
        )
