import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import argv
from time import sleep
from typing import Literal

from core.interpolate import product_dict
from core import log
from core.ascii_table import ascii_table
import pandas as pd


class Task:
    """
    Base class for tasks implementing a dependency tree structure for execution workflow.

    This class provides the foundation for creating tasks that can be organized in a
    dependency graph and executed in the correct order. Tasks can be generated and run
    using the `gen` function with different execution modes (sync, executor, or prefect).

    Attributes:
        output: Optional attribute for storing the task's output path or result.
        deps: Optional list of Task instances that this task depends on.
        status: Task execution status

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


def print_execution_summary(tasks: list):
    """
    Print a summary table showing count of tasks by status and class name.
    
    Args:
        tasks: List of Task instances to summarize
    """
    print('')
    print("=== EXECUTION SUMMARY ===")
    
    # Get unique statuses and classes directly from list comprehensions + set
    all_statuses = sorted(set(getattr(task, 'status', 'unknown') for task in tasks))
    all_classes = sorted(set(task.__class__.__name__ for task in tasks))
    
    if all_classes and all_statuses:
        # Create DataFrame directly using double nested loop
        columns = ['Class'] + all_statuses + ['Total']
        
        # Initialize DataFrame with the right shape
        df = pd.DataFrame(index=range(len(all_classes) + 1), columns=columns)
        
        # Fill rows for each class
        for i, class_name in enumerate(all_classes):
            df.iloc[i, 0] = class_name  # Class column
            class_total = 0
            
            for j, status in enumerate(all_statuses):
                # Count directly from tasks list
                count = sum(1 for task in tasks 
                           if task.__class__.__name__ == class_name 
                           and getattr(task, 'status', 'unknown') == status)
                df.iloc[i, j + 1] = str(count) if count > 0 else ''
                class_total += count
            
            df.iloc[i, -1] = str(class_total)  # Total column
        
        # Fill totals row
        totals_row_idx = len(all_classes)
        df.iloc[totals_row_idx, 0] = 'TOTAL'
        grand_total = 0
        
        for j, status in enumerate(all_statuses):
            # Count directly from tasks list
            status_total = sum(1 for task in tasks 
                             if getattr(task, 'status', 'unknown') == status)
            df.iloc[totals_row_idx, j + 1] = str(status_total)
            grand_total += status_total
        
        df.iloc[totals_row_idx, -1] = str(grand_total)
        
        # Replace NaN with empty strings
        df = df.fillna('')
        
        print("\nTask Summary by Status and Class:")
        
        colors = dict(
            Class       = log.rgb.blue,
            Pending     = log.rgb.orange,
            Success     = log.rgb.green,
            Error       = log.rgb.red,
            Skipped     = log.rgb.blue,
            Canceled    = log.rgb.red,
            Total       = log.rgb.blue,
        )
        
        sides = dict(
            Class       = "left",
            Pending     = "right",
            Success     = "right",
            Error       = "right",
            Skipped     = "right",
            Canceled    = "right",
            Total       = "right",
        )
        
        table = ascii_table(df, colors=colors, sides=sides)
        table.print()
    
    # Count and display failed tasks
    failed_tasks = [t for t in tasks if getattr(t, 'status', None) == 'error']
    nfailed = 5  # number of displayed failed tasks
    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for t in failed_tasks[:nfailed]:
            print(f"  - {t} with error {str(t.__class__)}")
        if len(failed_tasks) > 5:
            print("[...]")
        print("=========================")


def gen_executor(
    task: Task,
    executors: dict | None = None,
    default_executor=None,
    graph_executor=None,
    verbose: bool = False,
    _tasks: list | None = None,
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
        _tasks: internal parameter to track of the tasks status and to provide an
            execution summary.

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

    is_root_call = _tasks is None
    if _tasks is None:
        _tasks = []

    if graph_executor is None:
        graph_executor = ThreadPoolExecutor(512)
    
    if default_executor is None:
        default_executor = ThreadPoolExecutor()

    # Set initial status
    if not hasattr(task, "status"):
        setattr(task, "status", "pending")

    _tasks.append(task)
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
                _tasks=_tasks,
            )
            for d in deps
        ]

        # wait for all tasks to finish
        # if any error occurred in the task, it is raised now
        for t in futures:
            t.result()

        # Check if any dependency has an error status
        for dep in deps:
            if dep.status in ["error", "canceled"]:
                setattr(task, "status", "canceled")
                if verbose:
                    log.info(f"Task {task} canceled due to dependency error")

        if getattr(task, "status") != "canceled":
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
            try:
                result = future.result()
                setattr(task, "status", "success")
                if not is_root_call:
                    return result
            except Exception as e:
                setattr(task, "status", "error")
                if verbose:
                    log.warning(f"Task {task.__class__} failed with error {str(e.__class__)}")
    else:
        # Task is already done, mark as skipped if no status set
        if not hasattr(task, "status"):
            setattr(task, "status", "skipped")
    
    if is_root_call and verbose:
        print_execution_summary(_tasks)


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
    python -m core.process.deptree sync
    python -m core.process.deptree executor
    python -m core.process.deptree prefect
    """

    composite = sample()

    if argv[1] == 'sync':
        gen_sync(composite)
    elif argv[1] == 'executor':
        gen_executor(
            composite,
            executors={"Level1": ThreadPoolExecutor(2)},
            verbose=True,
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
