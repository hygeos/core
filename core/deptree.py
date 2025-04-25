import os
import socket
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from sys import argv
from time import sleep
from typing import Literal

from core import log


class BaseProduct:
    """
    Base class for products, which are vertices of the dependency tree.

    Should typically hold an attribute `path` for storing the product path.
    """
    def __init__(self):
        self.path = None

    def dependencies(self) -> list:
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
        if hasattr(self, 'path'):
            return Path(self.path).exists()
        else:
            return False


def gen(
    prod: BaseProduct,
    mode: Literal['sync', 'executor', 'prefect'] = 'sync',
    **kwargs
):
    if mode == 'sync':
        return gen_sync(prod, **kwargs)
    elif mode == 'executor':
        return gen_executor(prod, **kwargs)
    elif mode == 'prefect':
        return gen_prefect(prod, **kwargs)
    else:
        raise ValueError(f"Invalid mode '{mode}'")


def gen_sync(
    prod: BaseProduct,
    verbose: bool = False,
):
    """Generate a product in sequential mode

    Args:
        obj (Type[BaseProduct]): The object to generate
    """
    if not prod.exists():
        for d in prod.dependencies():
            gen_sync(d, verbose=verbose)
        if verbose:
            log.info(f"Generating {prod}...")
        return prod.run()


def gen_prefect(
    prod: BaseProduct,
    concurrency_limit: dict | None = None,
    task_runner=None,
):
    """
    Generate a product in a flow using Prefect

    Note: a prefect server will be started automatically. To use an existing server,
    you can use the following variable:
        PREFECT_API_URL=http://127.0.0.1:4200/api
    (here with a local server started with `prefect server start`)

    task_runner can be a DaskTaskRunner() or DaskTaskRunner(get_htcondor_cluster())
    """
    from prefect import flow

    from core.deptree_prefect import (decorator_concurrency_limit,
                                      gen_prefect_flow)

    flowname = f'{prod.__class__.__name__}'
    if concurrency_limit is None:
        concurrency_limit = {}
    return decorator_concurrency_limit(concurrency_limit)(
        flow(name=flowname, task_runner=task_runner)(gen_prefect_flow)
    )(prod)


def gen_executor(
    prod: BaseProduct,
    executors: dict | None = None,
    default_executor=None,
    graph_executor=None,
    verbose: bool = False,
):
    """
    Recursively generate dependencies of `obj`, then call its `run` method in a
    custom executor.

    Args:
        prod: product derived from BaseProduct
        executors: dictionary of executors, whose keys must include obj.executor_id
        default_executor: the executor to use for all tasks not included in executors. Defaults
            to ThreadPoolExecutor
        graph_executor: an instance of ThreadPoolExecutor used for traversing the
            dependency tree with the `gen` function. It is useful to override the
            default ThreadPoolExecutor(512) if the tree is too large.

    Example:
        from concurrent.futures import ThreadPoolExecutor
        from distributed import Client # dask executor

        gen(prod, executors={'Level1' : ThreadPoolExecutor()},
            default_executor=Client()  # dask executor
                # Note: can also use dask_jobqueue to instantiate
                # an executor backed by a job queuing system like
                # HTCondor
        })

    For details on how to implement `obj`, please check function core.deptree.example.
    """
    # Note: `gen` is defined as a function instead of an obj method because it includes
    # non-serializable code, and obj can be serialized (for example by the dask
    # distributed executor).

    if graph_executor is None:
        graph_executor = ThreadPoolExecutor(512)
    
    if default_executor is None:
        default_executor = ThreadPoolExecutor()

    if not prod.exists():
        # run all dependencies on the graph executor
        deps = prod.dependencies()
        if len(deps) > graph_executor._max_workers:
            raise RuntimeError(
                f"Object {prod} has {len(deps)} dependencies, but graph_executor only "
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

        # get the executor for current `obj`
        clsname = prod.__class__.__name__
        if (executors is not None) and (clsname in executors):
            executor = executors[clsname]
        else:
            executor = default_executor

        # execute `obj.run` on this executor and wait for result
        future = executor.submit(prod.run)
        if verbose:
            log.info(f"Generating {prod}...")
        return future.result()


def get_htcondor_logdir(logdir: Path | str | None) -> Path:
    """
    A default log directory for condor
    """
    # TODO: move to another module ?
    if logdir is None:
        username = os.getlogin()
        return f'/tmp/condor_logs_{username}/{datetime.now().isoformat()}'
    else:
        return Path(logdir)


def get_htcondor_cluster(
    cores=1,
    memory="1GB",
    disk="1GB",
    death_timeout = '600',
    logdir: Path | str | None = None,
    maximum_jobs: int = 100,
    **kwargs
):
    """
    Create a condor cluster

    kwargs are passed to HTCondorCluster
    """
    # TODO: move to another module ?
    from dask_jobqueue.htcondor import HTCondorCluster

    cluster = HTCondorCluster(
        cores=cores,
        memory=memory,
        disk=disk,
        death_timeout=death_timeout,
        log_directory=get_htcondor_logdir(logdir),
        **kwargs
    )
    # cluster.scale(jobs=2)  # ask for 10 jobs
    cluster.adapt(maximum_jobs=maximum_jobs)

    return cluster


def get_htcondor_client(logdir: Path | str | None = None, **kwargs):
    """
    Get a client for HTCondor cluster
    """
    # TODO: move to another module ?
    from dask.distributed import Client

    logdir = get_htcondor_logdir(logdir)
    cluster = get_htcondor_cluster(logdir=logdir, **kwargs)
    client = Client(cluster)
    print('Dashboard address is', client.dashboard_link)
    print('HTCondor log directory is', logdir)
    return client


def sample():
    """
    Generate a sample Product for demonstration purposes
    """
    class Level1(BaseProduct):
        def __init__(self, id):
            self.id = id

        def run(self):
            log.info(f"Downloading level1 {self.id}")
            sleep(1)

    class Level2(BaseProduct):
        def __init__(self, id, **kwargs):
            self.id = id
            self.kwargs = kwargs

        def dependencies(self) -> list:
            # each Level2 depends on a single Level1
            return [Level1(self.id, **self.kwargs)]

        def run(self):
            log.info(f"Generating level2 {self.id} on {socket.gethostname()}")
            sleep(1)

    class Composite(BaseProduct):
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
    elif argv[1] == 'prefect':
        # from prefect_dask import DaskTaskRunner
        gen_prefect(
            composite,
            # task_runner=DaskTaskRunner(get_htcondor_cluster()),
            concurrency_limit={"Level1": 2},
        )
    else:
        raise ValueError(f"Invalid mode {argv[1]}")

