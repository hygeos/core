import warnings
from core.process.condor import (
    get_htcondor_logdir,
    get_htcondor_params,
    get_htcondor_runner,
    get_htcondor_client,
)

warnings.warn("Please import from core.process.condor", DeprecationWarning)
