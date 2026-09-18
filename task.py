from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import bash


@task
def sandbox_smoke_test():
    """Minimal task to confirm the model + Docker sandbox wiring works.

    The model is asked to run a shell command inside the isolated
    container (no network) and report the output.
    """
    return Task(
        dataset=[
            Sample(
                input=(
                    "Use the bash tool to run `echo hello-from-sandbox` "
                    "and report the exact output."
                ),
                target="hello-from-sandbox",
            )
        ],
        solver=[use_tools([bash(timeout=60)]), generate()],
        scorer=includes(),
        sandbox="docker",
    )
