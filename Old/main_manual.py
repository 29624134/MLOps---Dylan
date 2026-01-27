
from orchestrator import WorkflowExecutor

if __name__ == "__main__":
    executor = WorkflowExecutor("../config/workflow.yaml")
    executor.start_workflow()
