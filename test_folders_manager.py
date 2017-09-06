from monitoring.folders_manager import FoldersManager
from models import test_model


manager = FoldersManager(
    loaded_model=test_model,
    logs_dir='logs',
    test_mode=True)
print("model_id:", manager.model_id)
print("model_run:", manager.model_run)
print("model_pred_dir:", manager.model_pred_dir)
