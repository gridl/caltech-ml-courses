import os
import pathlib
import logging
import json
from shutil import copyfile

# TODO: handle loaded model graph str declaraion as string, not module
# TODO: handle path to model file and copy it if provided
# TODO: rename properties
# TODO: rewrite last run

logging.basicConfig(level=logging.INFO)


class FoldersManagerException(Exception):
    pass


class FoldersManager:
    """Manager to create required directories.

    Directories are created for storing logs, saves, etc.
    Also manager will guess and return model id and required pathes"""
    graph_decl_f_name = 'graph_str.txt'
    description_f_name = 'description.txt'
    train_params_f_name = 'train_params.json'
    data_params_f_name = 'data_params.json'

    def __init__(self, model_graph_decl, logs_dir,
                 saves_dir=None, pred_dir=None,
                 train_params={}, data_params={},
                 model_description='',
                 model_file_path=None,
                 test_mode=False):
        """
        Args:
            model_graph_decl(str): string graph representation of the model
            logs_dir(str): Directory where all logs should be stored
            saves_dir(str): Directory where saves files should be stored
            pred_dir(str): Directory where predictions should be stored
            train_params(dict): Train parameters that were used
            data_params(dict): Parameters that were used for data preprocessing
            model_description(str): Some text description of the model
            model_file_path(sre): Path to .py file with model. If was provided
                will be copied to logs dir
            test_mode(bool): Run manager in test mode, so no any folders will
                be created and tmpdir will be used as source
        """
        self.logger = logging.getLogger("FoldersManager")
        if test_mode:
            self.logger.info("Manager in test mode")
            logs_dir = saves_dir = pred_dir = '/tmp/folders_manager_dir'

        self.model_graph_decl = model_graph_decl
        self.train_params = train_params
        self.data_params = data_params
        self.model_description = model_description
        self.model_file_path = model_file_path

        # create initial required folders
        saves_dir = logs_dir if saves_dir is None else saves_dir
        pred_dir = logs_dir if pred_dir is None else pred_dir
        self.logs_dir = pathlib.Path(logs_dir)
        self.saves_dir = pathlib.Path(saves_dir)
        self.pred_dir = pathlib.Path(pred_dir)
        self._prepare_model_run()

    def _prepare_model_run(self):
        self._create_main_folders()
        self._get_model_id()
        self._create_model_folders()
        self._store_model_info()
        self._get_model_run()
        self._save_model_description_and_params()
        self._view_info()

    def _create_main_folders(self):
        for attr in ['logs_dir', 'saves_dir', 'pred_dir']:
            getattr(self, attr).mkdir(exist_ok=True)

    def _get_model_id(self):
        """Try to find same run of the model. It such exists - return it's id.
        Otherwise create new folder for model
        """
        same_run_was_found = False
        models_pathes = [path for path in self.logs_dir.iterdir()
                         if path.is_dir()]
        # try to find same model by graph_str
        for path in models_pathes:
            metadata_file = path / self.graph_decl_f_name
            try:
                existed_graph_str = metadata_file.open('r').read()
            except FileNotFoundError:
                continue
            if existed_graph_str == self.model_graph_decl:
                self._model_id = path.name
                same_run_was_found = True
                break
        if not same_run_was_found:
            models_numbers = [
                int(path.name.split('_')[1]) for path in models_pathes]
            if not models_numbers:
                self._model_id = 'model_1'
            else:
                new_mode_num = max(models_numbers) + 1
                self._model_id = 'model_{}'.format(new_mode_num)
        return self._model_id

    def _create_model_folders(self):
        for attr in ['logs_dir', 'saves_dir', 'pred_dir']:
            sub_path = getattr(self, attr) / self.model_id
            sub_path.mkdir(exist_ok=True)

    def _store_model_info(self):
        """Copy model description, model file itself to the corresponding
        directory"""
        metadata_file = self.model_logs_dir / self.graph_decl_f_name
        metadata_file.write_text(self.model_graph_decl)
        if self.model_file_path:
            copyfile(self.model_file_path,
                     str(self.model_logs_dir / 'model.py'))

    def _get_model_run(self):
        existed_runs = [
            int(run.name) for run in
            self.model_logs_dir.iterdir() if run.is_dir()]
        if not existed_runs:
            self._model_run = '1'
        else:
            self._model_run = str(max(existed_runs) + 1)
        # create required folders for runs
        for attr in ['logs_dir', 'saves_dir', 'pred_dir']:
            sub_path = getattr(self, 'model_%s' % attr) / self._model_run
            sub_path.mkdir(exist_ok=True)
        return self._model_run

    def _save_model_description_and_params(self):
        (self.model_run_logs_dir / self.description_f_name).write_text(
            self.model_description)
        # with (self.model_run_logs_dir / self.train_params_f_name).open('w') as f:
        #     json.dump(self.train_params, f, indent=2)
        (self.model_run_logs_dir / self.train_params_f_name).write_text(
            json.dumps(self.train_params, indent=2))
        (self.model_run_logs_dir / self.data_params_f_name).write_text(
            json.dumps(self.data_params, indent=2))

    def _view_info(self):
        self.logger.info("model_id: %s" % self.model_id)
        self.logger.info("model_run: %s" % self.model_run)

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_run(self):
        return self._model_run

    @property
    def model_logs_dir(self):
        return self.logs_dir / self.model_id

    @property
    def model_saves_dir(self):
        return self.saves_dir / self.model_id

    @property
    def model_pred_dir(self):
        return self.pred_dir / self.model_id

    @property
    def model_run_logs_dir(self):
        return self.model_logs_dir / self.model_run

    @property
    def model_run_saves_dir(self):
        return self.model_saves_dir / self.model_run

    @property
    def model_run_pred_dir(self):
        return self.model_pred_dir / self.model_run
