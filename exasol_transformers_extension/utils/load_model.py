

class LoadModel:
    def __init__(self,
                 pipeline,
                 base_model,
                 tokenizer,
                 task_name,
                 device
                 ):
        self.pipeline = pipeline
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.device = device
        self.last_loaded_model = None
        self.last_loaded_tokenizer = None
        self.last_loaded_model_key = None

    def load_models(self, model_name: str,
                    current_model_key,
                    cache_dir,
                    token_conn_obj) -> None:
        """
        Load model and tokenizer model from the cached location in bucketfs.
        If the desired model is not cached, this method will attempt to
        download the model to the read-only path /bucket/.. and cause an error.
        This error will be addressed in ticket
        https://github.com/exasol/transformers-extension/issues/43.

        :param model_name: The model name to be loaded
        """
        token = False
        if token_conn_obj:
            token = token_conn_obj.password

        self.last_loaded_model = self.base_model.from_pretrained(
            model_name, cache_dir=cache_dir, use_auth_token=token)
        self.last_loaded_tokenizer = self.tokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_auth_token=token)
        last_created_pipeline = self.pipeline(
            self.task_name,
            model=self.last_loaded_model,
            tokenizer=self.last_loaded_tokenizer,
            device=self.device,
            framework="pt")
        self.last_loaded_model_key = current_model_key
        return last_created_pipeline
