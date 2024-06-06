

class ModelSpecificationString:
    def __init__(self, model_name: str): # taks_taype, model_version, optional model_seed
        self.specification_string = model_name

    def deconstruct(self):
        return self.specification_string

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, ModelSpecificationString):
            return self.specification_string == other.specification_string
        return False

    #todo add function returning pah part instead of deconstructs into seperate parts?
