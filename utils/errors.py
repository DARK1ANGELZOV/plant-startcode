class AgroAIError(Exception):
    pass


class ModelNotLoadedError(AgroAIError):
    pass


class InferenceError(AgroAIError):
    pass
