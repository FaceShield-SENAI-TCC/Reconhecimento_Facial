"""
Exceções personalizadas para tratamento consistente de erros
"""

class FaceRecognitionError(Exception):
    """Exceção base para erros do sistema de reconhecimento facial"""
    pass

class DatabaseError(FaceRecognitionError):
    """Erros relacionados ao banco de dados"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Erro de conexão com o banco de dados"""
    pass

class AuthenticationError(FaceRecognitionError):
    """Erros de autenticação"""
    pass

class TokenExpiredError(AuthenticationError):
    """Token JWT expirado"""
    pass

class InvalidTokenError(AuthenticationError):
    """Token JWT inválido"""
    pass

class ImageValidationError(FaceRecognitionError):
    """Erros de validação de imagem"""
    pass

class FaceDetectionError(FaceRecognitionError):
    """Erros na detecção facial"""
    pass

class FaceRecognitionServiceError(FaceRecognitionError):
    """Erros no serviço de reconhecimento facial"""
    pass

class ModelInitializationError(FaceRecognitionError):
    """Erros na inicialização do modelo"""
    pass