from .providers import QdrantProvider
from .VDBEnums import VectorDBType
from helpers.config import get_settings

class VDBFactory:

    @staticmethod
    def create(provider: str):
        settings = get_settings()
        
        if provider == VectorDBType.QDRANT.value:
            qdrant_provider = QdrantProvider(
                host= settings.VECTOR_DB_HOST,
                port= settings.VECTOR_DB_PORT,
                grpc_port= settings.VECTOR_DB_GRPC_PORT,
                distance_method= settings.VECTOR_DB_DISTANCE_METHOD,
            )
            return qdrant_provider
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        