import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    host: str
    port: int
    log_level: str
    model_cache_ttl: int
    max_models: int
    model_list_cache_ttl: int
    
    # Prompt Cache configuration
    default_max_kv_size: Optional[int]
    enable_cache_by_default: bool

    @classmethod
    def from_env(
        cls,
        host: Optional[str] = None,
        port: Optional[int] = None,
        log_level: Optional[str] = None,
        model_cache_ttl: Optional[int] = None,
        max_models: Optional[int] = None,
        model_list_cache_ttl: Optional[int] = None,
        default_max_kv_size: Optional[int] = None,
        enable_cache_by_default: Optional[bool] = None,
    ) -> "Config":
        # Parse enable cache from env
        env_enable_cache = os.getenv("ENABLE_CACHE", "true").lower()
        parsed_enable_cache = env_enable_cache in ("true", "1", "yes")
        
        return cls(
            host=host or os.getenv("HOST", "127.0.0.1"),
            port=port or int(os.getenv("PORT", "8008")),
            log_level=log_level or os.getenv("LOG_LEVEL", "info"),
            model_cache_ttl=model_cache_ttl or int(os.getenv("MODEL_CACHE_TTL", "600")),
            max_models=max_models or int(os.getenv("MAX_MODELS", "4")),
            model_list_cache_ttl=model_list_cache_ttl
            or int(os.getenv("MODEL_LIST_CACHE_TTL", "600")),
            default_max_kv_size=default_max_kv_size
            or (int(os.getenv("DEFAULT_MAX_KV_SIZE")) if os.getenv("DEFAULT_MAX_KV_SIZE") else None),
            enable_cache_by_default=enable_cache_by_default if enable_cache_by_default is not None else parsed_enable_cache,
        )


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config):
    global _config
    _config = config
