from __future__ import annotations


class DomainError(Exception):
    """A rejected tool call or claim that the application can report."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class UnknownBrandError(DomainError):
    def __init__(self, brand_id: str) -> None:
        super().__init__("unknown_brand", f"Unknown brand: {brand_id}")


class IllegalToolError(DomainError):
    def __init__(self, tool_name: str) -> None:
        super().__init__("illegal_tool", f"Illegal tool: {tool_name}")


class CatalogueFormatError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__("catalogue_format", message)


class SpanValidationError(DomainError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)


class ProviderTimeout(DomainError):
    def __init__(self) -> None:
        super().__init__("timeout", "The model request timed out")


class ProviderFailure(DomainError):
    def __init__(self) -> None:
        super().__init__("provider_failure", "The model request failed")
