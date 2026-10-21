from __future__ import annotations

import os

import uvicorn

from speaker_claims.api.app import create_app
from speaker_claims.runtime import build_service


def main() -> None:
    """Apply migrations and serve the classifier."""
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(build_service()), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
