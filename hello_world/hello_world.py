from typing import Any

import bentoml


my_image = bentoml.images.Image(python_version="3.11") \
            .python_packages("xgboost", "scikit-learn")

@bentoml.service(
    name="hello_world",
    image=my_image,
)
class HelloWorldService:
    """A minimal greeting service."""

    @bentoml.api
    def greet(self, name: str = "World", context: dict[str, Any] | None = None) -> str:
        """Return a greeting with optional PhenoWorks execution context.

        Args:
            name: Name to include in the greeting.
            context: JSON execution input and previous outputs, accepted for
                model extensions and unused by this minimal greeting.

        Returns:
            Greeting for the supplied name.
        """
        return f"Hello, {name}!"
