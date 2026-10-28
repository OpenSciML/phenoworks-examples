from pathlib import Path

from lgopy.core import Block, BlockContext, ModelReference
from lgopy.models import HttpModelResolver


class HelloWorldFile(Block):
    """Request greetings from a Hello World model.

    Args:
        model: Deployed Hello World model reference.
    """

    name = "hello_world"
    display_name = "Hello World"
    description = "Greet the first input filename using the Hello World model."
    version = "0.1.0"
    authors = ["PhenoWorks Team"]
    categories = ["Examples", "Model Inference"]
    tags = ["phenoworks", "example", "hello-world", "model-reference"]
    extras = {
        "supported_inputs": [], #["rgb:image"],
        "transform_scope": "dataset_item"
    }

    def __init__(self, model: ModelReference) -> None:
        """Select the model.

        Args:
            model: Model reference resolved by the execution host.
        """
        super().__init__()
        self.model = model

    def call(self, context: BlockContext) -> str:
        """Greet the first file, assuming an item without a plot.

        Args:
            context: Item containing one file grouped by type in ``input["files"]``.

        Returns:
            Greeting text returned by the model.

        Raises:
            RuntimeError: If the model returns an unsuccessful response.
            StopIteration: If the item contains no files.
        """
        file = next(file for files in context.input["files"].values() for file in files)
        filename = Path(file.get("relative_path") or file["file_uri"]).name
        response = self.models.invoke(self.model, "greet", json={"name": filename})
        response.raise_for_status()
        return response.content.decode("utf-8")


def generate_zip() -> Path:
    """Build the block with the root README and citation into dist.zip.

    Returns:
        Path to the generated ZIP archive.
    """
    root = Path(__file__).parent
    result = HelloWorldFile.build(
        output_dir=root / "dist",
        format="zip",
        readme_file=root / "README.md",
        citation_file=root / "CITATION.cff",
    )
    return Path(result.archive_path)


def main() -> None:
    """Test the block against the local service on port 4000 and build its ZIP.

    Start the Hello World BentoML service before running this file.
    """
    model_tag = "hello_world:latest"
    resolver = HttpModelResolver({model_tag: "http://localhost:4000"})
    try:
        block = HelloWorldFile(model_tag)
        block.models = resolver
        context = BlockContext(
            input={"files": {"image": [{"relative_path": "example.png"}]}}
        )
        greeting = block.call(context)
        assert greeting == "Hello, example.png!", greeting
        print(greeting)
    finally:
        resolver.close()

    print(f"Generated ZIP: {generate_zip()}")


if __name__ == "__main__":
    main()
