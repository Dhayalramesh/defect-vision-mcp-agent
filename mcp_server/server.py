from mcp.server.mcpserver import MCPServer
from tools import classify_image as _classify_image, get_model_metrics as _get_model_metrics

mcp = MCPServer("defect-vision-mcp")


@mcp.tool()
def classify_image(image_path: str) -> dict:
    """Classify a defect image and return predicted class + confidence."""
    return _classify_image(image_path)


@mcp.tool()
def get_model_metrics() -> dict:
    """Return aggregate model performance stats from logged predictions."""
    return _get_model_metrics()


if __name__ == "__main__":
    mcp.run()