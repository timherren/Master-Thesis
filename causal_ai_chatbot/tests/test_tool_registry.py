"""
Tests for ToolRegistry — the tool definitions the AI Agent sees.

Validates that:
  - All 9 tools are defined
  - Each tool has the required fields (name, description, parameters)
  - LLM function-calling schemas are valid
  - Tool names are unique
  - Helper methods work correctly
"""

import pytest
from .conftest import ToolRegistry


# ============================================================================
# Tool Definition Tests
# ============================================================================

EXPECTED_TOOLS = [
    "propose_dag",
    "test_dag",
    "fit_model",
    "sample",
    "compute_ate",
    "create_plots",
    "show_associations",
    "generate_report",
    "open_dag_editor",
]


class TestToolDefinitions:
    """Verify the structure and completeness of tool definitions."""

    def test_all_9_tools_are_defined(self):
        assert len(ToolRegistry.TOOLS) == 9

    def test_expected_tool_names_present(self):
        names = [t["name"] for t in ToolRegistry.TOOLS]
        for expected in EXPECTED_TOOLS:
            assert expected in names, f"Tool '{expected}' missing from ToolRegistry.TOOLS"

    def test_tool_names_are_unique(self):
        names = [t["name"] for t in ToolRegistry.TOOLS]
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_each_tool_has_required_fields(self):
        for tool in ToolRegistry.TOOLS:
            assert "name" in tool, f"Tool missing 'name': {tool}"
            assert "description" in tool, f"Tool '{tool.get('name')}' missing 'description'"
            assert "parameters" in tool, f"Tool '{tool.get('name')}' missing 'parameters'"

    def test_each_tool_has_nonempty_description(self):
        for tool in ToolRegistry.TOOLS:
            assert len(tool["description"]) > 20, (
                f"Tool '{tool['name']}' has a too-short description"
            )

    def test_parameters_have_type_object(self):
        """Function calling requires parameters.type == 'object'."""
        for tool in ToolRegistry.TOOLS:
            params = tool["parameters"]
            assert params.get("type") == "object", (
                f"Tool '{tool['name']}' parameters.type must be 'object', got '{params.get('type')}'"
            )

    def test_parameters_have_properties_or_empty(self):
        """Each tool's parameters should have a 'properties' dict."""
        for tool in ToolRegistry.TOOLS:
            params = tool["parameters"]
            assert "properties" in params, (
                f"Tool '{tool['name']}' parameters missing 'properties'"
            )
            assert isinstance(params["properties"], dict)


# ============================================================================
# LLM Schema Tests
# ============================================================================

class TestLLMSchemas:
    """Verify the function-calling schema output."""

    def test_get_llm_tool_schemas_returns_list(self):
        schemas = ToolRegistry.get_llm_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) == 9

    def test_each_schema_has_type_function(self):
        for schema in ToolRegistry.get_llm_tool_schemas():
            assert schema["type"] == "function"

    def test_each_schema_has_function_with_name_and_description(self):
        for schema in ToolRegistry.get_llm_tool_schemas():
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_schema_names_match_tool_names(self):
        schema_names = [s["function"]["name"] for s in ToolRegistry.get_llm_tool_schemas()]
        tool_names = [t["name"] for t in ToolRegistry.TOOLS]
        assert schema_names == tool_names


# ============================================================================
# Helper Method Tests
# ============================================================================

class TestHelperMethods:
    """Verify ToolRegistry helper methods."""

    def test_get_tool_names_returns_all(self):
        names = ToolRegistry.get_tool_names()
        assert names == EXPECTED_TOOLS

    def test_get_tool_descriptions_is_readable(self):
        desc = ToolRegistry.get_tool_descriptions()
        assert isinstance(desc, str)
        # Should contain each tool name
        for name in EXPECTED_TOOLS:
            assert name in desc


# ============================================================================
# Individual Tool Schema Validation
# ============================================================================

class TestSpecificToolSchemas:
    """Validate specific tools have the correct required parameters."""

    def _get_tool(self, name):
        for tool in ToolRegistry.TOOLS:
            if tool["name"] == name:
                return tool
        pytest.fail(f"Tool '{name}' not found")

    def test_propose_dag_requires_vars(self):
        tool = self._get_tool("propose_dag")
        assert "vars" in tool["parameters"]["properties"]
        assert "vars" in tool["parameters"].get("required", [])

    def test_compute_ate_requires_X_and_Y(self):
        tool = self._get_tool("compute_ate")
        props = tool["parameters"]["properties"]
        assert "X" in props
        assert "Y" in props
        required = tool["parameters"].get("required", [])
        assert "X" in required
        assert "Y" in required

    def test_create_plots_requires_plot_types(self):
        tool = self._get_tool("create_plots")
        assert "plot_types" in tool["parameters"]["properties"]
        assert "plot_types" in tool["parameters"].get("required", [])

    def test_test_dag_has_no_required_params(self):
        """test_dag reads DAG from session state, so no required params."""
        tool = self._get_tool("test_dag")
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_fit_model_has_no_required_params(self):
        """fit_model reads DAG and data from session state."""
        tool = self._get_tool("fit_model")
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_open_dag_editor_has_empty_properties(self):
        tool = self._get_tool("open_dag_editor")
        assert tool["parameters"]["properties"] == {}
