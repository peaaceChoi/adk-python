# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for McpToolset."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import pytest


@pytest.mark.asyncio
async def test_mcp_toolset_with_prefix():
  """Test that McpToolset correctly applies the tool_name_prefix."""
  # Mock the connection parameters
  mock_connection_params = MagicMock()
  mock_connection_params.timeout = None

  # Mock the MCPSessionManager and its create_session method
  mock_session_manager = MagicMock()
  mock_session = MagicMock()

  # Mock the list_tools response from the MCP server
  mock_tool1 = MagicMock()
  mock_tool1.name = "tool1"
  mock_tool1.description = "tool 1 desc"
  mock_tool2 = MagicMock()
  mock_tool2.name = "tool2"
  mock_tool2.description = "tool 2 desc"
  list_tools_result = MagicMock()
  list_tools_result.tools = [mock_tool1, mock_tool2]
  mock_session.list_tools = AsyncMock(return_value=list_tools_result)
  mock_session_manager.create_session = AsyncMock(return_value=mock_session)

  # Create an instance of McpToolset with a prefix
  toolset = McpToolset(
      connection_params=mock_connection_params,
      tool_name_prefix="my_prefix",
  )

  # Replace the internal session manager with our mock
  toolset._mcp_session_manager = mock_session_manager

  # Get the tools from the toolset
  tools = await toolset.get_tools()

  # The get_tools method in McpToolset returns MCPTool objects, which are
  # instances of BaseTool. The prefixing is handled by the BaseToolset,
  # so we need to call get_tools_with_prefix to get the prefixed tools.
  prefixed_tools = await toolset.get_tools_with_prefix()

  # Assert that the tools are prefixed correctly
  assert len(prefixed_tools) == 2
  assert prefixed_tools[0].name == "my_prefix_tool1"
  assert prefixed_tools[1].name == "my_prefix_tool2"

  # Assert that the original tools are not modified
  assert tools[0].name == "tool1"
  assert tools[1].name == "tool2"


def _create_mock_session_manager():
  """Helper to create a mock MCPSessionManager."""
  mock_session_manager = MagicMock()
  mock_session = MagicMock()

  mock_tool1 = MagicMock()
  mock_tool1.name = "tool1"
  mock_tool1.description = "tool 1 desc"
  mock_tool2 = MagicMock()
  mock_tool2.name = "tool2"
  mock_tool2.description = "tool 2 desc"
  list_tools_result = MagicMock()
  list_tools_result.tools = [mock_tool1, mock_tool2]

  mock_session.list_tools = AsyncMock(return_value=list_tools_result)
  mock_session_manager.create_session = AsyncMock(return_value=mock_session)
  return mock_session_manager, mock_session


@pytest.mark.asyncio
async def test_mcp_toolset_cache_disabled():
  """Test that list_tools is called every time when cache is disabled."""
  mock_connection_params = MagicMock()
  mock_connection_params.timeout = None
  mock_session_manager, mock_session = _create_mock_session_manager()

  toolset = McpToolset(connection_params=mock_connection_params, cache=False)
  toolset._mcp_session_manager = mock_session_manager

  await toolset.get_tools()
  await toolset.get_tools()

  assert mock_session.list_tools.call_count == 2


@pytest.mark.asyncio
async def test_mcp_toolset_cache_enabled():
  """Test that list_tools is called only once when cache is enabled."""
  mock_connection_params = MagicMock()
  mock_connection_params.timeout = None
  mock_session_manager, mock_session = _create_mock_session_manager()

  toolset = McpToolset(connection_params=mock_connection_params, cache=True)
  toolset._mcp_session_manager = mock_session_manager

  tools1 = await toolset.get_tools()
  tools2 = await toolset.get_tools()

  mock_session.list_tools.assert_called_once()
  assert len(tools1) == 2
  assert len(tools2) == 2
  assert tools1[0].name == tools2[0].name


@pytest.mark.asyncio
async def test_mcp_toolset_cache_with_ttl_not_expired():
  """Test that cache is used when TTL has not expired."""
  mock_connection_params = MagicMock()
  mock_connection_params.timeout = None
  mock_session_manager, mock_session = _create_mock_session_manager()

  toolset = McpToolset(
      connection_params=mock_connection_params, cache=True, cache_ttl_seconds=10
  )
  toolset._mcp_session_manager = mock_session_manager

  await toolset.get_tools()
  await toolset.get_tools()

  mock_session.list_tools.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_toolset_cache_with_ttl_expired():
  """Test that list_tools is called again after TTL expires."""
  mock_connection_params = MagicMock()
  mock_connection_params.timeout = None
  mock_session_manager, mock_session = _create_mock_session_manager()

  toolset = McpToolset(
      connection_params=mock_connection_params, cache=True, cache_ttl_seconds=1
  )
  toolset._mcp_session_manager = mock_session_manager

  await toolset.get_tools()
  mock_session.list_tools.assert_called_once()

  await asyncio.sleep(1.1)

  await toolset.get_tools()
  assert mock_session.list_tools.call_count == 2
