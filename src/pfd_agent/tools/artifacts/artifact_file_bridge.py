"""
Bridge tool to convert artifacts to temporary files for processing
"""

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import tempfile
import os
from pathlib import Path
import shutil
import json
from typing import Optional, Union, Dict, Any


async def list_available_files(tool_context: ToolContext) -> str:
    """List both local files and artifacts (uploaded files)"""
    try:
        result = "📁 Available files:\n\n"
        
        # List local files in workspace directory
        current_dir = Path.cwd()
        workspace_dir = current_dir / 'workspace'
        local_files = []
        
        # Check workspace directory if it exists
        if workspace_dir.exists():
            for ext in ['*.csv', '*.xlsx', '*.txt', '*.json', '*.md', '*.png', '*.jpg', '*.jpeg', '*.extxyz', '*.pt', '*.pth']:
                local_files.extend(workspace_dir.glob(ext))
        else:
            # Fallback to current directory
            for ext in ['*.csv', '*.xlsx', '*.txt', '*.json', '*.md', '*.png', '*.jpg', '*.jpeg', '*.extxyz', '*.pt', '*.pth']:
                local_files.extend(current_dir.glob(ext))
        
        if local_files:
            result += "**Local files (workspace):**\n"
            for file in local_files:
                result += f"- {file.name}\n"
            result += "\n"
        
        # List files in temp_artifacts directory
        temp_dir = current_dir / 'uploaded_files'
        temp_files = []
        if temp_dir.exists():
            for ext in ['*.csv', '*.xlsx', '*.txt', '*.json', '*.md', '*.png', '*.jpg', '*.jpeg', '*.tmp', '*.extxyz', '*.pt', '*.pth']:
                temp_files.extend(temp_dir.glob(ext))

        if temp_files:
            result += "**Uploaded files (from artifacts):**\n"
            for file in temp_files:
                result += f"- {file.name} (path: {file.absolute()})\n"
            result += "\n"
        
        # List artifacts (uploaded files)
        artifact_keys = []
        try:
            artifact_keys = await tool_context.list_artifacts()
            if artifact_keys:
                result += "**Uploaded files (artifacts):**\n"
                for key in artifact_keys:
                    result += f"- {key} (use file_read_artifact to convert to temp file)\n"
                result += "\n"
        except Exception as e:
            result += f"Error listing artifacts: {str(e)}\n"
        
        if not local_files and not temp_files and not artifact_keys:
            result += "No files found. Upload files or place them in the current directory.\n"
        
        return result.strip()
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


async def file_read_artifact(tool_context: ToolContext, filename: str) -> str:
    """
    Read a file using ADK ToolContext artifact methods.
    Works with artifacts that have been uploaded to the session.

    Returns a JSON string with structured information about the artifact:
    {
        "status": "success" | "error",
        "filename": str,
        "path": str,  # Full path to the file (for binary files saved to temp)
        "type": "text" | "image" | "model" | "binary",
        "size": int,  # Size in bytes
        "content": str,  # Only for text files
        "message": str  # Human-readable message
    }
    """
    try:
        # List available artifacts using ToolContext
        artifact_keys = await tool_context.list_artifacts()

        if filename not in artifact_keys:
            available = ', '.join(artifact_keys) if artifact_keys else 'No artifacts available'
            return json.dumps({
                "status": "error",
                "filename": filename,
                "path": "",
                "type": "unknown",
                "size": 0,
                "message": f"❌ Artifact not found: {filename}\n\nAvailable artifacts: {available}\n\nTip: Upload files through the ADK web interface to make them available as artifacts."
            }, indent=2)

        # Load the artifact using ToolContext
        artifact_part = await tool_context.load_artifact(filename=filename)

        if not artifact_part or not artifact_part.inline_data:
            return json.dumps({
                "status": "error",
                "filename": filename,
                "path": "",
                "type": "unknown",
                "size": 0,
                "message": f"❌ Could not load artifact data: {filename}"
            }, indent=2)

        data_size = len(artifact_part.inline_data.data)

        # For text-based files (CSV, TXT, JSON, EXTXYZ), return content directly
        try:
            # Try to decode as text
            content = artifact_part.inline_data.data.decode('utf-8')

            # Determine text file type
            file_type = "text"
            if filename.lower().endswith(('.json',)):
                file_type = "json"
            elif filename.lower().endswith(('.csv',)):
                file_type = "csv"
            elif filename.lower().endswith(('.extxyz',)):
                file_type = "extxyz"

            return json.dumps({
                "status": "success",
                "filename": filename,
                "path": "",  # Text files don't need a path
                "type": file_type,
                "size": data_size,
                "content": content,
                "message": f"✅ Text artifact loaded: {filename}"
            }, indent=2)

        except UnicodeDecodeError:
            # Binary file - handle different types
            temp_dir = Path.cwd() / 'uploaded_files'
            temp_dir.mkdir(exist_ok=True)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # For images
                file_ext = Path(filename).suffix or '.png'
                temp_file_path = temp_dir / f"{filename}_temp{file_ext}"

                with open(temp_file_path, 'wb') as f:
                    f.write(artifact_part.inline_data.data)

                return json.dumps({
                    "status": "success",
                    "filename": filename,
                    "path": str(temp_file_path.absolute()),
                    "type": "image",
                    "size": data_size,
                    "message": f"✅ IMAGE ARTIFACT READY: {temp_file_path.absolute()}\nThis image artifact is ready for vision analysis."
                }, indent=2)

            elif filename.lower().endswith(('.pt', '.pth', '.ckpt')):
                # For PyTorch models
                file_ext = Path(filename).suffix
                temp_file_path = temp_dir / filename

                with open(temp_file_path, 'wb') as f:
                    f.write(artifact_part.inline_data.data)

                return json.dumps({
                    "status": "success",
                    "filename": filename,
                    "path": str(temp_file_path.absolute()),
                    "type": "model",
                    "size": data_size,
                    "message": f"✅ MODEL ARTIFACT READY: {temp_file_path.absolute()}\nPyTorch model saved and ready for training/inference."
                }, indent=2)
            else:
                # Other binary files
                temp_file_path = temp_dir / filename

                with open(temp_file_path, 'wb') as f:
                    f.write(artifact_part.inline_data.data)

                return json.dumps({
                    "status": "success",
                    "filename": filename,
                    "path": str(temp_file_path.absolute()),
                    "type": "binary",
                    "size": data_size,
                    "message": f"✅ Binary artifact loaded: {filename}"
                }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "filename": filename,
            "path": "",
            "type": "unknown",
            "size": 0,
            "message": f"❌ Error accessing artifact: {str(e)}\n\nMake sure the file has been properly uploaded and promoted to artifact."
        }, indent=2)


async def get_artifact_path(tool_context: ToolContext, filename: str) -> str:
    """
    Get the local file path for an artifact.
    This is a convenience function that saves any artifact to uploaded_files and returns the path.

    Useful when you need to pass a file path to other tools (like training functions).

    Returns a JSON string with:
    {
        "status": "success" | "error",
        "filename": str,
        "path": str,  # Full absolute path to the saved file
        "message": str
    }
    """
    try:
        # List available artifacts
        artifact_keys = await tool_context.list_artifacts()

        if filename not in artifact_keys:
            available = ', '.join(artifact_keys) if artifact_keys else 'No artifacts available'
            return json.dumps({
                "status": "error",
                "filename": filename,
                "path": "",
                "message": f"❌ Artifact not found: {filename}\n\nAvailable: {available}"
            }, indent=2)

        # Load the artifact
        artifact_part = await tool_context.load_artifact(filename=filename)

        if not artifact_part or not artifact_part.inline_data:
            return json.dumps({
                "status": "error",
                "filename": filename,
                "path": "",
                "message": f"❌ Could not load artifact: {filename}"
            }, indent=2)

        # Save to temp directory
        temp_dir = Path.cwd() / 'uploaded_files'
        temp_dir.mkdir(exist_ok=True)

        temp_file_path = temp_dir / filename

        with open(temp_file_path, 'wb') as f:
            f.write(artifact_part.inline_data.data)

        return json.dumps({
            "status": "success",
            "filename": filename,
            "path": str(temp_file_path.absolute()),
            "message": f"✅ File saved to: {temp_file_path.absolute()}"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "filename": filename,
            "path": "",
            "message": f"❌ Error: {str(e)}"
        }, indent=2)


async def cleanup_temp_artifacts(tool_context: ToolContext) -> str:
    """Clean up uploaded artifact files"""
    try:
        temp_dir = Path.cwd() / 'uploaded_files'
        if temp_dir.exists():
            # Remove all files in temp directory
            for file in temp_dir.glob('*'):
                if file.is_file():
                    file.unlink()

            # Remove directory if empty
            try:
                temp_dir.rmdir()
                return "✅ Uploaded artifact files cleaned up successfully."
            except OSError:
                return "✅ Uploaded artifact files cleaned up (directory not empty)."
        else:
            return "No uploaded artifact files to clean up."

    except Exception as e:
        return f"Error cleaning up uploaded files: {str(e)}"


async def artifact_write_tool(tool_context: ToolContext, filename: str, content: str, mime_type: str = "text/plain") -> str:
    """
    Save content as a new downloadable artifact using ToolContext.
    This creates an artifact that can be downloaded via the UI.
    
    Args:
        tool_context: The tool context for artifact operations
        filename: Name for the artifact (e.g., 'results.csv')
        content: The content to save (text or binary as string)
        mime_type: MIME type of the content (default: text/plain)
    
    Returns:
        Success message with artifact key and download info
    """
    try:
        # Determine mime type if not provided
        if mime_type == "text/plain":
            if filename.endswith('.csv'):
                mime_type = "text/csv"
            elif filename.endswith('.json'):
                mime_type = "application/json"
            elif filename.endswith('.xlsx'):
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif filename.endswith('.md'):
                mime_type = "text/markdown"
            elif filename.endswith('.extxyz'):
                mime_type = "text/plain"
            elif filename.endswith(('.pt', '.pth', '.ckpt')):
                mime_type = "application/octet-stream"
        
        # Convert content to bytes if string
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        
        # Create the artifact using types.Part.from_bytes
        artifact_part = types.Part(
            inline_data=types.Blob(
                mime_type=mime_type,
                data=content_bytes,
            )
        )
        
        # Save the artifact using ToolContext
        await tool_context.save_artifact(filename=filename, artifact=artifact_part)
        
        result = f"✅ **File saved as downloadable artifact!**\n\n"
        result += f"📁 **Filename:** `{filename}`\n"
        result += f"📊 **Type:** {mime_type}\n"
        result += f"📏 **Size:** {len(content_bytes)} bytes\n\n"
        result += f"💾 **To download:** The file is now available in your artifacts list.\n"
        result += f"You can download it from the UI or use the download endpoint.\n\n"
        result += f"🔗 **Artifact Key:** `{filename}`"
        
        return result
        
    except Exception as e:
        return f"❌ Error creating artifact: {str(e)}"


# Create the tools
list_files_and_artifacts_tool = FunctionTool(func=list_available_files)
file_read_artifact_tool = FunctionTool(func=file_read_artifact)
get_artifact_path_tool = FunctionTool(func=get_artifact_path)
cleanup_temp_artifacts_tool = FunctionTool(func=cleanup_temp_artifacts)
artifact_write_tool = FunctionTool(func=artifact_write_tool)
