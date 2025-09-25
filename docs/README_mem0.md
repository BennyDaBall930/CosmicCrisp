# Mem0 Local Server Setup Guide

This guide provides instructions for setting up a local mem0 server instance with preloaded Apple Zero contextual memories to enhance CosmicCrisp's AI capabilities.

## Overview

Mem0 is an advanced memory management system that provides persistent, contextual memory capabilities for AI agents. This implementation includes:

- Local SQLite-backed storage for offline operation
- Preloaded Apple Zero contextual memories for enhanced reasoning
- Automatic fallback to SQLite when mem0 service is unavailable
- Seamless integration with existing CosmicCrisp memory pipeline

## Prerequisites

- Python 3.8+
- CosmicCrisp environment
- mem0ai>=0.1.10 (included in requirements.txt)

## Quick Start

### 1. Enable Local Mem0 Mode

Add the following environment variables to your `.env` file:

```env
# Mem0 Configuration for Local Mode
MEM0_LOCAL_MODE=true
MEM0_NAMESPACE=applezero
MEM0_BASE_URL=
MEM0_API_KEY=
```

### 2. Run Setup Script

Execute the setup script to initialize mem0 and preload memories:

```bash
cd CosmicCrisp
python scripts/setup_mem0.py
```

This script will:
- Initialize a local mem0 instance with SQLite backend
- Load Apple Zero contextual memories
- Verify the setup and integration

### 3. Verify Integration

Check the logs to confirm successful initialization. You should see messages like:
```
✓ Mem0 local instance initialized successfully
✓ Successfully preloaded X Apple Zero memories
✓ Mem0 integration verified (X memories found)
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MEM0_LOCAL_MODE` | Enable local mode (true/1/yes) | false | Yes for local mode |
| `MEM0_NAMESPACE` | Memory namespace/user ID | applezero | Optional |
| `MEM0_BASE_URL` | mem0 server URL | None | Optional |
| `MEM0_API_KEY` | mem0 API key | None | Optional |

### Local vs Cloud Mode

**Local Mode** (recommended for offline):
- Uses SQLite for storage
- Preloads Apple Zero memories
- Works without internet connection
- Faster for repeated queries

**Cloud Mode**:
- Connects to external mem0 service
- Higher-dimensional memory embeddings
- Requires API key and connectivity
- Better for collaborative scenarios

## Memory Management

### Memory Categories

The preloaded Apple Zero memories include context about:

- **expertise**: Programming languages, frameworks, tools
- **methodology**: Problem-solving approaches, systematic thinking
- **platform_integration**: How Apple Zero works with CosmicCrisp
- **technical_skills**: Languages, frameworks, cloud services
- **memory_systems**: How Apple Zero uses memory-based reasoning
- **ml_expertise**: MLX framework, Apple Silicon optimization
- **architecture**: System design patterns, scalability
- **code_quality**: Best practices, testing, CI/CD
- **enterprise_practices**: Security, monitoring, deployment
- **ai_ml_practices**: RAG, fine-tuning, prompt engineering

### Memory Operations

The mem0 adapter supports:
- **Semantic Search**: `similar(query, k)` - Find relevant memories by meaning
- **Recent Access**: `recent(k)` - Get most recent memories
- **Add Memory**: `add(item)` - Store new contextual information
- **Delete Memory**: `delete(id)` - Remove specific memories
- **Memory Count**: `count()` - Get total number of stored memories

### Fallback Behavior

When mem0 is unavailable, the system automatically falls back to SQLite:
- All operations continue working
- Memory persistence is maintained
- Performance may be reduced but functionality is preserved

## Troubleshooting

### Common Issues

**"mem0 client not installed"**
- Ensure mem0ai>=0.1.10 is in requirements.txt
- Run `pip install -r requirements.txt`
- Verify Python environment

**Setup script fails**
- Check MEM0_LOCAL_MODE is set to true
- Ensure scripts directory is executable
- Verify Python path and imports

**No memories preloaded**
- Check `scripts/mem0_preload_data.json` exists
- Verify JSON format is valid
- Check setup script logs for errors

**Memory search not working**
- Verify mem0 initialization completed successfully
- Check namespace configuration
- Ensure fallback store is properly configured

### Debug Commands

```bash
# Test mem0 connection
python -c "
from cosmiccrisp.python.runtime.memory.stores.mem0_adapter import Mem0Adapter
import asyncio

async def test():
    adapter = Mem0Adapter()
    count = await adapter.count()
    print(f'Total memories: {count}')

asyncio.run(test())
"
```

### Log Levels

Set logging level to see detailed mem0 operations:

```python
import logging
logging.getLogger('cosmiccrisp.python.runtime.memory.stores.mem0_adapter').setLevel(logging.DEBUG)
```

## Automated Setup

The mem0 setup is automatically included when running the main setup script:

```bash
./dev/macos/setup.sh
```

This will install all dependencies and configure mem0 with Apple Zero memories.

## Security Considerations

- API keys (if used) should be stored securely
- Local SQLite database contains sensitive conversation data
- Use appropriate file permissions for memory files
- Consider encryption for production deployments

## References

- [Mem0 Documentation](https://docs.mem0.ai/)
- [CosmicCrisp Memory Architecture](./docs/architecture.md)
- [Implementation Plan](./implementation_plan.md)
