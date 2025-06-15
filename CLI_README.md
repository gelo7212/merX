# merX CLI - Production-Ready Command Line Interface

The merX CLI is a comprehensive command-line interface for the merX Memory System, designed for both testing and production environments.

## Features

- **Multi-Environment Support**: Development, Testing, and Production modes
- **Configuration Management**: JSON-based configuration with environment-specific defaults
- **Comprehensive Logging**: File and console logging with configurable levels
- **Production Safety**: Error handling, graceful shutdown, and health monitoring
- **Interactive Mode**: Real-time memory operations with command history
- **Automated Testing**: Built-in test suite for system verification
- **Data Management**: Backup, recovery, and statistics monitoring

## Quick Start

### Development Mode (Default)
```bash
# Start interactive mode
python merx_cli.py --interactive

# Quick insert and search
python merx_cli.py --insert "Python is great for AI" --tags python,AI
python merx_cli.py --search "artificial intelligence"
```

### Production Mode
```bash
# Production interactive mode (minimal output)
python merx_cli.py --env production --interactive

# Production operations with JSON output
python merx_cli.py --env production --search "AI" --limit 5
python merx_cli.py --env production --stats
python merx_cli.py --env production --health
```

### Testing Mode
```bash
# Run comprehensive tests
python merx_cli.py --env testing --run-tests

# Testing with custom config
python merx_cli.py --env testing --config config/testing.json --interactive
```

## Environment Types

### Development (default)
- RAM Capacity: 100,000 nodes
- Logging: INFO level to console and file
- Output: Verbose with emojis and formatting
- Data Path: `data/memory.mex`

### Testing
- RAM Capacity: 10,000 nodes
- Logging: DEBUG level
- Data Path: `data/test_memory.mex`
- Fast flush intervals for testing

### Production
- RAM Capacity: 500,000 nodes
- Logging: WARNING level only
- Output: JSON format only
- Enhanced performance settings
- Auto-backup enabled

## Configuration

### Using Configuration Files
```bash
# Use specific config file
python merx_cli.py --config config/production.json --interactive

# Save current config to file
python merx_cli.py --save-config my_config.json
```

### Command Line Overrides
```bash
# Override specific settings
python merx_cli.py --data-path "custom/path.mex" --ram-capacity 200000 --interactive
python merx_cli.py --log-level DEBUG --env production --stats
```

## Command Line Options

### Basic Operations
```bash
# Insert memory
python merx_cli.py --insert "Content here" --tags tag1,tag2 --type fact

# Search by content
python merx_cli.py --search "query here" --limit 20

# Search by tags
python merx_cli.py --search-tags tag1,tag2

# Get specific memory
python merx_cli.py --get "uuid-here"

# Show statistics
python merx_cli.py --stats

# Health check
python merx_cli.py --health

# Create backup
python merx_cli.py --backup
python merx_cli.py --backup "custom/backup/path.mex"
```

### System Management
```bash
# Run comprehensive tests
python merx_cli.py --run-tests

# Interactive mode
python merx_cli.py --interactive

# Show configuration
python merx_cli.py --interactive
# Then type: config
```

## Interactive Mode Commands

Once in interactive mode (`--interactive`), you can use these commands:

### Memory Operations
```
insert Hello world #greeting #test
insert Python is great for AI #programming #python #AI
insert Remember to buy milk #todo #shopping
```

### Search Operations
```
search programming
search AI python
search --tags todo,shopping
```

### System Commands
```
get 12345678-1234-5678-9012-123456789abc
stats
health
backup
backup custom/path.mex
config
help
quit
```

## Production Usage

### JSON Output
In production mode, all output is in JSON format for easy parsing:

```bash
# Search returns JSON array
python merx_cli.py --env production --search "AI"
# Output: [{"id": "uuid", "content": "...", "activation": 0.95}]

# Stats returns JSON object
python merx_cli.py --env production --stats
# Output: {"total_memories": 1000, "ram_usage_mb": 45.2, ...}

# Health check returns status
python merx_cli.py --env production --health
# Output: {"health_check": "passed"}
```

### Error Handling
Production mode returns proper exit codes:
- `0`: Success
- `1`: Error occurred

### Logging
Production logs are written to `logs/merx_prod.log` with WARNING level and above.

## Configuration Reference

### Environment-Specific Defaults

**Development:**
- `ram_capacity`: 100000
- `log_level`: "INFO"
- `data_path`: "data/memory.mex"
- `flush_interval`: 5.0

**Testing:**
- `ram_capacity`: 10000
- `log_level`: "DEBUG"
- `data_path`: "data/test_memory.mex"
- `flush_interval`: 1.0

**Production:**
- `ram_capacity`: 500000
- `log_level`: "WARNING"
- `data_path`: "data/prod_memory.mex"
- `flush_interval`: 2.0
- `auto_backup`: true

### Full Configuration Options
```json
{
  "data_path": "data/memory.mex",
  "ram_capacity": 100000,
  "flush_interval": 5.0,
  "flush_threshold": 100,
  "log_level": "INFO",
  "log_file": "logs/merx_cli.log",
  "enable_compression": true,
  "compression_level": 5,
  "enable_distributed": false,
  "shard_count": 4,
  "activation_threshold": 0.01,
  "spreading_decay": 0.7,
  "max_hops": 3,
  "decay_interval": 60,
  "performance_monitoring": true,
  "auto_backup": true,
  "backup_interval": 3600
}
```

## Examples

### Building a Knowledge Base
```bash
# Start interactive mode
python merx_cli.py --interactive

# Add knowledge
insert Machine learning is a subset of AI #AI #ML #definition
insert Python has excellent ML libraries #python #ML #libraries
insert Neural networks mimic brain structure #AI #neural #brain

# Search knowledge
search neural networks
search --tags AI,ML
```

### Production Monitoring
```bash
# Health monitoring script
#!/bin/bash
python merx_cli.py --env production --health
if [ $? -eq 0 ]; then
    echo "System healthy"
else
    echo "System unhealthy" >&2
    exit 1
fi
```

### Automated Backup
```bash
# Daily backup script
#!/bin/bash
timestamp=$(date +%Y%m%d_%H%M%S)
python merx_cli.py --env production --backup "backups/daily_backup_$timestamp.mex"
```

## Troubleshooting

### Common Issues

**Memory not found:**
- Check the UUID format
- Verify the memory exists: `python merx_cli.py --stats`

**Performance issues:**
- Check RAM capacity: increase `--ram-capacity`
- Review configuration: use `config` command in interactive mode
- Run health check: `python merx_cli.py --health`

**Configuration problems:**
- Validate JSON config files
- Check file permissions for data and log directories
- Use `--log-level DEBUG` for detailed error information

### Debug Mode
```bash
# Enable debug logging
python merx_cli.py --log-level DEBUG --interactive

# Check system status
python merx_cli.py --health --stats
```

## Integration Examples

### Bash Scripts
```bash
#!/bin/bash
# Add memory from command line
python merx_cli.py --env production --insert "$1" --tags "$2"
```

### Python Integration
```python
import subprocess
import json

# Search memories
result = subprocess.run([
    'python', 'merx_cli.py', '--env', 'production', 
    '--search', 'AI', '--limit', '5'
], capture_output=True, text=True)

memories = json.loads(result.stdout)
```

## Performance Guidelines

### Development
- Suitable for < 100K memories
- Interactive exploration and testing

### Testing
- Optimized for test suites
- Fast startup and shutdown
- Small memory footprint

### Production
- Handles 100K+ memories efficiently
- Optimized for throughput
- Minimal resource usage
- Automated monitoring and backup

---

For more information, see the main merX documentation in `IMPLEM.md`.
