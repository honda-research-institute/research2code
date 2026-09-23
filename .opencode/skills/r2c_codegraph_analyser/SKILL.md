---
name: r2c_codegraph_analyser
description: Use when navigating, understanding, or debugging the R2C codebase. Provides structural code intelligence via the MCP codegraph tools — call chains, complexity, dead code, dependency graphs. DO NOT load for routine file reads (use grep/glob/read instead); load only when you need code-level relationships that text search cannot provide.
---

## When to Load This Skill

Load this skill when the task requires understanding **relationships between code objects** rather than reading individual files:

- "Why is function X slow?" → cyclomatic complexity
- "What calls function Y?" / "What does Y call?" → call graph
- "Is this code actually used?" → dead code detection
- "How do modules A and B relate?" → dependency graph
- "What's the overall structure of this project?" → repository stats
- "I need to find a function/class by keyword" → code search

Do NOT load for:
- Reading a specific file's contents (use `read`)
- Finding files by name pattern (use `glob`)
- Searching file contents for a string/regex (use `grep`)
- Understanding a single file's logic in isolation

## Available Tools

### Finding and Exploring Code

| Tool | Use When | Example |
|------|----------|---------|
| `find_code` | Keyword search across the entire codebase | Find all references to `train_policy` |
| `list_indexed_repositories` | Check what repos are indexed | Verify R2C repo is indexed |
| `get_repository_stats` | Get counts of files, functions, classes | "How many functions are in R2C?" |

### Analyzing Relationships

| Tool | Use When | Example |
|------|----------|---------|
| `analyze_code_relationships` | Call chains, imports, inheritance, modifiers | "Who calls `run_pipeline`?" |
| `calculate_cyclomatic_complexity` | Measure function complexity | "How complex is `stage_review_focus`?" |
| `find_datasource_nodes` | Database/Redis/MySQL schemas | "What tables does this use?" |
| `find_dead_code` | Find potentially unused functions | "What code is never called?" |

### Advanced Queries

| Tool | Use When | Example |
|------|----------|---------|
| `execute_cypher_query` | Custom graph queries | Complex multi-hop relationships |
| `visualize_graph_query` | Generate visualization URL | Interactive exploration of relationships |

## Common Query Types for `analyze_code_relationships`

- **`find_callers`** — who calls this function (direct callers only)
- **`find_all_callers`** — full call chain (recursive)
- **`find_callees`** — what functions this one calls
- **`find_all_callees`** — full call chain downstream (recursive)
- **`find_importers`** — what modules import this module
- **`who_modifies`** — what code modifies this variable
- **`class_hierarchy`** — inheritance tree for a class
- **`overrides`** — which methods override a parent method
- **`dead_code`** — potentially unused code
- **`call_chain`** — full path from caller to callee
- **`module_deps`** — module dependency graph
- **`variable_scope`** — where a variable is defined and used
- **`find_complexity`** — complexity analysis for a function
- **`find_functions_by_argument`** — find functions by parameter name
- **`find_functions_by_decorator`** — find functions by decorator

## Practical R2C Examples

### Debugging a pipeline stage failure

```
1. find_code(query="validate_method_spec")    # locate the validator
2. analyze_code_relationships(                 # who uses it?
     query_type="find_callers",
     target="validate_method_spec.main")
3. calculate_cyclomatic_complexity(            # is it complex?
     function_name="main",
     path="scripts/validate_method_spec.py")
```

### Understanding the orchestrator architecture

```
1. get_repository_stats()                      # overall structure
2. find_code(query="dispatch")                 # find dispatch logic
3. analyze_code_relationships(                 # call chain
     query_type="call_chain",
     target="dispatch_producer")
4. analyze_code_relationships(                 # dependencies
     query_type="module_deps",
     target="run_pipeline")
```

### Finding unused code before refactoring

```
1. find_dead_code()                            # dead code scan
2. analyze_code_relationships(                 # verify a candidate
     query_type="find_all_callers",
     target="deprecated_function_name")
```

## Best Practices

1. **Start broad, then narrow.** Use `find_code` to locate relevant code, then use relationship tools to understand connections.
2. **Use `context` or `repo_path`** when available to scope queries and improve precision.
3. **Prefer specific tools over Cypher.** Only use `execute_cypher_query` when the higher-level tools can't express your question.
4. **Validate findings with file reads.** The codegraph is an index — always read the actual source to confirm before making changes.
5. **Check repository stats first** when unfamiliar with a codebase to understand its scale and structure.
